import struct

class BitWriter:
    def __init__(self):
        self.bytes = bytearray()
        self.buffer = 0
        self.bits_in_buffer = 0

    def write_bits(self, value, num_bits):
        if num_bits == 0:
            return
        value &= (1 << num_bits) - 1
        self.buffer = (self.buffer << num_bits) | value
        self.bits_in_buffer += num_bits
        while self.bits_in_buffer >= 8:
            self.bits_in_buffer -= 8
            self.bytes.append((self.buffer >> self.bits_in_buffer) & 0xFF)
        self.buffer &= (1 << self.bits_in_buffer) - 1

    def flush(self):
        if self.bits_in_buffer > 0:
            self.bytes.append((self.buffer << (8 - self.bits_in_buffer)) & 0xFF)
            self.buffer = 0
            self.bits_in_buffer = 0
        return bytes(self.bytes)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bits(self, num_bits):
        if num_bits == 0:
            return 0
        result = 0
        bits_remaining = num_bits
        while bits_remaining > 0:
            if self.byte_pos >= len(self.data):
                raise ValueError("BitReader: no more data")
            available = 8 - self.bit_pos
            to_read = min(available, bits_remaining)
            shift = available - to_read
            result = (result << to_read) | ((self.data[self.byte_pos] >> shift) & ((1 << to_read) - 1))
            self.bit_pos += to_read
            bits_remaining -= to_read
            if self.bit_pos == 8:
                self.bit_pos = 0
                self.byte_pos += 1
        return result


def _leading_zeros(v):
    if v == 0:
        return 64
    n = 0
    while (v & (1 << 63)) == 0:
        n += 1
        v <<= 1
    return n


def _trailing_zeros(v):
    if v == 0:
        return 64
    n = 0
    while (v & 1) == 0:
        n += 1
        v >>= 1
    return n


class TimestampCompressor:
    """
    Gorilla-style delta-of-delta encoding for timestamps.

    Why it compresses so well:
    - Timestamps in time series are typically evenly spaced (e.g. every 10s).
    - The first delta = t1 - t0 captures the interval.
    - The second delta (delta-of-delta) = (t2-t1) - (t1-t0) is almost always 0
      for regular series, or very small for slight jitter.
    - Instead of storing 64 bits per timestamp, we store:
        * 1 bit '0' → delta-of-delta = 0 (most common case!)
        * '10' + 7 bits → value fits in [-64,63]
        * '110' + 9 bits → fits in [-256,255]
        * '1110' + 12 bits → fits in [-2048,2047]
        * '11110' + 16 bits → fits in [-32768,32767]
        * '111110' + 32 bits → large delta-of-delta
    - For a 1-day series at 10s intervals: 8640 timestamps.
      Raw: 8640 * 64 = 552,960 bits.
      Compressed: ~8640 * 1 bit = 8,640 bits (99% compression!).
    """
    def __init__(self):
        self.writer = BitWriter()
        self.prev_ts = None
        self.prev_delta = None
        self.first = True

    def append(self, ts):
        if self.first:
            self.writer.write_bits(ts, 64)
            self.prev_ts = ts
            self.first = False
            return
        delta = ts - self.prev_ts
        if self.prev_delta is None:
            self.writer.write_bits(1, 1)
            self.writer.write_bits(delta, 64)
            self.prev_delta = delta
            self.prev_ts = ts
            return
        dod = delta - self.prev_delta
        self.prev_delta = delta
        self.prev_ts = ts
        if dod == 0:
            self.writer.write_bits(0, 1)
        elif -63 <= dod <= 64:
            self.writer.write_bits(0b10, 2)
            self.writer.write_bits(dod + 63, 7)
        elif -255 <= dod <= 256:
            self.writer.write_bits(0b110, 3)
            self.writer.write_bits(dod + 255, 9)
        elif -2047 <= dod <= 2048:
            self.writer.write_bits(0b1110, 4)
            self.writer.write_bits(dod + 2047, 12)
        elif -32767 <= dod <= 32768:
            self.writer.write_bits(0b11110, 5)
            self.writer.write_bits(dod + 32767, 16)
        else:
            self.writer.write_bits(0b111110, 6)
            self.writer.write_bits(dod & 0xFFFFFFFF, 32)

    def finish(self):
        return self.writer.flush()


class TimestampDecompressor:
    def __init__(self, data):
        self.reader = BitReader(data)
        self.prev_ts = None
        self.prev_delta = None
        self.first = True
        self.second = True

    def read(self):
        if self.first:
            self.prev_ts = self.reader.read_bits(64)
            self.first = False
            return self.prev_ts
        if self.second:
            flag = self.reader.read_bits(1)
            delta = self.reader.read_bits(64)
            self.prev_delta = delta
            self.prev_ts = self.prev_ts + delta
            self.second = False
            return self.prev_ts
        prefix = 0
        while True:
            bit = self.reader.read_bits(1)
            if bit == 0:
                break
            prefix += 1
            if prefix >= 6:
                break
        if prefix == 0:
            dod = 0
        elif prefix == 1:
            dod = self.reader.read_bits(7) - 63
        elif prefix == 2:
            dod = self.reader.read_bits(9) - 255
        elif prefix == 3:
            dod = self.reader.read_bits(12) - 2047
        elif prefix == 4:
            dod = self.reader.read_bits(16) - 32767
        else:
            raw = self.reader.read_bits(32)
            if raw & (1 << 31):
                raw -= (1 << 32)
            dod = raw
        delta = self.prev_delta + dod
        self.prev_delta = delta
        self.prev_ts = self.prev_ts + delta
        return self.prev_ts


class ValueCompressor:
    """
    Gorilla-style XOR compression for floating-point values.

    Why it compresses so well:
    - Consecutive float values in time series are often very similar
      (e.g. temperature changes slowly).
    - XOR of two similar IEEE 754 doubles produces a value with:
        * Many leading zero bits (high bits identical)
        * Many trailing zero bits (low bits identical)
    - Instead of 64 bits per value, we store:
        * 1st value: raw 64 bits (only once)
        * Subsequent: XOR with previous value, then:
          - '0' bit → XOR is 0 (value unchanged!) → 1 bit total
          - '10' + 6-bit leading-zero count + 6-bit meaningful-bit count
            + meaningful bits → only the changing region
          - '11' → same leading/trailing zero pattern as last time
            (just store the meaningful bits)
    - For slowly-changing values, XOR often has ~10 meaningful bits.
      So: 1+6+6+10 = 23 bits vs 64 bits → ~64% compression per value.
    - Combined with timestamp compression, overall ratio can reach 90%+.
    """
    def __init__(self):
        self.writer = BitWriter()
        self.prev_raw = None
        self.prev_leading = -1
        self.prev_trailing = -1
        self.first = True

    def append(self, value):
        raw = struct.unpack('>Q', struct.pack('>d', value))[0]
        if self.first:
            self.writer.write_bits(raw, 64)
            self.prev_raw = raw
            self.first = False
            return
        xor = raw ^ self.prev_raw
        self.prev_raw = raw
        if xor == 0:
            self.writer.write_bits(0, 1)
            return
        self.writer.write_bits(1, 1)
        leading = _leading_zeros(xor)
        trailing = _trailing_zeros(xor)
        if self.prev_leading >= 0 and leading >= self.prev_leading and trailing >= self.prev_trailing:
            self.writer.write_bits(1, 1)
            meaningful_bits = 64 - self.prev_leading - self.prev_trailing
            self.writer.write_bits(xor >> self.prev_trailing, meaningful_bits)
        else:
            self.writer.write_bits(0, 1)
            self.writer.write_bits(leading, 6)
            meaningful_bits = 64 - leading - trailing
            self.writer.write_bits(meaningful_bits - 1, 6)
            if meaningful_bits > 0:
                self.writer.write_bits(xor >> trailing, meaningful_bits)
            self.prev_leading = leading
            self.prev_trailing = trailing

    def finish(self):
        return self.writer.flush()


class ValueDecompressor:
    def __init__(self, data):
        self.reader = BitReader(data)
        self.prev_raw = None
        self.prev_leading = -1
        self.prev_trailing = -1
        self.first = True

    def read(self):
        if self.first:
            raw = self.reader.read_bits(64)
            self.prev_raw = raw
            self.first = False
            return struct.unpack('>d', struct.pack('>Q', raw))[0]
        is_zero = self.reader.read_bits(1)
        if is_zero == 0:
            return struct.unpack('>d', struct.pack('>Q', self.prev_raw))[0]
        control = self.reader.read_bits(1)
        if control == 1:
            meaningful_bits = 64 - self.prev_leading - self.prev_trailing
            xor = self.reader.read_bits(meaningful_bits) << self.prev_trailing
        else:
            leading = self.reader.read_bits(6)
            meaningful_bits = self.reader.read_bits(6) + 1
            trailing = 64 - leading - meaningful_bits
            xor = self.reader.read_bits(meaningful_bits) << trailing
            self.prev_leading = leading
            self.prev_trailing = trailing
        raw = self.prev_raw ^ xor
        self.prev_raw = raw
        return struct.unpack('>d', struct.pack('>Q', raw))[0]
