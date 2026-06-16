from tsdb.encoding import (
    TimestampCompressor, TimestampDecompressor,
    ValueCompressor, ValueDecompressor,
)

CHUNK_SIZE = 3600


class DataPoint:
    __slots__ = ('ts', 'value')

    def __init__(self, ts, value):
        self.ts = ts
        self.value = value


class Chunk:
    """
    A time-bounded data block for one time series.

    How time-based chunking enables efficient range queries:
    - Each chunk covers a fixed time span [start_time, end_time).
    - The storage engine maintains chunks sorted by time.
    - When querying [t_start, t_end), we binary-search to find only
      chunks whose time range overlaps the query range.
    - All other chunks are completely skipped — no I/O, no decompression.
    - This is analogous to how LSM-tree SSTables use bloom filters and
      min/max keys to skip irrelevant files.

    Example: 1 day of data at 10s interval = 8640 points.
    If we chunk every 3600 points (10 hours), a 1-hour query only
    reads 1 chunk instead of all 3 → 3x less data to decompress.
    """

    def __init__(self, series_id, chunk_id):
        self.series_id = series_id
        self.chunk_id = chunk_id
        self.points = []
        self._ts_compressor = TimestampCompressor()
        self._val_compressor = ValueCompressor()
        self._compressed_ts = None
        self._compressed_val = None
        self._count = 0
        self._min_ts = None
        self._max_ts = None
        self._sealed = False

    @property
    def min_ts(self):
        return self._min_ts

    @property
    def max_ts(self):
        return self._max_ts

    @property
    def count(self):
        return self._count

    def append(self, ts, value):
        if self._sealed:
            raise RuntimeError("Cannot append to sealed chunk")
        self._ts_compressor.append(ts)
        self._val_compressor.append(value)
        self.points.append(DataPoint(ts, value))
        self._count += 1
        if self._min_ts is None or ts < self._min_ts:
            self._min_ts = ts
        if self._max_ts is None or ts > self._max_ts:
            self._max_ts = ts

    def seal(self):
        if self._sealed:
            return
        self._compressed_ts = self._ts_compressor.finish()
        self._compressed_val = self._val_compressor.finish()
        self._ts_compressor = None
        self._val_compressor = None
        self.points = []
        self._sealed = True

    def is_full(self):
        return self._count >= CHUNK_SIZE

    def iter_points(self):
        if not self._sealed:
            for p in self.points:
                yield p.ts, p.value
            return
        ts_dec = TimestampDecompressor(self._compressed_ts)
        val_dec = ValueDecompressor(self._compressed_val)
        for _ in range(self._count):
            ts = ts_dec.read()
            val = val_dec.read()
            yield ts, val

    def iter_range(self, t_start, t_end):
        for ts, val in self.iter_points():
            if ts > t_end:
                break
            if ts >= t_start:
                yield ts, val

    def compressed_size(self):
        if not self._sealed:
            return 0
        return len(self._compressed_ts) + len(self._compressed_val)

    def raw_size(self):
        return self._count * 16
