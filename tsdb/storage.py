from tsdb.chunk import Chunk
from tsdb.index import TagIndex
from tsdb.downsample import DownsampleStore, PREDEFINED_GRANULARITIES


class StorageEngine:
    """
    Storage engine: manages chunks, tag index, and pre-aggregations.

    Architecture overview:
    ┌─────────────────────────────────────────────────────┐
    │                   StorageEngine                      │
    │                                                      │
    │  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
    │  │ TagIndex  │  │ Chunks   │  │ DownsampleStore   │  │
    │  │ (inverted │  │ (time-   │  │ (multi-granularity │  │
    │  │  index)   │  │  blocked)│  │  pre-aggregation)  │  │
    │  └─────┬────┘  └────┬─────┘  └────────┬──────────┘  │
    │        │             │                  │             │
    │        └─────────────┴──────────────────┘             │
    │                      │                                │
    │              ┌───────┴───────┐                        │
    │              │  QueryEngine  │                        │
    │              └───────────────┘                        │
    └─────────────────────────────────────────────────────┘

    Write path:
      write(metric, tags, ts, value)
        → TagIndex.add_series(metric, tags) → series_id
        → Chunk.append(ts, value) into current chunk for series
        → DownsampleStore.add_point(series_id, ts, value)
        → If chunk is full, seal it (compress) and create new chunk

    Read path (via QueryEngine):
      query(metric, tags, t_start, t_end, agg, interval)
        → TagIndex.match(metric, tags) → set of series_ids
        → For each series_id, find overlapping chunks
        → Read & decompress only relevant chunks
        → Apply aggregation / downsampling
    """

    def __init__(self):
        self.index = TagIndex()
        self.downsample = DownsampleStore()
        self._series_chunks = {}
        self._series_current = {}
        self._total_points = 0
        self._total_raw_bytes = 0
        self._total_compressed_bytes = 0

    def write(self, metric, tags, ts, value):
        series_id = self.index.add_series(metric, tags)
        if series_id not in self._series_chunks:
            self._series_chunks[series_id] = []
            self._series_current[series_id] = None
        current = self._series_current[series_id]
        if current is None or current.is_full():
            if current is not None:
                current.seal()
                self._total_raw_bytes += current.raw_size()
                self._total_compressed_bytes += current.compressed_size()
            chunk_id = len(self._series_chunks[series_id])
            current = Chunk(series_id, chunk_id)
            self._series_chunks[series_id].append(current)
            self._series_current[series_id] = current
        current.append(ts, value)
        self.downsample.add_point(series_id, ts, value)
        self._total_points += 1

    def flush(self):
        for series_id, current in self._series_current.items():
            if current is not None and not current._sealed:
                current.seal()
                self._total_raw_bytes += current.raw_size()
                self._total_compressed_bytes += current.compressed_size()

    def get_relevant_chunks(self, series_id, t_start, t_end):
        """
        Find only chunks overlapping [t_start, t_end].

        How this skips irrelevant data:
        - Each chunk has min_ts and max_ts metadata.
        - A chunk is relevant iff: chunk.max_ts >= t_start AND chunk.min_ts <= t_end
        - Since chunks are ordered by time within a series, we can
          binary-search for the first overlapping chunk, then scan
          forward until we pass t_end.
        - In practice with ~1000 chunks per series and a 1-hour query
          on 10-hour chunk boundaries, we read 1-2 chunks instead of
          1000 — a ~500x reduction in data read.
        """
        chunks = self._series_chunks.get(series_id, [])
        relevant = []
        for chunk in chunks:
            if chunk.max_ts is not None and chunk.max_ts < t_start:
                continue
            if chunk.min_ts is not None and chunk.min_ts > t_end:
                break
            relevant.append(chunk)
        return relevant

    def read_series_range(self, series_id, t_start, t_end):
        chunks = self.get_relevant_chunks(series_id, t_start, t_end)
        for chunk in chunks:
            if chunk._sealed:
                for ts, val in chunk.iter_range(t_start, t_end):
                    yield ts, val
            else:
                for ts, val in chunk.iter_points():
                    if ts > t_end:
                        break
                    if ts >= t_start:
                        yield ts, val

    def compression_stats(self):
        ratio = 0
        if self._total_raw_bytes > 0:
            ratio = (1 - self._total_compressed_bytes / self._total_raw_bytes) * 100
        return {
            'total_points': self._total_points,
            'raw_bytes': self._total_raw_bytes,
            'compressed_bytes': self._total_compressed_bytes,
            'compression_ratio_pct': ratio,
            'total_series': len(self._series_chunks),
        }

    def chunk_stats(self):
        total_chunks = sum(len(chunks) for chunks in self._series_chunks.values())
        sealed_chunks = sum(
            1 for chunks in self._series_chunks.values()
            for c in chunks if c._sealed
        )
        return {
            'total_chunks': total_chunks,
            'sealed_chunks': sealed_chunks,
            'open_chunks': total_chunks - sealed_chunks,
        }
