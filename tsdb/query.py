from tsdb.aggregation import aggregate
from tsdb.downsample import PREDEFINED_GRANULARITIES


class QueryResult:
    def __init__(self, series_id, tags, points):
        self.series_id = series_id
        self.tags = tags
        self.points = points

    def __repr__(self):
        n = len(self.points)
        preview = self.points[:5] if n > 5 else self.points
        pts_str = ', '.join(f'({t}, {v:.2f})' for t, v in preview)
        if n > 5:
            pts_str += f', ... (+{n-5} more)'
        return f'QueryResult(series={self.series_id}, points=[{pts_str}])'


class QueryEngine:
    """
    Query engine with time range filtering, tag matching, aggregation,
    and multi-granularity downsampling.

    Query execution flow:
    ┌────────────────────────────────────────────────────────────┐
    │ 1. TagIndex.match(metric, tag_filters) → set[series_id]   │
    │                                                            │
    │ 2. For each series_id:                                     │
    │    a. Find relevant chunks (skip non-overlapping)          │
    │    b. If downsampling requested:                           │
    │       - Try pre-aggregated data first                      │
    │       - Stitch with raw data for partial buckets           │
    │    c. Else:                                                │
    │       - Stream decompressed points from relevant chunks    │
    │                                                            │
    │ 3. If aggregation (no downsampling):                       │
    │    - Collect all points, apply agg function                │
    │                                                            │
    │ 4. If cross-series aggregation:                            │
    │    - Merge points from all matching series                 │
    │    - Group by time bucket, apply agg function              │
    └────────────────────────────────────────────────────────────┘
    """

    def __init__(self, storage):
        self.storage = storage

    def query(self, metric, tag_filters=None, t_start=0, t_end=2**63,
              agg_func=None, interval=None):
        """
        Execute a query.

        Args:
            metric: metric name
            tag_filters: dict of {tag_key: tag_value_or_condition}
            t_start: start timestamp (inclusive)
            t_end: end timestamp (inclusive)
            agg_func: aggregation function name (sum, avg, min, max, count)
            interval: downsampling interval in seconds (None = no downsampling)

        Returns:
            list of QueryResult, one per matching series
        """
        if tag_filters is None:
            tag_filters = {}
        matching = self.storage.index.match(metric, tag_filters)
        if not matching:
            return []
        results = []
        for series_id in sorted(matching):
            tags = self.storage.index.get_series_tags(series_id)
            if interval is not None and interval > 0:
                points = self._query_downsampled(series_id, t_start, t_end,
                                                  agg_func or 'avg', interval)
            elif agg_func is not None:
                points = self._query_aggregate(series_id, t_start, t_end, agg_func)
            else:
                points = self._query_raw(series_id, t_start, t_end)
            results.append(QueryResult(series_id, tags, points))
        if agg_func and len(results) > 1 and interval is None:
            merged = self._cross_series_aggregate(results, agg_func)
            return [QueryResult(f'{metric}::__aggregate__', {}, merged)]
        return results

    def _query_raw(self, series_id, t_start, t_end):
        return list(self.storage.read_series_range(series_id, t_start, t_end))

    def _query_aggregate(self, series_id, t_start, t_end, agg_func):
        points = list(self.storage.read_series_range(series_id, t_start, t_end))
        if not points:
            return []
        val = aggregate(points, agg_func)
        return [(points[0][0], val)]

    def _query_downsampled(self, series_id, t_start, t_end, agg_func, interval):
        """
        Downsampled query with seamless pre-agg / raw-data stitching.

        Algorithm:
        1. Find the best matching pre-aggregation granularity:
           - Pick the coarsest granularity g where g <= interval.
           - E.g., for interval=1800s (30min), pick g=900s (15min)
             then aggregate 2 pre-agg buckets into each output bucket.
        2. Fetch complete pre-agg buckets covering [t_start, partial_start).
        3. For the partial bucket region [partial_start, t_end]:
           - Read raw data from relevant chunks.
           - Aggregate on the fly into the output interval.
        4. Merge both sets of results.
        """
        best_gran = self._find_best_granularity(interval)
        if best_gran is not None:
            complete, partial_info = self.storage.downsample.get_all_for_series(
                series_id, best_gran, t_start, t_end, agg_func
            )
            partial_key, partial_bucket = partial_info
            raw_points = []
            if partial_bucket is not None:
                raw_t_start = partial_key
                raw_points = list(self.storage.read_series_range(
                    series_id, raw_t_start, t_end
                ))
            else:
                uncovered = self._find_uncovered_range(
                    complete, best_gran, t_start, t_end
                )
                if uncovered:
                    raw_points = list(self.storage.read_series_range(
                        series_id, uncovered[0], uncovered[1]
                    ))
            if best_gran != interval:
                complete = self._reaggregate(complete, best_gran, interval, agg_func)
            raw_bucketed = self._bucket_raw(raw_points, interval, agg_func)
            merged = self._merge_results(complete, raw_bucketed)
            return merged
        return self._bucket_raw(
            list(self.storage.read_series_range(series_id, t_start, t_end)),
            interval, agg_func
        )

    def _find_best_granularity(self, interval):
        best = None
        for name, gran in sorted(PREDEFINED_GRANULARITIES.items(), key=lambda x: -x[1]):
            if gran <= interval:
                best = name
                break
        return best

    def _find_uncovered_range(self, complete_points, gran_name, t_start, t_end):
        if not complete_points:
            return (t_start, t_end)
        gran = PREDEFINED_GRANULARITIES[gran_name]
        last_covered = max(ts for ts, _ in complete_points) + gran
        if last_covered < t_end:
            return (last_covered, t_end)
        return None

    def _reaggregate(self, points, src_gran, dst_interval, agg_func):
        if not points:
            return []
        src_interval = PREDEFINED_GRANULARITIES[src_gran]
        if src_interval == dst_interval:
            return points
        buckets = {}
        for ts, val in points:
            bucket_ts = (ts // dst_interval) * dst_interval
            buckets.setdefault(bucket_ts, []).append(val)
        results = []
        for bts in sorted(buckets.keys()):
            vals = buckets[bts]
            if agg_func == 'sum':
                results.append((bts, sum(vals)))
            elif agg_func == 'avg':
                results.append((bts, sum(vals) / len(vals)))
            elif agg_func == 'min':
                results.append((bts, min(vals)))
            elif agg_func == 'max':
                results.append((bts, max(vals)))
            elif agg_func == 'count':
                results.append((bts, sum(vals)))
            else:
                results.append((bts, sum(vals) / len(vals)))
        return results

    def _bucket_raw(self, raw_points, interval, agg_func):
        if not raw_points:
            return []
        buckets = {}
        for ts, val in raw_points:
            bucket_ts = (ts // interval) * interval
            buckets.setdefault(bucket_ts, []).append((ts, val))
        results = []
        for bts in sorted(buckets.keys()):
            results.append((bts, aggregate(buckets[bts], agg_func)))
        return results

    def _merge_results(self, pre_agg, raw_bucketed):
        seen = set()
        merged = []
        for ts, val in pre_agg:
            merged.append((ts, val))
            seen.add(ts)
        for ts, val in raw_bucketed:
            if ts not in seen:
                merged.append((ts, val))
        merged.sort(key=lambda x: x[0])
        return merged

    def _cross_series_aggregate(self, results, agg_func):
        all_points = []
        for r in results:
            all_points.extend(r.points)
        all_points.sort(key=lambda x: x[0])
        if not all_points:
            return []
        val = aggregate(all_points, agg_func)
        return [(all_points[0][0], val)]
