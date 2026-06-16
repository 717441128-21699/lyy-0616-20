from tsdb.aggregation import aggregate
from tsdb.downsample import PREDEFINED_GRANULARITIES


class QueryResult:
    """
    Query result with statistics about how the data was computed.

    Attributes:
        series_id: unique identifier for the time series
        tags: tag key-value pairs for the series
        points: list of (timestamp, value) pairs
        stats: dictionary with performance/debug info:
            - pre_agg_buckets_used: number of output buckets from pre-agg merge
            - raw_buckets_used: number of output buckets computed from raw data
            - source_granularity: pre-agg granularity name used (e.g. '15m')
            - total_buckets: total output buckets
            - pre_agg_hit_ratio: pre_agg_buckets_used / total_buckets
    """
    def __init__(self, series_id, tags, points, stats=None):
        self.series_id = series_id
        self.tags = tags
        self.points = points
        self.stats = stats or {
            'pre_agg_buckets_used': 0,
            'raw_buckets_used': 0,
            'source_granularity': None,
            'total_buckets': len(points),
            'pre_agg_hit_ratio': 0.0,
        }

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
                points, stats = self._query_downsampled(series_id, t_start, t_end,
                                                          agg_func or 'avg', interval)
                results.append(QueryResult(series_id, tags, points, stats))
            elif agg_func is not None:
                points = self._query_aggregate(series_id, t_start, t_end, agg_func)
                results.append(QueryResult(series_id, tags, points))
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
        Downsampled query with strict boundary enforcement and weighted merging.

        ARCHITECTURE:
        ┌───────────────────────────────────────────────────────────────┐
        │  [t_start: first partial bucket]  [full pre-agg buckets]     │
        │  [last partial bucket: t_end]                                │
        │                                                               │
        │  1st & last bucket  → always from RAW DATA (strict boundary)  │
        │  Middle buckets     → pre-agg merge (count-weighted avg)      │
        │  Timestamp output  → first bucket bt = max(bt, t_start)      │
        │                       last bucket bt = bt (floor-aligned)     │
        │                       but bt must be <= t_end                  │
        └───────────────────────────────────────────────────────────────┘

        This guarantees:
        - No data outside [t_start, t_end] ever leaks in
        - First bucket timestamp never < t_start
        - Last bucket timestamp never implies data beyond t_end
        - Middle buckets truly use pre-aggregation (not raw scan)
        - avg uses count-weighted merging for uneven point distributions
        """
        best_gran = self._find_best_granularity(interval)

        first_bucket_bt = (t_start // interval) * interval
        last_bucket_bt = (t_end // interval) * interval
        total_buckets = (last_bucket_bt - first_bucket_bt) // interval + 1

        if total_buckets <= 0:
            return [], self._make_stats(0, 0, None)

        first_bucket_end = first_bucket_bt + interval - 1
        last_bucket_start = last_bucket_bt

        has_first = total_buckets >= 1
        has_last = total_buckets >= 2 and last_bucket_bt > first_bucket_bt
        has_middle = total_buckets >= 3

        pre_agg_count = 0
        raw_count = 0
        gran_name = None

        pre_agg_results = []
        raw_results = []

        if has_first:
            raw_start = max(first_bucket_bt, t_start)
            raw_end = min(first_bucket_end, t_end)
            first_points = list(self.storage.read_series_range(series_id, raw_start, raw_end))
            if first_points:
                val = aggregate(first_points, agg_func)
                first_bt_adjusted = max(first_bucket_bt, t_start)
                raw_results.append((first_bt_adjusted, val))
            raw_count += 1

        if has_last and last_bucket_bt > first_bucket_bt:
            raw_start = max(last_bucket_start, t_start)
            raw_end = min(last_bucket_start + interval - 1, t_end)
            last_points = list(self.storage.read_series_range(series_id, raw_start, raw_end))
            if last_points:
                val = aggregate(last_points, agg_func)
                raw_results.append((last_bucket_bt, val))
            raw_count += 1

        if has_middle and best_gran is not None:
            gran_interval = PREDEFINED_GRANULARITIES[best_gran]
            gran_name = best_gran

            middle_start_bt = first_bucket_bt + interval
            middle_end_bt = last_bucket_bt - interval

            for dst_bt in range(middle_start_bt, middle_end_bt + 1, interval):
                dst_end = dst_bt + interval - 1
                if dst_bt < t_start or dst_end > t_end:
                    raw_count += 1
                    raw_pts = list(self.storage.read_series_range(series_id, dst_bt, dst_end))
                    if raw_pts:
                        raw_results.append((dst_bt, aggregate(raw_pts, agg_func)))
                    continue

                if interval % gran_interval != 0:
                    raw_count += 1
                    raw_pts = list(self.storage.read_series_range(series_id, dst_bt, dst_end))
                    if raw_pts:
                        raw_results.append((dst_bt, aggregate(raw_pts, agg_func)))
                    continue

                num_sub_buckets = interval // gran_interval
                sub_buckets = []
                all_found = True
                for i in range(num_sub_buckets):
                    sub_bt = dst_bt + i * gran_interval
                    raw_b = self.storage.downsample.get_raw_buckets(
                        series_id, best_gran, sub_bt, sub_bt + gran_interval
                    )
                    if not raw_b or raw_b[0][0] != sub_bt:
                        all_found = False
                        break
                    sub_buckets.append(raw_b[0][1])
                if all_found and sub_buckets:
                    val = self._combine_bucket(sub_buckets, agg_func)
                    pre_agg_results.append((dst_bt, val))
                    pre_agg_count += 1
                else:
                    raw_count += 1
                    raw_pts = list(self.storage.read_series_range(series_id, dst_bt, dst_end))
                    if raw_pts:
                        raw_results.append((dst_bt, aggregate(raw_pts, agg_func)))
        elif has_middle and best_gran is None:
            middle_start_bt = first_bucket_bt + interval
            middle_end_bt = last_bucket_bt - interval
            for dst_bt in range(middle_start_bt, middle_end_bt + 1, interval):
                dst_end = dst_bt + interval - 1
                raw_pts = list(self.storage.read_series_range(series_id, dst_bt, dst_end))
                if raw_pts:
                    raw_results.append((dst_bt, aggregate(raw_pts, agg_func)))
                raw_count += 1

        merged = self._merge_results(pre_agg_results, raw_results)
        stats = self._make_stats(pre_agg_count, raw_count, gran_name)
        return merged, stats

    def _make_stats(self, pre_agg_count, raw_count, gran_name):
        total = pre_agg_count + raw_count
        return {
            'pre_agg_buckets_used': pre_agg_count,
            'raw_buckets_used': raw_count,
            'source_granularity': gran_name,
            'total_buckets': total,
            'pre_agg_hit_ratio': pre_agg_count / total if total > 0 else 0.0,
        }

    def _combine_bucket(self, bucket_list, agg_func):
        """
        Combine multiple raw pre-agg buckets into a single aggregate value.

        Uses count/sum/min/max from the raw bucket dicts so 'avg' is correctly
        weighted even with uneven point counts across sub-buckets.
        """
        total_count = sum(b['count'] for b in bucket_list)
        total_sum = sum(b['sum'] for b in bucket_list)
        if agg_func == 'sum':
            return total_sum
        elif agg_func == 'avg':
            return total_sum / total_count if total_count > 0 else 0.0
        elif agg_func == 'min':
            return min(b['min'] for b in bucket_list)
        elif agg_func == 'max':
            return max(b['max'] for b in bucket_list)
        elif agg_func == 'count':
            return total_count
        elif agg_func == 'first':
            earliest = min(bucket_list, key=lambda b: b['first_ts'])
            return earliest['first_val']
        elif agg_func == 'last':
            latest = max(bucket_list, key=lambda b: b['last_ts'])
            return latest['last_val']
        else:
            return total_sum / total_count if total_count > 0 else 0.0

    def _find_best_granularity(self, interval):
        best = None
        for name, gran in sorted(PREDEFINED_GRANULARITIES.items(), key=lambda x: -x[1]):
            if gran <= interval:
                best = name
                break
        return best

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
