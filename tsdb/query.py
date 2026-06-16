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
        Downsampled query with strict boundary enforcement and weighted merging.

        Key correctness guarantees:
        1. EVERY output bucket is strictly within [t_start, t_end].
           No data outside the query range leaks into the result.
        2. Pre-aggregation buckets are only used when they are FULLY CONTAINED
           within [t_start, t_end]. If a pre-agg bucket overlaps the boundary
           (e.g. t_start falls in the middle of a 15m bucket), we DO NOT use
           that pre-agg value — we read raw points for the partial region
           and aggregate on the fly.
        3. When merging smaller pre-agg buckets into larger output intervals
           (e.g. 2x 15min → 30min), 'avg' and 'sum' use count-weighted merging:
               merged_avg = (count1*avg1 + count2*avg2) / (count1+count2)
                          = (sum1 + sum2) / (count1 + count2)
           This ensures correctness even when data points are unevenly
           distributed across sub-buckets.
        """
        best_gran = self._find_best_granularity(interval)

        if best_gran is None:
            all_raw = list(self.storage.read_series_range(series_id, t_start, t_end))
            return self._bucket_raw_strict(all_raw, t_start, t_end, interval, agg_func)

        gran_interval = PREDEFINED_GRANULARITIES[best_gran]

        raw_buckets = self.storage.downsample.get_raw_buckets(
            series_id, best_gran, t_start, t_end
        )

        raw_regions = self._find_raw_regions(t_start, t_end, raw_buckets, gran_interval, interval)

        all_raw_points = []
        for r_start, r_end in raw_regions:
            all_raw_points.extend(
                list(self.storage.read_series_range(series_id, r_start, r_end))
            )

        pre_agg_bucketed = self._reaggregate_raw_buckets(
            raw_buckets, gran_interval, interval, agg_func
        )

        raw_bucketed = self._bucket_raw_strict(
            all_raw_points, t_start, t_end, interval, agg_func
        )

        merged = self._merge_results(pre_agg_bucketed, raw_bucketed)
        return merged

    def _find_raw_regions(self, t_start, t_end, raw_buckets, gran_interval, dst_interval):
        """
        Identify time regions that MUST be computed from raw data because
        they either:
        1. Overlap a query boundary (pre-agg bucket not fully inside [t_start, t_end])
        2. Contain a dst_interval boundary not aligned to the pre-agg granularity,
           meaning pre-agg buckets can only partially contribute

        Returns list of (r_start, r_end) inclusive timestamp ranges.
        """
        if not raw_buckets:
            return [(t_start, t_end)]

        covered = set(bts for bts, _ in raw_buckets)

        first_gran_bucket = (t_start // gran_interval) * gran_interval
        last_gran_bucket = ((t_end) // gran_interval) * gran_interval

        regions = []

        current_region_start = None
        current_region_end = None

        gran_bts = first_gran_bucket
        while gran_bts <= last_gran_bucket:
            gran_end = gran_bts + gran_interval - 1

            is_fully_inside = gran_bts >= t_start and gran_end <= t_end

            if is_fully_inside and gran_bts in covered:
                bucket_starts_at_dst_boundary = (gran_bts % dst_interval == 0)
                bucket_ends_at_dst_boundary = ((gran_end + 1) % dst_interval == 0)

                if bucket_starts_at_dst_boundary and bucket_ends_at_dst_boundary:
                    if current_region_start is not None:
                        regions.append((current_region_start, current_region_end))
                        current_region_start = None
                        current_region_end = None
                    gran_bts += gran_interval
                    continue

            overlap_start = max(gran_bts, t_start)
            overlap_end = min(gran_end, t_end)

            if current_region_start is None:
                current_region_start = overlap_start
                current_region_end = overlap_end
            else:
                if overlap_start <= current_region_end + 1:
                    current_region_end = max(current_region_end, overlap_end)
                else:
                    regions.append((current_region_start, current_region_end))
                    current_region_start = overlap_start
                    current_region_end = overlap_end

            gran_bts += gran_interval

        if current_region_start is not None:
            regions.append((current_region_start, current_region_end))

        return regions

    def _reaggregate_raw_buckets(self, raw_buckets, src_interval, dst_interval, agg_func):
        """
        Merge pre-aggregation buckets into target output intervals using
        COUNT-WEIGHTED combining.

        CRITICAL SAFETY FILTER: Only includes pre-agg buckets whose
        [bts, bts+src_interval) is FULLY ALIGNED to dst_interval boundaries.
        Buckets that span a dst_interval boundary are excluded — they're
        computed from raw data instead to guarantee correctness.
        """
        if not raw_buckets:
            return []
        if src_interval == dst_interval:
            results = []
            for bts, b in raw_buckets:
                if bts % dst_interval == 0:
                    results.append((bts, self._combine_bucket([b], agg_func)))
            return results
        groups = {}
        for bts, b in raw_buckets:
            if bts % dst_interval != 0:
                continue
            if (bts + src_interval) % dst_interval != 0:
                continue
            dst_bts = (bts // dst_interval) * dst_interval
            groups.setdefault(dst_bts, []).append(b)
        results = []
        for dst_bts in sorted(groups.keys()):
            results.append((dst_bts, self._combine_bucket(groups[dst_bts], agg_func)))
        return results

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

    def _bucket_raw_strict(self, raw_points, t_start, t_end, interval, agg_func):
        """
        Bucket raw points into output intervals, strictly respecting [t_start, t_end].

        Each output bucket's time label is the floor-aligned start of the interval
        that contains it. Buckets outside [t_start, t_end) are discarded.
        """
        if not raw_points:
            return []
        buckets = {}
        for ts, val in raw_points:
            if ts < t_start or ts > t_end:
                continue
            bucket_ts = (ts // interval) * interval
            buckets.setdefault(bucket_ts, []).append((ts, val))
        results = []
        for bts in sorted(buckets.keys()):
            results.append((bts, aggregate(buckets[bts], agg_func)))
        return results

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
