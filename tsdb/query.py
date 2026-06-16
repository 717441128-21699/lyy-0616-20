from tsdb.aggregation import aggregate, AGGREGATORS
from tsdb.downsample import PREDEFINED_GRANULARITIES


SOURCE_RAW = 'raw'
SOURCE_PRE_AGG_SINGLE = 'pre_agg:1'
SOURCE_PRE_AGG_MERGED = 'pre_agg:n'
SOURCE_CROSS_MERGED = 'cross_merged'


class QueryResult:
    """
    Query result with detailed diagnostics about how each bucket was computed.

    Attributes:
        series_id: unique identifier for the time series (or group name)
        tags: tag key-value pairs for the series (or grouping tags)
        points: list of (timestamp, value) pairs
        bucket_sources: list of source descriptions, one per point:
            'raw'                — computed by scanning raw data points
            'pre_agg:1'          — came from a single pre-agg bucket directly
            'pre_agg:n'          — merged from multiple smaller pre-agg buckets
            'cross_merged'       — merged across multiple series (cross-series agg)
        bucket_series_sources: for cross-series/grouped results, list of dicts
            mapping series_id -> source for each output bucket. Shows which
            source each contributing series used (raw/pre_agg:1/pre_agg:n/missing).
        group_key: for grouped results, the group's tag key=value string
        stats: dictionary with performance/debug info
    """
    def __init__(self, series_id, tags, points, bucket_sources=None,
                 stats=None, bucket_series_sources=None, group_key=None):
        self.series_id = series_id
        self.tags = tags
        self.points = points
        self.bucket_sources = bucket_sources or ([SOURCE_RAW] * len(points))
        self.bucket_series_sources = bucket_series_sources or ([None] * len(points))
        self.group_key = group_key
        default_counts = {
            SOURCE_RAW: sum(1 for s in self.bucket_sources if s == SOURCE_RAW),
            SOURCE_PRE_AGG_SINGLE: sum(1 for s in self.bucket_sources if s == SOURCE_PRE_AGG_SINGLE),
            SOURCE_PRE_AGG_MERGED: sum(1 for s in self.bucket_sources if s == SOURCE_PRE_AGG_MERGED),
            SOURCE_CROSS_MERGED: sum(1 for s in self.bucket_sources if s == SOURCE_CROSS_MERGED),
        }
        pre_agg_count = default_counts[SOURCE_PRE_AGG_SINGLE] + default_counts[SOURCE_PRE_AGG_MERGED]
        self.stats = stats or {
            'pre_agg_buckets_used': pre_agg_count,
            'raw_buckets_used': default_counts[SOURCE_RAW],
            'source_granularity': None,
            'total_buckets': len(points),
            'pre_agg_hit_ratio': pre_agg_count / len(points) if points else 0.0,
            'bucket_source_counts': default_counts,
        }

    def __repr__(self):
        n = len(self.points)
        preview_lines = []
        for i in range(min(5, n)):
            t, v = self.points[i]
            src = self.bucket_sources[i] if i < len(self.bucket_sources) else '?'
            parts = [f't={t}', f'v={v:.2f}', f'src={src}']
            ss = self.bucket_series_sources[i] if i < len(self.bucket_series_sources) else None
            if ss:
                ss_str = ','.join(f'{k}:{v}' for k, v in sorted(ss.items()))
                parts.append(f'lines={{{ss_str}}}')
            preview_lines.append('(' + ', '.join(parts) + ')')
        pts_str = ', '.join(preview_lines)
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
              agg_func=None, interval=None, cross_series_agg=None, group_by=None):
        """
        Execute a query.

        Args:
            metric: metric name
            tag_filters: dict of {tag_key: tag_value_or_condition}
            t_start: start timestamp (inclusive)
            t_end: end timestamp (inclusive)
            agg_func: per-series aggregation function name (sum, avg, min, max, count)
            interval: downsampling interval in seconds (None = no downsampling)
            cross_series_agg: if provided AND multiple series match, merge all
                              series into one output using this aggregation
                              function across series per time bucket.
                              SEMANTICS: per-series downsample first, then
                              aggregate the per-series values. e.g. every host
                              compute 30min avg first, then avg those avgs.
                              Also reuses per-series pre-aggregation.
            group_by: tag key name. If provided, matching series are grouped by
                         this tag's values. Each group is aggregated using
                         cross_series_agg (which must also be provided), and
                         one QueryResult is returned per distinct group value.

        Returns:
            list of QueryResult, one per matching series (or one merged
            result, or one per group)
        """
        if tag_filters is None:
            tag_filters = {}
        matching = self.storage.index.match(metric, tag_filters)
        if not matching:
            return []

        if interval is not None and interval > 0:
            if group_by is not None:
                if cross_series_agg is None:
                    raise ValueError("group_by requires cross_series_agg")
                return self._query_grouped_downsampled(
                    metric, matching, t_start, t_end,
                    agg_func or 'avg', interval,
                    cross_series_agg, group_by
                )
            if cross_series_agg is not None:
                return [self._query_cross_series_downsampled(
                    metric, matching, t_start, t_end,
                    agg_func or 'avg', interval, cross_series_agg
                )]

        results = []
        for series_id in sorted(matching):
            tags = self.storage.index.get_series_tags(series_id)
            if interval is not None and interval > 0:
                points, sources, gran_name = self._query_downsampled(
                    series_id, t_start, t_end, agg_func or 'avg', interval
                )
                stats = self._make_stats_from_sources(sources, gran_name)
                results.append(QueryResult(series_id, tags, points, sources, stats))
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
        Downsampled query with PER-BUCKET source decision.

        For every output bucket (aligned to dst_interval boundaries), check:
        - Is the bucket fully contained in [t_start, t_end] AND all its
          sub-pre-agg buckets exist? If yes → use pre-agg merge.
        - Otherwise → read raw data points strictly within the overlap.

        Timestamp output:
        - If first floor-aligned bucket bt < t_start → output bt = t_start
        - Last bucket bt = floor-aligned bt (<= t_end by construction)
        - Middle buckets = floor-aligned bt

        This works for 1-bucket, 2-bucket, and many-bucket queries alike —
        every bucket gets an independent decision, no artificial distinction
        between "first/last" and "middle".
        """
        best_gran = self._find_best_granularity(interval)
        gran_name = best_gran

        first_bt_floor = (t_start // interval) * interval
        last_bt_floor = (t_end // interval) * interval

        results = []
        sources = []

        for dst_bt in range(first_bt_floor, last_bt_floor + 1, interval):
            dst_end = dst_bt + interval - 1

            overlap_start = max(dst_bt, t_start)
            overlap_end = min(dst_end, t_end)

            if overlap_start > overlap_end:
                continue

            use_pre_agg = False
            merged_val = None
            source_kind = SOURCE_RAW

            if best_gran is not None:
                gran_interval = PREDEFINED_GRANULARITIES[best_gran]
                bucket_fully_inside = (dst_bt >= t_start and dst_end <= t_end)
                divisible = (interval % gran_interval == 0)

                if bucket_fully_inside and divisible:
                    num_sub = interval // gran_interval
                    sub_buckets = []
                    all_found = True
                    for i in range(num_sub):
                        sub_bt = dst_bt + i * gran_interval
                        raw_b = self.storage.downsample.get_raw_buckets(
                            series_id, best_gran, sub_bt, sub_bt + gran_interval
                        )
                        if not raw_b or raw_b[0][0] != sub_bt:
                            all_found = False
                            break
                        sub_buckets.append(raw_b[0][1])
                    if all_found and sub_buckets:
                        merged_val = self._combine_bucket(sub_buckets, agg_func)
                        use_pre_agg = True
                        source_kind = SOURCE_PRE_AGG_MERGED if num_sub > 1 else SOURCE_PRE_AGG_SINGLE

            if use_pre_agg:
                out_bt = dst_bt if dst_bt >= t_start else t_start
                results.append((out_bt, merged_val))
                sources.append(source_kind)
            else:
                raw_pts = list(self.storage.read_series_range(
                    series_id, overlap_start, overlap_end
                ))
                if raw_pts:
                    val = aggregate(raw_pts, agg_func)
                    out_bt = dst_bt if dst_bt >= t_start else t_start
                    results.append((out_bt, val))
                    sources.append(SOURCE_RAW)

        return results, sources, gran_name

    def _query_cross_series_downsampled(self, metric, series_ids, t_start, t_end,
                                         per_series_agg, interval, cross_agg,
                                         group_key=None, group_tags=None):
        """
        Downsample across multiple series and aggregate them into one curve.

        SEMANTICS (NEW): per-series downsample FIRST, then aggregate.

        Step 1. For each series independently: compute downsampled values
                    (30min avg for example), using per_series_agg).
        Step 2. For each time bucket: collect the per-series values
                    (some series' downsampled value for that bucket.
        Step 3. Apply cross_agg across those per-series values.

        Example: 3 hosts. Each computes 30min avg. Then avg those 3 avgs.
        This is NOT the same as mixing all raw points and averaging —
        because each host gets equal weight regardless of point count.

        Also tracks per-series source info for diagnostics.

        Args:
            group_key: for grouped results, pass the display name
            group_tags: for grouped results pass the group tag dict

        Returns a single QueryResult with enhanced diagnostics:
            bucket_series_sources[i] = {series_id: 'raw'|'pre_agg:1|'pre_agg:n'|'missing'}
        """
        first_bt_floor = (t_start // interval) * interval
        last_bt_floor = (t_end // interval) * interval

        per_series_results = {}
        per_series_sources_by_time = {}
        series_names = {}
        gran_name = None

        for series_id in sorted(series_ids):
            points, sources, gran = self._query_downsampled(
                series_id, t_start, t_end, per_series_agg, interval
            )
            gran_name = gran
            series_names[series_id] = self.storage.index.get_series_tags(series_id)
            per_series_results[series_id] = {t: v for t, v in points}
            src_map = {}
            for i, (t, _) in enumerate(points):
                if i < len(sources):
                    src_map[t] = sources[i]
            per_series_sources_by_time[series_id] = src_map

        output_points = []
        output_sources = []
        output_series_sources = []

        from tsdb.aggregation import aggregate as aggr_func

        sorted_series_ids = sorted(series_ids)

        for dst_bt in range(first_bt_floor, last_bt_floor + 1, interval):
            dst_end = dst_bt + interval - 1
            overlap_start = max(dst_bt, t_start)
            overlap_end = min(dst_end, t_end)
            if overlap_start > overlap_end:
                continue

            per_series_vals = []
            series_src_map = {}

            for sid in sorted_series_ids:
                src = per_series_sources_by_time.get(sid, {})
                if dst_bt in per_series_results.get(sid, {}):
                    val = per_series_results[sid][dst_bt]
                    per_series_vals.append((dst_bt, val))
                    series_src_map[sid] = src.get(dst_bt, 'missing')
                else:
                    series_src_map[sid] = 'missing'

            if not per_series_vals:
                continue

            combined_val = aggr_func(per_series_vals, cross_agg)
            out_bt = dst_bt if dst_bt >= t_start else t_start
            output_points.append((out_bt, combined_val))
            output_sources.append(SOURCE_CROSS_MERGED)
            output_series_sources.append(series_src_map)

        if group_tags is None:
            group_tags = {}
        if group_key is None:
            group_key = f'{metric}::__cross_agg__'

        stats = self._make_stats_from_sources(output_sources, gran_name)
        return QueryResult(
            group_key, group_tags, output_points, output_sources,
            stats, output_series_sources, group_key
        )

    def _query_grouped_downsampled(self, metric, series_ids, t_start, t_end,
                              per_series_agg, interval, cross_agg, group_by):
        """
        Group matching series by a tag value and do cross-series agg per group.

        Args:
            group_by: tag key name to group on.

        Returns one QueryResult per distinct group value, each with group_key set.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        group_tag_values = {}

        for sid in series_ids:
            tags = self.storage.index.get_series_tags(sid)
            val = tags.get(group_by)
            if val is not None:
                groups[val].append(sid)
                group_tag_values[val] = {group_by: val}

        results = []
        for group_val in sorted(groups.keys()):
            group_series = groups[group_val]
            group_key = f'{metric}::{group_by}={group_val}'
            group_tags = group_tag_values[group_val]
            res = self._query_cross_series_downsampled(
                metric, group_series, t_start, t_end,
                per_series_agg, interval, cross_agg,
                group_key, group_tags
            )
            results.append(res)
        return results

    def _points_to_stats(self, points):
        """Convert list of (ts, val) into a bucket stats dict."""
        if not points:
            return None
        vals = [v for _, v in points]
        return {
            'count': len(points),
            'sum': sum(vals),
            'min': min(vals),
            'max': max(vals),
            'first_ts': points[0][0],
            'first_val': points[0][1],
            'last_ts': points[-1][0],
            'last_val': points[-1][1],
        }

    def _merge_bucket_stats(self, bucket_list):
        """Merge multiple bucket stats dicts into one (across sub-buckets of same series)."""
        total_count = sum(b['count'] for b in bucket_list)
        total_sum = sum(b['sum'] for b in bucket_list)
        overall_min = min(b['min'] for b in bucket_list)
        overall_max = max(b['max'] for b in bucket_list)
        first = min(bucket_list, key=lambda b: b['first_ts'])
        last = max(bucket_list, key=lambda b: b['last_ts'])
        return {
            'count': total_count,
            'sum': total_sum,
            'min': overall_min,
            'max': overall_max,
            'first_ts': first['first_ts'],
            'first_val': first['first_val'],
            'last_ts': last['last_ts'],
            'last_val': last['last_val'],
        }

    def _cross_agg_from_stats(self, stats_list, cross_agg):
        """Combine per-series bucket stats into one cross-series value."""
        if cross_agg == 'sum':
            return sum(s['sum'] for s in stats_list)
        elif cross_agg == 'avg':
            total_sum = sum(s['sum'] for s in stats_list)
            total_count = sum(s['count'] for s in stats_list)
            return total_sum / total_count if total_count > 0 else 0.0
        elif cross_agg == 'min':
            return min(s['min'] for s in stats_list)
        elif cross_agg == 'max':
            return max(s['max'] for s in stats_list)
        elif cross_agg == 'count':
            return sum(s['count'] for s in stats_list)
        elif cross_agg == 'first':
            earliest = min(stats_list, key=lambda s: s['first_ts'])
            return earliest['first_val']
        elif cross_agg == 'last':
            latest = max(stats_list, key=lambda s: s['last_ts'])
            return latest['last_val']
        else:
            raise ValueError(f"Unknown cross-series agg: {cross_agg}")

    def _make_stats_from_sources(self, sources, gran_name):
        counts = {
            SOURCE_RAW: sum(1 for s in sources if s == SOURCE_RAW),
            SOURCE_PRE_AGG_SINGLE: sum(1 for s in sources if s == SOURCE_PRE_AGG_SINGLE),
            SOURCE_PRE_AGG_MERGED: sum(1 for s in sources if s == SOURCE_PRE_AGG_MERGED),
            SOURCE_CROSS_MERGED: sum(1 for s in sources if s == SOURCE_CROSS_MERGED),
        }
        pre_agg = counts[SOURCE_PRE_AGG_SINGLE] + counts[SOURCE_PRE_AGG_MERGED]
        total = len(sources)
        return {
            'pre_agg_buckets_used': pre_agg,
            'raw_buckets_used': counts[SOURCE_RAW],
            'source_granularity': gran_name,
            'total_buckets': total,
            'pre_agg_hit_ratio': pre_agg / total if total > 0 else 0.0,
            'bucket_source_counts': counts,
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
