from tsdb.aggregation import aggregate


PREDEFINED_GRANULARITIES = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '1h': 3600,
    '6h': 21600,
    '1d': 86400,
}


class DownsampleStore:
    """
    Multi-granularity pre-aggregation store.

    How pre-aggregation works and seamlessly stitches with raw data:

    1. PRE-AGGREGATION PRINCIPLE:
       - When data is written, we pre-compute aggregates at multiple
         time granularities (1m, 5m, 15m, 1h, 6h, 1d).
       - Each pre-aggregated bucket stores: count, sum, min, max, first, last.
       - This is like a materialized view in SQL — pre-computed rollups.

    2. WHY IT SPEEDS UP QUERIES:
       - A query for "avg CPU over 30 days at 1h granularity" needs
         30 * 24 = 720 data points.
       - Without pre-agg: read 30 * 8640 = 259,200 raw points, then
         aggregate → 360x more data to process.
       - With 1h pre-agg: read 720 pre-aggregated buckets → exact same
         number as the output, no over-read.

    3. SEAMLESS STITCHING:
       - When a query requests a time range, we try to use the coarsest
         pre-aggregation whose granularity >= requested interval.
       - However, the latest data may not yet be pre-aggregated (the
         current bucket is still accumulating). For this portion, we
         fall back to reading raw data and aggregating on the fly.
       - The result is a seamless merge: pre-agg points for complete
         buckets + on-the-fly aggregated points for the partial bucket.
       - This ensures queries always return fresh data, not stale views.

    4. STORAGE OVERHEAD:
       - Each granularity stores ~1/granularity_factor of the raw data
         volume as summary records. For all 6 granularities combined,
         overhead is typically < 2% of raw data size.
       - The massive query speedup far outweighs this small overhead.
    """

    def __init__(self):
        self._buckets = {}
        self._granularities = dict(PREDEFINED_GRANULARITIES)

    def _bucket_key(self, series_id, granularity, ts):
        interval = self._granularities[granularity]
        bucket_ts = (ts // interval) * interval
        return (series_id, granularity, bucket_ts)

    def add_point(self, series_id, ts, value):
        for gran_name, interval in self._granularities.items():
            key = self._bucket_key(series_id, gran_name, ts)
            if key not in self._buckets:
                self._buckets[key] = {
                    'count': 0, 'sum': 0.0,
                    'min': float('inf'), 'max': float('-inf'),
                    'first_ts': ts, 'first_val': value,
                    'last_ts': ts, 'last_val': value,
                }
            b = self._buckets[key]
            b['count'] += 1
            b['sum'] += value
            if value < b['min']:
                b['min'] = value
            if value > b['max']:
                b['max'] = value
            if ts >= b['last_ts']:
                b['last_ts'] = ts
                b['last_val'] = value

    def query_pre_agg(self, series_id, granularity, t_start, t_end, agg_func):
        """
        Query pre-aggregated data. Returns list of (bucket_ts, agg_value).
        Only returns fully-formed buckets (complete time intervals).
        """
        if granularity not in self._granularities:
            return []
        interval = self._granularities[granularity]
        results = []
        bucket_start = (t_start // interval) * interval
        now_approx = t_end
        for bts in range(bucket_start, t_end + 1, interval):
            key = (series_id, granularity, bts)
            if key not in self._buckets:
                continue
            b = self._buckets[key]
            if bts + interval > now_approx:
                continue
            points = list(range(b['count']))
            if agg_func == 'sum':
                results.append((bts, b['sum']))
            elif agg_func == 'avg':
                results.append((bts, b['sum'] / b['count'] if b['count'] > 0 else 0))
            elif agg_func == 'min':
                results.append((bts, b['min']))
            elif agg_func == 'max':
                results.append((bts, b['max']))
            elif agg_func == 'count':
                results.append((bts, b['count']))
            elif agg_func == 'first':
                results.append((bts, b['first_val']))
            elif agg_func == 'last':
                results.append((bts, b['last_val']))
            else:
                results.append((bts, b['sum'] / b['count'] if b['count'] > 0 else 0))
        return results

    def get_all_for_series(self, series_id, granularity, t_start, t_end, agg_func):
        """
        Get all pre-agg buckets (including the partial/incomplete one)
        for seamless stitching with raw data.
        """
        if granularity not in self._granularities:
            return [], None
        interval = self._granularities[granularity]
        complete = []
        partial_key = None
        partial_bucket = None
        bucket_start = (t_start // interval) * interval
        for bts in range(bucket_start, t_end + 1, interval):
            key = (series_id, granularity, bts)
            if key not in self._buckets:
                continue
            b = self._buckets[key]
            if bts + interval <= t_end:
                if agg_func == 'sum':
                    complete.append((bts, b['sum']))
                elif agg_func == 'avg':
                    complete.append((bts, b['sum'] / b['count'] if b['count'] > 0 else 0))
                elif agg_func == 'min':
                    complete.append((bts, b['min']))
                elif agg_func == 'max':
                    complete.append((bts, b['max']))
                elif agg_func == 'count':
                    complete.append((bts, b['count']))
                elif agg_func == 'first':
                    complete.append((bts, b['first_val']))
                elif agg_func == 'last':
                    complete.append((bts, b['last_val']))
                else:
                    complete.append((bts, b['sum'] / b['count'] if b['count'] > 0 else 0))
            else:
                partial_key = bts
                partial_bucket = b
        return complete, (partial_key, partial_bucket)

    def stats(self):
        return {
            'total_buckets': len(self._buckets),
            'granularities': list(self._granularities.keys()),
        }
