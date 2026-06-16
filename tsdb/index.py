from bisect import insort


class TagIndex:
    """
    Inverted index for tag-based time series lookup.

    How it enables fast lookup of series matching tag conditions:
    - Data structure: mapping from (tag_key, tag_value) → set of series_ids
    - This is exactly like a search engine's inverted index: each tag
      pair is a "term", and the posting list is the set of series that
      carry that tag.
    - To find series matching {host=server1, region=us-east}:
      1. Look up posting list for (host, server1)   → e.g. {s1, s3, s5}
      2. Look up posting list for (region, us-east)  → e.g. {s1, s2, s5}
      3. Intersection = {s1, s5}  ← only 2 set operations, O(min list)
    - Without the index, we'd have to scan ALL series and check their
      tags → O(N). With the index, we only touch series that have at
      least one matching tag → O(|posting_list|).
    - For "not equal" filters, we compute the complement: find all
      series with the tag key, subtract the ones matching the value.
    - For "regex" filters, we enumerate all values for the key and
      union the matching posting lists.

    Example: 10,000 series, query {host=srv1, region=us-east}.
    - Index: host=srv1 → 50 series, region=us-east → 2000 series.
    - Intersection: ~10 series. Cost: 50 + 2000 set operations.
    - Without index: scan 10,000 series. 5x-500x speedup.
    """

    def __init__(self):
        self._postings = {}
        self._key_values = {}
        self._series_tags = {}
        self._next_id = 0
        self._metric_series = {}

    def _make_series_id(self, metric, tags):
        tag_str = ','.join(f'{k}={v}' for k, v in sorted(tags.items()))
        return f'{metric}{{{tag_str}}}'

    def add_series(self, metric, tags):
        series_id = self._make_series_id(metric, tags)
        if series_id in self._series_tags:
            return series_id
        self._series_tags[series_id] = dict(tags)
        self._metric_series.setdefault(metric, set()).add(series_id)
        for key, value in tags.items():
            posting_key = (key, value)
            self._postings.setdefault(posting_key, set()).add(series_id)
            self._key_values.setdefault(key, set()).add(value)
        return series_id

    def match(self, metric, tag_filters):
        """
        Find all series_ids for a metric matching tag filters.

        tag_filters: dict of {tag_key: tag_value} for exact match,
                     or {tag_key: ('!=', value)} for not-equal,
                     or {tag_key: ('=~', regex_pattern)} for regex.
        """
        candidates = self._metric_series.get(metric, set()).copy()
        if not candidates:
            return set()
        for key, condition in tag_filters.items():
            if isinstance(condition, tuple):
                op, val = condition
                if op == '!=':
                    exclude = self._postings.get((key, val), set())
                    candidates -= exclude
                elif op == '=~':
                    import re
                    pattern = re.compile(val)
                    matching = set()
                    for v in self._key_values.get(key, set()):
                        if pattern.search(v):
                            matching |= self._postings.get((key, v), set())
                    candidates &= matching
            else:
                posting = self._postings.get((key, condition), set())
                candidates &= posting
            if not candidates:
                return set()
        return candidates

    def get_series_tags(self, series_id):
        return self._series_tags.get(series_id, {})

    def get_all_series_for_metric(self, metric):
        return self._metric_series.get(metric, set()).copy()

    def stats(self):
        return {
            'total_series': len(self._series_tags),
            'total_postings': len(self._postings),
            'total_tag_keys': len(self._key_values),
        }
