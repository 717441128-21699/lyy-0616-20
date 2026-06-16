import math


AGGREGATORS = {}


def register(name):
    def decorator(fn):
        AGGREGATORS[name] = fn
        return fn
    return decorator


@register('sum')
def agg_sum(values):
    return sum(v for _, v in values)


@register('avg')
def agg_avg(values):
    if not values:
        return 0.0
    return sum(v for _, v in values) / len(values)


@register('min')
def agg_min(values):
    return min(v for _, v in values)


@register('max')
def agg_max(values):
    return max(v for _, v in values)


@register('count')
def agg_count(values):
    return len(values)


@register('first')
def agg_first(values):
    return values[0][1] if values else 0.0


@register('last')
def agg_last(values):
    return values[-1][1] if values else 0.0


def aggregate(values, func_name):
    if not values:
        return None
    agg = AGGREGATORS.get(func_name)
    if agg is None:
        raise ValueError(f"Unknown aggregation function: {func_name}")
    return agg(values)
