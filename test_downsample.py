import random
import math
from tsdb import StorageEngine, QueryEngine


def baseline_downsample(points, t_start, t_end, interval, agg_func):
    """
    Ground-truth downsampling implementation:
    Scan every raw point in [t_start, t_end], bucket by interval, aggregate.
    This is the reference that the optimized query must match exactly.
    """
    from tsdb.aggregation import aggregate
    buckets = {}
    for ts, val in points:
        if ts < t_start or ts > t_end:
            continue
        bucket_ts = (ts // interval) * interval
        buckets.setdefault(bucket_ts, []).append((ts, val))
    results = []
    for bts in sorted(buckets.keys()):
        results.append((bts, aggregate(buckets[bts], agg_func)))
    return results


def bucket_raw_in_range(storage, series_id, t_start, t_end):
    """Get ALL raw points strictly within [t_start, t_end] for baseline."""
    return list(storage.read_series_range(series_id, t_start, t_end))


def assert_close(name, expected, actual, eps=1e-9):
    """Compare two lists of (ts, val) pairs. Timestamps must be exact."""
    ok = True
    if len(expected) != len(actual):
        print(f"  ✗ [{name}] 长度不一致: 期望 {len(expected)}, 实际 {len(actual)}")
        if len(expected) <= 50:
            print(f"    期望: {expected}")
            print(f"    实际: {actual}")
        ok = False
    else:
        for i, ((e_t, e_v), (a_t, a_v)) in enumerate(zip(expected, actual)):
            if e_t != a_t:
                print(f"  ✗ [{name}] 第{i}个桶时间戳不同: 期望 {e_t}, 实际 {a_t}")
                ok = False
            elif abs(e_v - a_v) > eps:
                print(f"  ✗ [{name}] 第{i}个桶(t={e_t})值不同: 期望 {e_v}, 实际 {a_v}, 差={abs(e_v-a_v)}")
                ok = False
    if ok:
        print(f"  ✓ [{name}] 通过 ({len(expected)} 个桶完全一致)")
    return ok


def test_misaligned_boundaries():
    """
    TEST 1: 不对齐的查询起止时间。
    - 数据: 整点对齐, 每10秒一个点, 覆盖完整小时
    - 查询: 从 12:37 开始, 到 14:53 结束 (都不落在桶边界)
    - 验证: 优化查询结果与全扫原始点得到的降采样结果逐桶一致
    """
    print("=" * 70)
    print("  TEST 1: 不对齐查询边界 (起止时间跨半个桶)")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400
    n_hours = 6
    n = n_hours * 360
    for i in range(n):
        ts = base_ts + i * 10
        val = 100.0 + i * 0.01
        storage.write('cpu', {'host': 'srv1'}, ts, val)
    storage.flush()

    t_start = base_ts + 37 * 60
    t_end = base_ts + 2 * 3600 + 53 * 60

    series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
    raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)
    print(f"  查询范围: [{t_start}, {t_end}]  ({t_start-base_ts}s ~ {t_end-base_ts}s 从起点)")
    print(f"  原始点数: {len(raw_points)}")
    print(f"  前3点: {raw_points[:3]}")
    print(f"  后3点: {raw_points[-3:]}\n")

    all_ok = True
    for interval in [60, 300, 900, 1800, 3600]:
        for agg in ['sum', 'avg', 'min', 'max', 'count']:
            baseline = baseline_downsample(raw_points, t_start, t_end, interval, agg)
            opt_result = query.query(
                'cpu', {'host': 'srv1'}, t_start, t_end,
                agg_func=agg, interval=interval
            )
            actual = opt_result[0].points if opt_result else []
            name = f"{interval}s {agg}"
            if not assert_close(name, baseline, actual):
                all_ok = False
    return all_ok


def test_uneven_points():
    """
    TEST 2: 不均匀采样 + 合并预聚合桶求 avg。
    - 数据: 15分钟内某段极密, 某段极疏 (故意让 count 差异极大)
    - 查询: 30 分钟粒度 avg (由两个 15min 预聚合桶合并)
    - 验证: avg 必须按真实点数加权, 即 (sum1+sum2)/(count1+count2)
    """
    print()
    print("=" * 70)
    print("  TEST 2: 不均匀采样, 合并预聚合桶求 avg 的加权正确性")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400

    bucket_1_start = base_ts
    bucket_2_start = base_ts + 900

    for i in range(100):
        ts = bucket_1_start + i * 1
        val = 10.0
        storage.write('cpu', {'host': 'srv2'}, ts, val)

    for i in range(5):
        ts = bucket_2_start + i * 60
        val = 100.0
        storage.write('cpu', {'host': 'srv2'}, ts, val)

    storage.flush()

    t_start = base_ts
    t_end = base_ts + 1800 - 1
    interval = 1800

    series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
    raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)

    raw_sum = sum(v for _, v in raw_points)
    raw_count = len(raw_points)
    expected_avg = raw_sum / raw_count
    wrong_avg = (10.0 + 100.0) / 2.0

    print(f"  bucket1: 100 个点, 每个 val=10")
    print(f"  bucket2:   5 个点, 每个 val=100")
    print(f"  正确 avg:  ({raw_sum}) / {raw_count} = {expected_avg:.4f}")
    print(f"  错误 avg:  (10+100)/2 = {wrong_avg:.1f}  (未按点数加权)\n")

    all_ok = True
    baseline = baseline_downsample(raw_points, t_start, t_end, interval, 'avg')
    opt_result = query.query(
        'cpu', {'host': 'srv2'}, t_start, t_end,
        agg_func='avg', interval=interval
    )
    actual = opt_result[0].points if opt_result else []
    all_ok = assert_close("30min avg (加权)", baseline, actual)

    if actual and abs(actual[0][1] - wrong_avg) < 1e-6:
        print("  ✗ 检测到使用了错误的等权平均!")
        all_ok = False
    elif actual:
        print(f"  实际 avg={actual[0][1]:.4f}, 正确={expected_avg:.4f}")

    return all_ok


def test_partial_bucket_edges():
    """
    TEST 3: 查询范围完全在单个预聚合桶内部。
    - 整个查询范围都是 "部分桶", 优化路径必须完全退化为原始数据计算
    - 验证结果与全扫原始点逐点一致
    """
    print()
    print("=" * 70)
    print("  TEST 3: 查询完全落在单个预聚合桶内部 (全量原始数据)")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400
    for i in range(360):
        ts = base_ts + i * 10
        val = math.sin(i * 0.1) * 50 + 100
        storage.write('cpu', {'host': 'srv3'}, ts, val)
    storage.flush()

    t_start = base_ts + 125
    t_end = base_ts + 789
    interval = 60

    series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
    raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)

    print(f"  查询范围: [{t_start}, {t_end}]")
    print(f"  原始点数: {len(raw_points)}\n")

    all_ok = True
    for agg in ['sum', 'avg', 'min', 'max', 'count', 'first', 'last']:
        baseline = baseline_downsample(raw_points, t_start, t_end, interval, agg)
        opt_result = query.query(
            'cpu', {'host': 'srv3'}, t_start, t_end,
            agg_func=agg, interval=interval
        )
        actual = opt_result[0].points if opt_result else []
        all_ok = assert_close(f"{interval}s {agg}", baseline, actual) and all_ok
    return all_ok


def test_mixed_granularities():
    """
    TEST 4: 多种降采样粒度组合验证。
    - 1m, 5m, 15m, 30m, 1h, 2h 各种间隔
    - 每个都同时验证 misaligned 和 非 misaligned 情况
    """
    print()
    print("=" * 70)
    print("  TEST 4: 多种降采样粒度 (1m~2h) 综合验证")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400
    random.seed(99)
    for i in range(8640):
        ts = base_ts + i * 10
        val = random.gauss(50, 10)
        storage.write('cpu', {'host': 'srv4'}, ts, val)
    storage.flush()

    scenarios = [
        ("对齐",   base_ts + 3600, base_ts + 5 * 3600),
        ("不对齐", base_ts + 3600 + 137, base_ts + 5 * 3600 - 251),
    ]

    intervals = [60, 300, 900, 1800, 3600, 7200]
    aggs = ['sum', 'avg', 'min', 'max', 'count']

    all_ok = True
    for scenario_name, t_start, t_end in scenarios:
        series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
        raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)
        print(f"\n  [{scenario_name}] 查询 [{t_start}, {t_end}], {len(raw_points)} 原始点")
        for interval in intervals:
            for agg in aggs:
                baseline = baseline_downsample(raw_points, t_start, t_end, interval, agg)
                opt_result = query.query(
                    'cpu', {'host': 'srv4'}, t_start, t_end,
                    agg_func=agg, interval=interval
                )
                actual = opt_result[0].points if opt_result else []
                name = f"{scenario_name} {interval}s {agg}"
                if not assert_close(name, baseline, actual):
                    all_ok = False
    return all_ok


def test_boundary_no_leak():
    """
    TEST 5: 严格验证查询范围外的数据不会污染结果。
    - 在查询范围外故意放极端值
    - 验证优化查询和基线都不包含这些值
    """
    print()
    print("=" * 70)
    print("  TEST 5: 边界外数据不泄漏 (放极端值验证)")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400
    t_start = base_ts + 3600
    t_end = base_ts + 7200 - 1

    for i in range(1000):
        ts = base_ts + i * 10
        if ts < t_start:
            val = -99999.9
        elif ts > t_end:
            val = 99999.9
        else:
            val = 50.0 + i * 0.01
        storage.write('cpu', {'host': 'srv5'}, ts, val)
    storage.flush()

    series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
    raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)

    all_ok = True
    for interval in [300, 1800]:
        for agg in ['min', 'max', 'avg', 'sum']:
            baseline = baseline_downsample(raw_points, t_start, t_end, interval, agg)
            opt_result = query.query(
                'cpu', {'host': 'srv5'}, t_start, t_end,
                agg_func=agg, interval=interval
            )
            actual = opt_result[0].points if opt_result else []
            all_ok = assert_close(f"{interval}s {agg}", baseline, actual) and all_ok

            for bts, v in actual:
                if abs(v) > 50000:
                    print(f"  ✗ 检测到范围外数据泄漏! t={bts}, val={v}")
                    all_ok = False
    return all_ok


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    降采样正确性自检 — 5 个测试场景                               ║")
    print("║    每一项都与 \"全扫原始点再聚合\" 的基线结果逐桶比对             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    results = []
    results.append(("TEST 1 不对齐边界", test_misaligned_boundaries()))
    results.append(("TEST 2 不均匀采样加权 avg", test_uneven_points()))
    results.append(("TEST 3 完全部分桶", test_partial_bucket_edges()))
    results.append(("TEST 4 多粒度综合", test_mixed_granularities()))
    results.append(("TEST 5 边界数据不泄漏", test_boundary_no_leak()))

    print()
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        all_pass = all_pass and ok
    print()
    if all_pass:
        print("  🎉 全部测试通过! 降采样查询与原始点聚合结果在所有场景下完全一致。")
    else:
        print("  ⚠ 存在失败测试, 请检查上方输出。")
    return all_pass


if __name__ == '__main__':
    ok = main()
    exit(0 if ok else 1)
