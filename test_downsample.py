import random
import math
from tsdb import StorageEngine, QueryEngine


def baseline_downsample(points, t_start, t_end, interval, agg_func):
    """
    Ground-truth downsampling implementation matching the new semantics:
    - For first bucket: if floor-aligned bt < t_start, output bt = t_start
      (so no bucket timestamp precedes the query start)
    - For last bucket: keep floor-aligned bt (which is <= t_end by construction)
    - All values computed exactly from raw points strictly within [t_start, t_end]
    """
    from tsdb.aggregation import aggregate
    buckets = {}
    for ts, val in points:
        if ts < t_start or ts > t_end:
            continue
        bucket_ts = (ts // interval) * interval
        buckets.setdefault(bucket_ts, []).append((ts, val))
    results = []
    sorted_bts = sorted(buckets.keys())
    for idx, bts in enumerate(sorted_bts):
        if idx == 0 and bts < t_start:
            out_bts = t_start
        else:
            out_bts = bts
        results.append((out_bts, aggregate(buckets[bts], agg_func)))
    return results


def bucket_raw_in_range(storage, series_id, t_start, t_end):
    """Get ALL raw points strictly within [t_start, t_end] for baseline."""
    return list(storage.read_series_range(series_id, t_start, t_end))


FAILURE_KIND = {
    'TIMESTAMP_LEAK': '桶时间戳跑到查询范围外',
    'VALUE_MISMATCH': '桶聚合值与基线不一致',
    'LENGTH_MISMATCH': '返回桶数与基线不一致',
    'PRE_AGG_MISS': '应该命中预聚合合并但没有',
    'DATA_LEAK': '范围外数据污染结果',
}


def assert_close(name, expected, actual, t_start=None, t_end=None, eps=1e-9):
    """Compare two lists of (ts, val) pairs with clear failure categorization."""
    ok = True
    failure_kind = None

    if len(expected) != len(actual):
        print(f"  ✗ [{name}] {FAILURE_KIND['LENGTH_MISMATCH']}")
        print(f"    期望 {len(expected)} 个桶, 实际 {len(actual)} 个桶")
        if len(expected) <= 50:
            print(f"    期望: {expected}")
            print(f"    实际: {actual}")
        failure_kind = 'LENGTH_MISMATCH'
        ok = False
    else:
        for i, ((e_t, e_v), (a_t, a_v)) in enumerate(zip(expected, actual)):
            if e_t != a_t:
                print(f"  ✗ [{name}] {FAILURE_KIND['TIMESTAMP_LEAK']} 第{i}个桶")
                print(f"    期望时间戳 {e_t}, 实际 {a_t}")
                if t_start is not None:
                    if a_t < t_start:
                        print(f"    原因: 桶时间戳 {a_t} < 查询起点 {t_start}")
                    elif a_t > t_end:
                        print(f"    原因: 桶时间戳 {a_t} > 查询终点 {t_end}")
                failure_kind = failure_kind or 'TIMESTAMP_LEAK'
                ok = False
            elif abs(e_v - a_v) > eps:
                print(f"  ✗ [{name}] {FAILURE_KIND['VALUE_MISMATCH']} 第{i}个桶(t={e_t})")
                print(f"    期望 {e_v}, 实际 {a_v}, 差={abs(e_v-a_v)}")
                failure_kind = failure_kind or 'VALUE_MISMATCH'
                ok = False

    if t_start is not None and t_end is not None and actual:
        for bts, _ in actual:
            if bts < t_start:
                print(f"  ✗ [{name}] {FAILURE_KIND['TIMESTAMP_LEAK']}: 桶 {bts} < 查询起点 {t_start}")
                ok = False
                failure_kind = failure_kind or 'TIMESTAMP_LEAK'
            if bts > t_end:
                print(f"  ✗ [{name}] {FAILURE_KIND['TIMESTAMP_LEAK']}: 桶 {bts} > 查询终点 {t_end}")
                ok = False
                failure_kind = failure_kind or 'TIMESTAMP_LEAK'

    if ok:
        extra = f" (范围 [{t_start}, {t_end}])" if t_start else ""
        print(f"  ✓ [{name}] 通过 ({len(expected)} 个桶完全一致{extra})")
    return ok, failure_kind


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
            ok, _ = assert_close(name, baseline, actual, t_start, t_end)
            all_ok = all_ok and ok
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
    ok, _ = assert_close("30min avg (加权)", baseline, actual, t_start, t_end)
    all_ok = all_ok and ok

    if actual and abs(actual[0][1] - wrong_avg) < 1e-6:
        print(f"  ✗ {FAILURE_KIND['VALUE_MISMATCH']}: 检测到使用了错误的等权平均!")
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
        ok, _ = assert_close(f"{interval}s {agg}", baseline, actual, t_start, t_end)
        all_ok = all_ok and ok
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
                ok, _ = assert_close(name, baseline, actual, t_start, t_end)
                all_ok = all_ok and ok
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
            ok, _ = assert_close(f"{interval}s {agg}", baseline, actual, t_start, t_end)
            all_ok = all_ok and ok

            for bts, v in actual:
                if abs(v) > 50000:
                    print(f"  ✗ {FAILURE_KIND['DATA_LEAK']}! t={bts}, val={v}")
                    all_ok = False
    return all_ok


def test_bucket_timestamps_in_range():
    """
    TEST 6: 所有返回桶的时间戳都严格落在 [t_start, t_end] 范围内。
    - 构造各种非对齐查询
    - 检查每个桶的 bt 都 >= t_start 且 <= t_end
    - 同时验证第一个桶的 bt 被调整为 max(bt, t_start)
    """
    print()
    print("=" * 70)
    print("  TEST 6: 桶时间戳严格落在查询范围内 (无越界标签)")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400
    for i in range(8640):
        ts = base_ts + i * 10
        val = 100 + i * 0.01
        storage.write('cpu', {'host': 'srv6'}, ts, val)
    storage.flush()

    scenarios = [
        ("37分起 53分止 1h降采样", base_ts + 37*60, base_ts + 3600 + 53*60, 3600),
        ("12分起 47分止 30m降采样", base_ts + 12*60, base_ts + 2*1800 + 47*60, 1800),
        ("7秒起 55分止 5m降采样", base_ts + 7, base_ts + 4*300 + 55*60, 300),
    ]

    all_ok = True
    for name, t_start, t_end, interval in scenarios:
        print(f"\n  [{name}] 范围 [{t_start}, {t_end}], 间隔 {interval}s")
        for agg in ['avg', 'sum']:
            opt_result = query.query(
                'cpu', {'host': 'srv6'}, t_start, t_end,
                agg_func=agg, interval=interval
            )
            actual = opt_result[0].points if opt_result else []
            stats = opt_result[0].stats if opt_result else None

            series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
            raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)
            baseline = baseline_downsample(raw_points, t_start, t_end, interval, agg)

            ok, kind = assert_close(f"{interval}s {agg}", baseline, actual, t_start, t_end)
            all_ok = all_ok and ok

            if stats:
                print(f"    命中统计: 预聚合={stats['pre_agg_buckets_used']} 桶, "
                      f"原始={stats['raw_buckets_used']} 桶, "
                      f"粒度={stats['source_granularity']}, "
                      f"命中率={stats['pre_agg_hit_ratio']*100:.0f}%")

            if actual:
                first_bt = actual[0][0]
                if first_bt < t_start:
                    print(f"    ✗ {FAILURE_KIND['TIMESTAMP_LEAK']}: 首桶 {first_bt} < t_start {t_start}")
                    all_ok = False
                else:
                    expected_first_bt = max((t_start // interval) * interval, t_start)
                    if first_bt == t_start:
                        print(f"    ✓ 首桶时间戳调整正确: {first_bt} (t_start, 无越界)")
                    elif first_bt == expected_first_bt:
                        print(f"    ✓ 首桶时间戳正确: {first_bt} (floor对齐且>=t_start)")

                last_bt = actual[-1][0]
                if last_bt > t_end:
                    print(f"    ✗ {FAILURE_KIND['TIMESTAMP_LEAK']}: 末桶 {last_bt} > t_end {t_end}")
                    all_ok = False
                else:
                    print(f"    ✓ 末桶时间戳正确: {last_bt} (<= t_end {t_end})")
    return all_ok


def test_pre_agg_merge_hit_for_uneven_avg():
    """
    TEST 7: 不均匀采样 + 30m/2h 粒度查询, 明确验证走了预聚合合并路径。
    - 构造数据: 15min 桶内点数差异极大 (100:1)
    - 查询: 对齐范围、30m 间隔 → 中间桶必须命中预聚合
    - 验证: (a) 值正确 (加权平均) (b) stats.pre_agg_buckets_used > 0
    """
    print()
    print("=" * 70)
    print("  TEST 7: 预聚合合并路径真实命中 (不均匀 avg)")
    print("=" * 70)

    storage = StorageEngine()
    query = QueryEngine(storage)

    base_ts = 1700006400

    for hour in range(4):
        hour_start = base_ts + hour * 3600
        for min15 in range(4):
            b_start = hour_start + min15 * 900
            if min15 % 2 == 0:
                n_pts = 100
                val = 10.0
                for i in range(n_pts):
                    ts = b_start + i * 1
                    storage.write('cpu', {'host': 'srv7'}, ts, val)
            else:
                n_pts = 5
                val = 100.0
                for i in range(n_pts):
                    ts = b_start + i * 60
                    storage.write('cpu', {'host': 'srv7'}, ts, val)
    storage.flush()

    t_start = base_ts + 3600
    t_end = base_ts + 3 * 3600 - 1
    interval = 1800

    print(f"  查询范围: [{t_start}, {t_end}] (对齐 2 小时)")
    print(f"  降采样间隔: {interval}s (30 分钟)")
    print(f"  预聚合粒度: 15m (每 2 个 15m 桶合并为 1 个 30m 输出桶)")
    print(f"  每个 30m 输出桶: 100 个点 val=10 + 5 个点 val=100")
    print(f"  加权 avg = (100*10 + 5*100) / 105 = 1500 / 105 ≈ 14.2857\n")

    series_id = list(storage.index.get_all_series_for_metric('cpu'))[0]
    raw_points = bucket_raw_in_range(storage, series_id, t_start, t_end)
    baseline = baseline_downsample(raw_points, t_start, t_end, interval, 'avg')

    opt_result = query.query(
        'cpu', {'host': 'srv7'}, t_start, t_end,
        agg_func='avg', interval=interval
    )
    actual = opt_result[0].points if opt_result else []
    stats = opt_result[0].stats if opt_result else None

    all_ok = True
    ok, kind = assert_close("30m avg", baseline, actual, t_start, t_end)
    all_ok = all_ok and ok

    if stats:
        print(f"\n  命中统计:")
        print(f"    总桶数: {stats['total_buckets']}")
        print(f"    预聚合桶: {stats['pre_agg_buckets_used']}")
        print(f"    原始桶: {stats['raw_buckets_used']}")
        print(f"    源粒度: {stats['source_granularity']}")
        print(f"    命中率: {stats['pre_agg_hit_ratio']*100:.0f}%")

        if stats['pre_agg_buckets_used'] > 0:
            print(f"\n  ✓ {FAILURE_KIND.get('PRE_AGG_MISS', '预聚合命中')}: "
                  f"确实使用了预聚合合并路径 ({stats['pre_agg_buckets_used']} 个桶)")
        else:
            print(f"\n  ✗ {FAILURE_KIND['PRE_AGG_MISS']}: 应该命中预聚合但没有! "
                  f"所有 {stats['total_buckets']} 个桶都走了原始数据")
            all_ok = False

        expected_middle_pre_agg = max(0, stats['total_buckets'] - 2)
        if stats['pre_agg_buckets_used'] < expected_middle_pre_agg:
            print(f"    注意: 中间桶应全部命中预聚合, 期望 {expected_middle_pre_agg}, 实际 {stats['pre_agg_buckets_used']}")

    print(f"\n  桶值验证 (应该都≈14.2857):")
    for i, (bts, val) in enumerate(actual):
        is_close = abs(val - 14.2857142857) < 0.01
        status = "✓" if is_close else "✗"
        print(f"    桶 {i}: t={bts}, avg={val:.4f} {status}")
        if not is_close:
            all_ok = False

    return all_ok


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║    降采样正确性自检 — 7 个测试场景                               ║")
    print("║    每一项都与 \"全扫原始点再聚合\" 的基线结果逐桶比对             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  失败类型说明:")
    for kind, desc in FAILURE_KIND.items():
        print(f"    {kind}: {desc}")
    print()

    results = []
    results.append(("TEST 1 不对齐边界", test_misaligned_boundaries()))
    results.append(("TEST 2 不均匀采样加权 avg", test_uneven_points()))
    results.append(("TEST 3 完全部分桶", test_partial_bucket_edges()))
    results.append(("TEST 4 多粒度综合", test_mixed_granularities()))
    results.append(("TEST 5 边界数据不泄漏", test_boundary_no_leak()))
    results.append(("TEST 6 桶时间戳在范围内", test_bucket_timestamps_in_range()))
    results.append(("TEST 7 预聚合合并真实命中", test_pre_agg_merge_hit_for_uneven_avg()))

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
        print("     其中 TEST 6 验证了桶时间戳不越界, TEST 7 验证了预聚合路径命中。")
    else:
        print("  ⚠ 存在失败测试, 请检查上方输出中标记的错误类型。")
    return all_pass


if __name__ == '__main__':
    ok = main()
    exit(0 if ok else 1)
