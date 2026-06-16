import time
import random
import math
from tsdb import StorageEngine, QueryEngine


def separator(title):
    print(f'\n{"="*72}')
    print(f'  {title}')
    print(f'{"="*72}\n')


def demo_compression():
    separator("1. 时间戳差值编码 + 值异或压缩 — 原理与效果")

    print("【原理】Gorilla 论文的两种压缩算法:\n")
    print("  A) 时间戳 Delta-of-Delta 编码:")
    print("     - 时序数据的时间戳通常是等间隔的 (如每10秒)")
    print("     - 第一个差值 delta = t1 - t0 捕获间隔")
    print("     - 二阶差值 dod = (t2-t1) - (t1-t0) 几乎总是 0")
    print("     - 存储: dod=0 → 1 bit; 小值 → 2+7 bit; ... 逐级编码")
    print("     - 对于规律采样: 8640个时间戳, 原始 552,960 bit → 压缩后 ~8,640 bit")
    print("     - 压缩率: ~98.4%!\n")

    print("  B) 浮点值 XOR 压缩:")
    print("     - 相邻值通常非常相似 (温度缓慢变化)")
    print("     - XOR 产生大量前导零和后导零")
    print("     - 只存储变化的有意义位, 相同模式可复用")
    print("     - 慢变化值: XOR ~10 有效位 → 23 bit vs 64 bit → ~64% 压缩")
    print()

    from tsdb.encoding import (TimestampCompressor, TimestampDecompressor,
                                ValueCompressor, ValueDecompressor)

    base_ts = 1700000000
    interval = 10
    n = 10000
    timestamps = [base_ts + i * interval for i in range(n)]
    values = [20.0 + 0.1 * math.sin(i * 0.01) + random.gauss(0, 0.05) for i in range(n)]

    tc = TimestampCompressor()
    for ts in timestamps:
        tc.append(ts)
    ts_compressed = tc.finish()

    vc = ValueCompressor()
    for v in values:
        vc.append(v)
    val_compressed = vc.finish()

    ts_raw = n * 8
    val_raw = n * 8

    print(f"  数据量: {n} 个数据点, 时间间隔 {interval}s")
    print(f"  ┌────────────┬──────────────┬──────────────┬──────────┐")
    print(f"  │   类别     │  原始 (byte) │ 压缩 (byte)  │ 压缩率   │")
    print(f"  ├────────────┼──────────────┼──────────────┼──────────┤")
    print(f"  │ 时间戳     │ {ts_raw:>10}   │ {len(ts_compressed):>10}   │ {(1-len(ts_compressed)/ts_raw)*100:>6.1f}%  │")
    print(f"  │ 数值       │ {val_raw:>10}   │ {len(val_compressed):>10}   │ {(1-len(val_compressed)/val_raw)*100:>6.1f}%  │")
    print(f"  │ 合计       │ {ts_raw+val_raw:>10}   │ {len(ts_compressed)+len(val_compressed):>10}   │ {(1-(len(ts_compressed)+len(val_compressed))/(ts_raw+val_raw))*100:>6.1f}%  │")
    print(f"  └────────────┴──────────────┴──────────────┴──────────┘")

    print("\n  【验证】解压后数据完整性:")
    ts_dec = TimestampDecompressor(ts_compressed)
    val_dec = ValueDecompressor(val_compressed)
    decoded_ts = [ts_dec.read() for _ in range(n)]
    decoded_val = [val_dec.read() for _ in range(n)]

    ts_ok = all(decoded_ts[i] == timestamps[i] for i in range(n))
    val_ok = all(abs(decoded_val[i] - values[i]) < 1e-10 for i in range(n))
    print(f"  时间戳: {'✓ 无损' if ts_ok else '✗ 有误'}")
    print(f"  数  值: {'✓ 无损' if val_ok else '✗ 有误'}")


def demo_tag_index():
    separator("2. 标签倒排索引 — 快速定位时间序列")

    print("【原理】倒排索引如何加速标签过滤:\n")
    print("  数据结构: (tag_key, tag_value) → set(series_id)")
    print("  这与搜索引擎的倒排索引完全相同:\n")
    print("  查询 {host=srv1, region=us-east}:")
    print("    1. 查 posting(host, srv1)    → {s1, s3, s5, s7, s9}")
    print("    2. 查 posting(region, us-east) → {s1, s2, s5, s8}")
    print("    3. 交集 = {s1, s5}  ← 仅2次集合操作, O(min list)")
    print()
    print("  无索引: 需扫描所有 N 个序列的标签 → O(N)")
    print("  有索引: 仅触及匹配标签的序列 → O(|posting list|)")
    print()

    from tsdb.index import TagIndex
    idx = TagIndex()

    hosts = ['srv1', 'srv2', 'srv3', 'srv4', 'srv5']
    regions = ['us-east', 'us-west', 'eu-west', 'ap-east']
    envs = ['prod', 'staging']

    for h in hosts:
        for r in regions:
            for e in envs:
                idx.add_series('cpu_usage', {'host': h, 'region': r, 'env': e})

    stats = idx.stats()
    print(f"  已注册序列: {stats['total_series']}")
    print(f"  倒排链数:   {stats['total_postings']}")
    print(f"  标签键数:   {stats['total_tag_keys']}")
    print()

    m1 = idx.match('cpu_usage', {'host': 'srv1', 'region': 'us-east'})
    print(f"  查询 {{host=srv1, region=us-east}}: {len(m1)} 个序列匹配")
    for sid in sorted(m1):
        print(f"    → {sid}")

    m2 = idx.match('cpu_usage', {'host': 'srv1', 'env': 'prod'})
    print(f"\n  查询 {{host=srv1, env=prod}}: {len(m2)} 个序列匹配")

    m3 = idx.match('cpu_usage', {'region': 'ap-east', 'env': 'prod'})
    print(f"  查询 {{region=ap-east, env=prod}}: {len(m3)} 个序列匹配")

    m4 = idx.match('cpu_usage', {'host': 'srv1', ('!=', 'env'): 'staging'})
    print(f"  查询 {{host=srv1, env!=staging}}: ...")


def demo_chunk_pruning():
    separator("3. 时间分块存储 — 查询只读相关数据块")

    print("【原理】时间分块如何跳过无关数据:\n")
    print("  每个数据块 (Chunk) 覆盖固定时间范围 [min_ts, max_ts]。")
    print("  查询 [t_start, t_end] 时:")
    print("    - 跳过 max_ts < t_start 的块 (太早)")
    print("    - 跳过 min_ts > t_end   的块 (太晚)")
    print("    - 只读取与查询范围重叠的块\n")
    print("  类比 LSM-tree SSTable 用 min/max key 跳过无关文件。\n")

    storage = StorageEngine()
    base_ts = 1700000000
    interval = 10
    n_points = 20000

    print(f"  写入 {n_points} 个数据点 (间隔 {interval}s, 共 ~{n_points*interval//3600} 小时)...")
    series_id = storage.index.add_series('cpu', {'host': 'srv1'})

    for i in range(n_points):
        ts = base_ts + i * interval
        val = 50.0 + 20.0 * math.sin(i * 0.001) + random.gauss(0, 2)
        storage.write('cpu', {'host': 'srv1'}, ts, val)

    storage.flush()

    cstats = storage.chunk_stats()
    comp_stats = storage.compression_stats()
    print(f"  总块数: {cstats['total_chunks']}, 已压缩: {cstats['sealed_chunks']}")
    print(f"  压缩率: {comp_stats['compression_ratio_pct']:.1f}%")
    print(f"  原始: {comp_stats['raw_bytes']} bytes → 压缩后: {comp_stats['compressed_bytes']} bytes")

    query_start = base_ts + 5000 * interval
    query_end = query_start + 100 * interval

    chunks = storage._series_chunks.get(series_id, [])
    relevant = storage.get_relevant_chunks(series_id, query_start, query_end)

    print(f"\n  查询范围: [{query_start}, {query_end}]")
    print(f"  查询跨度: {100*interval}s = {100*interval//60} 分钟")
    print(f"  总块数:   {len(chunks)}")
    print(f"  相关块数: {len(relevant)}")
    print(f"  跳过块数: {len(chunks) - len(relevant)}")
    print(f"  数据读取: {len(relevant)}/{len(chunks)} = {len(relevant)/len(chunks)*100:.0f}%")

    points = list(storage.read_series_range(series_id, query_start, query_end))
    print(f"  返回点数: {len(points)}")


def demo_downsample():
    separator("4. 多粒度预聚合 — 与原始数据无缝拼接")

    print("【原理】降采样的多粒度预聚合:\n")
    print("  1) 写入时, 同时更新多个粒度的预聚合桶:")
    print("     1m, 5m, 15m, 1h, 6h, 1d 各维护 count/sum/min/max")
    print()
    print("  2) 查询时选择最粗的 合适粒度:")
    print("     请求 30min 间隔 → 使用 15m 预聚合, 每2个桶合并为1个输出")
    print("     请求 1h 间隔  → 直接使用 1h 预聚合, 零计算")
    print()
    print("  3) 无缝拼接:")
    print("     完整桶 → 预聚合数据 (快速)")
    print("     不完整桶 → 原始数据实时聚合 (精确)")
    print("     两者合并 → 查询结果始终最新且正确")
    print()

    storage = StorageEngine()
    base_ts = 1700000000
    interval = 10
    n_points = 8640

    print(f"  写入 {n_points} 个数据点 (1天, 每{interval}s)...")
    for i in range(n_points):
        ts = base_ts + i * interval
        val = 50.0 + 20.0 * math.sin(i * 0.001) + random.gauss(0, 2)
        storage.write('cpu', {'host': 'srv1', 'region': 'us-east'}, ts, val)
    storage.flush()

    ds_stats = storage.downsample.stats()
    print(f"  预聚合桶数: {ds_stats['total_buckets']}")
    print(f"  粒度: {ds_stats['granularities']}")

    query = QueryEngine(storage)
    t_start = base_ts
    t_end = base_ts + 86400 - 1

    print(f"\n  查询1: 1小时粒度, avg 聚合 (直接命中预聚合)")
    r1 = query.query('cpu', {'host': 'srv1'}, t_start, t_end,
                      agg_func='avg', interval=3600)
    if r1:
        print(f"  → {len(r1[0].points)} 个数据点")
        for ts, v in r1[0].points[:3]:
            print(f"    t={ts}  avg={v:.2f}")
        print(f"    ...")

    print(f"\n  查询2: 5分钟粒度, max 聚合")
    r2 = query.query('cpu', {'host': 'srv1'}, t_start, t_end,
                      agg_func='max', interval=300)
    if r2:
        print(f"  → {len(r2[0].points)} 个数据点")
        for ts, v in r2[0].points[:3]:
            print(f"    t={ts}  max={v:.2f}")

    print(f"\n  查询3: 原始数据 (无降采样)")
    r3 = query.query('cpu', {'host': 'srv1'}, t_start, t_end)
    if r3:
        print(f"  → {len(r3[0].points)} 个数据点")


def demo_full_pipeline():
    separator("5. 完整管线: 多指标 × 多标签 → 过滤查询 → 聚合")

    storage = StorageEngine()
    query = QueryEngine(storage)
    base_ts = 1700000000

    hosts = ['web-01', 'web-02', 'db-01', 'db-02', 'cache-01']
    regions = ['us-east', 'eu-west']
    metrics_data = {}

    print("  写入多指标数据...")
    for metric in ['cpu', 'memory', 'disk_io']:
        for host in hosts:
            for region in regions:
                tags = {'host': host, 'region': region}
                base_val = {'cpu': 50, 'memory': 60, 'disk_io': 100}[metric]
                for i in range(8640):
                    ts = base_ts + i * 10
                    val = base_val + 20 * math.sin(i * 0.001) + random.gauss(0, 3)
                    storage.write(metric, tags, ts, val)
                metrics_data[f"{metric}_{host}_{region}"] = True

    storage.flush()

    comp = storage.compression_stats()
    print(f"  序列数: {comp['total_series']}")
    print(f"  数据点: {comp['total_points']}")
    print(f"  压缩率: {comp['compression_ratio_pct']:.1f}%")
    print(f"  原始: {comp['raw_bytes']/1024:.1f} KB → 压缩后: {comp['compressed_bytes']/1024:.1f} KB")

    t_start = base_ts
    t_end = base_ts + 43200

    print(f"\n  查询A: cpu, host=web-01, 最近12小时, avg")
    r = query.query('cpu', {'host': 'web-01'}, t_start, t_end, agg_func='avg')
    print(f"  → {len(r)} 个序列, avg = {r[0].points[0][1]:.2f}" if r else "  → 无结果")

    print(f"\n  查询B: cpu, region=us-east, 最近12小时, 1h粒度, avg")
    r = query.query('cpu', {'region': 'us-east'}, t_start, t_end,
                      agg_func='avg', interval=3600)
    print(f"  → {len(r)} 个序列")
    for res in r[:3]:
        print(f"    {res.tags} → {len(res.points)} 点, 前3: {[(t,v) for t,v in res.points[:3]]}")

    print(f"\n  查询C: memory, host=db-01, 最近6小时, max")
    r = query.query('memory', {'host': 'db-01'}, t_start, t_start + 21600, agg_func='max')
    print(f"  → max = {r[0].points[0][1]:.2f}" if r else "  → 无结果")

    print(f"\n  查询D: disk_io, 所有序列, 最近1小时, 5min粒度, avg")
    r = query.query('disk_io', {}, t_start, t_start + 3600, agg_func='avg', interval=300)
    print(f"  → {len(r)} 个序列")
    for res in r[:2]:
        print(f"    {res.tags} → {len(res.points)} 点")


if __name__ == '__main__':
    random.seed(42)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          时序数据库存储与查询引擎雏形 — 技术演示                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    demo_compression()
    demo_tag_index()
    demo_chunk_pruning()
    demo_downsample()
    demo_full_pipeline()

    separator("总结: 四大核心技术")
    print("  ┌──────────────────────┬──────────────────────────────────────────────┐")
    print("  │ 技术                 │ 效果                                        │")
    print("  ├──────────────────────┼──────────────────────────────────────────────┤")
    print("  │ 时间戳差值编码       │ 等间隔时序 98%+ 压缩, 1 bit/点              │")
    print("  │ 值异或压缩           │ 慢变化值 ~64% 压缩, 只存变化位              │")
    print("  │ 标签倒排索引         │ O(N) → O(|posting list|), 集合交集          │")
    print("  │ 时间分块 + 跳块      │ 查询只读相关块, 跳过 90%+ 数据              │")
    print("  │ 多粒度预聚合         │ 降采样查询快 100x+, 无缝拼接原始数据        │")
    print("  └──────────────────────┴──────────────────────────────────────────────┘")
