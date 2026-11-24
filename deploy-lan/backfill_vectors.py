#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回填向量（v9.1）：在虚拟机上把条目向量与原文块向量算齐。

为什么必须离线跑：730 条目 × 平均 17 块 = 12000+ 块，实测 hash 后端约 110 秒。
放在服务启动路径上会把重启拖成两分钟；放在 systemd 启动前跑又会让「重启即不可用」。
所以做成独立命令，由部署或运维在低峰期触发。

用法（在项目根执行）：
    python3 deploy-lan/backfill_vectors.py           # 条目向量 + 块向量
    python3 deploy-lan/backfill_vectors.py --entries # 只补条目向量
    python3 deploy-lan/backfill_vectors.py --chunks  # 只补块向量
    python3 deploy-lan/backfill_vectors.py --force   # 无视 text_hash 全量重算

幂等：默认按 text_hash 增量，重复执行只补变化的条目/块。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from molbase.config import DB_PATH, EMBEDDING_BACKEND, VECTOR_PRECOMPUTE_BATCH  # noqa: E402
from molbase.embedding import create_embedder  # noqa: E402
from molbase.kb_layers import rebuild_layers  # noqa: E402
from molbase.store import Store  # noqa: E402
from molbase.vectorize import ensure_entry_vectors  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MolBase 向量离线回填")
    parser.add_argument("--entries", action="store_true", help="只回填条目向量")
    parser.add_argument("--chunks", action="store_true", help="只回填原文块向量")
    parser.add_argument("--force", action="store_true", help="无视 text_hash 全量重算")
    parser.add_argument("--db", default="", help="数据库路径（默认取 config.DB_PATH）")
    parser.add_argument(
        "--backend", default="",
        help="向量后端，必须与服务一致（默认取 config.EMBEDDING_BACKEND）；"
             "hash / ollama / sentence-transformers",
    )
    args = parser.parse_args()

    # 两个开关都不给时视为「全都要」
    do_entries = args.entries or not (args.entries or args.chunks)
    do_chunks = args.chunks or not (args.entries or args.chunks)

    db_path = args.db or DB_PATH
    print(f"[回填] 数据库: {db_path}")
    store = Store(db_path)
    # 后端必须与服务一致：向量按 model_fp 分区，后端不同等于两套互不可见的向量。
    # 踩过的坑：VM 的 systemd 里写了 Environment=MOLBASE_EMBEDDING_BACKEND=hash，
    # 但直接 ssh 跑脚本时 shell 没有这个变量，config 默认 ollama（bge-m3），
    # 结果 12000+ 块按 bge-m3 逐条推理，要跑几个小时，而服务按 hash 查根本用不到。
    backend = args.backend or EMBEDDING_BACKEND
    embedder = create_embedder(backend)
    print(f"[回填] 向量后端: {getattr(embedder, 'name', '?')} dim={getattr(embedder, 'dim', 0)}")

    t0 = time.time()
    if do_entries:
        stat = ensure_entry_vectors(
            store, embedder, "entry", force=args.force, batch_size=VECTOR_PRECOMPUTE_BATCH
        )
        print(f"[回填] 条目向量: {stat} 耗时 {time.time() - t0:.1f}s")

    if do_chunks:
        t1 = time.time()
        # 原文层必须先把切块建齐，块向量才有对象可算
        result = rebuild_layers(store, embedder, with_vectors=True)
        print(f"[回填] 四层重建: {result} 耗时 {time.time() - t1:.1f}s")

    # 清孤儿向量：条目改稿后旧块 id 消失、以及「微块不算向量」规则生效后的历史残留
    pruned = store.prune_orphan_chunk_vectors()
    if pruned:
        print(f"[回填] 清理孤儿块向量 {pruned} 条")

    print(f"[回填] 合计耗时 {time.time() - t0:.1f}s")
    print(f"[回填] 覆盖检查: {store.chunk_stats()} 向量总数={store.count_vectors()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
