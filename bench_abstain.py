#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
弃答率回放台：把一批历史查询重跑一遍，统计弃答率与置信度分布。

用途：调完闸门/校准参数后，用它做「同一批查询、同一份数据」的前后对照，
避免拿少数几条金标问题自证。v9.2 的弃答线（0.43）就是用这个台子定的。

用法：
    # 从数据库里导出最近 100 条去重查询，直接回放
    python bench_abstain.py

    # 指定数据库 / 指定查询文件（每行一条）/ 指定条数
    python bench_abstain.py --db data/molbase.db --queries qs.txt --limit 200

输出一行 RESULT，便于脚本比对：
    RESULT n=100 abstain=13 rate=0.130 conf_mean=0.652 ...
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from molbase.config import DB_PATH
from molbase.orchestrator import Orchestrator


def load_queries(db_path: str, limit: int) -> list[str]:
    """从 metrics 表取最近的去重查询；库里没有指标表时退回空列表。"""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT query FROM metrics "
            "WHERE query IS NOT NULL AND length(query) > 0 "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description="MolBase 弃答率回放台")
    parser.add_argument("--db", default="", help="数据库路径（默认 config.DB_PATH）")
    parser.add_argument("--queries", default="", help="查询文件，每行一条（默认从 metrics 取）")
    parser.add_argument("--limit", type=int, default=100, help="最多回放多少条")
    args = parser.parse_args()

    db_path = args.db or str(DB_PATH)
    if args.queries:
        qs = [l.strip() for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        qs = load_queries(db_path, args.limit)
    qs = qs[: args.limit]
    if not qs:
        print("没有可回放的查询（metrics 为空，或 --queries 文件为空）")
        return 1

    orch = Orchestrator(db_path=db_path)
    abstain = 0
    confs: list[float] = []
    started = time.time()
    for q in qs:
        resp = orch.query(q)
        if resp.abstain:
            abstain += 1
        confs.append(resp.confidence)

    n = len(qs)
    print(
        f"RESULT n={n} abstain={abstain} rate={abstain / n:.3f} "
        f"conf_mean={sum(confs) / n:.3f} conf_min={min(confs):.3f} "
        f"conf_max={max(confs):.3f} elapsed={time.time() - started:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
