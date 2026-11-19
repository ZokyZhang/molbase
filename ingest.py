# -*- coding: utf-8 -*-
"""
MolBase 知识库标准摄入入口（幂等，可重复运行）。

把三个正式语料目录摄入知识库：
- knowledge_repo/  -> 知识条目（kb::<文件名>）
- skills_repo/     -> 技能代管（skill::<技能名>，SKILL.md 解析）
- code_repo/       -> 代码符号索引（AST + 调用图）

用法：
    python ingest.py              # 全量摄入（upsert，不会重复）
    python ingest.py --fresh      # 先清空数据库再摄入（重建）

说明：
- 所有 ingest_* 均为 upsert（稳定 entry_id），重复运行安全。
- 运行后即可用 `python -m molbase.server` 起服务对外查询。
"""
from __future__ import annotations

import sys
from pathlib import Path

from molbase.config import CODE_REPO_DIR, KNOWLEDGE_DIR, SKILLS_DIR, DB_PATH
from molbase.orchestrator import Orchestrator


def main() -> None:
    fresh = "--fresh" in sys.argv
    if fresh:
        try:
            Path(DB_PATH).unlink(missing_ok=True)
            print(f"[fresh] 已删除旧库 {DB_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 删除旧库失败（可能被占用）：{exc}")

    orc = Orchestrator()
    nk = orc.ingest_knowledge_dir(KNOWLEDGE_DIR)
    ns = orc.ingest_skill_dir(SKILLS_DIR)
    nc = orc.ingest_code_repo(CODE_REPO_DIR)

    print(f"[ok] knowledge={nk}  skill={ns}  code={nc}")
    print(f"[stats] {orc.stats()}")
    orc.close()


if __name__ == "__main__":
    main()
