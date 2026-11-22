# -*- coding: utf-8 -*-
"""
存储底座：SQLite 持久化层。

职责（底座 = 存储引擎）：
- 统一管理三类条目：knowledge（知识）、code（代码）、skill（技能）。
- 记录每条条目的热度信号（heat）与新鲜度信号（last_access / created_at）。
- 提供增删改查 + 反馈回写接口。

设计说明：
- 使用标准库 sqlite3，零外部依赖，保证底座轻量可移植。
- 所有方法均有注释；所有参数硬编码于 config.py。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import ABSTAIN_CONFIDENCE, DB_PATH, ensure_dirs


@dataclass
class Entry:
    """知识库条目（统一抽象：知识 / 代码 / 技能）。"""

    entry_type: str                 # knowledge | code | skill
    entry_id: str                   # 唯一 ID
    title: str                      # 标题
    content: str                    # 正文（知识=文档正文，代码=源码，技能=SKILL 描述）
    summary: str = ""               # 语义保持摘要（空=未生成，回退返回原文）
    source: str = ""                # 来源路径或 URL
    heat: float = 0.0               # 热度信号（命中/采纳/引用累加）
    created_at: float = field(default_factory=time.time)   # 创建时间戳
    last_access: float = field(default_factory=time.time)  # 最近访问时间戳
    meta: dict = field(default_factory=dict)               # 扩展元数据（如符号名、关联关系）


class Store:
    """SQLite 存储底座。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """
        初始化存储，必要时建库建表。

        线程安全设计（服务化 HTTP API 需要）：SQLite 连接默认绑定创建它的线程，
        而 ThreadingHTTPServer 每个请求在独立线程处理。因此这里用 threading.local
        为每个线程维护独立连接，check_same_thread=False 关闭同线程校验，
        timeout 处理多连接并发写时的锁等待。其余方法通过 self._conn（property）
        自动取当前线程的连接，无需改动。
        """
        ensure_dirs()
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._local = threading.local()
        conn = self._connect()
        self._create_tables(conn)
        self._local.conn = conn

    def _connect(self) -> sqlite3.Connection:
        """新建一个 SQLite 连接（每线程一个，允许跨线程 + 写锁等待）。"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """返回当前线程的连接；不存在则惰性创建并建表。"""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._create_tables(conn)
            self._local.conn = conn
        return conn

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """建表：entries 表保存全部条目，meta 以 JSON 字符串存扩展字段。"""
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                entry_id    TEXT PRIMARY KEY,
                entry_type  TEXT NOT NULL,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                source      TEXT DEFAULT '',
                heat        REAL DEFAULT 0.0,
                created_at  REAL NOT NULL,
                last_access REAL NOT NULL,
                summary     TEXT DEFAULT '',
                meta        TEXT DEFAULT '{}'
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type)"
        )
        # 增量迁移：老库补 summary 列（已存在则静默跳过）
        try:
            cur.execute("ALTER TABLE entries ADD COLUMN summary TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        # 共现统计表：记录「同一查询一起命中」的条目对次数，驱动自我进化自动建边
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cooccur (
                a   TEXT NOT NULL,
                b   TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (a, b)
            )
            """
        )
        # 冲突待裁决表：记录写入冲突（同 ID、不同来源、内容矛盾）的被拒内容，供人工裁决
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conflicts (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id         TEXT NOT NULL,
                rejected_title   TEXT,
                rejected_content TEXT,
                rejected_source  TEXT,
                reason           TEXT,
                created_at       REAL
            )
            """
        )
        # token 埋点表：每次 query 持久化一条 token 优化记录，供监控大屏实时读取。
        # 定位（对齐「部署后埋点实时监控 token 优化内容」）：落库即「知识库本体」的一部分，
        # 与条目/共现/冲突同属一套 SQLite，重启不丢，服务化后大屏直接聚合展示。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL NOT NULL,
                query             TEXT DEFAULT '',
                route             TEXT DEFAULT '',
                hits              INTEGER DEFAULT 0,
                actual_tokens     INTEGER DEFAULT 0,
                baseline_tokens   INTEGER DEFAULT 0,
                confidence        REAL DEFAULT 0.0,
                is_test           INTEGER DEFAULT 0,
                recall_candidates INTEGER DEFAULT 0,
                recall_tokens     INTEGER DEFAULT 0
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts)")
        # 迁移：旧库 metrics 表平滑补列（is_test 测试隔离 / recall_candidates 召回候选数 /
        # recall_tokens 召回候选 token，即真实基线），幂等，重复执行无害。
        self._ensure_column(conn, "metrics", "is_test", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "recall_candidates", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "recall_tokens", "INTEGER DEFAULT 0")
        # 会话链路：query 埋点归属到具体会话（可选，未传则空），
        # 用于监控「每个会话走了 query/feedback/harvest/tooltrace 哪几步、缺哪步」。
        self._ensure_column(conn, "metrics", "session_id", "TEXT DEFAULT ''")
        # 接口调用日志表：记录各接口真实调用次数（驱动「调用率」指标），
        # 同样带 is_test 标记，测试期调用可隔离出真实统计。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_calls (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL NOT NULL,
                api     TEXT NOT NULL,
                is_test INTEGER DEFAULT 0
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls(ts)")
        # 工具调用轨迹表：记录每次会话喂入的工具调用序列，驱动「操作模式自进化」。
        # 相同「操作指纹」（工具名+参数键签名，值不参与）累计计数，达到阈值后自动
        # 生成可复用 skill，实现 N+1 -> N 的 token 收敛。持久化于知识库本体，重启不丢。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tooltraces (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               REAL NOT NULL,
                session_id       TEXT DEFAULT '',
                intent           TEXT DEFAULT '',
                fingerprint      TEXT NOT NULL,
                tool_calls       TEXT DEFAULT '[]'
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tooltraces_fp ON tooltraces(fingerprint)")
        # 反馈日志表：记录每次反馈动作（hit/adopt/cite/reject），驱动飞轮「反馈」环节统计
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                session_id  TEXT DEFAULT '',
                entry_id    TEXT NOT NULL,
                action      TEXT NOT NULL
            )
            """
        )
        # 入库日志表：记录每次摄入的冲突四态（new/merge/duplicate/conflict/pinned），
        # 驱动监控「有没有入、入成什么样」，而非只看接口被调了几次。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                entry_id    TEXT NOT NULL,
                state       TEXT NOT NULL,
                title       TEXT DEFAULT ''
            )
            """
        )
        # 进化日志表：记录每次自我进化的结果（建边/冷衰减/归档），驱动飞轮「进化」环节统计
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evolve_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                added_edges INTEGER DEFAULT 0,
                decayed     INTEGER DEFAULT 0,
                archived    INTEGER DEFAULT 0
            )
            """
        )
        # 弃答日志表：记录每次「宁可不答」的归因，让飞轮的「弃答」环节不再是死胡同，
        # 而是可反馈、可整改的缺口信号。
        #   reason    整改方向：no_data（没数据）/ weak_match（弱相关但不准）/ ambiguous（区分度不足）
        #   subreason 更细原因：empty_recall / weak_semantic / low_strength / low_margin
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS abstain_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                REAL NOT NULL,
                session_id        TEXT DEFAULT '',
                query             TEXT NOT NULL,
                route             TEXT NOT NULL,
                reason            TEXT NOT NULL,
                subreason         TEXT DEFAULT '',
                recall_candidates INTEGER DEFAULT 0,
                top1_score        REAL DEFAULT 0,
                confidence        REAL DEFAULT 0
            )
            """
        )
        conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        """
        幂等加列：列不存在时执行 ALTER TABLE ADD COLUMN（旧库平滑迁移）。

        用于给已存在的 metrics 表补齐新增列（is_test / recall_candidates /
        recall_tokens），避免「库已建、表已存」时 CREATE TABLE IF NOT EXISTS
        不生效导致缺列报错。
        """
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            conn.commit()

    # ------------------------------------------------------------------ 写操作
    def upsert(self, entry: Entry) -> None:
        """插入或更新条目（以 entry_id 为主键）。"""
        import json
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO entries (entry_id, entry_type, title, content, source,
                                 heat, created_at, last_access, summary, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                entry_type=excluded.entry_type,
                title=excluded.title,
                content=excluded.content,
                source=excluded.source,
                heat=excluded.heat,
                last_access=excluded.last_access,
                summary=excluded.summary,
                meta=excluded.meta
            """,
            (
                entry.entry_id, entry.entry_type, entry.title, entry.content,
                entry.source, entry.heat, entry.created_at, entry.last_access,
                entry.summary,
                json.dumps(entry.meta, ensure_ascii=False),
            ),
        )
        self._conn.commit()

    def update_summary(self, entry_id: str, summary: str) -> None:
        """批量回填用：按 ID 回写语义保持摘要。"""
        cur = self._conn.cursor()
        cur.execute("UPDATE entries SET summary = ? WHERE entry_id = ?", (summary, entry_id))
        self._conn.commit()

    def list_no_summary(self, limit: int = 50) -> list[Entry]:
        """批量回填用：列出 summary 为空的条目。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM entries WHERE summary = '' OR summary IS NULL LIMIT ?", (limit,)
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def delete(self, entry_id: str) -> bool:
        """按 ID 删除条目，返回是否删除成功。"""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, entry_id: str) -> Optional[Entry]:
        """按 ID 读取单条条目，不存在返回 None。"""
        import json
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM entries WHERE entry_id = ?", (entry_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def list_by_type(self, entry_type: str) -> list[Entry]:
        """按类型列出全部条目。"""
        import json
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM entries WHERE entry_type = ? ORDER BY heat DESC",
            (entry_type,),
        )
        return [self._row_to_entry(r) for r in cur.fetchall()]

    def all_entries(self) -> list[Entry]:
        """列出全部条目（供索引构建）。"""
        import json
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM entries")
        return [self._row_to_entry(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ 反馈回写
    def add_heat(self, entry_id: str, weight: float) -> None:
        """热度回写：按权重累加热度，并刷新 last_access 时间戳。"""
        now = time.time()
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE entries SET heat = heat + ?, last_access = ? WHERE entry_id = ?",
            (weight, now, entry_id),
        )
        self._conn.commit()

    def touch(self, entry_id: str) -> None:
        """仅刷新最近访问时间戳（不算热度，只算新鲜度）。"""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE entries SET last_access = ? WHERE entry_id = ?",
            (time.time(), entry_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ 自我进化
    def incr_cooccur(self, a: str, b: str) -> None:
        """共现计数 +1：同一查询一起命中的条目对（按字典序规范化 a<b 后持久化）。"""
        if a > b:
            a, b = b, a
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO cooccur (a, b, cnt) VALUES (?, ?, 1)
            ON CONFLICT(a, b) DO UPDATE SET cnt = cnt + 1
            """,
            (a, b),
        )
        self._conn.commit()

    def cooccur_pairs(self, min_count: int) -> list[tuple[str, str, int]]:
        """返回共现次数达到阈值的条目对 (a, b, cnt)。"""
        cur = self._conn.cursor()
        cur.execute("SELECT a, b, cnt FROM cooccur WHERE cnt >= ?", (min_count,))
        return [(r["a"], r["b"], r["cnt"]) for r in cur.fetchall()]

    def decay_cold(self, days: float, factor: float) -> int:
        """
        冷知识衰减：对超过 days 天未访问且热度大于 0 的条目，热度乘以 factor，
        返回被衰减的条目数（自然淘汰：不用的知识自动下沉）。
        """
        cutoff = time.time() - days * 86400.0
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE entries SET heat = heat * ? WHERE last_access < ? AND heat > 0",
            (factor, cutoff),
        )
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ 质量底线
    def update_meta(self, entry_id: str, patch: dict) -> bool:
        """
        合并更新条目 meta（质量底线各标志位共用入口）。

        patch 键值会被合并进条目现有 meta（不覆盖未提及的键），
        用于 pinned / quarantined / archived / reject_count 等状态演进。
        返回是否更新成功（条目不存在返回 False）。
        """
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.meta.update(patch)
        self.upsert(entry)
        return True

    def archive_dead(self, days: float, heat_max: float) -> int:
        """
        死知识归档：对「长期未访问」且满足「已隔离或热度 <= heat_max」的条目
        打 archived 标志，返回本次归档条数。

        - 归档条目不参与召回（_rebuild 会过滤），但保留在库，可人工恢复
          （清除 meta.archived 标志即可）。
        - 这是进化侧的「刹车」：防止低质/被隔离的死知识长期堆积污染索引。
        """
        cutoff = time.time() - days * 86400.0
        n = 0
        for e in self.all_entries():
            if e.meta.get("archived"):
                continue  # 已归档跳过
            if e.last_access < cutoff and (
                e.meta.get("quarantined") or e.heat <= heat_max
            ):
                e.meta["archived"] = True
                self.upsert(e)
                n += 1
        return n

    def record_conflict(
        self,
        entry_id: str,
        rejected_title: str,
        rejected_content: str,
        rejected_source: str,
        reason: str,
    ) -> None:
        """
        记录一次写入冲突（质量底线输入闸门）：被拒绝覆盖的内容落库，供人工裁决。

        - 不修改已有条目，只在 conflicts 表新增一条待裁决记录。
        - 保留被拒内容与来源，方便判断「以哪个为准」。
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO conflicts (entry_id, rejected_title, rejected_content,
                                   rejected_source, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (entry_id, rejected_title, rejected_content, rejected_source, reason, time.time()),
        )
        self._conn.commit()

    def list_conflicts(self) -> list[dict]:
        """列出全部待裁决冲突（供查询/人工处理）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entry_id, rejected_title, rejected_content, rejected_source, reason, created_at "
            "FROM conflicts ORDER BY created_at DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ token 埋点
    def record_metric(
        self,
        query: str,
        route: str,
        hits: int,
        actual_tokens: int,
        baseline_tokens: int,
        confidence: float = 0.0,
        ts: float | None = None,
        is_test: int = 0,
        recall_candidates: int = 0,
        recall_tokens: int = 0,
        session_id: str = "",
    ) -> None:
        """
        持久化一条 token 埋点（每次 query 调用一次）。

        这是「知识库本体」的监控表：把 token 优化效果落库，服务化后由
        /metrics 与监控大屏实时读取，实现「部署后埋点实时监控 token 优化内容」。
        - is_test           ：测试隔离标记（1=开发期 demo 查询，聚合默认排除）。
        - recall_candidates ：召回候选数（rerank 前命中的候选条数）。
        - recall_tokens     ：召回候选 token（真实基线，替代旧版「全库塞」口径）。
        - session_id        ：可选会话标识，用于监控「每个会话走了哪几步」。
        baseline_tokens <= 0（如未开启 TOKEN_TRACE）时仍记录，但聚合端会跳过，
        避免拉低整体节省比例（口径与 TokenLedger 一致）。
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO metrics (ts, query, route, hits, actual_tokens,
                                 baseline_tokens, confidence, is_test,
                                 recall_candidates, recall_tokens, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts if ts is not None else time.time(), query, route, hits,
             actual_tokens, baseline_tokens, confidence, is_test,
             recall_candidates, recall_tokens, session_id),
        )
        self._conn.commit()

    def list_metrics(self, limit: int = 200, include_test: bool = False) -> list[dict]:
        """按时间倒序返回最近 N 条埋点明细（供趋势图 / 明细表），默认排除测试数据。"""
        cur = self._conn.cursor()
        cols = ("ts, query, route, hits, actual_tokens, baseline_tokens, confidence, "
                "is_test, recall_candidates, recall_tokens")
        if include_test:
            cur.execute(f"SELECT {cols} FROM metrics ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cur.execute(
                f"SELECT {cols} FROM metrics WHERE is_test = 0 ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]

    def metric_aggregate(self, include_test: bool = False) -> dict:
        """
        聚合查询类埋点（route IN knowledge/code/skill），返回累计 token 节省 +
        全口径真实效能指标（默认排除测试数据 is_test=1）。

        真实基线口径：baseline_tokens 已改为「召回候选全集 token」（由上层写入），
        而非旧版「全库塞」；故 saved_ratio 反映的是「召回候选 → 最小上下文」的
        真实收敛效果，而非虚高的全库压缩率。

        额外产出的调优指标：
        - intercept_rate ：拦截率（命中且置信度达标，未被 abstain 拦下）；
        - abstain_rate   ：abstain 率（置信度低于阈值，宁可不答）；
        - avg_hits / avg_recall_candidates / recall_return_ratio ：召回收敛。
        """
        cur = self._conn.cursor()
        test_clause = "" if include_test else "AND is_test = 0"
        cur.execute(
            "SELECT COUNT(*) AS n, "
            "COALESCE(SUM(actual_tokens), 0) AS act, "
            "COALESCE(SUM(baseline_tokens), 0) AS base, "
            "COALESCE(AVG(confidence), 0) AS conf, "
            "COALESCE(AVG(hits), 0) AS avg_hits, "
            "COALESCE(SUM(hits), 0) AS hits_sum, "
            "COALESCE(AVG(recall_candidates), 0) AS avg_recall, "
            "SUM(CASE WHEN confidence >= ? AND hits > 0 THEN 1 ELSE 0 END) AS intercepted, "
            "SUM(CASE WHEN confidence < ? THEN 1 ELSE 0 END) AS abstained "
            "FROM metrics WHERE baseline_tokens > 0 "
            "AND route IN ('knowledge', 'code', 'skill') "
            f"{test_clause}",
            (ABSTAIN_CONFIDENCE, ABSTAIN_CONFIDENCE),
        )
        row = cur.fetchone()
        n = int(row["n"] or 0)
        act = int(row["act"] or 0)
        base = int(row["base"] or 0)
        saved = base - act
        ratio = (saved / base) if base > 0 else 0.0
        intercepted = int(row["intercepted"] or 0)
        abstained = int(row["abstained"] or 0)
        avg_hits = float(row["avg_hits"] or 0.0)
        avg_recall = float(row["avg_recall"] or 0.0)
        return {
            "queries": n,
            "actual_tokens": act,
            "baseline_tokens": base,
            "saved_tokens": saved,
            "saved_ratio": round(ratio, 4),
            "saved_ratio_pct": f"{ratio * 100:.1f}%",
            "confidence_avg": round(float(row["conf"] or 0.0), 4),
            "intercept_rate": round(intercepted / n, 4) if n else 0.0,
            "abstain_rate": round(abstained / n, 4) if n else 0.0,
            "avg_hits": round(avg_hits, 2),
            "hits_total": int(row["hits_sum"] or 0),
            "avg_recall_candidates": round(avg_recall, 2),
            "recall_return_ratio": round(avg_hits / avg_recall, 4) if avg_recall else 0.0,
        }

    def record_api_call(self, api: str, is_test: int = 0, ts: float | None = None) -> None:
        """
        记录一次接口调用（驱动「调用率」指标），带测试隔离标记。

        各 HTTP 接口（query/feedback/harvest/tooltrace/ingest/summarize/skill）
        每次被调用都落一条，供监控大屏统计真实调用分布。
        """
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO api_calls (ts, api, is_test) VALUES (?, ?, ?)",
            (ts if ts is not None else time.time(), api, is_test),
        )
        self._conn.commit()

    def api_calls_aggregate(self, include_test: bool = False) -> dict:
        """聚合各接口调用次数（默认排除测试），返回 {by_api: {api: count}, total}。"""
        cur = self._conn.cursor()
        if include_test:
            cur.execute("SELECT api, COUNT(*) AS cnt FROM api_calls GROUP BY api ORDER BY cnt DESC")
        else:
            cur.execute(
                "SELECT api, COUNT(*) AS cnt FROM api_calls WHERE is_test = 0 GROUP BY api ORDER BY cnt DESC"
            )
        by_api = {r["api"]: r["cnt"] for r in cur.fetchall()}
        return {"total": sum(by_api.values()), "by_api": by_api}

    # ------------------------------------------------------------------ 工具调用轨迹
    def record_tooltrace(
        self,
        session_id: str,
        intent: str,
        fingerprint: str,
        tool_calls_json: str,
        ts: float | None = None,
    ) -> int:
        """
        持久化一条工具调用轨迹，返回该「操作指纹」的累计出现次数。

        这是「操作模式自进化」的观测底座：大模型每次会话实际调用了哪些工具
        被落库，相同指纹（工具名+参数键签名）累计计数，达到阈值后由上层
        自动生成可复用 skill，实现 N+1 -> N 的 token 收敛。
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO tooltraces (ts, session_id, intent, fingerprint, tool_calls)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts if ts is not None else time.time(), session_id, intent, fingerprint, tool_calls_json),
        )
        self._conn.commit()
        cur.execute("SELECT COUNT(*) FROM tooltraces WHERE fingerprint = ?", (fingerprint,))
        return int(cur.fetchone()[0])

    def list_tooltraces(self, limit: int = 200) -> list[dict]:
        """按时间倒序返回最近 N 条工具调用轨迹（供明细查看）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, session_id, intent, fingerprint, tool_calls "
            "FROM tooltraces ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_tooltraces_by_session(self, session_id: str) -> list[dict]:
        """
        返回指定会话的全部工具调用轨迹（按时间正序），供会话回写关联「结论↔过程」。

        会话收尾时，harvest 把结论沉淀为资产，同时通过本方法取回同一 session
        的操作指纹写进资产 meta.related_fingerprints，从而把「得出了什么结论」
        与「是怎么干出来的」绑定在一起，避免下次依赖模型从结论反推过程。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT fingerprint, intent, tool_calls FROM tooltraces "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def tooltrace_aggregate(self, top_n: int = 8) -> dict:
        """
        聚合工具调用轨迹，返回「操作模式收敛」统计（供监控大屏展示「越用越聪明」）。

        返回结构：
            traces         ：轨迹总数
            fingerprints   ：不同操作指纹数
            top            ：高频指纹 Top 列表（fingerprint/count/intent），
                             体现「哪些重复操作已被观察、等待固化」。
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tooltraces")
        traces = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(DISTINCT fingerprint) FROM tooltraces")
        fingerprints = int(cur.fetchone()[0])
        cur.execute(
            "SELECT fingerprint, COUNT(*) AS cnt, MAX(intent) AS intent "
            "FROM tooltraces GROUP BY fingerprint ORDER BY cnt DESC LIMIT ?",
            (top_n,),
        )
        top = [
            {"fingerprint": r["fingerprint"], "count": r["cnt"], "intent": r["intent"] or ""}
            for r in cur.fetchall()
        ]
        return {"traces": traces, "fingerprints": fingerprints, "top": top}

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> Entry:
        """把 sqlite 行对象转成 Entry 数据类。"""
        import json
        return Entry(
            entry_type=row["entry_type"],
            entry_id=row["entry_id"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"] if "summary" in row.keys() else "",
            source=row["source"],
            heat=row["heat"],
            created_at=row["created_at"],
            last_access=row["last_access"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    # ------------------------------------------------------------------ 环节日志（飞轮监控）
    def record_feedback(self, session_id: str, entry_id: str, action: str, ts: float | None = None) -> None:
        """记录一次反馈动作（hit/adopt/cite/reject），驱动飞轮「反馈」环节统计。"""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO feedback_log (ts, session_id, entry_id, action) VALUES (?, ?, ?, ?)",
            (ts if ts is not None else time.time(), session_id, entry_id, action),
        )
        self._conn.commit()

    def record_ingest(self, entry_id: str, state: str, title: str = "", ts: float | None = None) -> None:
        """记录一次摄入的冲突四态结果，驱动「有没有入、入成什么样」统计。"""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO ingest_log (ts, entry_id, state, title) VALUES (?, ?, ?, ?)",
            (ts if ts is not None else time.time(), entry_id, state, title),
        )
        self._conn.commit()

    def record_evolve(self, added_edges: int, decayed: int, archived: int, ts: float | None = None) -> None:
        """记录一次自我进化的结果（建边/冷衰减/归档），驱动飞轮「进化」环节统计。"""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO evolve_log (ts, added_edges, decayed, archived) VALUES (?, ?, ?, ?)",
            (ts if ts is not None else time.time(), added_edges, decayed, archived),
        )
        self._conn.commit()

    def record_abstain(
        self,
        query: str,
        route: str,
        reason: str,
        subreason: str = "",
        recall_candidates: int = 0,
        top1_score: float = 0.0,
        confidence: float = 0.0,
        session_id: str = "",
        ts: float | None = None,
    ) -> None:
        """
        记录一次「弃答」及其归因，让弃答不再是死胡同，而是可反馈、可整改的缺口信号。

        reason 整改方向：no_data（没数据）/ weak_match（弱相关但不准）/ ambiguous（区分度不足）。
        subreason 更细原因：empty_recall / weak_semantic / low_strength / low_margin。
        """
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO abstain_log "
            "(ts, session_id, query, route, reason, subreason, recall_candidates, top1_score, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts if ts is not None else time.time(), session_id, query, route,
                reason, subreason, recall_candidates, top1_score, confidence,
            ),
        )
        self._conn.commit()

    def abstain_aggregate(self, limit: int = 30) -> dict:
        """
        聚合弃答归因：返回总量 + 一级/二级归因分布 + 最近弃答缺口清单。
        用于监控大屏「为什么拦截、怎么整改」。
        """
        cur = self._conn.cursor()
        cur.execute("SELECT reason, COUNT(*) AS cnt FROM abstain_log GROUP BY reason ORDER BY cnt DESC")
        by_reason = {r["reason"]: r["cnt"] for r in cur.fetchall()}
        cur.execute(
            "SELECT reason || ':' || subreason AS k, COUNT(*) AS cnt "
            "FROM abstain_log GROUP BY reason, subreason ORDER BY cnt DESC"
        )
        by_subreason = {r["k"]: r["cnt"] for r in cur.fetchall()}
        cur.execute(
            "SELECT query, route, reason, subreason, recall_candidates, "
            "ROUND(top1_score, 4) AS top1_score, ROUND(confidence, 4) AS confidence "
            "FROM abstain_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        recent = [dict(r) for r in cur.fetchall()]
        return {
            "total": sum(by_reason.values()),
            "by_reason": by_reason,
            "by_subreason": by_subreason,
            "recent": recent,
        }

    def abstain_gaps(self, limit: int = 20) -> list[dict]:
        """
        返回「弃答缺口清单」：按归因分组，给出最近未命中/不准的查询词，
        供后续补数据 / 调阈值 / 跑进化的整改依据。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT query, route, reason, subreason, recall_candidates, "
            "ROUND(confidence, 4) AS confidence, ts "
            "FROM abstain_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def feedback_aggregate(self) -> dict:
        """聚合反馈动作分布，返回 {by_action: {action: count}, total}。"""
        cur = self._conn.cursor()
        cur.execute("SELECT action, COUNT(*) AS cnt FROM feedback_log GROUP BY action ORDER BY cnt DESC")
        by_action = {r["action"]: r["cnt"] for r in cur.fetchall()}
        return {"total": sum(by_action.values()), "by_action": by_action}

    def ingest_aggregate(self) -> dict:
        """聚合入库结果分布，返回 {by_state: {state: count}, total}。"""
        cur = self._conn.cursor()
        cur.execute("SELECT state, COUNT(*) AS cnt FROM ingest_log GROUP BY state ORDER BY cnt DESC")
        by_state = {r["state"]: r["cnt"] for r in cur.fetchall()}
        return {"total": sum(by_state.values()), "by_state": by_state}

    def evolve_aggregate(self) -> dict:
        """聚合进化历史，返回 {steps, added_edges, decayed, archived}。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(added_edges),0) AS e, "
            "COALESCE(SUM(decayed),0) AS d, COALESCE(SUM(archived),0) AS a FROM evolve_log"
        )
        row = cur.fetchone()
        return {
            "steps": int(row["n"] or 0),
            "added_edges": int(row["e"] or 0),
            "decayed": int(row["d"] or 0),
            "archived": int(row["a"] or 0),
        }

    def writeback_recent(self, limit: int = 10) -> list[dict]:
        """
        聚合「回写」明细：把入库(ingest_log)、反馈(feedback_log)、进化(evolve_log)、
        工具轨迹(tooltraces) 四类写库动作按时间倒序合并，返回最近 N 条。

        会话回写(harvest) 复用 ingest_knowledge 落库、entry_id 以 session:: 开头，
        在此单独识别为 harvest 类型，便于大屏「回写明细」一屏看清所有写库动作。

        每条结构：{ts, type, type_cn, obj, state, detail}
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT ts, 'ingest' AS type, entry_id AS obj, state AS state, title AS detail
            FROM ingest_log
            UNION ALL
            SELECT ts, 'feedback' AS type, entry_id AS obj, action AS state, '' AS detail
            FROM feedback_log
            UNION ALL
            SELECT ts, 'evolve' AS type, '' AS obj, '' AS state,
                   CAST(added_edges AS TEXT) || ',' || CAST(decayed AS TEXT) || ',' || CAST(archived AS TEXT) AS detail
            FROM evolve_log
            UNION ALL
            SELECT ts, 'tooltrace' AS type, session_id AS obj, intent AS state, fingerprint AS detail
            FROM tooltraces
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        type_cn = {
            "ingest": "知识入库", "harvest": "会话回写", "feedback": "反馈",
            "evolve": "进化", "tooltrace": "工具轨迹",
        }
        out: list[dict] = []
        for r in rows:
            typ = r["type"]
            obj = r["obj"] or ""
            # 会话回写(harvest) 复用 ingest 落库，entry_id 以 session:: 开头，单列展示
            if typ == "ingest" and obj.startswith("session::"):
                typ = "harvest"
            detail = r["detail"] or ""
            # 进化日志的三元组 -> 可读摘要
            if typ == "evolve" and detail:
                try:
                    e, d, a = detail.split(",")
                    detail = f"建边 {e} · 冷衰减 {d} · 归档 {a}"
                except Exception:
                    detail = ""
            out.append({
                "ts": round(float(r["ts"]), 3),
                "type": typ,
                "type_cn": type_cn.get(typ, typ),
                "obj": obj,
                "state": r["state"] or "",
                "detail": detail,
            })
        return out

    def tag_stats(self) -> dict:
        """统计各标签（meta.tags）覆盖的条目数，返回 {tag: count}（按数量倒序）。"""
        counts: dict[str, int] = {}
        for e in self.all_entries():
            tags = e.meta.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            for t in tags:
                key = str(t).strip()
                if key:
                    counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

    def session_summary(self, limit: int = 30) -> list[dict]:
        """
        会话链路追踪：汇总每个出现过的 session_id 走完了哪几步，缺哪步。

        四步（对应一次理想会话的完整闭环）：
            query -> feedback -> harvest -> tooltrace
        数据来源：
            query     来自 metrics.session_id；
            feedback  来自 feedback_log.session_id；
            tooltrace 来自 tooltraces.session_id；
            harvest   来自 entries 中 entry_id = session::<id> 的沉淀条目。
        返回列表按「最近活跃」倒序，每项含四步计数与 missing（缺失步骤清单）。
        """
        import json as _json
        cur = self._conn.cursor()
        sessions: set[str] = set()
        for table in ("metrics", "feedback_log", "tooltraces"):
            for r in cur.execute(f"SELECT DISTINCT session_id FROM {table} WHERE session_id != ''"):
                sessions.add(r["session_id"])
        for r in cur.execute("SELECT entry_id FROM entries WHERE entry_id LIKE 'session::%'"):
            sessions.add(r["entry_id"].replace("session::", "", 1))

        rows: list[dict] = []
        for sid in sessions:
            q = cur.execute("SELECT COUNT(*) FROM metrics WHERE session_id = ?", (sid,)).fetchone()[0]
            f = cur.execute("SELECT COUNT(*) FROM feedback_log WHERE session_id = ?", (sid,)).fetchone()[0]
            t = cur.execute("SELECT COUNT(*) FROM tooltraces WHERE session_id = ?", (sid,)).fetchone()[0]
            h = cur.execute("SELECT COUNT(*) FROM entries WHERE entry_id = ?", (f"session::{sid}",)).fetchone()[0]
            step = {"query": int(q), "feedback": int(f), "harvest": int(h), "tooltrace": int(t)}
            missing = [k for k, v in step.items() if v == 0]
            rows.append({"session_id": sid, **step, "missing": missing})
        # 按活跃度（四步总数）倒序，取前 limit 个
        rows.sort(key=lambda r: r["query"] + r["feedback"] + r["harvest"] + r["tooltrace"], reverse=True)
        return rows[:limit]

    def session_participation(self, limit: int = 30) -> list[dict]:
        """
        会话参与度追踪：统计「每个会话 MolBase 到底参与了几次」，并判定逃票。

        直接回答三个监控问题：
            1. 一个会话里 MolBase 参与了几次（query_count，= 查询埋点数）；
            2. 是每次查询都参与、还是只查了第一次（query_count 的多少 + 命中/弃答分布）；
            3. 第一次查询无返回后，是不是就「带标签逃票」了（first_abstain 且未再查）。

        会话集合 = metrics / feedback_log / tooltraces 里出现过的 session_id
                  + entries 里 entry_id = session::<id> 的沉淀条目，保证「有动作但没
                  查询」的会话（未参与）也能被暴露出来。

        每条结构：{session_id, query_count, hit_count, abstain_count,
                   first_abstain, requery_after_abstain, evasion}
        """
        cur = self._conn.cursor()
        sessions: set[str] = set()
        for table in ("metrics", "feedback_log", "tooltraces"):
            for r in cur.execute(f"SELECT DISTINCT session_id FROM {table} WHERE session_id != ''"):
                sessions.add(r["session_id"])
        for r in cur.execute("SELECT entry_id FROM entries WHERE entry_id LIKE 'session::%'"):
            sessions.add(r["entry_id"].replace("session::", "", 1))

        query_route_sql = "route IN ('knowledge','code','skill')"
        out: list[dict] = []
        for sid in sessions:
            q = cur.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN confidence >= ? AND hits > 0 THEN 1 ELSE 0 END) AS hit, "
                "SUM(CASE WHEN confidence < ? OR hits <= 0 THEN 1 ELSE 0 END) AS abstain, "
                "MIN(id) AS first_id "
                f"FROM metrics WHERE session_id = ? AND {query_route_sql}",
                (ABSTAIN_CONFIDENCE, ABSTAIN_CONFIDENCE, sid),
            ).fetchone()
            qc = int(q["n"] or 0)
            hit = int(q["hit"] or 0)
            abstain = int(q["abstain"] or 0)
            first_id = q["first_id"]

            first_abstain = False
            requery = False
            if first_id is not None:
                fr = cur.execute(
                    "SELECT confidence, hits FROM metrics WHERE id = ?", (first_id,),
                ).fetchone()
                first_abstain = bool(
                    fr and (fr["confidence"] < ABSTAIN_CONFIDENCE or fr["hits"] <= 0)
                )
                if first_abstain:
                    after = cur.execute(
                        f"SELECT COUNT(*) FROM metrics WHERE session_id = ? "
                        f"AND {query_route_sql} AND id > ?",
                        (sid, first_id),
                    ).fetchone()[0]
                    requery = int(after) > 0

            # 逃票判定（按严重程度排序）
            if qc == 0:
                evasion = "未参与（有动作无查询）"
            elif first_abstain and not requery:
                evasion = "首次无返回即逃票"
            elif qc == 1:
                evasion = "仅查1次（疑似逃票）"
            else:
                evasion = "正常参与"

            out.append({
                "session_id": sid,
                "query_count": qc,
                "hit_count": hit,
                "abstain_count": abstain,
                "first_abstain": first_abstain,
                "requery_after_abstain": requery,
                "evasion": evasion,
            })

        # 逃票/未参与排在前面（更值得关注），同档按参与次数倒序
        order = {"首次无返回即逃票": 0, "未参与（有动作无查询）": 1, "仅查1次（疑似逃票）": 2, "正常参与": 3}
        out.sort(key=lambda r: (order.get(r["evasion"], 9), -r["query_count"]))
        return out[:limit]

    def close(self) -> None:
        """关闭当前线程的数据库连接（各线程连接随线程结束自动释放）。"""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def __enter__(self) -> "Store":
        """上下文管理器入口。"""
        return self

    def __exit__(self, *args: Any) -> None:
        """上下文管理器出口，自动关闭连接。"""
        self.close()
