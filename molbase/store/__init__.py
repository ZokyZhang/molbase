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

from ..config import DB_PATH, ensure_dirs


@dataclass
class Entry:
    """知识库条目（统一抽象：知识 / 代码 / 技能）。"""

    entry_type: str                 # knowledge | code | skill
    entry_id: str                   # 唯一 ID
    title: str                      # 标题
    content: str                    # 正文（知识=文档正文，代码=源码，技能=SKILL 描述）
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
                meta        TEXT DEFAULT '{}'
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_type ON entries(entry_type)"
        )
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
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               REAL NOT NULL,
                query            TEXT DEFAULT '',
                route            TEXT DEFAULT '',
                hits             INTEGER DEFAULT 0,
                actual_tokens    INTEGER DEFAULT 0,
                baseline_tokens  INTEGER DEFAULT 0,
                confidence       REAL DEFAULT 0.0
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts)")
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
        conn.commit()

    # ------------------------------------------------------------------ 写操作
    def upsert(self, entry: Entry) -> None:
        """插入或更新条目（以 entry_id 为主键）。"""
        import json
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO entries (entry_id, entry_type, title, content, source,
                                 heat, created_at, last_access, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                entry_type=excluded.entry_type,
                title=excluded.title,
                content=excluded.content,
                source=excluded.source,
                heat=excluded.heat,
                last_access=excluded.last_access,
                meta=excluded.meta
            """,
            (
                entry.entry_id, entry.entry_type, entry.title, entry.content,
                entry.source, entry.heat, entry.created_at, entry.last_access,
                json.dumps(entry.meta, ensure_ascii=False),
            ),
        )
        self._conn.commit()

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
    ) -> None:
        """
        持久化一条 token 埋点（每次 query 调用一次）。

        这是「知识库本体」的监控表：把 token 优化效果落库，服务化后由
        /metrics 与监控大屏实时读取，实现「部署后埋点实时监控 token 优化内容」。
        baseline_tokens <= 0（如未开启 TOKEN_TRACE）时仍记录，但聚合端会跳过，
        避免拉低整体节省比例（口径与 TokenLedger 一致）。
        """
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO metrics (ts, query, route, hits, actual_tokens,
                                 baseline_tokens, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts if ts is not None else time.time(), query, route, hits,
             actual_tokens, baseline_tokens, confidence),
        )
        self._conn.commit()

    def list_metrics(self, limit: int = 200) -> list[dict]:
        """按时间倒序返回最近 N 条埋点明细（供趋势图 / 明细表）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, query, route, hits, actual_tokens, baseline_tokens, confidence "
            "FROM metrics ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def metric_aggregate(self) -> dict:
        """
        聚合全部埋点，返回累计 token 节省统计（对齐 TokenLedger.snapshot 口径）。

        - 仅统计 baseline_tokens > 0 的记录（未开启埋点的记录跳过）；
        - 返回 queries / actual_tokens / baseline_tokens / saved_tokens /
          saved_ratio / saved_ratio_pct / confidence_avg，供监控大屏核心指标卡。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n, "
            "COALESCE(SUM(actual_tokens), 0) AS act, "
            "COALESCE(SUM(baseline_tokens), 0) AS base, "
            "COALESCE(AVG(confidence), 0) AS conf "
            "FROM metrics WHERE baseline_tokens > 0"
        )
        row = cur.fetchone()
        n = int(row["n"] or 0)
        act = int(row["act"] or 0)
        base = int(row["base"] or 0)
        saved = base - act
        ratio = (saved / base) if base > 0 else 0.0
        return {
            "queries": n,
            "actual_tokens": act,
            "baseline_tokens": base,
            "saved_tokens": saved,
            "saved_ratio": round(ratio, 4),
            "saved_ratio_pct": f"{ratio * 100:.1f}%",
            "confidence_avg": round(float(row["conf"] or 0.0), 4),
        }

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
            source=row["source"],
            heat=row["heat"],
            created_at=row["created_at"],
            last_access=row["last_access"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

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
