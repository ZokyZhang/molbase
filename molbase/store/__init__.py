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

import json
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

    # 全量明细埋点字段白名单（v6）：每次 query 的场景/标签/实体/置信度分量/延迟/模型等
    DETAIL_COLUMNS = (
        "scene_id", "label", "tags", "query_tags", "entities", "refs",
        "rewritten", "resolved_version_id", "refs_miss", "abstain",
        "abstain_reason", "abstain_subreason", "strength", "margin",
        "top1_score", "top1_entry_id", "top1_title", "hits_detail",
        "web_results", "embed_model", "embedding_backend", "rewrite_used",
        "latency_total_ms", "latency_recall_ms", "latency_rerank_ms",
        "confidence_threshold"
    )

    # 全量对比支持的分组维度
    ALLOWED_GROUP_BY = (
        "scene_id", "label", "route", "abstain", "abstain_reason",
        "embed_model", "embedding_backend", "session_id", "day"
    )

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
        # v6 全量明细监控：场景/标记/标签/实体/置信度分量/延迟/模型等 26 个维度，
        # 全部幂等补列，旧库可直接平滑升级。
        self._ensure_column(conn, "metrics", "scene_id", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "label", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "tags", "TEXT DEFAULT '[]'")
        self._ensure_column(conn, "metrics", "query_tags", "TEXT DEFAULT '[]'")
        self._ensure_column(conn, "metrics", "entities", "TEXT DEFAULT '[]'")
        self._ensure_column(conn, "metrics", "refs", "TEXT DEFAULT '[]'")
        self._ensure_column(conn, "metrics", "rewritten", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "resolved_version_id", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "refs_miss", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "abstain", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "abstain_reason", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "abstain_subreason", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "strength", "REAL DEFAULT 0")
        self._ensure_column(conn, "metrics", "margin", "REAL DEFAULT 0")
        self._ensure_column(conn, "metrics", "top1_score", "REAL DEFAULT 0")
        self._ensure_column(conn, "metrics", "top1_entry_id", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "top1_title", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "hits_detail", "TEXT DEFAULT '[]'")
        self._ensure_column(conn, "metrics", "web_results", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "embed_model", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "embedding_backend", "TEXT DEFAULT ''")
        self._ensure_column(conn, "metrics", "rewrite_used", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "latency_total_ms", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "latency_recall_ms", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "latency_rerank_ms", "INTEGER DEFAULT 0")
        self._ensure_column(conn, "metrics", "confidence_threshold", "REAL DEFAULT 0")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_scene ON metrics(scene_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_label ON metrics(label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_abstain ON metrics(abstain)")
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
        # v7 自进化闭环：区分人工反馈与命中后自动正反馈（auto_hit）
        self._ensure_column(conn, "feedback_log", "source", "TEXT DEFAULT 'manual'")
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
        # v7.2 弃答补数草稿表：弃答→草稿→确认入库→命中回验
        cur.execute("""
            CREATE TABLE IF NOT EXISTS draft_log (
                draft_id         TEXT PRIMARY KEY,
                query            TEXT NOT NULL,
                reason           TEXT DEFAULT '',
                subreason        TEXT DEFAULT '',
                status           TEXT DEFAULT 'pending',
                created_entry_id TEXT DEFAULT '',
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                confirmed_at     DATETIME
            )
        """)
        # 趋势快照表：驱动历史趋势曲线，重启不丢（v4 新增）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metric_snapshots (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                metric    TEXT NOT NULL,
                value     REAL NOT NULL,
                detail    TEXT DEFAULT '{}'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_metric_ts ON metric_snapshots(metric, ts)")
        # v3 场景锚定层：建6张表 + entries.scene 列（幂等，可安全重复调用）
        self._create_scene_tables(conn)
        conn.commit()

    def _create_scene_tables(self, conn: sqlite3.Connection) -> None:
        """
        创建场景锚定层 6 张表，并为 entries 表幂等添加 scene 列。
        包含：scenes, topic_tree, scene_entries, entity_index, knowledge_versions, semantic_cache。
        """
        cur = conn.cursor()

        # 1. 创建 scenes 表：存储场景元数据
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                scene_path TEXT,
                parent_id TEXT,
                facets TEXT,
                description TEXT,
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. 创建 topic_tree 表：存储场景内的主题树结构
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topic_tree (
                topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT,
                parent_id INTEGER,
                level INTEGER,
                scene_id TEXT,
                FOREIGN KEY (scene_id) REFERENCES scenes(scene_id)
            )
        """)

        # 3. 创建 scene_entries 表：建立场景与 entries 的关联
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scene_entries (
                scene_id TEXT,
                entry_id TEXT,
                via TEXT,
                weight REAL,
                PRIMARY KEY (scene_id, entry_id),
                FOREIGN KEY (entry_id) REFERENCES entries(entry_id)
            )
        """)

        # 4. 创建 entity_index 表：实体索引，用于知识检索
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entity_index (
                entity TEXT,
                entry_id TEXT,
                knowledge_id TEXT,
                source TEXT,
                confidence REAL,
                PRIMARY KEY (entity, entry_id),
                FOREIGN KEY (entry_id) REFERENCES entries(entry_id)
            )
        """)

        # 5. 创建 knowledge_versions 表：知识版本管理
        cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_versions (
                knowledge_id TEXT,
                version_id TEXT,
                scene_id TEXT,
                answer_entry_id TEXT,
                answer_digest TEXT,
                facets_match TEXT,
                authority REAL,
                confidence REAL,
                status TEXT DEFAULT 'active',
                valid_from DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (knowledge_id, version_id),
                FOREIGN KEY (scene_id) REFERENCES scenes(scene_id),
                FOREIGN KEY (answer_entry_id) REFERENCES entries(entry_id)
            )
        """)

        # 6. 创建 semantic_cache 表：语义缓存，加速问答检索
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                q_key TEXT,
                scene_id TEXT,
                question_vec BLOB,
                answer TEXT,
                evidence_ids TEXT,
                hit_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (q_key, scene_id)
            )
        """)

        # 幂等性操作：给 entries 表添加 scene 列，用于关联场景
        self._ensure_column(conn, "entries", "scene", "TEXT DEFAULT 'global'")

        # 创建常用查询索引
        # 索引 scene_entries.scene_id：加速场景内条目查询
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scene_entries_scene_id ON scene_entries(scene_id)")
    
        # 索引 entity_index.entity：加速实体检索
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_index_entity ON entity_index(entity)")
    
        # 索引 knowledge_versions.scene_id：加速版本查询
        cur.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_versions_scene_id ON knowledge_versions(scene_id)")
    
        # 索引 semantic_cache.scene_id：加速缓存命中查询
        cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_scene_id ON semantic_cache(scene_id)")

        # 7. 创建 vector_index 表：条目/原文块的向量索引（v9.0）
        # 主键含 owner_kind：同一 owner_id 在 entry 与 chunk 两个命名空间下可共存向量，
        # 且后续若引入多粒度（标题级/段落级）向量不必再改表。
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vector_index (
                owner_id TEXT NOT NULL, owner_kind TEXT NOT NULL, model_fp TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL, text_hash TEXT DEFAULT '',
                updated_at REAL NOT NULL,
                PRIMARY KEY (owner_id, owner_kind, model_fp)
            )
        """)
        # 回填查询与按后端统计都走 (model_fp, owner_kind) 组合，索引列序与查询条件对齐
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_index_kind ON vector_index(model_fp, owner_kind)")

        # 迁移：早期开发库里 vector_index 主键是 (owner_id, model_fp)，缺 owner_kind。
        # CREATE TABLE IF NOT EXISTS 对已存在的表不生效，若不重建，entry 与 chunk 的
        # 同 id 向量会互相覆盖。向量可重算，直接重建最安全。
        pk_cols = [r[1] for r in cur.execute("PRAGMA table_info(vector_index)").fetchall()
                   if len(r) > 5 and r[5]]
        if pk_cols and "owner_kind" not in pk_cols:
            cur.execute("DROP TABLE vector_index")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_index (
                    owner_id TEXT NOT NULL, owner_kind TEXT NOT NULL, model_fp TEXT NOT NULL,
                    dim INTEGER NOT NULL, vec BLOB NOT NULL, text_hash TEXT DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (owner_id, owner_kind, model_fp)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_index_kind ON vector_index(model_fp, owner_kind)")

        # 8. 原文层（v9.1）：条目切块后的原文块，承载「引用回溯」与「块级精准召回」
        #    path 是标题栈快照（如 "部署/VM/端口"），char_start/char_end 是块在条目正文中的字符区间，
        #    两者合起来足以把一句答案精确还原到原文位置。
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                heading TEXT DEFAULT '',
                path TEXT DEFAULT '',
                char_start INTEGER DEFAULT 0,
                char_end INTEGER DEFAULT 0,
                text_len INTEGER DEFAULT 0,
                body TEXT NOT NULL,
                text_hash TEXT DEFAULT '',
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_entry ON chunks(entry_id, ordinal)")

        # 9. 受控标签词表（v9.1）：把自由标签收敛成「归一化标签 -> 父标签 + 文档频次」
        #    df 用于给标签加成做 idf 加权，避免「通用/其他」这类高频标签贡献虚假置信度。
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tag_vocab (
                tag TEXT PRIMARY KEY,
                parent TEXT DEFAULT '',
                df INTEGER DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tag_vocab_parent ON tag_vocab(parent)")

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

    # ------------------------------------------------------------------ 向量索引（v9.0）
    @staticmethod
    def _pack_vec(vec: list) -> tuple:
        """把 float 列表打包成 float32 BLOB，返回 (blob, dim)。统一字节序避免跨平台误读。"""
        import struct
        dim = len(vec)
        return struct.pack(f"<{dim}f", *vec), dim

    @staticmethod
    def _unpack_vec(blob, dim: int) -> list | None:
        """把 float32 BLOB 解包成 float 列表；长度与 dim 不符时返回 None（脏数据自愈）。"""
        import struct
        if not blob:
            return None
        try:
            if len(blob) != int(dim) * 4:
                return None
            return list(struct.unpack(f"<{int(dim)}f", blob))
        except Exception:
            return None

    def upsert_vector(self, owner_id: str, owner_kind: str, model_fp: str, vec: list, text_hash: str = "") -> None:
        """插入或更新单条向量（主键 owner_id+owner_kind+model_fp）。"""
        self.upsert_vectors([(owner_id, owner_kind, model_fp, vec, text_hash)])

    def upsert_vectors(self, rows: list) -> int:
        """
        批量插入或更新向量，整批只提交一次。

        rows 为 (owner_id, owner_kind, model_fp, vec, text_hash) 元组列表。
        返回实际写入条数；空向量条目直接跳过（不做静默零向量占位，
        否则零向量会被当成「已有向量」进入重排，把真实条目压下去）。
        """
        import time
        if not rows:
            return 0
        now = time.time()
        payload = []
        for owner_id, owner_kind, model_fp, vec, text_hash in rows:
            if not vec:
                continue
            blob, dim = self._pack_vec(vec)
            payload.append((owner_id, owner_kind, model_fp, dim, blob, text_hash or "", now))
        if not payload:
            return 0
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO vector_index(owner_id, owner_kind, model_fp, dim, vec, text_hash, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            payload,
        )
        self._conn.commit()
        return len(payload)

    def get_vector(self, owner_id: str, model_fp: str, owner_kind: str = "entry") -> list | None:
        """按 owner_id + 后端指纹取单条向量，返回 float 列表或 None。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT vec, dim FROM vector_index WHERE owner_id = ? AND model_fp = ? AND owner_kind = ?",
            (owner_id, model_fp, owner_kind),
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._unpack_vec(row[0], row[1])

    def get_vectors(self, model_fp: str, owner_kind: str = "", owner_ids: list | None = None) -> dict:
        """
        按后端指纹批量取向量，返回 {owner_id: [float]}。

        owner_ids 为明确候选集时必须传入：重排只需要候选集的向量，
        不传会整表扫（条目上千时白读 MB 级 BLOB）。
        """
        result = {}
        cur = self._conn.cursor()
        if owner_ids is not None and len(owner_ids) > 0:
            # SQLite 默认变量上限 999，留余量按 900 分批
            batch_size = 900
            for i in range(0, len(owner_ids), batch_size):
                batch = owner_ids[i:i + batch_size]
                placeholders = ",".join(["?"] * len(batch))
                # 拼接进去的只有字面量占位符 ?，所有取值都走 params 绑定
                sql = f"SELECT owner_id, vec, dim FROM vector_index WHERE model_fp = ? AND owner_id IN ({placeholders})"  # nosec B608
                params = [model_fp] + list(batch)
                if owner_kind:
                    sql += " AND owner_kind = ?"
                    params.append(owner_kind)
                cur.execute(sql, params)
                for owner_id, vec_blob, dim in cur.fetchall():
                    v = self._unpack_vec(vec_blob, dim)
                    if v is not None:
                        result[owner_id] = v
            return result

        sql = "SELECT owner_id, vec, dim FROM vector_index WHERE model_fp = ?"
        params = [model_fp]
        if owner_kind:
            sql += " AND owner_kind = ?"
            params.append(owner_kind)
        cur.execute(sql, params)
        for owner_id, vec_blob, dim in cur.fetchall():
            v = self._unpack_vec(vec_blob, dim)
            if v is not None:
                result[owner_id] = v
        return result

    def get_vector_hashes(self, model_fp: str, owner_kind: str = "") -> dict:
        """返回 {owner_id: text_hash}，用于向量增量更新判断。"""
        query = "SELECT owner_id, text_hash FROM vector_index WHERE model_fp = ?"
        params = [model_fp]
        if owner_kind:
            query += " AND owner_kind = ?"
            params.append(owner_kind)
        cursor = self._conn.cursor()
        cursor.execute(query, params)
        result = {}
        for owner_id, text_hash in cursor.fetchall():
            result[owner_id] = text_hash if text_hash is not None else ""
        return result

    def count_vectors(self, model_fp: str = "", owner_kind: str = "") -> int:
        """统计向量数量，支持 model_fp 和 owner_kind 过滤。"""
        cur = self._conn.cursor()
        if model_fp or owner_kind:
            conditions = []
            params = []
            if model_fp:
                conditions.append("model_fp = ?")
                params.append(model_fp)
            if owner_kind:
                conditions.append("owner_kind = ?")
                params.append(owner_kind)
            # conditions 只由本方法内硬编码的 "列 = ?" 片段拼成，无外部字符串
            query = f"SELECT COUNT(*) FROM vector_index WHERE {' AND '.join(conditions)}"  # nosec B608
            cur.execute(query, params)
        else:
            cur.execute("SELECT COUNT(*) FROM vector_index")
        return cur.fetchone()[0]

    def delete_vectors(self, owner_id: str = "", model_fp: str = "", owner_kind: str = "") -> int:
        """
        删除向量，返回删除行数。

        安全约束：owner_id 为空时**拒绝全表删除**，必须显式给 model_fp 或 owner_kind
        做范围限定。整表清空只允许走 rebuild_vectors 这类显式维护入口，
        避免一个漏传参数把整库向量抹掉。
        """
        cur = self._conn.cursor()
        conditions = []
        params = []
        if owner_id:
            conditions.append("owner_id = ?")
            params.append(owner_id)
        if model_fp:
            conditions.append("model_fp = ?")
            params.append(model_fp)
        if owner_kind:
            conditions.append("owner_kind = ?")
            params.append(owner_kind)
        if not conditions:
            # 无任何限定条件：直接返回 0，不做删除
            return 0
        # conditions 只由本方法内硬编码的 "列 = ?" 片段拼成，无外部字符串
        query = f"DELETE FROM vector_index WHERE {' AND '.join(conditions)}"  # nosec B608
        cur.execute(query, params)
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ 弃答补数草稿（v7.2）
    def create_draft(self, query: str, reason: str = "", subreason: str = "") -> str:
        """创建补数草稿（按 query+reason 哈希幂等，重复返回同一 id）。"""
        import hashlib
        draft_id = "draft::" + hashlib.md5((query + "|" + reason).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        cur = self._conn.cursor()
        cur.execute("SELECT draft_id FROM draft_log WHERE draft_id=?", (draft_id,))
        if cur.fetchone():
            return draft_id
        cur.execute("INSERT INTO draft_log(draft_id,query,reason,subreason) VALUES (?,?,?,?)", (draft_id, query, reason, subreason))
        self._conn.commit()
        return draft_id

    def list_drafts(self, status: str = "pending", limit: int = 50) -> list[dict]:
        """列出草稿（status 为空则查全部）。"""
        cur = self._conn.cursor()
        if status:
            cur.execute("SELECT draft_id,query,reason,subreason,status,created_entry_id,created_at FROM draft_log WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            cur.execute("SELECT draft_id,query,reason,subreason,status,created_entry_id,created_at FROM draft_log ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def confirm_draft(self, draft_id: str, created_entry_id: str = "") -> bool:
        """确认草稿并回填入库条目 id。"""
        cur = self._conn.cursor()
        cur.execute("UPDATE draft_log SET status='confirmed',created_entry_id=?,confirmed_at=CURRENT_TIMESTAMP WHERE draft_id=?", (created_entry_id, draft_id))
        self._conn.commit()
        return cur.rowcount > 0

    def reject_draft(self, draft_id: str) -> bool:
        """驳回草稿。"""
        cur = self._conn.cursor()
        cur.execute("UPDATE draft_log SET status='rejected',confirmed_at=CURRENT_TIMESTAMP WHERE draft_id=?", (draft_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def draft_stats(self) -> dict:
        """草稿状态分布统计。"""
        cur = self._conn.cursor()
        cur.execute("SELECT status,COUNT(*) FROM draft_log GROUP BY status")
        by_status = {r[0]: r[1] for r in cur.fetchall()}
        return {"total": sum(by_status.values()), "by_status": by_status}

    def get_draft(self, draft_id: str) -> dict | None:
        """按 id 取草稿详情。"""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM draft_log WHERE draft_id=?", (draft_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ 场景锚定层写入（v7.2）
    def upsert_scene(self, scene_id: str, scene_path: str = "", parent_id: str = "", facets: str = "", description: str = "") -> None:
        """插入或更新场景（以 scene_id 为主键）。"""
        import json
        cur = self._conn.cursor()
        if isinstance(facets, (dict, list)):
            facets = json.dumps(facets, ensure_ascii=False)
        cur.execute(
            "INSERT OR REPLACE INTO scenes(scene_id,scene_path,parent_id,facets,description) VALUES (?,?,?,?,?)",
            (scene_id, scene_path, parent_id, facets, description)
        )
        self._conn.commit()

    def link_scene_entry(self, scene_id: str, entry_id: str, via: str = "auto", weight: float = 1.0) -> None:
        """链接场景与条目（以 scene_id,entry_id 为主键）。"""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO scene_entries(scene_id,entry_id,via,weight) VALUES (?,?,?,?)",
            (scene_id, entry_id, via, weight)
        )
        self._conn.commit()

    def upsert_entity(self, entity: str, entry_id: str, knowledge_id: str = "", source: str = "", confidence: float = 0.0) -> None:
        """插入或更新实体索引（以 entity,entry_id 为主键）。"""
        if not entity or not entry_id:
            return
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO entity_index(entity,entry_id,knowledge_id,source,confidence) VALUES (?,?,?,?,?)",
            (entity, entry_id, knowledge_id, source, confidence)
        )
        self._conn.commit()

    def upsert_version(self, knowledge_id: str, version_id: str, scene_id: str = "", answer_entry_id: str = "", answer_digest: str = "", facets_match: str = "", authority: float = 0.0, confidence: float = 0.0) -> None:
        """插入或更新知识版本（以 knowledge_id,version_id 为主键）。"""
        import json
        if isinstance(facets_match, (dict, list)):
            facets_match = json.dumps(facets_match, ensure_ascii=False)
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO knowledge_versions(knowledge_id,version_id,scene_id,answer_entry_id,answer_digest,facets_match,authority,confidence) VALUES (?,?,?,?,?,?,?,?)",
            (knowledge_id, version_id, scene_id, answer_entry_id, answer_digest, facets_match, authority, confidence)
        )
        self._conn.commit()

    def scene_stats(self) -> dict:
        """返回场景锚定层各表记录数统计。"""
        cur = self._conn.cursor()
        return {
            "scenes": cur.execute("SELECT COUNT(*) FROM scenes").fetchone()[0],
            "scene_entries": cur.execute("SELECT COUNT(*) FROM scene_entries").fetchone()[0],
            "entity_index": cur.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0],
            "knowledge_versions": cur.execute("SELECT COUNT(*) FROM knowledge_versions").fetchone()[0],
        }

    def get_scene_entries(self, scene_id: str, limit: int = 50) -> list[dict]:
        """获取指定场景下的条目列表（按权重降序）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entry_id,via,weight FROM scene_entries WHERE scene_id=? ORDER BY weight DESC LIMIT ?",
            (scene_id, limit)
        )
        return [dict(r) for r in cur.fetchall()]

    def get_entities_for_entry(self, entry_id: str) -> list[dict]:
        """获取指定条目关联的实体列表。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entity,knowledge_id,source,confidence FROM entity_index WHERE entry_id=?",
            (entry_id,)
        )
        return [dict(r) for r in cur.fetchall()]

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
        detail: dict | None = None,
    ) -> None:
        """
        持久化一条全量明细埋点（每次 query 调用一次）。

        原 11 参数签名保持不变（向后兼容）；detail 字典承载 v6 的 26 个维度：
        场景/标记/显式标签/解析标签/实体/引用/改写/版本/弃答归因/置信度分量/
        top1 命中/延迟/模型等。JSON 类字段自动序列化为字符串。
        """
        cur = self._conn.cursor()
        cols = ["ts", "query", "route", "hits", "actual_tokens", "baseline_tokens",
                "confidence", "is_test", "recall_candidates", "recall_tokens", "session_id"]
        vals = [ts if ts is not None else time.time(), query, route, hits,
                actual_tokens, baseline_tokens, confidence, is_test,
                recall_candidates, recall_tokens, session_id]
        json_fields = {"tags", "query_tags", "entities", "refs", "hits_detail"}
        if detail:
            for col in self.DETAIL_COLUMNS:
                val = detail.get(col)
                if val is not None:
                    cols.append(col)
                    if col in json_fields and isinstance(val, (list, dict)):
                        vals.append(json.dumps(val, ensure_ascii=False))
                    else:
                        vals.append(val)
        placeholders = ", ".join(["?"] * len(cols))
        cur.execute(
            f"INSERT INTO metrics ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        self._conn.commit()

    def list_metrics(self, limit: int = 200, include_test: bool = False) -> list[dict]:
        """按时间倒序返回最近 N 条全字段埋点明细，默认排除测试数据。"""
        cur = self._conn.cursor()
        if include_test:
            cur.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,))
        else:
            cur.execute(
                "SELECT * FROM metrics WHERE is_test = 0 ORDER BY id DESC LIMIT ?",
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

    def metric_compare(self, group_by: str = "scene_id", include_test: bool = False, limit: int = 50) -> dict:
        """
        全量明细对比：按任意维度分组，输出每组的置信度/强度/间隔/命中/token/延迟/
        拦截率/弃答率/召回收敛/节省率的均值，用于「所有维度明细对比」。
        """
        if group_by not in self.ALLOWED_GROUP_BY:
            group_by = "scene_id"
        expr = "strftime('%Y-%m-%d', ts, 'unixepoch')" if group_by == "day" else group_by
        where = "baseline_tokens > 0 AND route IN ('knowledge', 'code', 'skill')"
        if not include_test:
            where += " AND is_test = 0"
        sql = (
            f"SELECT {expr} AS g, COUNT(*) AS n, "
            "AVG(confidence) AS conf_avg, AVG(strength) AS strength_avg, "
            "AVG(margin) AS margin_avg, AVG(hits) AS hits_avg, "
            "AVG(recall_candidates) AS recall_avg, AVG(actual_tokens) AS act_avg, "
            "AVG(baseline_tokens) AS base_avg, AVG(latency_total_ms) AS latency_avg, "
            "SUM(CASE WHEN confidence >= ? AND hits > 0 THEN 1 ELSE 0 END) AS intercepted, "
            "SUM(CASE WHEN confidence < ? THEN 1 ELSE 0 END) AS abstained "
            f"FROM metrics WHERE {where} GROUP BY {expr} ORDER BY n DESC LIMIT ?"
        )
        cur = self._conn.cursor()
        cur.execute(sql, (ABSTAIN_CONFIDENCE, ABSTAIN_CONFIDENCE, limit))
        groups = []
        for row in cur.fetchall():
            g, n = row[0], row[1] or 0
            conf_avg, strength_avg, margin_avg = (round(row[i] or 0, 4) for i in (2, 3, 4))
            hits_avg, recall_avg = round(row[5] or 0, 4), round(row[6] or 0, 4)
            act_avg, base_avg, latency_avg = (round(row[i] or 0, 4) for i in (7, 8, 9))
            intercepted, abstained = row[10] or 0, row[11] or 0
            groups.append({
                "g": g, "n": n,
                "conf_avg": conf_avg, "strength_avg": strength_avg, "margin_avg": margin_avg,
                "hits_avg": hits_avg, "recall_avg": recall_avg,
                "act_avg": act_avg, "base_avg": base_avg, "latency_avg": latency_avg,
                "intercept_rate": round(intercepted / n, 4) if n else 0.0,
                "abstain_rate": round(abstained / n, 4) if n else 0.0,
                "recall_return_ratio": round(hits_avg / recall_avg, 4) if recall_avg else 0.0,
                "saved_ratio": round((base_avg - act_avg) / base_avg, 4) if base_avg else 0.0,
            })
        return {
            "group_by": group_by,
            "groups": groups,
            "available_group_by": list(self.ALLOWED_GROUP_BY),
            "overall": self.metric_aggregate(include_test=include_test),
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
    def record_feedback(
        self,
        session_id: str,
        entry_id: str,
        action: str,
        ts: float | None = None,
        source: str = "manual",
    ) -> None:
        """
        记录一次反馈动作（hit/adopt/cite/reject），驱动飞轮「反馈」环节统计。
        source="auto" 表示高置信命中后的隐式正反馈，source="manual" 为调用方显式反馈。
        """
        if ts is None:
            ts = time.time()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO feedback_log (ts, session_id, entry_id, action, source) VALUES (?, ?, ?, ?, ?)",
            (ts, session_id, entry_id, action, source),
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

    def count_positive_feedback_since_evolve(self) -> int:
        """
        统计「上次进化之后」的正面反馈数（hit/adopt/cite）。

        用途：v8.0 飞轮修复——进化触发计数原为 Orchestrator 实例内存，
        进程重启即清零，导致「5 次正反馈触发进化」在重启后永远攒不满。
        此方法让计数可从持久化日志恢复，飞轮不再因重启而停摆。
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COALESCE(MAX(ts), 0) FROM evolve_log")
        last_ts = float(cur.fetchone()[0] or 0)
        cur.execute(
            "SELECT COUNT(*) FROM feedback_log WHERE ts > ? AND action IN ('hit','adopt','cite')",
            (last_ts,),
        )
        return int(cur.fetchone()[0] or 0)

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
        """聚合反馈动作与来源分布，返回 {total, by_action, by_source}。"""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback_log")
        total = cur.fetchone()[0]
        cur.execute("SELECT action, COUNT(*) FROM feedback_log GROUP BY action")
        by_action = dict(cur.fetchall())
        cur.execute("SELECT source, COUNT(*) FROM feedback_log GROUP BY source")
        by_source = dict(cur.fetchall())
        return {"total": total, "by_action": by_action, "by_source": by_source}

    def ingest_aggregate(self) -> dict:
        """聚合入库结果分布，返回 {by_state: {state: count}, total}。"""
        cur = self._conn.cursor()
        cur.execute("SELECT state, COUNT(*) AS cnt FROM ingest_log GROUP BY state ORDER BY cnt DESC")
        by_state = {r["state"]: r["cnt"] for r in cur.fetchall()}
        return {"total": sum(by_state.values()), "by_state": by_state}

    def feedback_recent(self, limit: int = 100) -> list[dict]:
        """最近 N 条反馈明细（飞轮「反馈」下钻数据源）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, ts, session_id, entry_id, action, source FROM feedback_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["id", "ts", "session_id", "entry_id", "action", "source"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def evolve_recent(self, limit: int = 100) -> list[dict]:
        """最近 N 次进化步进明细（飞轮「进化」下钻数据源）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, ts, added_edges, decayed, archived FROM evolve_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        cols = ["id", "ts", "added_edges", "decayed", "archived"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def evolution_health(self) -> dict:
        """
        自进化健康度：反馈率 / 自动反馈占比 / 进化步进 / 弃答缺口 /
        工具轨迹数 / 会话完整率。全部为基础类型，供大屏与对比接口消费。
        """
        ma = self.metric_aggregate()
        queries = ma["queries"]
        fa = self.feedback_aggregate()
        ea = self.evolve_aggregate()
        aa = self.abstain_aggregate(200)
        ss = self.session_summary()
        tt = self.list_tooltraces(1000)

        feedback_total = fa["total"]
        by_action = fa["by_action"]
        by_source = fa["by_source"]
        feedback_rate = round(feedback_total / queries, 4) if queries else 0.0
        auto_feedback = int(by_source.get("auto", 0))
        auto_feedback_ratio = round(auto_feedback / feedback_total, 4) if feedback_total else 0.0
        complete_sessions = sum(1 for item in ss if not item.get("missing"))
        return {
            "queries": queries,
            "feedback_total": feedback_total,
            "by_action": by_action,
            "by_source": by_source,
            "feedback_rate": feedback_rate,
            "auto_feedback": auto_feedback,
            "auto_feedback_ratio": auto_feedback_ratio,
            "evolve": ea,
            "gap_total": aa["total"],
            "gap_no_data": aa["by_reason"].get("no_data", 0),
            "tooltrace_count": len(tt),
            "session_complete_rate": round(complete_sessions / len(ss), 4) if ss else 0.0,
        }

    def evolve_aggregate(self) -> dict:
        """
        聚合进化历史，返回 {steps, added_edges, decayed, archived, cleaned, net_delta, idle_steps}。

        v8.0：added_edges 是「连边动作数」而非质量提升，单看它会虚高（旧版 2279 边、
        decayed/archived 恒 0，飞轮看似在转实则只增不减）。新增：
        - cleaned    = decayed + archived：真实生命周期动作（衰减+归档）；
        - net_delta  = added_edges - cleaned：净变化，正数净增、负数说明在收敛淘汰；
        - idle_steps：decayed+archived==0 的步数（空转步），直接暴露「伪进化」。
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(added_edges),0) AS e, "
            "COALESCE(SUM(decayed),0) AS d, COALESCE(SUM(archived),0) AS a, "
            "COALESCE(SUM(CASE WHEN decayed + archived = 0 THEN 1 ELSE 0 END),0) AS idle "
            "FROM evolve_log"
        )
        row = cur.fetchone()
        e = int(row["e"] or 0)
        d = int(row["d"] or 0)
        a = int(row["a"] or 0)
        return {
            "steps": int(row["n"] or 0),
            "added_edges": e,
            "decayed": d,
            "archived": a,
            "cleaned": d + a,
            "net_delta": e - (d + a),
            "idle_steps": int(row["idle"] or 0),
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

    # ------------------------------------------------------------------ 原文层：切块（v9.1）
    def save_chunks(self, entry_id: str, chunks: list[dict]) -> int:
        """
        覆盖写入某个条目的全部切块（先删后插，整批一次提交）。

        chunks 每项需含 chunk_id / ordinal / heading / path / char_start / char_end /
        text_len / body / text_hash。返回写入块数。
        覆盖写入而非增量：条目正文一变，块边界全部位移，增量合并会留下孤儿块。
        """
        import time
        if not entry_id:
            return 0
        now = time.time()
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks WHERE entry_id = ?", (entry_id,))
        payload = []
        for c in chunks or []:
            payload.append((
                c.get("chunk_id") or f"{entry_id}::{c.get('ordinal', 0)}",
                entry_id,
                int(c.get("ordinal", 0)),
                c.get("heading") or "",
                c.get("path") or "",
                int(c.get("char_start", 0)),
                int(c.get("char_end", 0)),
                int(c.get("text_len", len(c.get("body") or ""))),
                c.get("body") or "",
                c.get("text_hash") or "",
                now,
            ))
        if payload:
            cur.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, entry_id, ordinal, heading, path, "
                "char_start, char_end, text_len, body, text_hash, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                payload,
            )
        self._conn.commit()
        return len(payload)

    def save_all_chunks(self, rows: list[dict], clear: bool = True, chunk_size: int = 500) -> int:
        """
        批量全量写入切块（重建原文层专用），返回写入块数。

        与 save_chunks 的区别：全库重建时逐条目 commit 会把 730 次 fsync 变成主要耗时
        （实测首次建 12426 块要 8 分钟）。这里 clear 一次、分批 executemany、
        最后只 commit 一次，把提交次数从 O(条目数) 降到 1。
        """
        import time
        now = time.time()
        cur = self._conn.cursor()
        if clear:
            cur.execute("DELETE FROM chunks")
        payload = []
        for c in rows or []:
            payload.append((
                c.get("chunk_id") or f"{c.get('entry_id', '')}::{c.get('ordinal', 0)}",
                c.get("entry_id", ""),
                int(c.get("ordinal", 0)),
                c.get("heading") or "",
                c.get("path") or "",
                int(c.get("char_start", 0)),
                int(c.get("char_end", 0)),
                int(c.get("text_len", len(c.get("body") or ""))),
                c.get("body") or "",
                c.get("text_hash") or "",
                now,
            ))
        for i in range(0, len(payload), chunk_size):
            cur.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, entry_id, ordinal, heading, path, "
                "char_start, char_end, text_len, body, text_hash, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                payload[i:i + chunk_size],
            )
        self._conn.commit()
        return len(payload)

    def get_chunks(self, entry_id: str) -> list[dict]:
        """按条目取切块列表（按 ordinal 升序），用于引用回溯与块级召回。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, ordinal, heading, path, char_start, char_end, text_len, body, text_hash "
            "FROM chunks WHERE entry_id = ? ORDER BY ordinal",
            (entry_id,),
        )
        return [
            {
                "chunk_id": r[0], "ordinal": r[1], "heading": r[2], "path": r[3],
                "char_start": r[4], "char_end": r[5], "text_len": r[6],
                "body": r[7], "text_hash": r[8],
            }
            for r in cur.fetchall()
        ]

    def get_chunk(self, chunk_id: str) -> dict | None:
        """按 chunk_id 取单个切块（引用回溯用）。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, entry_id, ordinal, heading, path, char_start, char_end, body "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "chunk_id": r[0], "entry_id": r[1], "ordinal": r[2], "heading": r[3],
            "path": r[4], "char_start": r[5], "char_end": r[6], "body": r[7],
        }

    def prune_orphan_chunk_vectors(self) -> int:
        """
        清掉在 chunks 表里已经找不到对应块的块向量，返回删除行数。

        产生原因：条目正文改动后切块重建，旧 chunk_id 消失但向量行还在；
        以及「微块不参与块向量」规则生效后，历史上给标题行算过的向量变成孤儿。
        孤儿向量不会被查到（按 chunk_id 取），但会白占空间、干扰覆盖率统计。
        """
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM vector_index WHERE owner_kind = 'chunk' "
            "AND owner_id NOT IN (SELECT chunk_id FROM chunks)"
        )
        self._conn.commit()
        return cur.rowcount

    def chunk_stats(self) -> dict:
        """切块覆盖统计：{entries_with_chunks, chunks, avg_chunks}，用于观测原文层建成度。"""
        cur = self._conn.cursor()
        total = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        owners = cur.execute("SELECT COUNT(DISTINCT entry_id) FROM chunks").fetchone()[0]
        return {
            "chunks": int(total),
            "entries_with_chunks": int(owners),
            "avg_chunks": round(total / owners, 2) if owners else 0.0,
        }

    def get_chunk_hashes(self, entry_id: str) -> dict:
        """返回 {chunk_id: text_hash}，用于判断某条目的切块是否需要重算。"""
        cur = self._conn.cursor()
        cur.execute("SELECT chunk_id, text_hash FROM chunks WHERE entry_id = ?", (entry_id,))
        return {r[0]: (r[1] or "") for r in cur.fetchall()}

    # ------------------------------------------------------------------ 标签词表（v9.1）
    def rebuild_tag_vocab(self, alias_map: dict | None = None) -> dict:
        """
        从全库条目的 meta.tags 重建受控词表，返回 {tags, parents}。

        alias_map 是「别名 -> 规范标签」的映射（如 {"部署":"运维"}），
        归一化后同一概念只占一个词条，df 才是真实文档频次。
        """
        import time
        alias_map = alias_map or {}
        df: dict[str, int] = {}
        for e in self.all_entries():
            tags = e.meta.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            seen = set()
            for t in tags:
                key = str(t).strip()
                if not key:
                    continue
                # 必须与 kb_layers.normalize_tag 同口径（别名映射 + 小写）：
                # 归一化只在写入侧做，读侧（tag_df）直接查表。两边不一致会出现
                # 「Deploy / deploy / 部署」三个词条，df 被拆散，idf 全失真。
                key = alias_map.get(key, key).lower()
                if key in seen:
                    continue
                seen.add(key)
                df[key] = df.get(key, 0) + 1

        now = time.time()
        cur = self._conn.cursor()
        cur.execute("DELETE FROM tag_vocab")
        payload = [(t, "", c, now) for t, c in df.items()]
        if payload:
            cur.executemany(
                "INSERT OR REPLACE INTO tag_vocab(tag, parent, df, updated_at) VALUES (?,?,?,?)",
                payload,
            )
        self._conn.commit()
        return {"tags": len(payload), "parents": 0}

    def tag_df(self) -> dict:
        """返回 {tag: 文档频次}。词表为空时退回实时统计，保证加成逻辑永远有值可用。"""
        cur = self._conn.cursor()
        rows = cur.execute("SELECT tag, df FROM tag_vocab").fetchall()
        if rows:
            return {r[0]: int(r[1]) for r in rows}
        return self.tag_stats()

    # ------------------------------------------------------------------ 主题树（v9.1）
    def rebuild_topic_tree(self, nodes: list[dict]) -> int:
        """
        重建主题树，返回写入节点数。

        nodes 每项含 topic_name / parent_name / level / scene_id；
        topic_id 由自增分配，parent_id 按 parent_name 反查（父节点必须先于子节点出现在列表里）。
        """
        cur = self._conn.cursor()
        cur.execute("DELETE FROM topic_tree")
        name_to_id: dict[str, int] = {}
        next_id = 1
        payload = []
        for n in nodes or []:
            name = str(n.get("topic_name") or "").strip()
            if not name:
                continue
            tid = next_id
            next_id += 1
            parent_name = str(n.get("parent_name") or "").strip()
            pid = name_to_id.get(parent_name, 0) if parent_name else 0
            name_to_id[name] = tid
            payload.append((tid, name, pid, int(n.get("level", 1)), n.get("scene_id") or ""))
        if payload:
            cur.executemany(
                "INSERT OR REPLACE INTO topic_tree(topic_id, topic_name, parent_id, level, scene_id) "
                "VALUES (?,?,?,?,?)",
                payload,
            )
        self._conn.commit()
        return len(payload)

    def tree_children(self, topic_id: int = 0) -> list[dict]:
        """取某节点的直接子节点（topic_id=0 取根层），用于树导航与路径回溯。"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT topic_id, topic_name, parent_id, level, scene_id FROM topic_tree "
            "WHERE parent_id = ? ORDER BY topic_name",
            (int(topic_id),),
        )
        return [
            {"topic_id": r[0], "topic_name": r[1], "parent_id": r[2], "level": r[3], "scene_id": r[4]}
            for r in cur.fetchall()
        ]

    def tree_path(self, topic_id: int) -> list[str]:
        """从根到该节点的主题名路径（最多 8 层，防脏数据成环导致死循环）。"""
        cur = self._conn.cursor()
        path = []
        cur_id = int(topic_id)
        for _ in range(8):
            row = cur.execute(
                "SELECT topic_name, parent_id FROM topic_tree WHERE topic_id = ?", (cur_id,)
            ).fetchone()
            if not row:
                break
            path.append(row[0])
            if not row[1]:
                break
            cur_id = int(row[1])
        return list(reversed(path))

    def tree_stats(self) -> dict:
        """树规模统计：{nodes, roots, max_level}。"""
        cur = self._conn.cursor()
        nodes = cur.execute("SELECT COUNT(*) FROM topic_tree").fetchone()[0]
        roots = cur.execute("SELECT COUNT(*) FROM topic_tree WHERE parent_id = 0").fetchone()[0]
        max_level = cur.execute("SELECT COALESCE(MAX(level), 0) FROM topic_tree").fetchone()[0]
        return {"nodes": int(nodes), "roots": int(roots), "max_level": int(max_level)}

    # ------------------------------------------------------------------ 关系网 PMI 治理（v9.1）
    def cooccur_pmi_pairs(self, min_count: int = 2, min_pmi: float = 0.0,
                          limit: int = 2000) -> list[tuple[str, str, float]]:
        """
        按 PMI 给共现边重新定权，返回 [(a, b, pmi)] 降序。

        PMI = log( cnt(a,b) * E / (deg(a) * deg(b)) )，其中
            deg(x) = 该节点所有边的 cnt 之和，E = 全表 cnt 之和 / 2（总边的"质量"）。
        动机：原始 cnt 会奖励「到处都出现」的高频节点对，PMI 惩罚纯高频、奖励专属关联。

        为什么用度数而不是 tag_vocab 的 df：本表的端点是 entry_id（kb::xxx / wb::xxx / tg_xxx），
        不是标签名。早期版本拿 tag_df() 当边际分布，端点查不到 df 就整条丢弃，
        实测 823 条边一条都留不下（函数恒返回空），等于「PMI 治理」根本没生效。
        改用本表自身的度数作边际，口径自洽且不依赖外部数据。
        """
        import math
        cur = self._conn.cursor()
        rows = cur.execute("SELECT a, b, cnt FROM cooccur WHERE cnt >= ?", (int(min_count),)).fetchall()
        if not rows:
            return []

        degree: dict[str, float] = {}
        total_mass = 0.0
        for a, b, cnt in rows:
            w = float(cnt)
            degree[a] = degree.get(a, 0.0) + w
            degree[b] = degree.get(b, 0.0) + w
            total_mass += w
        # E = 边的总质量的一半（每条边贡献两次），避免除以 0
        edge_mass = max(1.0, total_mass / 2.0)

        out = []
        for a, b, cnt in rows:
            da = degree.get(a, 0.0)
            db = degree.get(b, 0.0)
            if da <= 0 or db <= 0 or not cnt:
                continue
            pmi = math.log((float(cnt) * edge_mass) / (da * db))
            if pmi >= min_pmi:
                out.append((a, b, round(pmi, 4)))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:limit]

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

    def record_snapshot(self, metric: str, value: float, detail: dict | None = None) -> None:
        """记录指标快照到数据库"""
        import json as _json
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO metric_snapshots (ts, metric, value, detail) VALUES (?, ?, ?, ?)",
            (time.time(), metric, value, _json.dumps(detail or {}, ensure_ascii=False))
        )
        self._conn.commit()

    def list_snapshots(self, metric: str, limit: int = 100, since: float | None = None) -> list[dict]:
        """列出指定指标的快照记录"""
        import json as _json
        cur = self._conn.cursor()
        if since is not None:
            cur.execute(
                "SELECT ts, value, detail FROM metric_snapshots WHERE metric=? AND ts>=? ORDER BY ts ASC LIMIT ?",
                (metric, since, limit)
            )
        else:
            cur.execute(
                "SELECT ts, value, detail FROM metric_snapshots WHERE metric=? ORDER BY ts ASC LIMIT ?",
                (metric, limit)
            )
        rows = cur.fetchall()
        result = []
        for row in rows:
            ts, value, detail_str = row
            try:
                detail = _json.loads(detail_str) if detail_str else {}
            except Exception:
                detail = {}
            result.append({"ts": ts, "value": value, "detail": detail})
        return result

    def prune_snapshots(self, keep_days: float = 30) -> int:
        """清理过期的快照记录"""
        cur = self._conn.cursor()
        cutoff = time.time() - keep_days * 86400
        cur.execute("DELETE FROM metric_snapshots WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return cur.rowcount

    def list_ingest(self, limit: int = 500, offset: int = 0) -> list[dict]:
        """列出最近的摄入日志"""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT ts, entry_id, state, title FROM ingest_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cur.fetchall()
        return [
            {"ts": r[0], "entry_id": r[1], "state": r[2], "title": r[3]}
            for r in rows
        ]

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
