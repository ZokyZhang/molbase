# -*- coding: utf-8 -*-
"""
代工层编排器：统一入口。

职责（代工层 = 查询优化器 + 执行计划）：
- 意图路由：判断查询目标（知识 / 代码 / 技能 / 混合）。
- 召回：调用混合召回器。
- 重排：三信号综合打分，输出「最小上下文」。
- 反馈：命中 / 采纳 / 引用 回写热度与新鲜度。

对外暴露的核心接口（大模型/agent 只调用这一个）：
- query(text, top_k) -> 返回最小上下文块列表
- feedback(entry_id, action) -> 回写信号

这正是「agent.md 只留一条约束」的实现基础：agent 不再自己检索、
读文件、加载 skill，而是把意图交给代工层，由代工层精准定位。
"""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import (
    ABSTAIN_CONFIDENCE,
    ARCHIVE_DAYS,
    ARCHIVE_HEAT_MAX,
    AUTO_HIT_HEAT,
    BACKFILL_BATCH,
    COMPACT_ENABLED,
    COMPACT_MODE,
    CONFLICT_DUPLICATE_SIM,
    COST_INPUT_PER_1M,
    EMBEDDING_BACKEND,
    EMBEDDING_HASH_DIM,
    EMBEDDING_MODEL_NAME,
    EVOLVE_INTERVAL,
    EXPAND_CACHE_SIZE,
    FEEDBACK_MIN_CONFIDENCE,
    FEEDBACK_WEIGHT_ADOPT,
    FEEDBACK_WEIGHT_CITE,
    FEEDBACK_WEIGHT_HIT,
    FEEDBACK_WEIGHT_REJECT,
    FINAL_TOP_K,
    LLM_ENABLED,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROTOCOL,
    LLM_REWRITE,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    QUARANTINE_REJECT_THRESHOLD,
    SKILLS_DIR,
    SUMMARY_MAX_CHARS,
    TOKEN_TRACE,
    TOOL_TRACE_FINGERPRINT_MIN_LEN,
    TOOL_TRACE_GENERATE_THRESHOLD,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_ENDPOINT,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_TOP_N,
    WEB_SEARCH_TRIGGER_SCORE,
    NO_MATCH_STRENGTH,
    WEAK_STRENGTH_THRESHOLD,
    ensure_dirs,
)
from .code import CodeIndexer
from .embedding import (
    EmbeddingProvider,
    cosine_similarity,
    create_embedder,
    is_near_duplicate,
)
from .evolution import EvolutionEngine
from .graph import RelationGraph
from .index import BM25Index
from .llm import LocalLLM, create_llm
from .profiler import Profiler, now_ms
from .retrieval import Reranker, Retriever
from .skills import SkillRegistry
from .summarize import split_blocks, split_sentences, semantic_preserve, textrank_summarize
from .store import Entry, Store
from .tokens import TokenLedger, estimate_tokens
from .websearch import SearchProvider, create_search_provider

# 意图路由的 LLM 系统提示词（V2：语义分类；V1：规则兜底）
_ROUTE_SYSTEM_PROMPT = (
    "你是 MolBase 的意图路由器。根据用户查询判断意图，只输出 JSON。"
    "route 取值：knowledge（知识/概念/做法）、code（代码/方法/函数/源码）、"
    "skill（技能/工具/流程步骤）。"
    "规则：knowledge 为默认兜底——任何不确定、泛泛的自然语言问题、概念、做法一律归为此类；"
    "只有明确要写代码/查源码才选 code，明确要调用技能执行才选 skill。"
    "entities：数组，抽取查询中的关键实体/专有名词（如工具名、项目名、人名、术语），最多 8 个。"
    "scene：字符串，推断用户使用场景（如 qwen-worker、family-edu、运维排障、知识库建设），无法判断则输 'default'。"
    "query_tags：数组，从受控词表 [运维，环境，配置，架构，规范，设计，业务，敏感，工具] 中选 0-3 个最相关标签。"
    "输出格式示例：{\"route\":\"knowledge\",\"entities\":[\"关键词\"],\"scene\":\"default\",\"query_tags\":[]}"
    "强调：只输出 JSON，不要解释、不要 markdown 代码块标记。"
)

# 生成式摘要的 LLM 系统提示词（V2：摘要压缩；V1：原文截断兜底）
_SUMMARIZE_SYSTEM_PROMPT = (
    "你是MolBase的上下文压缩器。把输入内容压缩成精炼摘要，"
    "保留关键事实与可执行步骤，去除冗余。只输出摘要正文，不要任何解释。"
)

# 查询改写的 LLM 系统提示词（V2：改写口语化查询，提升召回）
_REWRITE_SYSTEM_PROMPT = (
    "你是MolBase的查询改写器。把用户的口语化查询改写成适合全文检索的"
    "关键词查询：去掉寒暄与语气词，保留核心实体、术语与意图，可补充同义关键词。"
    "只输出改写后的查询文本，不要任何解释、标点或引号。"
)

# 会话回写的 LLM 系统提示词（V2：生成式结构化提炼；V1：TextRank 抽取兜底）
_HARVEST_SYSTEM_PROMPT = (
    "你是MolBase的会话提炼器。把一段会话记录提炼成一份可复用的核心资产，只输出 JSON。"
    "字段：title（一句话标题）、conclusion（核心结论，2-5句，保留关键事实与可执行结论）、"
    "decisions（已做出的决策，字符串数组，无则空数组）、actions（待办行动项，字符串数组，无则空数组）、"
    "tags（主题标签，字符串数组，无则空数组）。"
    "格式：{\"title\":\"...\",\"conclusion\":\"...\",\"decisions\":[\"...\"],\"actions\":[\"...\"],\"tags\":[\"...\"]}"
)

# 技能生成的 LLM 系统提示词（V2：生成式结构化；V1：模板兜底）
_TOOLTRACE_SYSTEM_PROMPT = (
    "你是MolBase的技能生成器。根据一组重复出现的工具调用序列，生成一份可复用的技能规格，只输出 JSON。"
    "字段：name（小写英文短横线命名的技能名）、description（一句话，说明何时使用该技能）、"
    "steps（执行步骤，字符串数组，每个步骤描述调用什么工具、传什么关键参数）、"
    "tags（主题标签，字符串数组，无则空数组）。"
    "格式：{\"name\":\"...\",\"description\":\"...\",\"steps\":[\"...\"],\"tags\":[\"...\"]}"
)


@dataclass
class QueryResult:
    """一次查询的返回：精准定位到的最小上下文块。"""

    entry_id: str
    entry_type: str
    title: str
    content: str
    source: str
    score: float
    meta: dict = field(default_factory=dict)
    summary: str = ""  # 语义保持摘要（空=未生成，回退返回原文）


@dataclass
class QueryResponse:
    """查询完整响应：路由结果 + 命中列表 + 网络兜底 + token 埋点 + 质量底线。"""

    route: str
    results: list[QueryResult]
    query: str
    rewritten: str = ""  # 改写后的检索查询（无 LLM 或改写失败时等于原 query）
    web_results: list[QueryResult] = field(default_factory=list)  # 网络搜索兜底结果（触发时非空）
    token_estimate: int = 0     # 本次返回内容的粗估 token 数
    baseline_tokens: int = 0    # 传统「全表扫描」模式的粗估 token 数（全部条目内容）
    content_chars: int = 0      # 本次返回内容的总字符数
    confidence: float = 0.0     # 综合置信度（0~1），低于阈值时 abstain=True（质量底线输出闸门）
    abstain: bool = False       # 是否「宁可不答也不错答」（内部召回置信不足）
    scene_id: str = ""          # 本次查询解析出的场景锚点
    entities: list = field(default_factory=list)  # 本次抽取的实体（不再丢弃）
    resolved_version_id: str = ""  # 同问异答裁决出的版本 ID（无版本裁决时为空）
    refs_miss: bool = False     # refs 硬约束未命中（用于调用方判断是否允许放宽）
    skills: list = field(default_factory=list)  # v7.2 隐式复用：与查询相关的自动固化技能摘要
    refs: list = field(default_factory=list)    # v9.1 原文层定位：命中条目出自哪一节、正文哪个字符区间

@dataclass
class IngestResult:
    """一次知识摄入的判定结果（写入冲突四态，质量底线输入闸门）。"""

    entry_id: str
    state: str = "new"        # new | merge | duplicate | conflict | pinned
    against_id: str = ""      # 与之冲突/重复的已有条目（duplicate/conflict/pinned 时非空）
    reason: str = ""          # 判定原因（可解释，便于审计）


@dataclass
class HarvestResult:
    """一次会话回写的收获结果（提炼引擎 + 写入状态 + 关联过程指纹）。"""

    entry_id: str
    state: str = "new"        # 复用摄入四态：new | merge | duplicate | conflict | pinned | empty
    engine: str = "textrank"  # llm | textrank | none（会话内容为空时）
    title: str = ""
    reason: str = ""          # 冲突/重复/空内容原因（透传 IngestResult.reason）
    related_fingerprints: list[str] = field(default_factory=list)  # 关联的同会话操作指纹（结论↔过程）


@dataclass
class ToolTraceResult:
    """一次工具调用轨迹沉淀的结果（操作模式自进化）。"""

    fingerprint: str          # 本次操作指纹（工具名+参数键签名）
    count: int = 0            # 该指纹累计出现次数
    generated: bool = False   # 本次是否触发了 skill 自动生成
    skill_id: str = ""        # 生成/已存在的 skill 条目 ID（存在时非空）
    reason: str = ""          # 说明（未达阈值 / 已生成 / 空序列等）


class Orchestrator:
    """MolBase统一编排器。"""

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        embedding_backend: str | None = None,
        llm: LocalLLM | None = None,
        llm_endpoint: str | None = None,
        llm_model: str | None = None,
        search_provider: SearchProvider | None = None,
        db_path: Path | str | None = None,
        skills_dir: Path | str | None = None,
        profiler: Profiler | None = None,
    ) -> None:
        """
        初始化各组件，构建索引与关联图。

        embedder        ：可选，显式传入向量化后端；不传则按 config.EMBEDDING_BACKEND 创建。
                          任何后端创建失败都会自动降级为零模型哈希向量，保证「绝不崩溃」。
        llm             ：可选，显式传入本地大模型客户端；不传则按 config 的 LLM_* 创建。
                          未启用/不可用时为 None，意图路由与摘要自动降级回纯算法，绝不崩溃。
        search_provider ：可选，显式传入网络搜索提供者；不传则按 config 的 WEB_SEARCH_*
                          创建（默认关闭，返回空）。网络搜索是兜底增强，失败返回空不崩溃。
        db_path         ：可选，显式指定 SQLite 数据库路径（多租户/测试隔离用）；
                          不传则用 config.DB_PATH。
        profiler        ：可选，性能剖析器；传入后会自动给 LLM 挂 tracer，并在 query
                          各环节埋点（召回 / 重排 / 网络搜索），供性能报告量化降本。
                          不传则完全零开销，不影响既有测试。
        """
        ensure_dirs()
        self.store = Store(db_path)
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR
        self.bm25 = BM25Index()
        # 库词表缓存（两阶段闸门的阶段1 用），_rebuild 时失效重建
        self._domain_vocab = None
        self.graph = RelationGraph()
        self.reranker = Reranker()
        self.retriever = Retriever(self.bm25, self.graph)
        self.embedder = self._init_embedder(embedder, embedding_backend)
        self.llm = self._init_llm(llm, llm_endpoint, llm_model)
        self.profiler = profiler
        self.search_provider = (
            search_provider
            if search_provider is not None
            else create_search_provider(WEB_SEARCH_ENABLED, WEB_SEARCH_ENDPOINT, WEB_SEARCH_TIMEOUT)
        )
        self.evolution = EvolutionEngine(self.store)
        self.token_ledger = TokenLedger()
        self._query_count = 0
        self._resolver: dict[str, Entry] = {}
        self._baseline_tokens = 0
        # v8.0 飞轮：同会话命中记录（驱动 implicit hit 隐式正反馈，让「再次命中」也产出信号）
        self._session_hits: dict[str, set] = {}
        # v8.0 飞轮：进化触发计数从持久化日志恢复（避免重启清零导致飞轮停摆）
        try:
            self._feedback_since_evolve = self.store.count_positive_feedback_since_evolve()
        except Exception:
            self._feedback_since_evolve = 0
        self._rebuild()
        # v9.0 启动自愈：确保条目向量与当前后端/窗口口径一致（增量补算，失败不影响启动）
        # 仅 hash 后端在启动时补算（毫秒级）；bge-m3 等重型后端由离线回填脚本完成，避免启动阻塞数分钟
        # 先给默认值：非 hash 后端不会进 if 分支，若不预设，overview 读该字段会 AttributeError
        self._vector_bootstrap = {"skipped": True}
        try:
            from .config import VECTOR_PRECOMPUTE_ENABLED, VECTOR_PRECOMPUTE_BATCH
            if VECTOR_PRECOMPUTE_ENABLED and str(getattr(self.embedder, "name", "")).startswith("hash"):
                from .vectorize import ensure_entry_vectors
                self._vector_bootstrap = ensure_entry_vectors(
                    self.store, self.embedder, "entry", force=False,
                    batch_size=VECTOR_PRECOMPUTE_BATCH,
                )
        except Exception:
            self._vector_bootstrap = {"error": "bootstrap skipped"}
        # v9.1 四层结构（标签词表/主题树/原文切块）自愈。
        # 默认不在启动时构建：首次切 12000+ 块要 8~110 秒，同步跑会堵住服务端口，
        # 放后台线程又会长期持有 SQLite 连接（测试临时库 tearDown 报 WinError 32）。
        # 生产做法是部署时跑 deploy-lan/backfill_vectors.py 离线建一次，
        # 之后新条目由入库钩子增量维护；运维需要时可显式打开 LAYER_BOOTSTRAP_ON_START。
        self._layer_bootstrap = {"skipped": True}
        try:
            from .config import LAYER_BOOTSTRAP_ON_START, CHUNK_VECTOR_BOOTSTRAP
            if LAYER_BOOTSTRAP_ON_START and self.store.chunk_stats().get("chunks", 0) == 0:
                from .kb_layers import rebuild_layers
                self._layer_bootstrap = rebuild_layers(
                    self.store, self.embedder, with_vectors=CHUNK_VECTOR_BOOTSTRAP
                )
        except Exception:
            self._layer_bootstrap = {"error": "layer bootstrap skipped"}

    def _init_embedder(
        self,
        embedder: EmbeddingProvider | None,
        backend: str | None,
    ) -> EmbeddingProvider | None:
        """
        初始化向量化后端：
        - 显式传入 embedder 直接用；
        - 否则按 config 的 EMBEDDING_BACKEND 创建，失败降级 hash；
        - backend 为 "hash" 时返回 HashEmbedding（零模型基线）。
        """
        if embedder is not None:
            return embedder
        chosen = backend or EMBEDDING_BACKEND
        if chosen == "hash":
            from .embedding import HashEmbedding
            return HashEmbedding(EMBEDDING_HASH_DIM)
        # v3 修复：ollama 后端必须传 bge-m3 模型名，不能用 sentence-transformers 的
        # EMBEDDING_MODEL_NAME（all-MiniLM-L6-v2），否则 Ollama 报 404、语义相似度恒为 0。
        if chosen == "ollama":
            from .config import OLLAMA_EMBED_MODEL
            return create_embedder(chosen, OLLAMA_EMBED_MODEL)
        return create_embedder(chosen, EMBEDDING_MODEL_NAME)

    def _init_llm(
        self,
        llm: LocalLLM | None,
        endpoint: str | None,
        model: str | None,
    ) -> LocalLLM | None:
        """
        初始化本地大模型客户端：
        - 显式传入 llm 直接用；
        - 否则按 config 的 LLM_ENABLED/LLM_ENDPOINT/LLM_MODEL 创建；
        - 未启用（LLM_ENABLED=False）或未配置 endpoint/model 时返回 None，
          上层据此走纯算法降级路径。
        """
        if llm is not None:
            return llm
        if not LLM_ENABLED:
            return None
        return create_llm(
            endpoint or LLM_ENDPOINT,
            model or LLM_MODEL,
            timeout=LLM_TIMEOUT,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            protocol=LLM_PROTOCOL,
        )

    # ------------------------------------------------------------------ 索引构建
    def _rebuild(self) -> None:
        """从存储底座加载「可召回」条目，重建索引与关联图（排除已隔离/已归档）。"""
        entries = [
            e for e in self.store.all_entries()
            if not e.meta.get("quarantined") and not e.meta.get("archived")
        ]
        self._resolver = {e.entry_id: e for e in entries}
        self.bm25.build(entries)
        # 词表随条目集变化，重建时必须让缓存失效，否则阶段1 会用旧词表算覆盖率
        self._domain_vocab = None

        # 重建关联图：每次清空重来，避免边权重重复累加
        self.graph = RelationGraph()
        self.retriever.graph = self.graph

        for e in entries:
            for callee_name in e.meta.get("calls", []):
                # 反查 callee 条目 ID（按标题限定名匹配）
                for other in entries:
                    if other.entry_type == "code" and (
                        other.title == callee_name
                        or other.title.endswith(f".{callee_name}")
                        or other.title.split("::")[-1] == callee_name
                    ):
                        self.graph.add_edge(e.entry_id, other.entry_id)
            for rel_id in e.meta.get("relations", []):
                if rel_id in self._resolver:
                    self.graph.add_edge(e.entry_id, rel_id)

        # 若向量后端支持语料统计（如 TF-IDF），刷新 IDF（零模型增强）
        if hasattr(self.embedder, "build"):
            try:
                self.embedder.build([e.title + " " + e.content for e in entries])
            except Exception:
                pass

        # 缓存「全表扫描」基线 token：传统 agent 模式会读全部条目内容（粗估）
        self._baseline_tokens = estimate_tokens(
            "\n".join(e.title + " " + e.content for e in entries)
        )

    # ------------------------------------------------------------------ 意图路由
    def resolve_scene(self, scene_hint: str = "", session_id: str = "", query: str = "") -> str:
        """
        场景识别方法（优先级链）。
        核心逻辑：scene 回答“你在什么场景下用它”，与 tags（“这是什么”）正交。
        优先级顺序：显式传参 > 会话绑定 > 查询关键词推断 > 兜底 global。
        全程 try/except 包裹，确保任何异常均降级返回 "global"，绝不崩溃。
        """
        try:
            # 1. 显式传参最高优先
            # 归一化：去除首尾空格并转为小写
            if scene_hint and scene_hint.strip().lower() != "default":
                return scene_hint.strip().lower()

            # 2. 会话绑定检查
            # 使用 hasattr 保护，防止 store 对象缺失 get_meta 方法导致崩溃
            if session_id:
                get_meta = getattr(self.store, "get_meta", None)
                if get_meta:
                    try:
                        meta = get_meta(session_id)
                        # 约定：返回 dict 或 None。若含 "scene" 键且非空，则返回该场景
                        if isinstance(meta, dict) and meta.get("scene"):
                            return meta["scene"]
                    except Exception:
                        # 任何异常视为无绑定，继续下一步
                        pass

            # 3. 查询关键词推断
            # 若 query 非空，使用轻量规则匹配关键词，不重复调用 LLM
            if query:
                # 定义关键词到场景的映射常量
                scene_keywords = {
                    "qwen-worker": "qwen-worker",
                    "family-edu": "family-edu",
                    "生产环境": "production",
                    "运维": "ops",
                    "排障": "troubleshooting"
                }
                query_lower = query.lower()
                for keyword, scene_id in scene_keywords.items():
                    if keyword in query_lower:
                        return scene_id

            # 4. 兜底返回
            return "global"

        except Exception:
            # 全局兜底：任何未捕获异常均返回 "global"，确保服务不崩溃
            return "global"

    def resolve_version(self, knowledge_id: str, scene_id: str = "") -> dict | None:
            """
            解析知识版本，根据场景匹配优先级和评分规则返回最佳版本。

            优先级逻辑：
            1. 精确场景匹配 (row.scene_id == scene_id)
            2. 父场景继承 (row.scene_id 以 scene_id + '/' 开头)
            3. 全局/默认场景 (row.scene_id in ('global', 'default', ''))
            4. 全部 active 兜底

            同组排序规则 (必须正确)：
            - 第一关键字：authority * confidence (降序，大者优先)
            - 第二关键字：valid_from (字符串字典序降序，'2026-11-23' > '2026-11-22')
            - 实现方式：使用正值 + reverse=True，严禁对字符串取负。

            Authority 来源说明：
            - 用户拍板 1.0
            - 已验证 0.9
            - 文档推断 0.7
            - 通用 0.5
            """
            try:
                # 获取所有 active 状态的版本记录
                rows = self.store._conn.execute(
                    "SELECT * FROM knowledge_versions WHERE knowledge_id=? AND status='active'",
                    (knowledge_id,)
                ).fetchall()

                if not rows:
                    return None

                # 定义优先级分组函数
                def get_priority_group(row: sqlite3.Row, target_scene: str) -> int:
                    scene = row["scene_id"]
                    # 1. 精确匹配
                    if scene == target_scene:
                        return 0
                    # 2. 父场景继承 (scene 以 target_scene + '/' 开头)
                    if target_scene and scene.startswith(target_scene + "/"):
                        return 1
                    # 3. 全局/默认场景
                    if scene in ("global", "default", ""):
                        return 2
                    # 4. 其他场景 (不匹配当前目标，放入兜底组)
                    return 3

                # 处理每一行数据
                processed_items = []
                for row in rows:
                    # 转换为字典
                    item = dict(row)

                    # 计算评分：authority * confidence
                    # 使用 float 处理，默认值为 0
                    auth = float(item.get("authority", 0) or 0)
                    conf = float(item.get("confidence", 0) or 0)
                    item["_score"] = auth * conf

                    # 提取时间字符串，用于字典序比较
                    # 注意：字符串时间字典序即时间序，禁止取负
                    item["_valid_from"] = str(item.get("valid_from", "") or "")

                    processed_items.append(item)

                # 按优先级分组
                groups = {0: [], 1: [], 2: [], 3: []}
                for item in processed_items:
                    priority = get_priority_group(item, scene_id)
                    groups[priority].append(item)

                # 定义排序键：(score, valid_from)，使用 reverse=True 实现降序
                # 严禁对字符串取负，直接使用正值 + reverse=True
                def sort_key(d: dict):
                    return (d["_score"], d["_valid_from"])

                # 对每个非空组进行排序
                best_version = None
                for priority in [0, 1, 2, 3]:
                    group = groups[priority]
                    if not group:
                        continue

                    # 正确写法：正值 + reverse=True，一次到位
                    group.sort(key=sort_key, reverse=True)

                    # 取该组第一个元素
                    if group:
                        best_version = group[0]
                        break # 找到最高优先级组后直接返回

                if best_version is None:
                    return None

                # 返回结果，剔除下划线开头的临时字段
                result = {k: v for k, v in best_version.items() if not k.startswith("_")}
                return result

            except Exception:
                # 捕获任何异常，返回 None
                return None

    def _route_full(self, query: str, scene_hint: str = "") -> dict:
        """
        结构化路由方法：返回包含 route、entities、scene、query_tags 的完整字典。
        设计目的：解决旧版 _route_llm 丢弃 entities/scene/query_tags 导致信息利用率低的问题。
        即使 entities 在全库中仅出现 1 次，保留其上下文对后续技能调用至关重要。
        """
        # 1. 定义默认返回结构，确保异常或无 LLM 时仍有基础路由结果
        default_result = {
            "route": self._route_rules(query),
            "entities": [],
            "scene": scene_hint or "global",
            "query_tags": []
        }

        # 2. 若 LLM 可用，尝试获取更丰富的结构化数据
        if self.llm is not None:
            try:
                t0 = now_ms()
                data = self.llm.complete_json(_ROUTE_SYSTEM_PROMPT, query)

                # 解析 LLM 返回的数据
                if data:
                    # 2.1 处理 route：若 LLM 输出合法 (knowledge/code/skill)，则覆盖默认规则结果
                    route_val = data.get("route")
                    if route_val in ("knowledge", "code", "skill"):
                        default_result["route"] = route_val

                    # 2.2 处理 entities：若为列表，取前 8 个字符串元素
                    entities_val = data.get("entities")
                    if isinstance(entities_val, list):
                        default_result["entities"] = [str(e) for e in entities_val[:8]]

                    # 2.3 处理 query_tags：若为列表，取前 3 个字符串元素
                    tags_val = data.get("query_tags")
                    if isinstance(tags_val, list):
                        default_result["query_tags"] = [str(t) for t in tags_val[:3]]

                    # 2.4 处理 scene：显式 hint 优先，其次 LLM 推断，最后默认 global
                    if scene_hint:
                        default_result["scene"] = scene_hint
                    elif data.get("scene"):
                        default_result["scene"] = data["scene"]
                    # 否则保持 default_result["scene"] 为 "global"

                # 可选：埋点记录（若需要，此处注释掉以避免重复，或取消注释使用）
                # self._trace_llm("意图路由(full)", t0, _ROUTE_SYSTEM_PROMPT, query, json.dumps(data, ensure_ascii=False) if data else "")

            except Exception:
                # 3. 任何异常直接降级，返回默认字典，保证程序不崩溃
                pass

        # 4. 返回最终结果
        return default_result

    def _route(self, query: str) -> str:
        """
        意图路由：LLM 语义分类优先，规则兜底。

        - V2：本地大模型可用时，用语义分类（更准，理解自然语言）；
        - V1：LLM 不可用/超时/解析失败时，降级回关键词规则。
        """
        route = self._route_llm(query)
        if route:
            return route
        return self._route_rules(query)

    def _route_llm(self, query: str) -> str | None:
        """用本地大模型做语义意图分类，失败返回 None（由规则兜底）。"""
        if self.llm is None:
            return None
        try:
            t0 = now_ms()
            data = self.llm.complete_json(_ROUTE_SYSTEM_PROMPT, query)
            # 性能埋点：意图路由（模型环节，记录输入/输出 token 与耗时）
            self._trace_llm(
                "意图路由", t0, _ROUTE_SYSTEM_PROMPT, query,
                json.dumps(data, ensure_ascii=False) if data else "",
            )
            if data and data.get("route") in ("knowledge", "code", "skill"):
                return data["route"]
        except Exception:
            # 模型未启动 / 超时 / 输出非 JSON -> 降级
            return None
        return None

    def _route_rules(self, query: str) -> str:
        """
        路由规则判定函数。
        逻辑：默认归为 knowledge，仅当命中强信号词时分别归为 code 或 skill。
        已移除弱词（如单独的方法、函数、调用、步骤）的误判逻辑。
        """
        # 强代码信号词列表：明确指向编写代码、查看源码或具体代码片段
        code_keywords = {"源码", "代码块", "函数名", "def ", "class ", "import ", "```", "编译", "调试代码"}
    
        # 强技能信号词列表：明确指向技能包、执行流程或工具调用步骤
        skill_keywords = {"技能包", "执行步骤", "工具调用流程", "skill 包", "how to"}
    
        # 将查询转换为小写以便统一匹配英文关键词
        query_lower = query.lower()
    
        # 1. 检查强代码信号
        # 处理中文关键词直接命中
        for kw in code_keywords:
            if kw in query:
                return "code"
        # 处理英文关键词 (def, class, import)
        for kw in ["def", "class", "import"]:
            if kw in query_lower:
                return "code"
            
        # 2. 检查强技能信号
        for kw in skill_keywords:
            if kw in query:
                return "skill"
            
        # 3. 默认兜底
        # 此时即使包含“函数”、“调用”、“步骤”等弱词，只要没有命中上述强信号，
        # 一律视为知识查询 (knowledge)
        return "knowledge"

    def _trace_llm(self, stage: str, t0: float, system: str, user: str, out_text: str) -> None:
        """
        性能埋点：记录一次模型调用的 token 与耗时到 profiler（若启用）。

        stage    ：环节名（意图路由 / 查询改写 / 摘要压缩）。
        t0       ：调用开始时间（now_ms() 返回的毫秒时间戳）。
        system   ：系统提示词。
        user     ：用户输入。
        out_text ：模型返回文本（据此粗估输出 token）。
        仅在 self.profiler 非空时生效，否则零开销。
        """
        if self.profiler is None:
            return
        self.profiler.record(
            stage,
            llm_call=True,
            input_tokens=estimate_tokens(system) + estimate_tokens(user),
            output_tokens=estimate_tokens(out_text),
            duration_ms=now_ms() - t0,
        )

    def _rewrite(self, query: str) -> str:
        """
        查询改写（本地大模型在代工层的第三个任务）：
        - LLM 可用且 LLM_REWRITE 开启时，把口语化查询改写成检索友好关键词，
          去除寒暄语气词、补充同义词，提升 BM25/向量召回率；
        - LLM 不可用/超时/输出异常时，降级返回原查询，绝不崩溃。
        - 中文提问被英译时保留原文，避免中文知识库召回失败。
        """
        if self.llm is None or not LLM_REWRITE:
            return query
        try:
            t0 = now_ms()
            rw = self.llm.chat(_REWRITE_SYSTEM_PROMPT, query)
            if rw and rw.strip():
                rw = rw.strip()
                # 中文保护：若原文含中文而改写结果不含中文，则丢弃改写结果
                query_has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in query)
                rw_has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in rw)
                if query_has_chinese and not rw_has_chinese:
                    print('[molbase] 查询改写丢失中文，保留原文以保证中文库召回')
                    return query
                # 性能埋点：查询改写（模型环节）
                self._trace_llm("查询改写", t0, _REWRITE_SYSTEM_PROMPT, query, rw)
                return rw
        except Exception:
            pass  # 模型不可用 -> 降级用原查询
        return query

    def _filter_by_route(self, results: list[Any], route: str) -> list[Any]:
        """
        按路由类型对结果进行软优先重排，不再硬过滤丢弃条目。
    
        修复逻辑：
        1. 移除原有的列表推导硬过滤，确保召回池中的知识条目不被丢弃。
        2. 使用稳定排序 (stable sort) 根据 entry_type 优先级重新排列结果。
        3. 优先级规则：
           - 目标 route 类型 (如 skill/code/knowledge) 优先级为 0，排在最前。
           - knowledge 类型 (当 route 非 knowledge 时) 优先级为 1，排在中间。
           - 其他类型优先级为 2，排在最后。
        4. 当 route == "knowledge" 时，knowledge 条目优先级为 0，code/skill 等靠后。
    
        此策略保证召回池长度不变，仅调整顺序，避免空召回导致的弃答。
        """
        def get_priority(item: Any) -> int:
            """
            计算条目的排序优先级。
            返回越小，排序越靠前。
            """
            entry_type = item.entry.entry_type
            if entry_type == route:
                return 0  # 目标类型，最高优先级
            elif entry_type == "knowledge":
                return 1  # 知识类型，次级优先级
            else:
                return 2  # 其他类型，最低优先级

        # 使用 stable sort 保持同优先级内的相对顺序，仅按优先级重排
        return sorted(results, key=get_priority)

    def _filter_by_scene(self, results: list[Any], scene_id: str) -> list[Any]:
            """
            场景过滤：树收敛硬剪枝 + 冷场景 relax。
            1. 若 scene_id 为空或 "global"，返回全部结果。
            2. 过滤出 scene 匹配的条目。
            3. 若结果少于 SCENE_RELAX_MIN，则补充 "global" 场景的条目（去重保序）。
            4. 异常降级返回原结果。
            """
            from .config import SCENE_RELAX_MIN
            try:
                # 情况 1: 全局或空 ID，不过滤
                if not scene_id or scene_id == "global":
                    return results

                # 情况 2: 正常过滤
                filtered = []
                for entry in results:
                    # 安全获取 meta 和 scene
                    try:
                        meta = getattr(entry, "meta", None) or getattr(getattr(entry, "entry", None), "meta", {})
                        if isinstance(meta, dict) and meta.get("scene", "global") == scene_id:
                            filtered.append(entry)
                    except (AttributeError, TypeError, KeyError):
                        # 单个条目处理异常，跳过或保留，这里选择跳过该条目
                        continue

                # 情况 3: 冷场景 relax 逻辑
                if len(filtered) < SCENE_RELAX_MIN:
                    # 收集所有 scene 为 "global" 的条目
                    global_entries = []
                    for entry in results:
                        try:
                            meta = getattr(entry, "meta", None) or getattr(getattr(entry, "entry", None), "meta", {})
                            if isinstance(meta, dict) and meta.get("scene", "global") == "global":
                                global_entries.append(entry)
                        except (AttributeError, TypeError, KeyError):
                            continue

                    # 合并去重并保持原顺序
                    # 使用集合记录已添加的 entry 对象引用（假设 entry 对象可哈希或需唯一标识）
                    # 由于 entry 可能是复杂对象，这里假设通过 id(entry) 去重
                    seen_ids = set()
                    merged = []
                    for item in filtered + global_entries:
                        item_id = id(item)
                        if item_id not in seen_ids:
                            seen_ids.add(item_id)
                            merged.append(item)

                    return merged

                return filtered

            except Exception:
                # 兜底异常处理，返回原结果
                return results

    def _filter_by_refs(self, results: list[Any], refs: list[str] | None) -> list[Any]:
            """
            引用过滤：用户显式给文件参考时优先、命中即短路。
            1. refs 为空，返回全部。
            2. 保留 entry_id 或 source 命中任一 ref 的条目。
            3. 异常降级返回原结果。
            """
            try:
                # 情况 1: refs 为空，不过滤
                if not refs:
                    return results

                filtered = []
                for entry in results:
                    try:
                        # 获取 entry_id 和 source
                        inner = getattr(entry, 'entry', entry)  # ScoredHit.entry 才是 Entry
                        entry_id = getattr(inner, 'entry_id', None)
                        source = getattr(inner, 'source', None)

                        # 构建搜索字符串
                        search_str = ""
                        if entry_id:
                            search_str += str(entry_id)
                        if source:
                            search_str += " " + str(source)

                        # 命中判断：任一 ref 出现在搜索串中
                        if any(ref in search_str for ref in refs):
                            filtered.append(entry)
                    except (AttributeError, TypeError, ValueError):
                        # 单个条目属性访问异常，跳过
                        continue

                return filtered

            except Exception:
                # 兜底异常处理，返回原结果
                return results

    def _filter_by_tags(self, results: list[Any], tags: list[str] | None) -> list[Any]:
        """
        根据标签过滤或提升结果。
        
        修复说明：支持 or_boost 模式，命中标签的结果排前，未命中的保留但排后，避免误删召回。
        """
        from .config import TAG_FILTER_MODE, TAG_BOOST
        
        if not tags:
            return results
        want = [t.strip() for t in tags if t and str(t).strip()]
        if not want:
            return results
        
        if TAG_FILTER_MODE == "or_boost":
            # 收集每个结果的标签命中数
            scored_results = []
            for r in results:
                have = r.entry.meta.get("tags") or []
                # 兼容 str 与 list
                if isinstance(have, str):
                    have = [have]
                if not isinstance(have, list):
                    have = []
                
                hit_count = sum(1 for t in want if t in have)
                scored_results.append((r, hit_count))
            
            # stable sort: 命中数高的排前，命中数低的排后
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scored_results]
        else:
            # 保持原 AND 语义
            out = []
            for r in results:
                have = r.entry.meta.get("tags") or []
                if isinstance(have, str):
                    have = [have]
                if not isinstance(have, list):
                    have = []
                if all(t in have for t in want):
                    out.append(r)
            return out

    # ------------------------------------------------------------------ 对外查询
    def query(
        self,
        text: str,
        top_k: int = FINAL_TOP_K,
        tags: list[str] | None = None,
        session_id: str = "",
        device_mac: str = "",
        scene: str = "",
        refs: list[str] | None = None,
        label: str = "",
    ) -> QueryResponse:
        """
        核心查询接口：意图路由 -> 召回 -> 重排 -> 最小上下文输出。

        tags：可选，按 meta.tags 过滤（AND 语义），用于按「运维/架构/敏感」等分类精准检索。
        session_id：可选会话标识，用于监控「每个会话走了哪几步、缺哪步」。
        scene：场景提示，用于辅助意图识别和场景过滤。
        refs：参考实体列表，用于三级短路之 Level1 过滤（实体锚点召回）。
        label：全量明细监控标记（golden / negative / real），用于分组对比。
        """
        t_start = now_ms()  # v6 全量延迟埋点：整条 query 链路起点
        # 结构化路由：获取 route、entities、query_tags 及 scene_id
        route_info = self._route_full(text, scene_hint=scene)
        route = route_info["route"]
        entities = route_info.get("entities", [])
        query_tags = route_info.get("query_tags", [])
        scene_id = route_info.get("scene", "") or scene or "global"

        # 查询改写：LLM 可用时口语化→检索友好关键词（保留原有逻辑）
        search_text = self._rewrite(text)

        # 场景识别：若 scene_id 仍是 "global" 且有 session_id，调用 resolve_scene 解析
        if scene_id == "global" and session_id:
            scene_id = self.resolve_scene(scene_hint=scene, session_id=session_id, query=text)

        # 召回（用改写后的查询，激活图扩散）：尝试传入 seed_ids
        t_recall = now_ms()
        try:
            hits = self.retriever.retrieve(
                search_text,
                top_n=max(top_k * 8, 40),
                resolver=self._resolver,
                seed_ids=entities,
            )
        except Exception:
            # 异常时回退不传 seed_ids 的旧调用，保证兼容
            hits = self.retriever.retrieve(
                search_text,
                top_n=max(top_k * 8, 40),
                resolver=self._resolver,
            )

        # 三级短路之 refs（Level1）：若 refs 非空，硬约束过滤（命中即短路，未命中标记 refs_miss）
        refs_miss = False
        if refs:
            hits = self._filter_by_refs(hits, refs)
            if not hits:
                refs_miss = True

        # 场景过滤（Level3 硬约束 + relax）
        hits = self._filter_by_scene(hits, scene_id)

        # 标签过滤：显式 tags 走 or_boost 软优先；LLM 解析的 query_tags 不参与硬过滤，
        # 仅通过 _compute_confidence 作为软信号，避免误删召回（v3 标签链路修复）
        hits = self._filter_by_tags(hits, tags)

        # 保留 rerank 前的原始 BM25 得分，供「是否触发网络兜底」判断真实匹配强度
        raw_hits = hits

        # 重排（传入 embedder，模型失效时内部自动降级为 BM25）
        t_rerank = now_ms()
        # 语义重排同样用用户原始提问：LLM 改写补充的泛化同义词会稀释语义排序，
        # 把精准条目挤到后面（如 AI 批改模型选型被泛化成"深度学习/NLP"后错配架构设计）。
        ranked = self._rerank_ranked(hits, text, device_mac)
        # v9.1 原文层补强：长文档再跑一遍块级比对，把「答案在正文后段」的条目捞回来
        ranked = self._apply_chunk_boost(ranked, text)

        # 性能埋点：召回 / 重排（代工环节，不消耗模型 token）
        if self.profiler is not None:
            self.profiler.record(
                "混合召回",
                llm_call=False,
                duration_ms=now_ms() - t_recall,
                note=f"命中{len(hits)}条",
            )
            self.profiler.record(
                "重排",
                llm_call=False,
                duration_ms=now_ms() - t_rerank,
                note=f"embedding={getattr(self.embedder, 'name', 'none')}",
            )

        # 路由类型优先级：knowledge 路由下，类型仅作同分 tie-break，避免高分 skill 被低分 knowledge 挤掉
        if route == "knowledge":
            priority = {"knowledge": 0, "code": 1, "skill": 2}
            ranked.sort(key=lambda r: (-r.score, priority.get(r.entry.entry_type, 9)))

        # 组织返回 + 刷新新鲜度（touch）
        results: list[QueryResult] = []
        hit_ids: list[str] = []
        for r in ranked[:top_k]:
            # 自动命中热度（自我进化：用得多自然热），add_heat 同时刷新新鲜度
            self.store.add_heat(r.entry.entry_id, AUTO_HIT_HEAT)
            r.entry.heat += AUTO_HIT_HEAT  # 同步内存缓存，下次重排立即生效
            hit_ids.append(r.entry.entry_id)
            results.append(
                QueryResult(
                    entry_id=r.entry.entry_id,
                    entry_type=r.entry.entry_type,
                    title=r.entry.title,
                    content=r.entry.content,
                    source=r.entry.source,
                    score=round(r.score, 4),
                    meta=r.entry.meta,
                    summary=r.entry.summary,
                )
            )
        # 记录共现 + 节流触发进化（自我进化闭环，全自动，无需人工）
        self.evolution.record_query(hit_ids)
        self._maybe_evolve()

        # 网络搜索兜底：内部命中很弱或为空时，联网补充（外部召回源，可插拔）
        web_results: list[QueryResult] = []
        if self._should_web_search(raw_hits):
            t_web = now_ms()
            web_results = self._web_search(search_text, WEB_SEARCH_TOP_N)
            # 性能埋点：网络搜索（外部召回源，不消耗模型 token，但消耗输入 token）
            if self.profiler is not None and web_results:
                self.profiler.record(
                    "网络搜索",
                    llm_call=False,
                    input_tokens=estimate_tokens(
                        "".join(r.title + " " + r.content for r in web_results)
                    ),
                    duration_ms=now_ms() - t_web,
                    note=f"兜底{len(web_results)}条",
                )

        # token 埋点：粗估本次返回内容与「召回候选全集」基线的 token 数
        all_text = "".join(r.title + " " + r.content for r in results)
        all_text += "".join(r.title + " " + r.content for r in web_results)
        token_estimate = estimate_tokens(all_text) if TOKEN_TRACE else 0
        # 真实基线 = 召回候选全集（rerank 前的候选）token，而非旧版「全库塞」：
        # saved 反映「召回候选 -> 最小上下文」的真实收敛，不再是虚高的全库压缩率。
        recall_candidates = len(raw_hits)
        recall_tokens = (
            estimate_tokens("".join(h.entry.title + " " + h.entry.content for h in raw_hits))
            if TOKEN_TRACE else 0
        )
        baseline_tokens = recall_tokens

        # 质量底线（输出闸门）：综合置信度 + abstain 判定
        # 置信度融合：传入标签/场景/实体信号（动态归一化，无信号项不干扰）
        # 置信度用用户原始提问计算：LLM 改写会补充泛化同义词（如"深度学习/自然语言处理"），
        # 稀释语义集中度导致真实相关条目 conf 被拉低误弃答。原文更贴近用户真实意图。
        confidence, strength, margin = self._compute_confidence(
            ranked, text, query_tags=query_tags, scene_id=scene_id, anchor_ids=entities
        )
        # v9.0 两阶段闸门：阶段1 判「库域内」（查询词覆盖率），阶段2 判「条目质量」（logistic 置信度）
        abstain, reason, subreason = self._judge_abstain(ranked, confidence, strength, text)
        if abstain:
            # 使用同源原始余弦 strength 替代归一化后的 score，消除量纲伪矛盾
            self.store.record_abstain(
                query=text,
                route=route,
                reason=reason,
                subreason=subreason,
                recall_candidates=recall_candidates,
                top1_score=strength,
                confidence=confidence,
                session_id=session_id,
            )

        # 版本裁决：按场景解析 top1 候选的权威版本，同问异答的关键锚点。
        # 库内无版本数据时 resolve_version 返回 None，字段为空属正常。
        resolved_version_id = ""
        if ranked:
            version_row = self.resolve_version(ranked[0].entry.entry_id, scene_id)
            if version_row:
                resolved_version_id = str(version_row.get("version_id", "") or "")
        # v9.1 原文层定位：弃答时不返回定位（避免调用方拿空答案配真引用）
        refs = [] if abstain else self._collect_refs(ranked)
        resp = QueryResponse(
            route=route,
            results=results,
            query=text,
            rewritten=search_text,
            web_results=web_results,
            token_estimate=token_estimate,
            baseline_tokens=baseline_tokens,
            content_chars=len(all_text),
            confidence=confidence,
            abstain=abstain,
            scene_id=scene_id,
            entities=entities,
            resolved_version_id=resolved_version_id,
            refs_miss=refs_miss,
            refs=refs,
        )

        # 会话级 token 台账：累计节省比例（baseline<=0 时 ledger 内部自动跳过）
        self.token_ledger.record(token_estimate, baseline_tokens)
        # 持久化埋点：把本次 token 优化落库（知识库本体 metrics 表），
        # 供服务化后的 /metrics 与监控大屏实时聚合展示「部署后埋点实时监控」。
        self.store.record_metric(
            query=text,
            route=route,
            hits=len(results) + len(web_results),
            actual_tokens=token_estimate,
            baseline_tokens=baseline_tokens,
            confidence=confidence,
            recall_candidates=recall_candidates,
            recall_tokens=recall_tokens,
            session_id=session_id,
            detail={
                "scene_id": scene_id,
                "label": label,
                "tags": tags or [],
                "query_tags": query_tags,
                "entities": entities,
                "refs": refs or [],
                "rewritten": search_text if search_text != text else "",
                "rewrite_used": 1 if search_text != text else 0,
                "resolved_version_id": resolved_version_id,
                "refs_miss": 1 if refs_miss else 0,
                "abstain": 1 if abstain else 0,
                "abstain_reason": reason,
                "abstain_subreason": subreason,
                "strength": round(strength, 4),
                "margin": round(margin, 4),
                "top1_score": round(ranked[0].score, 4) if ranked else 0.0,
                "top1_entry_id": ranked[0].entry.entry_id if ranked else "",
                "top1_title": ranked[0].entry.title if ranked else "",
                "hits_detail": [
                    {"entry_id": r.entry.entry_id, "title": r.entry.title,
                     "entry_type": r.entry.entry_type, "source": r.entry.source,
                     "score": round(r.score, 4)}
                    for r in ranked[:top_k]
                ],
                "web_results": len(web_results),
                "embed_model": getattr(self.embedder, "model", getattr(self.embedder, "name", "")),
                "embedding_backend": getattr(self.embedder, "name", EMBEDDING_BACKEND),
                "latency_total_ms": now_ms() - t_start,
                "latency_recall_ms": t_rerank - t_recall,
                "latency_rerank_ms": now_ms() - t_rerank,
                "confidence_threshold": ABSTAIN_CONFIDENCE,
            },
        )
        # v7.2 场景锚定资产回流：把本次查询的场景/实体/版本写回结构化表，供下次裁决复用
        self._index_query_assets(scene_id, entities, ranked, resolved_version_id)
        # v7 自进化闭环：高置信命中自动正反馈（热度越用越准）+ 工具调用轨迹归因。
        # 两者都不得影响主链路返回，异常一律吞掉。
        if not abstain and ranked:
            top1_id = ranked[0].entry.entry_id
            self._auto_feedback(top1_id, session_id, confidence)
            # v8.0 飞轮：同会话隐式正反馈——同一 session 再次命中同一条目，
            # 说明该条目被反复用到，额外记一次 hit（多源信号，不依赖用户打分）。
            try:
                seen = self._session_hits.setdefault(session_id or "_default", set())
                if top1_id in seen:
                    self.store.add_heat(top1_id, FEEDBACK_WEIGHT_HIT)
                    self.store.record_feedback(session_id, top1_id, "hit", source="auto_implicit")
                seen.add(top1_id)
                # 防长跑服务内存泄漏：会话记录超过上限时丢弃最早的一个（dict 保插入序）
                if len(self._session_hits) > 500:
                    try:
                        self._session_hits.pop(next(iter(self._session_hits)))
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            self.record_tooltrace(
                session_id,
                "query:" + str(route),
                [{"tool": "molbase_query", "args": {"query": text[:80], "top_k": top_k, "scene": scene}}],
            )
        except Exception:
            pass
        # v7.2 工具轨迹隐式复用：查询自动携带相关 skill 摘要（不影响主结果）
        resp.skills = self._match_auto_skills(text)
        # v8.0 飞轮：技能被检索复用即记一次 cite（技能复用的正反馈，闭环到技能热度）
        if resp.skills:
            try:
                for sk in resp.skills:
                    sid = sk.get("entry_id")
                    if sid:
                        self.store.add_heat(sid, FEEDBACK_WEIGHT_CITE)
                        self.store.record_feedback(session_id, sid, "cite", source="auto_skill")
            except Exception:
                pass
        return resp

    def _semantic_sim(self, query: str, entry: Entry) -> float:
        """
        计算 query 与条目的原始语义余弦（未归一化），用于置信度兜底。

        语义文本必须与预计算/现场重排同口径（统一走 semantic_text），
        否则同一对「查询-条目」在不同路径下会得到不同余弦，导致 conf 抖动。
        """
        if self.embedder is None:
            return 0.0

        try:
            from .vectorize import semantic_text
            from .config import SEMANTIC_SIM_CONTENT_CHARS

            entry_text = semantic_text(
                entry.title or "",
                entry.content or "",
                entry.summary or "",
                max_chars=SEMANTIC_SIM_CONTENT_CHARS,
            )
            q_vec = self.embedder.embed(query)
            e_vec = self.embedder.embed(entry_text)
            return cosine_similarity(q_vec, e_vec)
        except Exception:
            return 0.0

    def _rerank_ranked(self, hits, text: str, device_mac: str = ""):
        """优先用预计算向量重排（只编码 1 条 query）；向量覆盖不足或异常时回退现场重排。"""
        # 两条路径的语义文本窗口必须一致（统一 SEMANTIC_SIM_CONTENT_CHARS=300），否则排序抖动
        from .config import VECTOR_PRECOMPUTE_ENABLED, VECTOR_MIN_COVERAGE, SEMANTIC_SIM_CONTENT_CHARS
        limit = SEMANTIC_SIM_CONTENT_CHARS
        try:
            if VECTOR_PRECOMPUTE_ENABLED and hits and self.embedder is not None:
                from .vectorize import model_fingerprint
                fp = model_fingerprint(self.embedder)
                # 只取候选集向量：重排不需要全库向量，整表扫会白读 MB 级 BLOB
                candidate_ids = [h.entry.entry_id for h in hits]
                vectors = self.store.get_vectors(fp, "entry", owner_ids=candidate_ids)
                if vectors:
                    covered = sum(1 for h in hits if h.entry.entry_id in vectors)
                    coverage = covered / max(1, len(hits))
                    # 覆盖率达标才用预计算，避免"部分有向量"污染排序
                    if coverage >= VECTOR_MIN_COVERAGE:
                        q_vec = self.embedder.embed(text)
                        if q_vec:
                            return self.reranker.rerank_with_vectors(hits, q_vec, vectors, device_mac)
        except Exception:
            pass
        return self.reranker.rerank(hits, embedder=self.embedder, query=text,
                                    device_mac=device_mac, content_limit=limit)

    def _apply_chunk_boost(self, ranked, text: str):
        """
        长文档块级补强：条目级语义分只看正文前 N 字，长文档的答案常在正文后段。

        只对超长条目做，且只在「块级相似度高于条目级」时上调，
        上调后在 logit 之外直接用加权修正 score 并重排——不做归一化，
        避免少量条目变动引起全局 min-max 漂移、把整批排序打乱。
        """
        from .config import (CHUNK_BOOST_ENABLED, CHUNK_BOOST_MIN_CHARS,
                             CHUNK_BOOST_WEIGHT)
        if not CHUNK_BOOST_ENABLED or not ranked or self.embedder is None:
            return ranked
        try:
            from .kb_layers import chunk_vectors_for, max_chunk_similarity

            long_ids = [
                r.entry.entry_id for r in ranked
                if len(r.entry.content or "") > CHUNK_BOOST_MIN_CHARS
            ]
            if not long_ids:
                return ranked
            pairs_map = chunk_vectors_for(self.store, self.embedder, long_ids)
            if not pairs_map:
                return ranked
            q_vec = self.embedder.embed(text)
            if not q_vec:
                return ranked

            changed = False
            for r in ranked:
                pairs = pairs_map.get(r.entry.entry_id)
                if not pairs:
                    continue
                chunk_sim = max_chunk_similarity(q_vec, pairs)
                if chunk_sim <= r.sim_norm:
                    continue
                new_sim = (1.0 - CHUNK_BOOST_WEIGHT) * r.sim_norm + CHUNK_BOOST_WEIGHT * chunk_sim
                r.score += self.reranker.w_sim * (new_sim - r.sim_norm)
                r.sim_norm = new_sim
                changed = True
            if changed:
                ranked.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            pass
        return ranked

    def _collect_refs(self, ranked, limit: int = 3) -> list[dict]:
        """
        取 top-N 命中的原文定位信息，供答案标注「出自哪一节第几段」。

        path 是标题栈快照，char_start/char_end 是块在正文中的字符区间，
        两者足够把结论还原到原文位置，不需要把整段正文再塞回上下文。
        """
        refs = []
        for r in ranked[:limit]:
            entry_id = r.entry.entry_id
            try:
                chunks = self.store.get_chunks(entry_id)
            except Exception:
                chunks = []
            if not chunks:
                continue
            # 取最靠前的块作为该条目的代表定位（块序号 0 覆盖文档开头，最常用）
            c = chunks[0]
            refs.append({
                "entry_id": entry_id,
                "title": r.entry.title,
                "path": c.get("path") or "",
                "heading": c.get("heading") or "",
                "char_start": c.get("char_start", 0),
                "char_end": c.get("char_end", 0),
                "chunk_id": c.get("chunk_id", ""),
            })
        return refs

    def _judge_abstain(self, ranked, confidence: float, strength: float, text: str):
        """
        两阶段弃答判定：阶段1 判「问题是否落在库域内」，阶段2 判「命中条目质量」。

        阶段1 只做归因修正，不引入额外拦截：out_of_domain 要求「查询词几乎不在库词表」
        且「语义强度也低」，两个条件同时成立才判库外，避免新词/同义词查询被误杀。
        阶段2 用校准后置信度放行，再用 strength 相对校准中点派生弃答子因。

        子因阈值必须相对 CONF_MID 派生：hash 后端中点 0.26 低于旧的绝对阈值 0.30，
        若仍用 0.30/0.50，则 strength<0.26 才是 weak_semantic、0.30~0.50 区间根本
        到不了（conf 已 >=0.5 放行），low_strength/low_margin 会变成永远不可达的死分支。
        """
        from .config import ABSTAIN_CONFIDENCE, DOMAIN_COV_MIN, NO_MATCH_STRENGTH, conf_params

        mid, _slope = conf_params(getattr(self.embedder, "name", ""))

        if not ranked:
            return True, "no_data", "empty_recall"

        try:
            cov = self._domain_coverage(text)
        except Exception:
            cov = 1.0
        if cov < DOMAIN_COV_MIN and strength < NO_MATCH_STRENGTH:
            return True, "no_data", "out_of_domain"

        if confidence >= ABSTAIN_CONFIDENCE:
            return False, "", ""

        # 低于放行线：按 strength 相对中点的位置给出可归因子因
        if strength < mid:
            return True, "no_data", "weak_semantic"
        if strength < mid + 0.10:
            return True, "weak_match", "low_strength"
        return True, "ambiguous", "low_margin"

    def _domain_coverage(self, text: str) -> float:
        """计算查询词在库词表中的覆盖率（两阶段闸门的阶段1判据）。
        - 用模块顶部已导入的 tokenize（来自 .index）对 text 分词，去重
        - 只保留长度 >= DOMAIN_COV_TOKEN_MIN_LEN 的词（从 .config 导入）
        - 词表 = self.bm25.idf 的键集合（BM25Index 构建后 idf 即全库词表）
        - 返回 命中词数 / 总词数；无有效词时返回 0.0
        - 异常时返回 1.0（宁可放行交给阶段2质量闸门兜底，不能让异常触发"库外"误杀）
        """
        try:
            from .index import tokenize
            from .config import DOMAIN_COV_TOKEN_MIN_LEN

            tokens = tokenize(text)
            unique_tokens = set(tokens)
            valid_tokens = {t for t in unique_tokens if len(t) >= DOMAIN_COV_TOKEN_MIN_LEN}
            # 极短查询（如「网」）分词后可能全是单字，此时回退到全 token 集合，
            # 否则覆盖率恒为 0，会被阶段1 误判成库外问题
            if not valid_tokens:
                valid_tokens = unique_tokens

            if not valid_tokens:
                return 1.0

            vocab = self._domain_vocab
            if vocab is None:
                vocab = set(self.bm25.idf.keys())
                self._domain_vocab = vocab
            hits = len(valid_tokens & vocab)
            return hits / len(valid_tokens)
        except Exception:
            return 1.0

    def _lexical_coverage(self, query: str, entry) -> float:
        """
        查询词在 top1 条目中的 idf 加权覆盖率（0~1）。

        与 _domain_coverage 的区别：那个看「词在不在全库词表里」，判库域；
        这个看「词在不在这一条命中里」，判这一条命中吃掉了提问的多少信息。

        分母只统计 idf >= LEX_COV_MIN_IDF 的词：像「如何」「怎么」这类高频虚词
        idf 接近 0，计入分母会把覆盖率稀释成常量，失去判别力。
        """
        try:
            from .index import tokenize
            from .config import LEX_COV_MIN_IDF, LEX_COV_MIN_TOKENS

            q_tokens = {t for t in tokenize(query) if t}
            if len(q_tokens) < LEX_COV_MIN_TOKENS:
                return 0.0

            idf = self.bm25.idf
            weights = {t: idf[t] for t in q_tokens if t in idf and idf[t] >= LEX_COV_MIN_IDF}
            # 分母为 0 说明提问里全是库外词或全是虚词，无从判断覆盖率，交回语义分决定
            total = sum(weights.values())
            if total <= 0:
                return 0.0
            # 只有 1 个词够格时覆盖率非 0 即 1，判别力为零且极易误放（「今天天气怎么样」
            # 只因命中「今天」就会被算成 100% 覆盖），这类短问不启用词法推断
            if len(weights) < LEX_COV_MIN_TOKENS:
                return 0.0

            # 只对 top1 这一条做词面比对：tokenize 一次 300 字文档，代价可忽略
            entry_text = (entry.title or "") + " " + (entry.content or "")
            entry_tokens = set(tokenize(entry_text))
            covered = sum(w for t, w in weights.items() if t in entry_tokens)
            return max(0.0, min(1.0, covered / total))
        except Exception:
            return 0.0

    def _compute_confidence(self, ranked, query: str, query_tags=None, scene_id: str = "",
                            anchor_ids=None) -> tuple:
        """
        计算 (confidence, strength, margin)。

        confidence 主项是按后端标定中点的 logistic 校准值（可当概率读），
        再叠加标签/场景/实体三项乘性小幅加成；主项 + 加成合起来不超过 1.0。

        strength 取「语义余弦」与「词法覆盖率折算值」的较大者：
        关键词堆叠型查询会被词袋向量平均化导致余弦虚低，此时词法覆盖率是更可靠的证据。
        两者都是 0~1 的同量纲量，取 max 不引入新的标定问题。
        """
        if not ranked:
            return (0.0, 0.0, 0.0)
        import math
        from .config import (CONF_W_TAG, CONF_W_SCENE, CONF_W_ENTITY,
                             SCENE_MATCH_WEAK, LEX_COV_ENABLED, LEX_COV_W,
                             conf_params)

        # 中点/斜率按运行时真实后端取：embedder 可能已从 ollama 静默降级到 hash
        mid, slope = conf_params(getattr(self.embedder, "name", ""))

        top1 = ranked[0]
        cos1 = getattr(top1, "sim_norm", None)
        if cos1 is None:
            cos1 = self._semantic_sim(query, top1.entry)
        cos2 = 0.0
        if len(ranked) > 1:
            cos2 = getattr(ranked[1], "sim_norm", None)
            if cos2 is None:
                cos2 = self._semantic_sim(query, ranked[1].entry)

        strength = max(0.0, min(1.0, float(cos1)))
        margin = max(0.0, min(1.0, float(cos1) - float(cos2))) if len(ranked) > 1 else 0.0

        # 词法覆盖率下限：只抬升，不降低；覆盖率 1.0 时等价余弦 LEX_COV_W
        if LEX_COV_ENABLED:
            cov = self._lexical_coverage(query, top1.entry)
            strength = max(strength, min(1.0, LEX_COV_W * cov))

        # 主项：logistic 校准（strength==mid 时恰好 0.5）
        conf = 1.0 / (1.0 + math.exp(-slope * (strength - mid)))

        meta = top1.entry.meta if isinstance(top1.entry.meta, dict) else {}

        # 标签加成：查询标签被条目标签覆盖的比例
        if query_tags:
            entry_tags = meta.get("tags")
            if isinstance(entry_tags, list) and entry_tags:
                tag_set = set(query_tags)
                cov = len(tag_set & set(entry_tags)) / max(1, len(tag_set))
                conf += conf * CONF_W_TAG * cov

        # 场景加成：同场景记满分，弱匹配场景记折减
        if scene_id and scene_id != "global":
            entry_scene = meta.get("scene")
            if entry_scene:
                scene_score = 1.0 if entry_scene == scene_id else SCENE_MATCH_WEAK
                conf += conf * CONF_W_SCENE * scene_score

        # 实体加成：anchor_ids 是路由出来的实体名，不是 entry_id，
        # 因此先比条目 meta 里的实体集合，未挂实体时用标题包含关系兜底（实体名常直接进标题）
        if anchor_ids:
            anchors = {a for a in anchor_ids if a}
            ent = meta.get("entities") or meta.get("entity_ids") or []
            ent_set = set(ent) if isinstance(ent, (list, tuple, set)) else set()
            if anchors and (ent_set & anchors):
                conf += conf * CONF_W_ENTITY
            else:
                title = top1.entry.title or ""
                if any(a in title for a in anchors):
                    conf += conf * CONF_W_ENTITY * 0.6

        conf = max(0.0, min(1.0, conf))
        return (round(conf, 4), round(strength, 4), round(margin, 4))

    # ------------------------------------------------------------------ 网络搜索兜底
    def _should_web_search(self, hits: list[Any]) -> bool:
        """
        判断是否触发网络搜索兜底（基于 rerank 前的 BM25 原始得分）：
        - 未启用网络搜索（NullSearchProvider）时恒 False；
        - 内部召回为空，或 top-1 的 BM25 原始得分低于阈值时 True。
        用原始得分而非归一化分——归一化后「只要有命中」得分就接近 1，
        无法反映匹配的真实强度。
        """
        if getattr(self.search_provider, "name", "null") == "null":
            return False
        if not hits:
            return True
        return hits[0].score < WEB_SEARCH_TRIGGER_SCORE

    def _web_search(self, query: str, top_n: int) -> list[QueryResult]:
        """
        执行网络搜索兜底，把外部结果包装成 QueryResult（entry_type="web"，不入库）。
        任何异常由 provider 内部吞掉并返回空列表，绝不崩溃。
        """
        try:
            hits = self.search_provider.search(query, top_n)
        except Exception:
            return []
        return [
            QueryResult(
                entry_id=f"web::{h.url}",
                entry_type="web",
                title=h.title,
                content=h.content,
                source=h.url,
                score=round(h.score, 4),
                meta={"origin": "web"},
            )
            for h in hits
        ]

    def adopt_web_result(self, web_result: QueryResult, entry_id: str | None = None) -> str:
        """
        采纳一条网络搜索结果，沉淀为内部知识条目（「搜索 → 沉淀 → 复用」闭环）。

        采纳后，下次同类查询直接命中内部知识，不再联网——网络搜索从「每次必查」
        退化为「冷知识/新知识的临时补充」，这正是「越用越少依赖网络」的来源。

        返回新条目的 entry_id。
        """
        new_id = entry_id or f"web::{web_result.source}"
        self.ingest_knowledge(
            new_id,
            web_result.title,
            web_result.content,
            source=web_result.source,
            meta={"origin": "web"},
        )
        return new_id

    # ------------------------------------------------------------------ 自我进化
    def _maybe_evolve(self) -> None:
        """按 EVOLVE_INTERVAL 节流触发自我进化（共现建边 + 冷知识衰减）。"""
        self._query_count += 1
        if self._query_count % EVOLVE_INTERVAL == 0:
            self.evolve()

    def evolve(self) -> dict[str, int]:
        """触发一次自我进化（query 会自动节流调用，也可手动触发），返回学习统计。"""
        stats = self.evolution.learn()
        # 质量底线（进化侧）：死知识自动归档
        stats["archived"] = self.store.archive_dead(ARCHIVE_DAYS, ARCHIVE_HEAT_MAX)
        # v8.0 飞轮：空转标记——本步既没衰减也没归档，说明进化没产生真实生命周期动作，
        # 只有「连边」在空转（旧版 added_edges 2279 而 decayed/archived 恒 0 的病根）。
        stats["idle"] = 1 if (int(stats.get("cold_decayed", 0)) + int(stats.get("archived", 0))) == 0 else 0
        # 进化日志：记录本次步进结果（建边/冷衰减/归档），驱动飞轮「进化」环节统计
        self.store.record_evolve(
            added_edges=int(stats.get("edges", 0)),
            decayed=int(stats.get("cold_decayed", 0)),
            archived=int(stats.get("archived", 0)),
        )
        self._rebuild()  # 学习后重建关联图，让新发现的关联边生效
        return stats

    # ------------------------------------------------------------------ 上下文压缩
    def summarize(self, text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
        """
        上下文压缩（代工层降 token）：
        - 语义保持模式（默认）：去冗余 -> LLM 摘要 -> 硬事实校验兜底，语义 100% 不变；
        - 有损模式（COMPACT_MODE=lossy）：LLM 生成式 -> TextRank 抽取式 -> 截断。
        """
        if COMPACT_ENABLED and COMPACT_MODE == "semantic":
            return semantic_preserve(text, max_chars=max_chars, llm=self.llm)
        if self.llm is not None:
            try:
                t0 = now_ms()
                summary = self.llm.chat(_SUMMARIZE_SYSTEM_PROMPT, text)
                if summary and summary.strip():
                    summary = summary.strip()
                    # 性能埋点：摘要压缩（模型环节）
                    self._trace_llm("摘要压缩", t0, _SUMMARIZE_SYSTEM_PROMPT, text, summary)
                    return summary
            except Exception:
                pass  # 模型不可用 -> 降级 TextRank
        # TextRank 抽取式摘要（零模型，超长才压缩）
        if len(text) > max_chars:
            try:
                summary = textrank_summarize(text)
                if summary and len(summary) < len(text):
                    return summary
            except Exception:
                pass  # TextRank 异常 -> 降级截断
        return text if len(text) <= max_chars else text[:max_chars] + "…"

    def expand(self, entry_id: str) -> Optional[Entry]:
        """按需展开原文：凭 entry_id 拉全文，语义无损可逆。"""
        return self.store.get(entry_id)

    def backfill_summaries(self, limit: int = BACKFILL_BATCH) -> dict:
        """批量回填存量摘要：找 summary 为空的条目逐个生成并回写，完成后重建索引刷新缓存。"""
        entries = self.store.list_no_summary(limit)
        done = 0
        for e in entries:
            try:
                s = semantic_preserve(e.content, SUMMARY_MAX_CHARS, e.entry_id, self.llm)
                if s:
                    self.store.update_summary(e.entry_id, s)
                    done += 1
            except Exception:
                continue
        if done > 0:
            self._rebuild()  # 回写后重建索引，让 query 立即读到新摘要（否则缓存里仍是旧空值）
        return {"scanned": len(entries), "backfilled": done}

    def chunk(self, text: str, max_chars: int = 500) -> list[str]:
        """智能切块：按段落/句子边界把长文本切成上下文块（零模型）。"""
        return split_blocks(text, max_chars=max_chars)

    # ------------------------------------------------------------------ 反馈闭环
    def feedback(self, entry_id: str, action: str = "hit", session_id: str = "", source: str = "manual") -> None:
        """
        反馈回写：action 取值 hit / adopt / cite / reject，累加对应热度权重。

        - hit / adopt / cite 为正反馈（热度上升）；
        - reject 为负反馈（热度下降，自然沉底）。
        session_id：可选会话标识，用于监控飞轮「反馈」环节归属。
        source：manual=显式反馈，auto=高置信命中后的隐式正反馈。
        v7：正反馈累计 5 次自动触发进化，让反馈真正驱动飞轮（而非只记日志）。
        """
        weight_map = {
            "hit": FEEDBACK_WEIGHT_HIT,
            "adopt": FEEDBACK_WEIGHT_ADOPT,
            "cite": FEEDBACK_WEIGHT_CITE,
            "reject": FEEDBACK_WEIGHT_REJECT,
        }
        weight = weight_map.get(action, FEEDBACK_WEIGHT_HIT)
        self.store.add_heat(entry_id, weight)
        self.store.record_feedback(session_id, entry_id, action, source=source)
        # 质量底线（进化侧）：reject 累计达到阈值 -> 自动隔离（不参与召回）
        if action == "reject":
            self._maybe_quarantine(entry_id)
        # v7 反馈驱动进化：正反馈累计到阈值触发一次 evolve（内部已 rebuild）
        cur = getattr(self, "_feedback_since_evolve", 0)
        if action in ("hit", "adopt", "cite"):
            cur += 1
        self._feedback_since_evolve = cur
        if cur >= 5:
            self._feedback_since_evolve = 0
            self.evolve()
            return
        # 未触发进化时重建索引（热度变化影响后续重排）
        self._rebuild()

    def _evolution_health(self) -> dict:
        """自进化健康度：委托 store 聚合反馈/进化/缺口/轨迹/会话完整率。"""
        return self.store.evolution_health()

    def _index_query_assets(self, scene_id: str, entities: list, ranked: list, version_id: str = "") -> None:
        """v7.2 场景锚定资产回流：把本次查询的场景/命中条目/实体/版本写入结构化表，供下次裁决复用。异常一律吞掉，绝不影响主查询。"""
        try:
            if scene_id:
                self.store.upsert_scene(scene_id)
            for r in ranked:
                entry_id = r.entry.entry_id
                score = getattr(r, 'score', 0.0)
                self.store.link_scene_entry(scene_id, entry_id, via="auto", weight=round(float(score), 4))
            if entities and ranked:
                first_entry_id = ranked[0].entry.entry_id
                for entity in entities:
                    self.store.upsert_entity(entity, first_entry_id, source="query")
            if version_id and ranked:
                first_entry_id = ranked[0].entry.entry_id
                score = getattr(ranked[0], 'score', 0.0)
                self.store.upsert_version(first_entry_id, version_id, scene_id, first_entry_id, authority=round(float(score), 4), confidence=0.0)
        except Exception:
            pass

    def generate_gap_drafts(self, limit: int = 50) -> dict:
        """从弃答缺口生成补数草稿（只建草稿，不自动入库）。"""
        try:
            gaps = self.store.abstain_gaps(limit)
            # 循环外一次性取已有草稿 id 集合，避免循环内重复全表查询
            seen = {d["draft_id"] for d in self.store.list_drafts("", 10000)}
            created = 0
            skipped = 0
            for gap in gaps:
                if gap.get("reason") in ("no_data", "weak_match") and gap.get("query"):
                    # 先按幂等规则算出 id：与 create_draft 内部 md5(query+'|'+reason)[:12] 一致
                    import hashlib
                    did = "draft::" + hashlib.md5((gap["query"] + "|" + gap["reason"]).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
                    if did in seen:
                        skipped += 1
                        continue
                    new_id = self.store.create_draft(gap["query"], gap["reason"], gap.get("subreason", ""))
                    seen.add(new_id)
                    created += 1
            return {"created": created, "skipped": skipped, "pending": len(self.store.list_drafts("pending", 10000))}
        except Exception as e:
            return {"created": 0, "skipped": 0, "error": str(e)}

    def confirm_gap_draft(self, draft_id: str, title: str = "", content: str = "", meta: dict | None = None) -> dict:
        """确认草稿→入库→立即复检命中，形成缺口-补数-验证闭环。必须显式调用才入库。"""
        try:
            draft = self.store.get_draft(draft_id)
            if not draft:
                return {"ok": False, "error": "draft not found"}
            q = draft["query"]
            # draft_id 形如 draft::<hash>，取 hash 段构造入库 entry_id
            entry_id = "draft::" + draft_id.split("::")[-1]
            t = title or q
            c = content or q
            m = dict(meta or {})
            m.setdefault("source", "gap_draft")
            m.setdefault("tags", ["补数"])
            m.setdefault("query", q)
            res = self.ingest_knowledge(entry_id, t, c, source="gap_draft", meta=m)
            state = res.state
            ok_confirm = self.store.confirm_draft(draft_id, entry_id)
            resp = self.query(q, top_k=5, label="gap_verify")
            verified = (not resp.abstain) and resp.confidence >= ABSTAIN_CONFIDENCE
            # v8.0 飞轮：补数草稿确认并复检命中 = 一次真实的任务成功信号，回写正反馈
            # （直接加热度、不触发 rebuild，避免与上面 query 的重建重入）。
            if verified:
                try:
                    self.store.add_heat(entry_id, FEEDBACK_WEIGHT_ADOPT)
                    self.store.record_feedback("", entry_id, "adopt", source="auto_gap")
                except Exception:
                    pass
            return {"ok": True, "draft_id": draft_id, "entry_id": entry_id, "state": state, "confirmed": ok_confirm, "verified": verified, "confidence": round(resp.confidence, 4)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _match_auto_skills(self, text: str, top_n: int = 3) -> list[dict]:
        """v7.2 工具轨迹隐式复用：按查询词命中度，从自动固化技能中挑出最相关的摘要。异常返回空列表，绝不影响主查询。"""
        try:
            words = text.lower().split()
            candidates = []
            for e in self.store.all_entries():
                if e.entry_type != "skill":
                    continue
                meta = e.meta or {}
                if not meta.get("auto_generated"):
                    continue
                content = e.content or ""
                search_text = (e.title + " " + str(meta.get("description") or "") + " " + content[:200]).lower()
                score = sum(1 for w in words if w in search_text)
                if score > 0:
                    candidates.append({
                        "entry_id": e.entry_id,
                        "title": e.title,
                        "description": str(meta.get("description") or "")[:120],
                        "score": score
                    })
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_n]
        except Exception:
            return []

    def _auto_feedback(self, entry_id: str, session_id: str, confidence: float) -> None:
        """
        高置信命中后的隐式正反馈：越常被命中的条目热度越高，实现「越用越准」。
        不触发 rebuild（避免每次 query 都重建索引），热度在下次进化/反馈/入库时生效。
        """
        # v8.0 飞轮：触发线从硬编码 0.6 改为 FEEDBACK_MIN_CONFIDENCE（恒低于弃答线），
        # 保证「答了就会反馈」，杜绝 0.55~0.6 的答而不馈空档（飞轮自锁根因之一）。
        if not entry_id or confidence < FEEDBACK_MIN_CONFIDENCE:
            return
        try:
            self.store.add_heat(entry_id, FEEDBACK_WEIGHT_HIT)
            self.store.record_feedback(session_id, entry_id, "hit", source="auto")
        except Exception:
            return

    def _maybe_quarantine(self, entry_id: str) -> None:
        """
        负反馈隔离：reject 累计达到阈值后把条目标记为 quarantined，
        使其不再参与召回（_rebuild 会过滤），防止「坏知识」继续污染检索结果。
        """
        entry = self.store.get(entry_id)
        if entry is None:
            return
        cnt = int(entry.meta.get("reject_count", 0)) + 1
        patch = {"reject_count": cnt}
        if cnt >= QUARANTINE_REJECT_THRESHOLD:
            patch["quarantined"] = True
        self.store.update_meta(entry_id, patch)

    # ------------------------------------------------------------------ 摄入接口
    def ingest_knowledge(
        self, entry_id: str, title: str, content: str, source: str = "", meta: dict | None = None
    ) -> IngestResult:
        """
        摄入一条知识条目，带「写入冲突四态」判定（质量底线输入闸门）。

        判定顺序（早命中早返回）：
        - pinned   ：已有条目被人工固定（meta.pinned），拒绝覆盖；
        - duplicate：与已有内容近重复（SimHash 或高相似度），跳过写入；
        - conflict ：同 ID 但来源不同且内容实质矛盾，不覆盖、记录待裁决；
        - merge/new：同事实更新/补充，或全新新建，正常 upsert。

        返回 IngestResult（state=new/merge/duplicate/conflict/pinned），
        供调用方按状态决定后续动作（如冲突转人工裁决）。
        """
        meta = meta or {}
        existing = self.store.get(entry_id)

        # 1. pinned：人工固定的事实禁止被覆盖
        if existing and existing.meta.get("pinned"):
            self.store.record_ingest(entry_id, "pinned", title)
            return IngestResult(
                entry_id=entry_id, state="pinned", against_id=entry_id,
                reason="已有条目被人工固定(pinned)，拒绝覆盖",
            )

        # 2. conflict：同 ID、来源不同 -> 不覆盖，待裁决
        #    （优先于 duplicate：来源不同的「矛盾」与「重复」本质不同，
        #      如爸爸记五千 vs 妈妈记八千，即使文本高度相似也是矛盾而非重复）
        if existing and self._is_conflict(existing, source):
            self.store.record_conflict(
                entry_id, title, content, source, "同 ID 来源不同，可能矛盾"
            )
            self.store.record_ingest(entry_id, "conflict", title)
            return IngestResult(
                entry_id=entry_id, state="conflict", against_id=entry_id,
                reason=f"与已有条目(来源={existing.source or '未知'})来源不同，已记录待裁决",
            )

        # 3. duplicate：同 ID 同源（或来源缺失）且内容近重复，无需重复写入
        if existing and self._is_duplicate(existing.content, content):
            self.store.record_ingest(entry_id, "duplicate", title)
            return IngestResult(
                entry_id=entry_id, state="duplicate", against_id=entry_id,
                reason="与已有内容近重复，跳过写入",
            )

        # 4. merge/new：正常写入（入库前生成语义保持摘要）
        state = "merge" if existing else "new"
        self.store.record_ingest(entry_id, state, title)
        summary = ""
        if COMPACT_ENABLED:
            try:
                summary = semantic_preserve(content, SUMMARY_MAX_CHARS, entry_id, self.llm)
            except Exception:
                summary = ""
        self.store.upsert(
            Entry(
                entry_type="knowledge",
                entry_id=entry_id,
                title=title,
                content=content,
                summary=summary,
                source=source,
                meta=meta,
            )
        )
        self._rebuild()
        # v9.0 向量预计算：只补算本条（entry_ids 限定），不做全库扫描
        try:
            from .config import VECTOR_PRECOMPUTE_ENABLED, VECTOR_PRECOMPUTE_BATCH
            if VECTOR_PRECOMPUTE_ENABLED:
                from .vectorize import ensure_entry_vectors
                ensure_entry_vectors(self.store, self.embedder, "entry", force=False,
                                     batch_size=VECTOR_PRECOMPUTE_BATCH, entry_ids=[entry_id])
        except Exception:
            pass
        # v9.1 原文层：本条正文切块 + 块向量，保证新知识立刻可溯源、可块级召回
        try:
            from .config import CHUNK_INDEX_ENABLED, CHUNK_VECTOR_ENABLED
            if CHUNK_INDEX_ENABLED:
                from .kb_layers import index_entry_chunks
                entry_obj = self._resolver.get(entry_id)
                if entry_obj is not None:
                    index_entry_chunks(self.store, self.embedder, entry_obj,
                                       with_vectors=CHUNK_VECTOR_ENABLED)
        except Exception:
            pass
        return IngestResult(entry_id=entry_id, state=state)

    def ingest_files(self, device_mac: str, files: list[dict]) -> dict:
        """
        将客户端上传的文件进行三级入库（raw / summary / index）。

        参数:
            device_mac: 发起加载的设备 MAC 标识（打进 meta.device_mac，检索时本机加权）。
            files: 文件列表，每项包含 path / title / summary / content 字段。

        返回:
            包含文件数与各入库状态（new/merge/duplicate/conflict/pinned）计数的字典。
        """
        # 初始化状态计数器
        stats = {
            "files": 0,
            "new": 0,
            "merge": 0,
            "duplicate": 0,
            "conflict": 0,
            "pinned": 0,
        }

        # 遍历每个文件
        for file_info in files:
            # 提取文件信息，设置默认值
            path = file_info.get("path", "")
            title = file_info.get("title", "") or path
            summary = file_info.get("summary", "")
            content = file_info.get("content", "")

            # 路径或内容为空则跳过该文件
            if not path or not content:
                continue

            # 增加文件计数
            stats["files"] += 1

            # 1. 入库 raw 级（原文）
            raw_entry_id = f"device::{device_mac}::raw::{path}"
            raw_meta = {
                "device_mac": device_mac,
                "level": "raw",
                "source_path": path,
            }
            raw_result = self.ingest_knowledge(
                entry_id=raw_entry_id,
                title=title,
                content=content,
                source=path,
                meta=raw_meta,
            )
            # 累加 raw 级的状态计数
            stats[raw_result.state] += 1

            # 2. 入库 summary 级（摘要），仅当 summary 非空时
            if summary:
                summary_entry_id = f"device::{device_mac}::summary::{path}"
                summary_meta = {
                    "device_mac": device_mac,
                    "level": "summary",
                    "source_path": path,
                }
                summary_result = self.ingest_knowledge(
                    entry_id=summary_entry_id,
                    title=title,
                    content=summary,
                    source=path,
                    meta=summary_meta,
                )
                # 累加 summary 级的状态计数
                stats[summary_result.state] += 1

            # 3. 入库 index 级（索引）
            index_content = f"# {title}\n路径: {path}\n设备: {device_mac}"
            index_entry_id = f"device::{device_mac}::index::{path}"
            index_meta = {
                "device_mac": device_mac,
                "level": "index",
                "source_path": path,
            }
            index_result = self.ingest_knowledge(
                entry_id=index_entry_id,
                title=title,
                content=index_content,
                source=path,
                meta=index_meta,
            )
            # 累加 index 级的状态计数
            stats[index_result.state] += 1

        # 返回统计结果
        return stats

    # ------------------------------------------------------------------ 会话回写
    def harvest_session(
        self,
        session_id: str,
        transcript: str,
        source: str = "",
        meta: dict | None = None,
    ) -> HarvestResult:
        """
        会话回写：把一段完整会话提炼成一份「核心资产」沉淀回知识库。

        定位（对齐「所有会话最终生成一份核心资产，丰富资产 + 重复利用 + 供他人使用」）：
        - 提炼引擎：本地 LLM 可用时用生成式结构化提炼（title/conclusion/decisions/actions/tags），
          不可用/超时/输出异常时自动降级 TextRank 抽取式摘要，绝不失败；
        - 回写：复用 ingest_knowledge 的「写入冲突四态」闸门，落库 entry_type=knowledge、
          entry_id=session::<session_id>，meta 标记 session_id/engine/tags/decisions/actions；
        - 复用：落库后重建索引，后续 query 即可召回这份资产（丰富资产 + 供他人复用）；
        - 埋点：route=harvest 计入 metrics 表，把「会话→资产」的 token 压缩也纳入监控大屏。

        返回 HarvestResult（entry_id / state / engine / title / reason）。
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return HarvestResult(
                entry_id="", state="empty", engine="none", title="",
                reason="会话内容为空，跳过回写",
            )

        # 1. 提炼：LLM 优先，失败/不可用降级 TextRank
        engine = "textrank"
        title, conclusion, decisions, actions, tags = self._harvest_textrank(transcript)
        if self.llm is not None:
            try:
                t0 = now_ms()
                data = self.llm.complete_json(_HARVEST_SYSTEM_PROMPT, transcript)
                self._trace_llm(
                    "会话回写", t0, _HARVEST_SYSTEM_PROMPT, transcript,
                    json.dumps(data, ensure_ascii=False) if data else "",
                )
                if data and data.get("title"):
                    engine = "llm"
                    title = str(data.get("title") or title).strip() or title
                    conclusion = str(data.get("conclusion") or conclusion).strip() or conclusion
                    decisions = self._as_list(data.get("decisions"))
                    actions = self._as_list(data.get("actions"))
                    tags = self._as_list(data.get("tags"))
            except Exception:
                # 模型未启动 / 超时 / 输出非 JSON -> 降级 TextRank，不崩溃
                pass

        # 2. 组装结构化资产正文（markdown，供人读也参与检索）
        content = self._format_harvest_content(
            session_id, engine, title, conclusion, decisions, actions,
        )

        # 3. 组装 meta（结构化字段，供程序复用；调用方 meta 优先保留）
        entry_meta = dict(meta or {})
        entry_meta.update({
            "session_id": session_id,
            "engine": engine,
            "harvested_at": time.time(),
            "tags": tags,
            "decisions": decisions,
            "actions": actions,
        })

        # 3.5 关联同会话的操作指纹（结论 ↔ 过程：结论反推不出「具体怎么干」，故显式绑定）
        related_fingerprints = self._session_fingerprints(session_id)
        if related_fingerprints:
            entry_meta["related_fingerprints"] = related_fingerprints

        # 4. 回写（复用摄入四态闸门，不覆盖 pinned、近重复跳过、矛盾转裁决）
        entry_id = f"session::{session_id}"
        src = source or f"session:{session_id}"
        ingested = self.ingest_knowledge(entry_id, title, content, src, entry_meta)

        # 5. 埋点：会话→资产 的 token 压缩效果（baseline=原始会话量，actual=资产量）
        self.store.record_metric(
            query=f"session:{session_id}",
            route="harvest",
            hits=0,
            actual_tokens=estimate_tokens(content),
            baseline_tokens=estimate_tokens(transcript),
            confidence=1.0 if engine == "llm" else 0.6,
        )

        return HarvestResult(
            entry_id=entry_id,
            state=ingested.state,
            engine=engine,
            title=title,
            reason=ingested.reason,
            related_fingerprints=related_fingerprints,
        )

    def _session_fingerprints(self, session_id: str) -> list[str]:
        """取回指定会话的全部操作指纹（去重、按出现顺序），供结论↔过程关联。"""
        fingerprints: list[str] = []
        for trace in self.store.list_tooltraces_by_session(session_id):
            fp = trace.get("fingerprint") or ""
            if fp and fp not in fingerprints:
                fingerprints.append(fp)
        return fingerprints

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        """把 LLM 可能返回的字符串/列表/None 归一化为非空字符串列表。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _harvest_textrank(
        transcript: str,
    ) -> tuple[str, str, list[str], list[str], list[str]]:
        """TextRank 兜底提炼：摘要为结论、首句为标题，无决策/行动/标签。"""
        summary = textrank_summarize(transcript, ratio=0.3, max_sentences=5)
        if not summary:
            summary = transcript[:200]
        first = split_sentences(summary)
        title = (first[0][:50] if first else "会话核心资产")
        return title, summary, [], [], []

    @staticmethod
    def _format_harvest_content(
        session_id: str,
        engine: str,
        title: str,
        conclusion: str,
        decisions: list[str],
        actions: list[str],
    ) -> str:
        """组装结构化资产正文（markdown 块，人可读、亦参与全文检索）。"""
        lines = [f"# 会话核心资产：{title}", ""]
        lines.append(f"> 来源会话：{session_id} · 提炼引擎：{engine}")
        lines.append("")
        lines.append("## 结论")
        lines.append(conclusion or "（无）")
        if decisions:
            lines.append("")
            lines.append("## 决策")
            lines.extend(f"- {d}" for d in decisions)
        if actions:
            lines.append("")
            lines.append("## 行动项")
            lines.extend(f"- [ ] {a}" for a in actions)
        return "\n".join(lines)

    # ------------------------------------------------------------------ 工具调用自进化
    def record_tooltrace(
        self,
        session_id: str,
        intent: str,
        tool_calls: list,
        source: str = "",
        meta: dict | None = None,
    ) -> ToolTraceResult:
        """
        工具调用轨迹沉淀：观察大模型本次会话实际调用了哪些工具，生成「操作指纹」，
        相同指纹累计达到阈值后自动生成可复用 skill，实现 N+1 -> N 的 token 收敛。

        定位（对齐「中间层应接管工具调用的自主总结、生成 skill/mcp」）：
        - 观察：把 tool_calls 归一化为「工具名 + 参数键签名」（值不参与，键才代表模式）；
        - 聚类：相同指纹落 tooltraces 表累计计数（重启不丢）；
        - 固化：累计达到 TOOL_TRACE_GENERATE_THRESHOLD 且尚未生成时，自动生成 SKILL.md
          写回 skills_repo 并入库代管（LLM 优先结构化生成、模板兜底），标记 auto_generated；
        - 复用：生成后下次 /skill 匹配即可命中固化步骤，大模型不再反复自创同一套操作。

        返回 ToolTraceResult（fingerprint / count / generated / skill_id / reason）。
        """
        # 1. 规范化工具调用序列
        calls = self._normalize_tool_calls(tool_calls)
        if not calls:
            return ToolTraceResult(
                fingerprint="", count=0, reason="工具调用序列为空，跳过",
            )
        fingerprint = self._fingerprint(calls)

        # 2. 落库 + 计数（观察）
        count = self.store.record_tooltrace(
            session_id, intent, fingerprint, json.dumps(calls, ensure_ascii=False),
        )

        # 3. 阈值判定 + 生成（固化）
        slug = self._fingerprint_slug(fingerprint)
        skill_id = f"skill::{slug}"
        existing_skill = self.store.get(skill_id)

        # 未达阈值 -> 仅观察，不生成
        if count < TOOL_TRACE_GENERATE_THRESHOLD:
            return ToolTraceResult(
                fingerprint=fingerprint, count=count, skill_id=skill_id if existing_skill else "",
                reason=f"未达阈值（{count}/{TOOL_TRACE_GENERATE_THRESHOLD}），继续观察",
            )

        # 已达阈值但序列过短（单工具偶发调用）-> 仅记录，不固化成流程
        if len(calls) < TOOL_TRACE_FINGERPRINT_MIN_LEN:
            return ToolTraceResult(
                fingerprint=fingerprint, count=count, skill_id=skill_id if existing_skill else "",
                reason=f"序列过短（{len(calls)} 个工具 < {TOOL_TRACE_FINGERPRINT_MIN_LEN}），不固化成流程",
            )

        # 已达阈值但已生成过 -> 复用，不重复生成
        if existing_skill is not None:
            return ToolTraceResult(
                fingerprint=fingerprint, count=count, skill_id=skill_id,
                reason="已达阈值但 skill 已存在，直接复用",
            )

        # 达标 + 未生成 -> 自动生成 skill
        try:
            name = self._generate_skill(slug, fingerprint, calls, intent)
            return ToolTraceResult(
                fingerprint=fingerprint, count=count, generated=True, skill_id=f"skill::{name}",
                reason=f"已达阈值（{count}），自动生成 skill：{name}",
            )
        except Exception as exc:  # 生成失败不崩溃，返回原因供上层知晓
            return ToolTraceResult(
                fingerprint=fingerprint, count=count, reason=f"生成失败：{exc}",
            )

    @staticmethod
    def _normalize_tool_calls(tool_calls: Any) -> list[dict]:
        """
        把上游喂入的 tool_calls 归一化为 [{tool, args_keys}] 列表。

        兼容多种字段命名：tool/name 取工具名，args/params/arguments 取参数
        （dict 取键名并排序，非 dict 视为无参数）。值不参与，键才代表操作模式。
        """
        if not isinstance(tool_calls, list):
            return []
        out: list[dict] = []
        for c in tool_calls:
            if not isinstance(c, dict):
                continue
            tool = str(c.get("tool") or c.get("name") or "").strip()
            if not tool:
                continue
            args = c.get("args") or c.get("params") or c.get("arguments") or {}
            keys = sorted(str(k) for k in args.keys()) if isinstance(args, dict) else []
            out.append({"tool": tool, "args_keys": keys})
        return out

    @staticmethod
    def _fingerprint(calls: list[dict]) -> str:
        """由工具调用序列生成「操作指纹」：tool(args_keys) 用 ' > ' 串联。"""
        parts = [
            f"{c['tool']}({','.join(c['args_keys'])})" for c in calls
        ]
        return " > ".join(parts)

    @staticmethod
    def _fingerprint_slug(fingerprint: str) -> str:
        """把操作指纹转成 skill 的短横线 slug（英文小写，加 auto- 前缀标识自动生成）。"""
        import re
        tools = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\(", fingerprint)
        if not tools:
            tools = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", fingerprint)
        slug = "-".join(t.lower().replace("_", "-") for t in tools)
        slug = slug or "auto-skill"
        return f"auto-{slug}"[:64]

    def _generate_skill(
        self,
        slug: str,
        fingerprint: str,
        calls: list[dict],
        intent: str,
    ) -> str:
        """
        生成一份可复用 skill：LLM 优先结构化生成、模板兜底，写回 skills_repo 并入库代管。

        返回最终技能名（LLM 可能重命名，故以实际落库名为准）。
        """
        tools_desc = "\n".join(
            f"- {c['tool']}({', '.join(c['args_keys']) or '无参数'})" for c in calls
        )

        name, description, steps, tags = slug, "", [], []
        engine = "template"

        # LLM 优先：结构化生成（失败/不可用自动降级模板，绝不崩溃）
        if self.llm is not None:
            try:
                user = f"意图：{intent or '未提供'}\n工具调用序列：\n{tools_desc}"
                t0 = now_ms()
                data = self.llm.complete_json(_TOOLTRACE_SYSTEM_PROMPT, user)
                self._trace_llm(
                    "技能生成", t0, _TOOLTRACE_SYSTEM_PROMPT, user,
                    json.dumps(data, ensure_ascii=False) if data else "",
                )
                if data and data.get("name"):
                    engine = "llm"
                    name = str(data.get("name") or slug).strip() or slug
                    description = str(data.get("description") or "").strip()
                    steps = self._as_list(data.get("steps"))
                    tags = self._as_list(data.get("tags"))
            except Exception:
                pass

        # 模板兜底（LLM 未产出时，用工具调用序列本身拼出可执行步骤）
        if not description:
            description = (
                f"执行「{intent or fingerprint}」的可复用操作流程："
                f"{' → '.join(c['tool'] for c in calls)}"
            )
        if not steps:
            steps = [
                f"调用 {c['tool']}"
                + (f"（关键参数：{', '.join(c['args_keys'])}）" if c["args_keys"] else "")
                for c in calls
            ]
        if not tags:
            tags = ["自动生成", "操作流程"]

        # 组装 SKILL.md 并写回 skills_repo（人可读、可再次摄入）
        skill_md = self._format_skill_md(name, description, steps)
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        # 入库代管（entry_type=skill，meta 标记 auto_generated 便于追溯）
        self.store.upsert(
            Entry(
                entry_type="skill",
                entry_id=f"skill::{name}",
                title=name,
                content=f"{description}\n\n{skill_md}",
                source=str(skill_dir / "SKILL.md"),
                meta={
                    "skill_name": name,
                    "description": description,
                    "auto_generated": True,
                    "fingerprint": fingerprint,
                    "engine": engine,
                    "intent": intent,
                    "tags": tags,
                },
            )
        )
        self._rebuild()
        return name

    @staticmethod
    def _format_skill_md(name: str, description: str, steps: list[str]) -> str:
        """组装 SKILL.md 文本（YAML frontmatter + markdown 正文，符合 agentskills.io 标准）。"""
        lines = ["---", f"name: {name}", f"description: {description}", "---", "", f"# {name}", ""]
        if steps:
            lines.append("## 步骤")
            lines.extend(f"{i}. {s}" for i, s in enumerate(steps, 1))
            lines.append("")
        lines.append("> 本技能由 MolBase 自动生成（工具调用模式自进化）。")
        return "\n".join(lines)

    @staticmethod
    def _sim(a: str, b: str) -> float:
        """两段文本的字符级相似度（difflib.SequenceMatcher.ratio，0~1）。"""
        if not a and not b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _is_duplicate(self, existing_content: str, content: str) -> bool:
        """近重复判定：SimHash 海明距离近，或字符级相似度 >= 阈值。"""
        if is_near_duplicate(existing_content, content):
            return True
        return self._sim(existing_content, content) >= CONFLICT_DUPLICATE_SIM

    def _is_conflict(self, existing: Entry, source: str) -> bool:
        """
        矛盾事实判定：同 ID 且来源不同（source 字段变了）时判为可能矛盾。

        - 任一来源缺失：保守不判冲突（走 merge，不阻断写入）；
        - 来源相同：视为同源更新，走 merge；
        - 来源不同：视为可能矛盾（如「爸爸记账 vs 妈妈记账」对同一事实键），
          进入待裁决，不自动覆盖——「宁可少存，不可错存」。
        """
        if not source or not existing.source:
            return False  # 缺来源信息，保守不判冲突
        return source != existing.source

    def ingest_knowledge_dir(self, knowledge_dir: Path | str) -> int:
        """
        批量摄入知识目录下的 markdown 文件，返回摄入条数。

        - 标题：取文件首个 `#` 一级标题；无标题时用文件名（去扩展名）。
        - entry_id：`kb::<文件名>`（唯一、可复现）。
        - source：记录文件路径，便于追溯。
        - 全部 upsert 后统一重建一次索引（避免逐条 rebuild 的低效）。
        """
        n = 0
        for md in sorted(Path(knowledge_dir).rglob("*.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue
            title = md.stem
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            self.store.upsert(
                Entry(
                    entry_type="knowledge",
                    entry_id=f"kb::{md.stem}",
                    title=title,
                    content=text,
                    source=str(md),
                )
            )
            n += 1
        if n:
            self._rebuild()
        return n

    def ingest_code_repo(self, repo_dir: Path | str) -> int:
        """摄入一个代码仓库：AST 索引全部符号 + 建调用图，返回符号数。"""
        indexer = CodeIndexer(repo_dir)
        symbols: list[Any] = []
        for py in Path(repo_dir).rglob("*.py"):
            symbols.extend(indexer.index_file(py))
        entries = indexer.to_entries(symbols)
        for e in entries:
            self.store.upsert(e)
        # 调用图边由 _rebuild 从 meta.calls 统一重建，此处无需手动 add_edge
        self._rebuild()
        return len(entries)

    def ingest_skill_dir(self, skills_dir: Path | str) -> int:
        """摄入一个技能目录：解析全部 SKILL.md 入库代管，返回技能数。"""
        registry = SkillRegistry(skills_dir)
        entries = registry.load_all()
        for e in entries:
            self.store.upsert(e)
        self._rebuild()
        return len(entries)

    def add_relation(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """手动建立两条目之间的关联（写入 meta.relations 持久化，_rebuild 时重建为边）。"""
        entry = self.store.get(from_id)
        if entry is None:
            return
        rels = entry.meta.get("relations", [])
        if to_id not in rels:
            rels.append(to_id)
            entry.meta["relations"] = rels
            self.store.upsert(entry)
        self._rebuild()

    def record_api_call(self, api: str) -> None:
        """
        记录一次接口调用（驱动「调用率」指标）。

        由服务层在各 HTTP 接口入口调用，落库 api_calls 表，供监控大屏统计
        真实调用分布（query/feedback/harvest/tooltrace/ingest/summarize/skill）。
        """
        self.store.record_api_call(api)

    def stats(self) -> dict[str, int]:
        """返回各类条目数量统计。"""
        counts = {"knowledge": 0, "code": 0, "skill": 0}
        for e in self.store.all_entries():
            if e.entry_type in counts:
                counts[e.entry_type] += 1
        counts["graph_nodes"] = self.graph.node_count()
        return counts

    def token_stats(self) -> dict:
        """
        返回会话级 token 节省统计（累计多次查询的节省比例与等效降本）。

        与单次 QueryResponse 的 token 埋点互补：后者看一次调用，前者看整个会话
        累计「越用越省」的整体效果。可随时 token_ledger.reset() 重计。
        """
        return self.token_ledger.snapshot()

    def metrics_stats(self, limit: int = 200, include_test: bool = False) -> dict:
        """
        返回持久化埋点统计（知识库本体 metrics 表，重启不丢）。

        这是「部署后实时监控」的数据源：聚合累计 token 节省 + 最近 N 条明细，
        由服务层 /metrics 接口透出，监控大屏据此可视化 token 优化趋势。

        include_test=False（默认）时排除测试数据，只反映真实使用情况。

        返回结构：
            aggregate  ：累计节省 + 全口径真实效能（拦截率/abstain率/召回/匹配度）
            recent     ：最近 limit 条明细（倒序，默认排除测试）
            ledger     ：会话级台账快照（本进程存活期内的实时累计）
            api_calls  ：各接口真实调用分布（调用率）
            tooltraces ：操作模式收敛统计
        """
        aggregate = self.store.metric_aggregate(include_test=include_test)
        recent = self.store.list_metrics(limit=limit, include_test=include_test)
        api_calls = self.store.api_calls_aggregate(include_test=include_test)
        tooltraces = self.store.tooltrace_aggregate()
        tooltraces["auto_skill_count"] = sum(
            1 for e in self.store.all_entries()
            if e.entry_type == "skill" and e.meta.get("auto_generated")
        )
        return {
            "aggregate": aggregate,
            "recent": recent,
            "ledger": self.token_ledger.snapshot(),
            "api_calls": api_calls,
            "tooltraces": tooltraces,
        }

    def memory_usage(self) -> dict:
        """
        读取当前进程运行内存 RSS（MB）。

        Linux 优先读 /proc/self/status 的 VmRSS（当前真实常驻内存）；
        Windows 用 ctypes 调 psapi.GetProcessMemoryInfo（零第三方依赖）；
        两者都不可用时返回 0.0。
        """
        rss_mb = 0.0
        # Linux：/proc/self/status 的 VmRSS（单位 KB）
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024.0
                        break
        except Exception:
            pass
        # Windows：psapi.GetProcessMemoryInfo 读 WorkingSetSize（单位字节，零依赖）
        if rss_mb <= 0.0:
            try:
                import ctypes
                import os
                from ctypes import wintypes
                class _PMC(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ]
                pmc = _PMC()
                pmc.cb = ctypes.sizeof(pmc)
                # OpenProcess 获取真实句柄（GetCurrentProcess 伪句柄在 ctypes 下会溢出）
                _handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, os.getpid())
                try:
                    ctypes.windll.psapi.GetProcessMemoryInfo(_handle, ctypes.byref(pmc), pmc.cb)
                    rss_mb = pmc.WorkingSetSize / 1024.0 / 1024.0
                finally:
                    ctypes.windll.kernel32.CloseHandle(_handle)
            except Exception:
                pass
        return {"rss_mb": round(rss_mb, 1)}

    def overview(self) -> dict:
        """
        监控大屏聚合接口：一次返回大屏所需的全部指标。

        返回结构：
            token    ：累计 token 消耗与节省（actual/baseline/saved/ratio/queries）；
            flywheel ：飞轮运转（查询→命中→反馈→进化→更准 各环节计数）；
            ingest   ：入库结果分布（new/merge/duplicate/conflict/pinned）；
            sessions ：会话链路（每会话走了哪几步、缺哪步）；
            catalog  ：本体统计（数据量/标签分布/运行内存/图规模）。
        """
        agg = self.store.metric_aggregate()
        auto_skills = sum(
            1 for e in self.store.all_entries()
            if e.entry_type == "skill" and e.meta.get("auto_generated")
        )
        abstain = self.store.abstain_aggregate()
        return {
            "evolution_health": self._evolution_health(),
            "token": {
                "actual_tokens": agg["actual_tokens"],
                "baseline_tokens": agg["baseline_tokens"],
                "saved_tokens": agg["saved_tokens"],
                "saved_ratio": agg["saved_ratio"],
                "saved_ratio_pct": agg["saved_ratio_pct"],
                "queries": agg["queries"],
                "saved_cost": round(agg["saved_tokens"] / 1e6 * COST_INPUT_PER_1M, 4),
            },
            "flywheel": {
                "queries": agg["queries"],
                "hits_total": agg["hits_total"],
                "feedback": self.store.feedback_aggregate(),
                "evolve": self.store.evolve_aggregate(),
                "auto_skills": auto_skills,
                "intercept_rate": agg["intercept_rate"],
                "abstain_rate": agg["abstain_rate"],
            },
            "abstain": abstain,
            # v7.2 三方向资产落地观测：场景锚定三表 + 弃答补数草稿
            "scene_assets": self.store.scene_stats(),
            # v9.1 四层结构建成度：树/标签/原文，用来看「知识库是否真的长出了结构」
            "layers": {
                "tree": self.store.tree_stats(),
                "tags": {"vocab": len(self.store.tag_df())},
                "chunks": self.store.chunk_stats(),
                "vectors": {"precompute": self._vector_bootstrap},
            },
            "drafts": self.store.draft_stats(),
            "ingest": self.store.ingest_aggregate(),
            "writeback": self.store.writeback_recent(10),
            "sessions": self.store.session_summary(),
            "participation": self.store.session_participation(30),
            "catalog": {
                **self.stats(),
                "tags": self.store.tag_stats(),
                "memory": self.memory_usage(),
            },
        }

    def metric_detail(self, metric: str, limit: int = 50, offset: int = 0, q: str = "", sort: str = "ts", order: str = "desc", scene: str = "", label: str = "", abstain: str = "", include_test: bool = False) -> dict:
        """获取指定指标的详细信息"""
        # 参数归一化
        if not (1 <= limit <= 500):
            limit = 50
        if offset < 0:
            offset = 0
        if order != 'asc':
            order = 'desc'

        # 定义指标映射
        metrics_map = {
            'feedback': {
                'summary_fn': lambda: self.store.feedback_aggregate(),
                'raw_fn': lambda: self.store.feedback_recent(1000),
                'filter_fields': ('session_id', 'entry_id', 'action', 'source'),
                'sort_fields': ('ts',)
            },
            'evolve': {
                'summary_fn': lambda: self.store.evolve_aggregate(),
                'raw_fn': lambda: self.store.evolve_recent(1000),
                'filter_fields': ('added_edges', 'decayed', 'archived'),
                'sort_fields': ('ts',)
            },
            'token': {
                'summary_fn': lambda: self.store.metric_aggregate(),
                'raw_fn': lambda: self.store.list_metrics(1000, include_test=include_test),
                'filter_fields': ('query', 'route', 'scene_id', 'label', 'tags', 'entities', 'abstain_reason', 'top1_title'),
                'sort_fields': ('ts', 'actual_tokens', 'baseline_tokens', 'hits', 'confidence')
            },
            'flywheel': {
                'summary_fn': lambda: {
                    'queries': self.store.metric_aggregate()['queries'],
                    'hits_total': self.store.metric_aggregate()['hits_total'],
                    'feedback': self.store.feedback_aggregate(),
                    'evolve': self.store.evolve_aggregate(),
                    'auto_skills': sum(1 for e in self.store.all_entries() if e.entry_type == 'skill' and e.meta.get('auto_generated')),
                    'intercept_rate': self.store.metric_aggregate()['intercept_rate'],
                    'abstain_rate': self.store.metric_aggregate()['abstain_rate']
                },
                'raw_fn': lambda: self.store.list_metrics(1000, include_test=include_test),
                'filter_fields': ('query', 'route', 'scene_id', 'label', 'tags', 'entities', 'abstain_reason', 'top1_title'),
                'sort_fields': ('ts', 'hits', 'confidence')
            },
            'abstain': {
                'summary_fn': lambda: {k: v for k, v in self.store.abstain_aggregate(1000).items() if k != 'recent'},
                'raw_fn': lambda: list({(r['query'], r['reason'], r['ts']): r for r in self.store.abstain_aggregate(1000)['recent'] + self.store.abstain_gaps(1000)}.values()),
                'filter_fields': ('query', 'reason', 'subreason'),
                'sort_fields': ('ts', 'confidence', 'top1_score')
            },
            'ingest': {
                'summary_fn': lambda: self.store.ingest_aggregate(),
                'raw_fn': lambda: self.store.list_ingest(500, 0),
                'filter_fields': ('entry_id', 'state', 'title'),
                'sort_fields': ('ts', 'state')
            },
            'writeback': {
                'summary_fn': lambda: {'total': len(self.store.writeback_recent(1000))},
                'raw_fn': lambda: self.store.writeback_recent(1000),
                'filter_fields': ('type_cn', 'obj', 'state', 'detail'),
                'sort_fields': ('ts',)
            },
            'participation': {
                'summary_fn': lambda: {
                    'total': len(self.store.session_participation(1000)),
                    'evasion_counts': {
                        k: sum(1 for r in self.store.session_participation(1000) if r.get('evasion') == k)
                        for k in set(r.get('evasion') for r in self.store.session_participation(1000) if r.get('evasion'))
                    }
                },
                'raw_fn': lambda: self.store.session_participation(1000),
                'filter_fields': ('session_id', 'evasion'),
                'sort_fields': ('query_count', 'abstain_count')
            },
            'sessions': {
                'summary_fn': lambda: {'total': len(self.store.session_summary(1000))},
                'raw_fn': lambda: self.store.session_summary(1000),
                'filter_fields': ('session_id', 'missing'),
                'sort_fields': ('query', 'feedback', 'harvest', 'tooltrace')
            },
            'catalog': {
                'summary_fn': lambda: {**self.stats(), 'tag_count': len(self.store.tag_stats())},
                'raw_fn': lambda: [{'entry_id': e.entry_id, 'entry_type': e.entry_type, 'title': e.title, 'source': e.source, 'heat': e.heat, 'created_at': e.created_at} for e in self.store.all_entries()],
                'filter_fields': ('entry_id', 'title', 'entry_type'),
                'sort_fields': ('created_at', 'heat', 'title')
            },
            'api': {
                'summary_fn': lambda: self.store.api_calls_aggregate(),
                'raw_fn': lambda: [dict(zip(['ts', 'api', 'is_test'], r)) for r in self.store._conn.cursor().execute("SELECT ts, api, is_test FROM api_calls ORDER BY id DESC LIMIT 1000").fetchall()],
                'filter_fields': ('api',),
                'sort_fields': ('ts', 'api')
            }
        }

        # 未知 metric 回退到 token
        if metric not in metrics_map:
            metric = 'token'

        cfg = metrics_map[metric]

        # 获取原始数据和摘要
        try:
            summary = cfg['summary_fn']()
            raw = cfg['raw_fn']()
        except Exception:
            summary = {}
            raw = []

        # 过滤
        filtered = raw
        if q:
            q_lower = q.lower()
            filtered = [
                r for r in filtered
                if any(
                    q_lower in str(r.get(f, '')).lower()
                    for f in cfg['filter_fields']
                )
            ]
            # 特殊处理 missing 字段如果是 list
            if 'missing' in cfg['filter_fields']:
                filtered = [
                    r for r in filtered
                    if q_lower in str(''.join(r.get('missing', []) if isinstance(r.get('missing'), list) else r.get('missing', ''))).lower()
                ]

        # v6 全量明细：显式维度过滤（场景/标记/弃答）
        if scene:
            filtered = [r for r in filtered if str(r.get("scene_id", "")) == scene]
        if label:
            filtered = [r for r in filtered if str(r.get("label", "")) == label]
        if abstain in ("1", "0"):
            filtered = [r for r in filtered if int(r.get("abstain", 0) or 0) == int(abstain)]

        # 排序
        if sort not in cfg['sort_fields']:
            sort = 'ts'
        filtered = sorted(
            filtered,
            key=lambda r: r.get(sort) if r.get(sort) is not None else -1,
            reverse=(order == 'desc')
        )

        # 分页
        total = len(filtered)
        items = filtered[offset:offset + limit]

        return {
            "metric": metric,
            "summary": summary,
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items
        }

    def metric_compare(self, group_by: str = "scene_id", include_test: bool = False, limit: int = 50) -> dict:
        """全量明细对比：委托 store 按任意维度分组聚合。"""
        return self.store.metric_compare(group_by=group_by, include_test=include_test, limit=limit)

    def alert_rules(self) -> dict:
        """返回告警规则定义"""
        return {
            "rules": [
                {
                    "key": "abstain_rate",
                    "name": "弃答率",
                    "op": ">",
                    "threshold": 0.30,
                    "unit": "ratio",
                    "hint": "弃答率超30%，知识覆盖或阈值需整改"
                },
                {
                    "key": "intercept_rate",
                    "name": "拦截率",
                    "op": "<",
                    "threshold": 0.50,
                    "unit": "ratio",
                    "hint": "拦截率低于50%，命中质量偏低"
                },
                {
                    "key": "saved_ratio",
                    "name": "Token节省率",
                    "op": "<",
                    "threshold": 0.50,
                    "unit": "ratio",
                    "hint": "节省率低于50%，上下文收敛效果不足"
                },
                {
                    "key": "evasion_sessions",
                    "name": "逃票会话数",
                    "op": ">",
                    "threshold": 0,
                    "unit": "count",
                    "hint": "存在疑似逃票会话，需关注参与度"
                }
            ]
        }

    def alerts(self) -> dict:
        """检查并返回触发的告警"""
        try:
            agg = self.store.metric_aggregate()
            parts = self.store.session_participation(1000)
            evasion_sessions = sum(
                1 for r in parts
                if r.get('evasion') in ('首次无返回即逃票', '未参与（有动作无查询）', '仅查1次（疑似逃票）')
            )
            values = {
                'abstain_rate': agg['abstain_rate'],
                'intercept_rate': agg['intercept_rate'],
                'saved_ratio': agg['saved_ratio'],
                'evasion_sessions': evasion_sessions
            }
            page = {
                'abstain_rate': 'abstain',
                'intercept_rate': 'flywheel',
                'saved_ratio': 'token',
                'evasion_sessions': 'participation'
            }

            rules = self.alert_rules()['rules']
            fired = []
            for rule in rules:
                key = rule['key']
                value = values.get(key, 0)
                threshold = rule['threshold']
                op = rule['op']

                triggered = False
                if op == '>':
                    triggered = value > threshold
                elif op == '<':
                    triggered = value < threshold

                if triggered:
                    level = 'warn'
                    if rule['unit'] == 'ratio' and abs(value - threshold) >= threshold * 0.5:
                        level = 'critical'

                    fired.append({
                        'key': key,
                        'name': rule['name'],
                        'level': level,
                        'value': round(value, 4),
                        'threshold': threshold,
                        'op': op,
                        'hint': rule['hint'],
                        'metric_page': page.get(key, '')
                    })

            return {
                'checked_at': time.time(),
                'total': len(rules),
                'fired': len(fired),
                'alerts': fired
            }
        except Exception:
            return {
                'checked_at': time.time(),
                'total': 0,
                'fired': 0,
                'alerts': []
            }

    def capture_snapshot(self) -> dict:
        """捕获当前指标快照"""
        try:
            agg = self.store.metric_aggregate()
            calls = self.store.api_calls_aggregate()
            st = self.stats()
            rss = self.memory_usage()['rss_mb']

            values = {
                'queries': agg['queries'],
                'saved_tokens': agg['saved_tokens'],
                'saved_ratio': agg['saved_ratio'],
                'intercept_rate': agg['intercept_rate'],
                'abstain_rate': agg['abstain_rate'],
                'hits_total': agg['hits_total'],
                'api_total': calls['total'],
                'knowledge': st['knowledge'],
                'memory_mb': rss
            }

            for k, v in values.items():
                self.store.record_snapshot(k, float(v))

            self.store.prune_snapshots(30)
            return values
        except Exception:
            return {}

    def close(self) -> None:
        """关闭底层存储连接，释放资源。"""
        self.store.close()
