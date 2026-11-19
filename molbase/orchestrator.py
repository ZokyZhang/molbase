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
    CONFLICT_DUPLICATE_SIM,
    EMBEDDING_BACKEND,
    EMBEDDING_HASH_DIM,
    EMBEDDING_MODEL_NAME,
    EVOLVE_INTERVAL,
    FEEDBACK_WEIGHT_ADOPT,
    FEEDBACK_WEIGHT_CITE,
    FEEDBACK_WEIGHT_HIT,
    FEEDBACK_WEIGHT_REJECT,
    FINAL_TOP_K,
    LLM_ENABLED,
    LLM_ENDPOINT,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REWRITE,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    QUARANTINE_REJECT_THRESHOLD,
    SKILLS_DIR,
    TOKEN_TRACE,
    TOOL_TRACE_FINGERPRINT_MIN_LEN,
    TOOL_TRACE_GENERATE_THRESHOLD,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_ENDPOINT,
    WEB_SEARCH_TIMEOUT,
    WEB_SEARCH_TOP_N,
    WEB_SEARCH_TRIGGER_SCORE,
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
from .summarize import split_blocks, split_sentences, textrank_summarize
from .store import Entry, Store
from .tokens import TokenLedger, estimate_tokens
from .websearch import SearchProvider, create_search_provider

# 意图路由的 LLM 系统提示词（V2：语义分类；V1：规则兜底）
_ROUTE_SYSTEM_PROMPT = (
    "你是MolBase的意图路由器。根据用户查询判断意图，只输出 JSON。"
    "route 取值：knowledge（知识/概念/做法）、code（代码/方法/函数/源码）、"
    "skill（技能/工具/流程步骤）。格式：{\"route\":\"knowledge\",\"entities\":[\"关键词\"]}"
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
        self._rebuild()

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
        规则版意图路由（零模型兜底）：
        - 含「方法/函数/代码/源码/调用」关键词 -> code
        - 含「技能/skill/流程/步骤」关键词 -> skill
        - 其余 -> knowledge
        """
        q = query.lower()
        code_kw = ("方法", "函数", "代码", "源码", "调用", "method", "function", "code", "def ")
        skill_kw = ("技能", "skill", "流程", "步骤", "how to")
        if any(k in q for k in code_kw):
            return "code"
        if any(k in q for k in skill_kw):
            return "skill"
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
        """
        if self.llm is None or not LLM_REWRITE:
            return query
        try:
            t0 = now_ms()
            rw = self.llm.chat(_REWRITE_SYSTEM_PROMPT, query)
            if rw and rw.strip():
                rw = rw.strip()
                # 性能埋点：查询改写（模型环节）
                self._trace_llm("查询改写", t0, _REWRITE_SYSTEM_PROMPT, query, rw)
                return rw
        except Exception:
            pass  # 模型不可用 -> 降级用原查询
        return query

    def _filter_by_route(
        self, results: list[Any], route: str
    ) -> list[Any]:
        """
        按路由类型过滤结果：code/skill 精确匹配；knowledge 不限制（允许混合）。
        """
        if route == "knowledge":
            return results
        return [r for r in results if r.entry.entry_type == route]

    def _filter_by_tags(self, results: list[Any], tags: list[str] | None) -> list[Any]:
        """
        按 meta.tags 过滤（AND 语义：仅保留同时含全部指定标签的条目）。
        tags 为空或全是空串时不过滤。
        """
        if not tags:
            return results
        want = [t.strip() for t in tags if t and str(t).strip()]
        if not want:
            return results
        out = []
        for r in results:
            have = r.entry.meta.get("tags") or []
            if all(t in have for t in want):
                out.append(r)
        return out

    # ------------------------------------------------------------------ 对外查询
    def query(
        self,
        text: str,
        top_k: int = FINAL_TOP_K,
        tags: list[str] | None = None,
    ) -> QueryResponse:
        """
        核心查询接口：意图路由 -> 召回 -> 重排 -> 最小上下文输出。

        tags：可选，按 meta.tags 过滤（AND 语义），用于按「运维/架构/敏感」等分类精准检索。
        """
        route = self._route(text)

        # 查询改写：LLM 可用时把口语化查询改写成检索友好关键词（提升召回）
        search_text = self._rewrite(text)

        # 召回（用改写后的查询）
        t_recall = now_ms()
        hits = self.retriever.retrieve(
            search_text, top_n=max(top_k * 4, 20), resolver=self._resolver
        )

        # 路由过滤（knowledge 不过滤）
        hits = self._filter_by_route(hits, route)

        # 标签过滤（按 meta.tags，AND 语义，可选）
        hits = self._filter_by_tags(hits, tags)

        # 保留 rerank 前的原始 BM25 得分，供「是否触发网络兜底」判断真实匹配强度
        raw_hits = hits

        # 重排（传入 embedder，模型失效时内部自动降级为 BM25）
        t_rerank = now_ms()
        ranked = self.reranker.rerank(hits, embedder=self.embedder, query=search_text)

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

        # 路由类型优先级：knowledge 路由下，目标类型优先，避免无关技能/代码喧宾夺主
        if route == "knowledge":
            priority = {"knowledge": 0, "code": 1, "skill": 2}
            ranked.sort(key=lambda r: (priority.get(r.entry.entry_type, 9), -r.score))

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

        # token 埋点：粗估本次返回内容与「全表扫描」基线的 token 数
        all_text = "".join(r.title + " " + r.content for r in results)
        all_text += "".join(r.title + " " + r.content for r in web_results)
        token_estimate = estimate_tokens(all_text) if TOKEN_TRACE else 0
        baseline_tokens = self._baseline_tokens if TOKEN_TRACE else 0

        # 质量底线（输出闸门）：综合置信度 + abstain 判定
        confidence = self._compute_confidence(ranked, search_text)
        abstain = confidence < ABSTAIN_CONFIDENCE

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
        )
        return resp

    def _semantic_sim(self, query: str, entry: Entry) -> float:
        """
        计算 query 与某条目的原始语义相似度（embedder 余弦，未归一化）。

        用「未归一化的余弦值」而非重排里的 min-max 归一化分——归一化会把
        弱命中也拉到接近 1，掩盖真实匹配强度。embedder 不可用/异常时返回 0
        （保守，倾向 abstain，宁可不答也不错答）。
        """
        if self.embedder is None:
            return 0.0
        try:
            q_vec = self.embedder.embed(query)
            e_vec = self.embedder.embed(entry.title + " " + entry.content)
            return cosine_similarity(q_vec, e_vec)
        except Exception:
            return 0.0

    def _compute_confidence(self, ranked: list[Any], query: str) -> float:
        """
        计算一次查询的综合置信度（0~1），供 abstain 闸门判断（质量底线输出侧）。

        核心信号是「未归一化的原始语义余弦」（hash/tfidf embedding 下，真命中
        通常 > 0.5，假命中接近 0），而非 BM25 绝对分或 min-max 归一化分——
        后两者在大语料下都会整体抬升、无法区分真伪。

        两因子（零模型、确定性、可解释）：
        - 匹配强度 strength：query 与 top1 的原始语义余弦（clip 0~1，权重 0.7）；
        - 唯一性 margin   ：top1 与 top2 语义余弦之差（权重 0.3）。

        无命中时置信度为 0（必然 abstain）。
        """
        if not ranked:
            return 0.0
        cos_top1 = self._semantic_sim(query, ranked[0].entry)
        strength = max(0.0, min(1.0, cos_top1))
        margin = 0.0
        if len(ranked) >= 2:
            cos_top2 = self._semantic_sim(query, ranked[1].entry)
            margin = max(0.0, min(1.0, cos_top1 - cos_top2))
        conf = 0.7 * strength + 0.3 * margin
        return round(min(1.0, conf), 4)

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
        self._rebuild()  # 学习后重建关联图，让新发现的关联边生效
        return stats

    # ------------------------------------------------------------------ 上下文压缩
    def summarize(self, text: str, max_chars: int = 200) -> str:
        """
        上下文压缩（代工层降 token，三级降级）：
        - 本地大模型可用：生成式摘要（最精准，保留关键事实与步骤）；
        - 否则 TextRank 抽取式摘要（零模型，保留高权重句子）；
        - 再退化为按 max_chars 截断原文。
        """
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

    def chunk(self, text: str, max_chars: int = 500) -> list[str]:
        """智能切块：按段落/句子边界把长文本切成上下文块（零模型）。"""
        return split_blocks(text, max_chars=max_chars)

    # ------------------------------------------------------------------ 反馈闭环
    def feedback(self, entry_id: str, action: str = "hit") -> None:
        """
        反馈回写：action 取值 hit / adopt / cite / reject，累加对应热度权重。

        - hit / adopt / cite 为正反馈（热度上升）；
        - reject 为负反馈（热度下降，自然沉底）。
        """
        weight = {
            "hit": FEEDBACK_WEIGHT_HIT,
            "adopt": FEEDBACK_WEIGHT_ADOPT,
            "cite": FEEDBACK_WEIGHT_CITE,
            "reject": FEEDBACK_WEIGHT_REJECT,
        }.get(action, FEEDBACK_WEIGHT_HIT)
        self.store.add_heat(entry_id, weight)
        # 质量底线（进化侧）：reject 累计达到阈值 -> 自动隔离（不参与召回）
        if action == "reject":
            self._maybe_quarantine(entry_id)
        # 反馈后立即重建索引（热度变化影响后续重排）
        self._rebuild()

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
            return IngestResult(
                entry_id=entry_id, state="conflict", against_id=entry_id,
                reason=f"与已有条目(来源={existing.source or '未知'})来源不同，已记录待裁决",
            )

        # 3. duplicate：同 ID 同源（或来源缺失）且内容近重复，无需重复写入
        if existing and self._is_duplicate(existing.content, content):
            return IngestResult(
                entry_id=entry_id, state="duplicate", against_id=entry_id,
                reason="与已有内容近重复，跳过写入",
            )

        # 4. merge/new：正常写入
        state = "merge" if existing else "new"
        self.store.upsert(
            Entry(
                entry_type="knowledge",
                entry_id=entry_id,
                title=title,
                content=content,
                source=source,
                meta=meta,
            )
        )
        self._rebuild()
        return IngestResult(entry_id=entry_id, state=state)

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

    def metrics_stats(self, limit: int = 200) -> dict:
        """
        返回持久化埋点统计（知识库本体 metrics 表，重启不丢）。

        这是「部署后实时监控」的数据源：聚合累计 token 节省 + 最近 N 条明细，
        由服务层 /metrics 接口透出，监控大屏据此可视化 token 优化趋势。

        返回结构：
            aggregate  ：累计节省（queries/actual/baseline/saved/ratio/pct/conf 均值）
            recent     ：最近 limit 条明细（倒序）
            ledger     ：会话级台账快照（本进程存活期内的实时累计）
        """
        aggregate = self.store.metric_aggregate()
        recent = self.store.list_metrics(limit=limit)
        tooltraces = self.store.tooltrace_aggregate()
        tooltraces["auto_skill_count"] = sum(
            1 for e in self.store.all_entries()
            if e.entry_type == "skill" and e.meta.get("auto_generated")
        )
        return {
            "aggregate": aggregate,
            "recent": recent,
            "ledger": self.token_ledger.snapshot(),
            "tooltraces": tooltraces,
        }

    def close(self) -> None:
        """关闭底层存储连接，释放资源。"""
        self.store.close()
