# -*- coding: utf-8 -*-
"""
检索与重排模块。

职责（代工层 = 查询优化器核心）：
- 混合召回：BM25（稀疏检索）+ 关联图（关联方法/知识）。
- 重排：综合「热度 + 新鲜度 + 语义相关度」三信号打分。

设计说明（对齐「对标 MySQL、杜绝模型版本失效」）：
- 语义相关度有两条路径：
    1. 基线（零模型）：用 BM25 得分归一化近似 —— 永远可用，不依赖任何模型。
    2. 增强（可插拔）：若传入 embedding 后端，则用向量余弦相似度加权 ——
       模型失效时由上层降级，此处绝不抛错。
- 热度、新鲜度信号来自存储底座的 heat / last_access 字段。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..config import (
    DEVICE_BOOST,
    FINAL_TOP_K,
    FRESHNESS_HALF_LIFE_DAYS,
    RERANK_WEIGHT_FRESH,
    RERANK_WEIGHT_HEAT,
    RERANK_WEIGHT_SIM,
)
from ..embedding import EmbeddingProvider, cosine_similarity
from ..index import BM25Index, ScoredHit
from ..store import Entry


@dataclass
class RankedResult:
    """重排后的最终结果。"""

    entry: Entry
    score: float           # 综合得分（0~1）
    heat_norm: float       # 归一化热度
    fresh_norm: float      # 归一化新鲜度
    sim_norm: float        # 归一化语义相关度


def _normalize(values: list[float]) -> list[float]:
    """把一组值归一化到 0~1（min-max，全相等时退化为 1.0）。"""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class Reranker:
    """热度 + 新鲜度 + 语义 三信号重排器。"""

    def __init__(
        self,
        w_heat: float = RERANK_WEIGHT_HEAT,
        w_fresh: float = RERANK_WEIGHT_FRESH,
        w_sim: float = RERANK_WEIGHT_SIM,
    ) -> None:
        """初始化重排权重。"""
        self.w_heat = w_heat
        self.w_fresh = w_fresh
        self.w_sim = w_sim

    def freshness(self, last_access: float) -> float:
        """新鲜度：按半衰期指数衰减（越近越接近 1）。"""
        age_days = (time.time() - last_access) / 86400.0
        return 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS)

    def rerank(
        self,
        hits: list[ScoredHit],
        embedder: Optional[EmbeddingProvider] = None,
        query: str = "",
        device_mac: Optional[str] = None,
    ) -> list[RankedResult]:
        """
        对召回命中做三信号综合重排，返回综合得分降序的 FINAL_TOP_K 条。

        语义分策略：
        - 无 embedder（或 embedder 异常）：退化为 BM25 得分归一化（零模型基线）。
        - 有 embedder：用 query 与各命中内容的余弦相似度作为语义分。
          计算全程 try/except，失败自动回退 BM25，绝不抛错。
        """
        if not hits:
            return []

        heats = [h.entry.heat for h in hits]
        freshes = [self.freshness(h.entry.last_access) for h in hits]

        # ---- 语义相关度：优先向量相似度，失败/缺失回退 BM25 ----
        sims: list[float]
        if embedder is not None and query:
            try:
                q_vec = embedder.embed(query)
                from ..config import SEMANTIC_SIM_CONTENT_CHARS
                sims = [
                    cosine_similarity(q_vec, embedder.embed(h.entry.title + " " + (h.entry.content or "")[:SEMANTIC_SIM_CONTENT_CHARS]))
                    for h in hits
                ]
            except Exception:
                sims = [h.score for h in hits]  # 模型失效 -> 回退 BM25
        else:
            sims = [h.score for h in hits]

        # 语义分用绝对余弦（bge-m3 有量纲意义），不做相对归一化，
        # 避免 0.70 与 0.66 的真实差距被抹平后被热度分反超。
        sim_n = [max(0.0, min(1.0, s)) for s in sims]
        heat_n = _normalize(heats)
        fresh_n = _normalize(freshes)

        results: list[RankedResult] = []
        for hit, s, h, f in zip(hits, sim_n, heat_n, fresh_n):
            # 计算基础综合得分
            score = self.w_sim * s + self.w_heat * h + self.w_fresh * f
            # 若 device_mac 非空且条目匹配，则增加设备加权分
            if device_mac and hit.entry.meta.get("device_mac") == device_mac:
                score += DEVICE_BOOST
            results.append(
                RankedResult(
                    entry=hit.entry,
                    score=score,
                    heat_norm=h,
                    fresh_norm=f,
                    sim_norm=s,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:FINAL_TOP_K]


class Retriever:
    """混合召回器：组合 BM25 与关联图结果。"""

    def __init__(self, bm25: BM25Index, graph: "RelationGraph") -> None:
        """初始化。"""
        self.bm25 = bm25
        self.graph = graph

    def retrieve(
        self,
        query: str,
        top_n: int,
        resolver: dict[str, Entry] | None = None,
        seed_ids: list[str] | None = None,
    ) -> list[ScoredHit]:
        """
        混合召回：
        1. BM25 全文召回；
        2. 若给定 seed_ids（已知关联的条目），追加其图邻居。
        去重后返回 top_n。
        """
        from ..config import RECALL_TOP_N

        hits: dict[str, ScoredHit] = {}
        # 1. BM25 召回
        for h in self.bm25.search(query, RECALL_TOP_N):
            hits[h.entry.entry_id] = h

        # 2. 图关联召回（seed 条目的一跳邻居）
        for sid in seed_ids or []:
            for rel in self.graph.neighbors(sid, resolver=resolver, top_n=10):
                if rel.entry.entry_id not in hits:
                    # 关联命中的语义分暂以边权近似（1.0 封顶）
                    hits[rel.entry.entry_id] = ScoredHit(
                        entry=rel.entry, score=min(rel.weight, 1.0)
                    )

        ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
        return ranked[:top_n]
