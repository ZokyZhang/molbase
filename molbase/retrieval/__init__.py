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

import math
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
        content_limit: int = 0,
    ) -> list[RankedResult]:
        """语义精排：BM25 召回池做向量相似度打分，融合热度与新鲜度。
        content_limit 用于与预计算向量口径对齐（传 0 时取 SEMANTIC_SIM_CONTENT_CHARS）。
        """
        if not hits:
            return []

        from ..config import SEMANTIC_SIM_CONTENT_CHARS, SEMANTIC_RERANK_LIMIT
        from ..vectorize import semantic_text
        limit = content_limit if content_limit > 0 else SEMANTIC_SIM_CONTENT_CHARS

        heats = [h.entry.heat for h in hits]
        freshes = [self.freshness(h.entry.last_access) for h in hits]

        sims: list[float]
        if embedder is not None and query:
            try:
                rerank_hits = hits[:SEMANTIC_RERANK_LIMIT]
                texts = [
                    semantic_text(h.entry.title, h.entry.content or "", h.entry.summary or "", limit)
                    for h in rerank_hits
                ]
                q_vec = embedder.embed(query)
                batch_fn = getattr(embedder, "embed_batch", None)
                if batch_fn is not None:
                    # 批量后端：一次前向拿全部条目向量，降低推理开销
                    entry_vecs = batch_fn(texts)
                else:
                    # 无批量接口的后端（tfidf 等）必须逐条 embed，不能把文本当向量参与余弦
                    entry_vecs = [embedder.embed(t) for t in texts]
                sims = [cosine_similarity(q_vec, v) for v in entry_vecs]
                # 超出精排窗口的条目语义分记 0（不用 BM25 冒充语义分）
                if len(sims) < len(hits):
                    sims = list(sims) + [0.0] * (len(hits) - len(sims))
                # 归一化必须在 try 内完成，避免上层拿到非法值时在 try 外崩掉整个查询
                sim_n = [max(0.0, min(1.0, float(s))) for s in sims]
            except Exception:
                sim_n = [max(0.0, min(1.0, h.score)) for h in hits]
        else:
            sim_n = [max(0.0, min(1.0, h.score)) for h in hits]
        heat_n = _normalize(heats)
        fresh_n = _normalize(freshes)

        results: list[RankedResult] = []
        for hit, s, h, f in zip(hits, sim_n, heat_n, fresh_n):
            score = self.w_sim * s + self.w_heat * h + self.w_fresh * f
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

    def rerank_with_vectors(self, hits, q_vec, vectors, device_mac: str = "") -> list:
        """
        使用预计算的条目向量进行重排打分（查询侧只编码 1 条 query）。

        vectors 形如 {entry_id: [float]}；向量缺失或维度不符的条目语义分记 0.0。
        精排窗口与 rerank() 保持一致（只对前 SEMANTIC_RERANK_LIMIT 条算语义分），
        否则两条路径在同一批候选上会产出不同排序。
        点积等价余弦的前提是双方都已 L2 归一化；此处显式做一次归一化兜底，
        防止某个后端返回未归一化向量时相似度被放大到 >1 后被 clamp 成常数。
        """
        if not hits:
            return []

        from ..config import SEMANTIC_RERANK_LIMIT

        heats = [h.entry.heat for h in hits]
        freshes = [self.freshness(h.entry.last_access) for h in hits]
        heat_n = _normalize(heats)
        fresh_n = _normalize(freshes)

        q_norm = math.sqrt(sum(a * a for a in q_vec)) or 1.0

        results = []
        for i, hit in enumerate(hits):
            entry = hit.entry
            entry_id = entry.entry_id

            s = 0.0
            v = vectors.get(entry_id)
            if i < SEMANTIC_RERANK_LIMIT and v is not None and len(v) == len(q_vec):
                v_norm = math.sqrt(sum(b * b for b in v)) or 1.0
                s = sum(a * b for a, b in zip(q_vec, v)) / (q_norm * v_norm)
                s = max(0.0, min(1.0, s))

            score = self.w_sim * s + self.w_heat * heat_n[i] + self.w_fresh * fresh_n[i]

            if device_mac and entry.meta.get("device_mac") == device_mac:
                score += DEVICE_BOOST

            results.append(RankedResult(
                entry=entry,
                score=score,
                heat_norm=heat_n[i],
                fresh_norm=fresh_n[i],
                sim_norm=s
            ))

        results.sort(key=lambda x: x.score, reverse=True)
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
        # BM25 召回上限取两者较大值：调用方需要更大候选池时（如精准知识点深召回）
        # 不能被固定 RECALL_TOP_N 卡住，否则语义最强但 BM25 排名靠后的条目永远召不回。
        for h in self.bm25.search(query, max(RECALL_TOP_N, top_n)):
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
