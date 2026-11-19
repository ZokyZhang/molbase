# -*- coding: utf-8 -*-
"""
关联图模块：纯 Python 实现（零第三方依赖）。

职责（代工层 = 查询优化器的一部分）：
- 维护「知识 / 代码 / 技能」条目之间的关联边（如：方法 A 调用方法 B，知识 X 关联知识 Y）。
- 支持「给定一个条目，找它的关联条目」（关联方法定位 / 关联知识推荐）。

设计说明：
- 不依赖 networkx，用 dict 邻接表实现有向图，对标「零依赖、自我运行」。
- PageRank 用幂迭代手写实现，仅依赖标准库，避免 scipy/numpy 重依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import PAGERANK_DAMPING, PAGERANK_MAX_ITER, PAGERANK_TOLERANCE
from ..store import Entry


@dataclass
class RelationHit:
    """关联命中：条目 + 关联强度。"""

    entry: Entry
    weight: float


class RelationGraph:
    """条目关联图（有向，邻接表实现）。"""

    def __init__(self) -> None:
        """初始化空有向图：_adj[from_id][to_id] = weight。"""
        self._adj: dict[str, dict[str, float]] = {}
        self._nodes: set[str] = set()

    def add_edge(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """新增/累加一条关联边。"""
        if from_id == to_id:
            return  # 忽略自环
        self._nodes.add(from_id)
        self._nodes.add(to_id)
        out = self._adj.setdefault(from_id, {})
        out[to_id] = out.get(to_id, 0.0) + weight

    def add_relation(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """别名：新增关联边。"""
        self.add_edge(from_id, to_id, weight)

    def neighbors(
        self,
        entry_id: str,
        resolver: Optional[dict[str, Entry]] = None,
        top_n: int = 20,
    ) -> list[RelationHit]:
        """
        返回与 entry_id 直接关联的邻居条目，按关联强度降序。

        resolver：entry_id -> Entry 的映射（由调用方传入，避免图内嵌对象）。
        """
        out = self._adj.get(entry_id)
        if not out:
            return []
        ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
        hits: list[RelationHit] = []
        for nid, weight in ranked[:top_n]:
            entry = resolver.get(nid) if resolver else None
            if entry is not None:
                hits.append(RelationHit(entry=entry, weight=weight))
        return hits

    def pagerank_related(
        self,
        entry_id: str,
        resolver: Optional[dict[str, Entry]] = None,
        top_n: int = 10,
    ) -> list[RelationHit]:
        """
        用 PageRank 计算「全局关联度」，返回与 entry_id 在图中关系最紧密的条目。

        适用于「关联方法定位」：给定一个方法，找调用链上最重要的相关方法。

        算法：幂迭代 r = (1-d)/N + d * A^T r，仅用标准库 list 实现矩阵乘法。
        """
        if entry_id not in self._nodes or not self._nodes:
            return []

        nodes = sorted(self._nodes)
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        d = PAGERANK_DAMPING

        # 构建转置邻接矩阵 A[j][i] = 从 i 到 j 的转移概率（列归一化）
        A = [[0.0] * n for _ in range(n)]
        for from_id, outs in self._adj.items():
            i = idx[from_id]
            total = sum(outs.values())
            if total <= 0:
                continue
            for to_id, w in outs.items():
                j = idx[to_id]
                A[j][i] = w / total

        # 处理悬挂节点（出度为 0）：均匀跳转到所有节点
        for i in range(n):
            col_sum = sum(A[j][i] for j in range(n))
            if col_sum == 0.0:
                for j in range(n):
                    A[j][i] = 1.0 / n

        # 幂迭代求解 PageRank
        r = [1.0 / n] * n
        for _ in range(PAGERANK_MAX_ITER):
            r_new = [0.0] * n
            for j in range(n):
                s = sum(A[j][i] * r[i] for i in range(n))
                r_new[j] = (1.0 - d) / n + d * s
            diff = sum(abs(r_new[i] - r[i]) for i in range(n))
            r = r_new
            if diff < PAGERANK_TOLERANCE:
                break

        # 排除自身，按 PageRank 值降序取 top_n
        target_idx = idx[entry_id]
        related = [
            (nodes[i], r[i])
            for i in range(n)
            if i != target_idx and resolver and nodes[i] in resolver
        ]
        related.sort(key=lambda kv: kv[1], reverse=True)

        hits: list[RelationHit] = []
        for nid, score in related[:top_n]:
            entry = resolver[nid]
            hits.append(RelationHit(entry=entry, weight=score))
        return hits

    def personalized_pagerank(
        self,
        seed_ids: list[str],
        resolver: Optional[dict[str, Entry]] = None,
        top_n: int = 10,
    ) -> list[RelationHit]:
        """
        Personalized PageRank（P2）：以给定种子节点为 teleport 目标，做带偏随机游走，
        返回与种子集合关系最紧密的节点（排除种子自身）。

        适用于「以查询命中的条目为种子，找全局关联知识/方法」。
        算法：r = (1-d)*p + d*A^T r，其中 p 只在种子节点上有质量。
        """
        if not self._nodes or not seed_ids:
            return []

        nodes = sorted(self._nodes)
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        d = PAGERANK_DAMPING

        # 构建转移矩阵 A[j][i] = 从 i 到 j 的转移概率（列归一化）
        A = [[0.0] * n for _ in range(n)]
        for from_id, outs in self._adj.items():
            i = idx[from_id]
            total = sum(outs.values())
            if total <= 0:
                continue
            for to_id, w in outs.items():
                A[idx[to_id]][i] = w / total
        # 处理悬挂节点（出度为 0）
        for i in range(n):
            if sum(A[j][i] for j in range(n)) == 0.0:
                for j in range(n):
                    A[j][i] = 1.0 / n

        # teleport 向量 p：只在种子节点上有质量（均分）
        valid_seeds = [s for s in seed_ids if s in idx]
        if not valid_seeds:
            return []
        p = [0.0] * n
        for s in valid_seeds:
            p[idx[s]] += 1.0
        total_p = sum(p)
        p = [x / total_p for x in p]

        # 幂迭代求解 Personalized PageRank
        r = p[:]
        for _ in range(PAGERANK_MAX_ITER):
            r_new = [(1.0 - d) * p[i] for i in range(n)]
            for j in range(n):
                acc = 0.0
                for i in range(n):
                    acc += A[j][i] * r[i]
                r_new[j] += d * acc
            if sum(abs(r_new[i] - r[i]) for i in range(n)) < PAGERANK_TOLERANCE:
                r = r_new
                break
            r = r_new

        # 排除种子自身，按分数降序取 top_n
        seed_set = set(valid_seeds)
        scored = [(nodes[i], r[i]) for i in range(n) if nodes[i] not in seed_set]
        scored.sort(key=lambda kv: kv[1], reverse=True)

        hits: list[RelationHit] = []
        for nid, score in scored[:top_n]:
            if resolver is not None:
                entry = resolver.get(nid)
                if entry is None:
                    continue
            else:
                # 无 resolver 时用占位条目（只含 id），便于测试与调用
                entry = Entry(entry_type="", entry_id=nid, title=nid, content="")
            hits.append(RelationHit(entry=entry, weight=score))
        return hits

    def node_count(self) -> int:
        """返回图中节点数。"""
        return len(self._nodes)
