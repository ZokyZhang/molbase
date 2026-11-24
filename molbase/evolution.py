# -*- coding: utf-8 -*-
"""
自我进化引擎：让知识库从「使用痕迹」中自动学习，无需人工干预。

闭环：使用 → 记录 → 学习 → 反哺检索

三个零模型、全自动、确定性的进化机制：
1. 自动命中热度：query 命中的条目自动累加热度（用得多自然热）。
2. 共现自动建边：同一查询经常一起命中的条目，自动建立关联边（自我发现关联）。
3. 冷知识自动衰减：长期未被访问的条目热度自然衰减（自然淘汰）。

设计说明（对齐「对标 MySQL、自我运行」）：
- 全部纯 Python 标准库实现，零模型、零幻觉。
- 进化信号（共现计数）持久化到 SQLite 的 cooccur 表，重启不丢。
- 这是「越用越顺手」的自动闭环：使用本身就在优化检索，无需人工整理。
"""

from __future__ import annotations

from .config import (
    COLD_DECAY_DAYS,
    COLD_DECAY_FACTOR,
    COOCCUR_PMI_MIN,
    COOCCUR_SAME_TYPE_ONLY,
    COOCCUR_THRESHOLD,
)
from .store import Store


class EvolutionEngine:
    """自我进化引擎：从使用痕迹中自动学习。"""

    def __init__(self, store: Store) -> None:
        """初始化，绑定存储底座。"""
        self.store = store

    def record_query(self, hit_ids: list[str]) -> None:
        """
        记录一次查询的命中集合，累计两两共现计数。

        供「共现自动建边」学习：频繁一起命中的条目会被自动关联，
        从而让系统自己发现「知识 A 和知识 B 总是一起用」。
        """
        ids = list(dict.fromkeys(hit_ids))  # 去重保序
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.store.incr_cooccur(ids[i], ids[j])

    def learn(self) -> dict[str, int]:
        """
        触发一次学习步进：共现建边 + 冷知识衰减，返回学习统计。

        1. 共现计数超阈值的条目对 -> 自动写双向关联（持久化 meta.relations）。
        2. 长期未访问的冷知识 -> 热度按因子衰减（自然下沉）。
        """
        stats: dict[str, int] = {"edges": 0, "cold_decayed": 0}

        # 1. 共现自动建边（类型感知 + PMI 去高频）
        # 只看 cnt 阈值会把「到处都和别人一起出现」的枢纽条目连成一张稠密网：
        # 这类边 cnt 很高但不是专属关联。用 PMI 把「纯高频」压掉，只留真正专属的强关联。
        # PMI 取不到（表为空/数据不足）时退回纯 cnt 阈值，不让治理逻辑反而把建边停掉。
        pmi_map = {}
        try:
            pmi_map = {(a, b): w for a, b, w in self.store.cooccur_pmi_pairs(COOCCUR_THRESHOLD)}
        except Exception:
            pmi_map = {}
        pruned = 0
        for a, b, _cnt in self.store.cooccur_pairs(COOCCUR_THRESHOLD):
            if not self._same_type(a, b):
                continue
            # pmi_map 缺失该边（异常或口径不一致）时按放行处理，宁可多建一条边
            if pmi_map and pmi_map.get((a, b), pmi_map.get((b, a))) is None:
                pass
            elif pmi_map and (pmi_map.get((a, b), pmi_map.get((b, a), 0.0)) < COOCCUR_PMI_MIN):
                pruned += 1
                continue
            self._link(a, b)
            stats["edges"] += 1
        stats["pmi_pruned"] = pruned

        # 2. 冷知识自动衰减
        stats["cold_decayed"] = self.store.decay_cold(COLD_DECAY_DAYS, COLD_DECAY_FACTOR)
        return stats

    def _same_type(self, a: str, b: str) -> bool:
        """
        类型感知过滤：仅当两个条目同类型（knowledge/code/skill）时才允许自动建边。

        - 开启（COOCCUR_SAME_TYPE_ONLY=True）：知识只关联知识、代码只关联代码、
          技能只关联技能，避免「知识条目与技能条目」这类跨类型噪音关联。
        - 关闭：允许跨类型关联（更宽松，但可能引入噪音）。
        """
        if not COOCCUR_SAME_TYPE_ONLY:
            return True
        ea = self.store.get(a)
        eb = self.store.get(b)
        if ea is None or eb is None:
            return False
        return ea.entry_type == eb.entry_type

    def _link(self, a: str, b: str) -> None:
        """
        把一对共现条目写成双向关联，持久化到 meta.relations。
        orchestrator._rebuild 时会据此重建关联图边。
        """
        for src, dst in ((a, b), (b, a)):
            entry = self.store.get(src)
            if entry is None:
                continue
            rels = list(entry.meta.get("relations", []))
            if dst not in rels:
                rels.append(dst)
                entry.meta["relations"] = rels
                self.store.upsert(entry)
