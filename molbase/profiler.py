# -*- coding: utf-8 -*-
"""
性能剖析器（Profiler）：像性能测试一样，逐环节记录一次调用的 token 与成本。

定位（对齐用户诉求「展示优化效果」）：
- 把一次端到端调用拆成若干「环节（stage）」，记录每个环节：
    调用次数（calls）、其中 LLM 调用次数（llm_calls）、
    输入 token、输出 token、耗时（duration_ms）。
- 区分两类环节：
    模型环节（消耗 token）：意图路由 / 查询改写 / 摘要压缩 / 最终生成；
    代工环节（不消耗模型 token，替代了模型劳动）：混合召回 / 重排 / 网络搜索。
- 零第三方依赖：纯标准库 dataclass + time。

设计原则：Profiler 是「可选增强」，不注入时 Orchestrator 完全照常运行，
不影响既有测试套件与零模型内核。
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StageStat:
    """单个环节的累计统计。"""

    name: str = ""          # 环节名（意图路由 / 查询改写 / 召回 / 重排 / 摘要 / 网络搜索 / 最终生成）
    calls: int = 0          # 该环节总调用次数
    llm_calls: int = 0      # 其中消耗模型（LLM）的调用次数
    input_tokens: int = 0   # 输入 token 累计（粗估）
    output_tokens: int = 0  # 输出 token 累计（粗估）
    duration_ms: float = 0.0  # 耗时累计（毫秒）
    note: str = ""          # 备注（如 embedding 后端名、降级说明）

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return self.input_tokens + self.output_tokens


class Profiler:
    """
    环节级性能剖析器。

    用法：贯穿一次调用，对各环节调用 record()；最后用 stages()/totals()
    汇总输出。reset() 清空，便于逐次查询独立统计。
    """

    def __init__(self) -> None:
        """初始化空的环节统计表。"""
        self._stages: dict[str, StageStat] = {}

    def record(
        self,
        stage: str,
        *,
        llm_call: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
        note: str = "",
    ) -> StageStat:
        """
        记录一次环节调用。

        stage         ：环节名（自定义字符串，报告按此聚合展示）。
        llm_call      ：本次调用是否消耗模型（True 计入模型调用次数）。
        input_tokens  ：本次输入 token 增量。
        output_tokens ：本次输出 token 增量。
        duration_ms   ：本次耗时增量（毫秒）。
        note          ：备注，仅保留最后一次写入的值。
        """
        s = self._stages.setdefault(stage, StageStat(name=stage))
        s.calls += 1
        if llm_call:
            s.llm_calls += 1
        s.input_tokens += input_tokens
        s.output_tokens += output_tokens
        s.duration_ms += duration_ms
        if note:
            s.note = note
        return s

    def get(self, stage: str) -> StageStat:
        """读取某环节统计（不存在时返回空统计，不自动建表）。"""
        return self._stages.get(stage, StageStat(name=stage))

    def stages(self) -> list[StageStat]:
        """返回全部环节统计列表（按名称稳定排序）。"""
        return [self._stages[k] for k in sorted(self._stages)]

    def total_llm_calls(self) -> int:
        """全部模型调用次数。"""
        return sum(s.llm_calls for s in self._stages.values())

    def total_input_tokens(self) -> int:
        """全部输入 token。"""
        return sum(s.input_tokens for s in self._stages.values())

    def total_output_tokens(self) -> int:
        """全部输出 token。"""
        return sum(s.output_tokens for s in self._stages.values())

    def total_tokens(self) -> int:
        """全部 token（输入 + 输出）。"""
        return self.total_input_tokens() + self.total_output_tokens()

    def total_duration_ms(self) -> float:
        """全部耗时（毫秒）。"""
        return sum(s.duration_ms for s in self._stages.values())

    def reset(self) -> None:
        """清空统计，供下一轮独立测量。"""
        self._stages.clear()


def now_ms() -> float:
    """返回当前时间戳（毫秒，单调时钟），用于计时埋点。"""
    return time.perf_counter() * 1000.0
