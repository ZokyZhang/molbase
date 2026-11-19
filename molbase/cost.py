# -*- coding: utf-8 -*-
"""
成本折算模型：把 token 粗估值换算成等价成本（人民币）。

定位（对齐用户诉求「降低了多少成本」）：
- 本地大模型「免费」（实际成本仅电费 / 算力），无法直接体现降本收益；
  因此用「若上云等效单价」折算——把省下的 token 换算成「如果走商用 API 要花多少钱」，
  从而量化「代工层帮模型省了多少调用成本」。
- 单价硬编码在 config.py，单位：人民币元 / 每百万 token，区间参考主流商用模型定价。
- 零第三方依赖，纯算术，不产生任何网络调用。

口径说明：
- 输入 token（prompt）与输出 token（生成）单价不同（生成通常更贵），分开计价。
- 这是「等效折算」，不是账单；真实本地推理成本远低于此（只有电费）。
"""

from __future__ import annotations

from .config import (
    COST_CURRENCY,
    COST_INPUT_PER_1M,
    COST_OUTPUT_PER_1M,
)

# 每百万 token 的基准单位（1M = 1_000_000）
_PER_1M = 1_000_000.0


def cost_of(
    input_tokens: int,
    output_tokens: int = 0,
    *,
    input_price: float = COST_INPUT_PER_1M,
    output_price: float = COST_OUTPUT_PER_1M,
) -> float:
    """
    按输入 / 输出 token 分别折算成本。

    input_tokens  ：输入 token 数（粗估）。
    output_tokens ：输出 token 数（粗估）。
    input_price   ：输入单价（元 / 百万 token）。
    output_price  ：输出单价（元 / 百万 token）。

    返回：成本（元），保留可读精度。
    """
    return round(
        input_tokens / _PER_1M * input_price
        + output_tokens / _PER_1M * output_price,
        4,
    )


def format_cost(value: float, currency: str = COST_CURRENCY) -> str:
    """
    把成本金额格式化为可读字符串。

    value    ：金额（元）。
    currency ：货币符号（默认 ¥）。
    """
    return f"{currency}{value:,.4f}"
