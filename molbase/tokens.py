# -*- coding: utf-8 -*-
"""
Token 估算工具（粗估埋点）。

用途：给代工层「降 token」能力做量化观测——在 query 响应里埋一个 token 粗估值，
用「返回内容字符数 / 字符-token 比」估算，供后续接入真实场景时校准。

说明：这是估算，不是实测。真实 token 数取决于分词器（中文 1 字约 0.6~1 token，
英文 1 token 约 4 字符）。这里采用行业常用粗估：
- 中文汉字：1 字 ≈ 1 token
- 英文字母/数字：4 字符 ≈ 1 token（向上取整）
- 空格、标点等不计入（占比小，忽略）
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """
    粗估一段文本的 token 数。

    - 中文汉字（CJK 基本区）按 1 字 = 1 token 计；
    - ASCII 字母/数字按 4 字符 = 1 token 计（向上取整）；
    - 其余字符（空格、标点）忽略。
    """
    if not text:
        return 0
    cn = 0
    en = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:  # CJK 统一汉字基本区
            cn += 1
        elif ch.isascii() and ch.isalnum():
            en += 1
    return cn + (en + 3) // 4


def estimate_texts(texts: list[str]) -> int:
    """粗估多段文本的 token 总数。"""
    return sum(estimate_tokens(t) for t in texts)


class TokenLedger:
    """
    会话级 token 台账：累积多次查询的 token 埋点，实时统计「节省比例」与「等效降本」。

    定位（对齐用户诉求「减少了多少 token、降低了多少成本」的会话级视角）：
    - 单次 query 只反映一次调用；会话级累计才能体现「越用越省」的整体效果。
    - 口径：
        actual_tokens   = Σ 每次 query 返回的最小上下文 token（代工层裁剪后）。
        baseline_tokens = Σ 每次 query 若不做代工、模型自读全表的 token。
        saved_ratio     = (baseline - actual) / baseline，即代工层帮模型省下的 token 占比。
    - 成本按「输入 token 单价」折算（返回的上下文对下游模型而言是输入），
      复用 cost.py 的 cost_of，保持「若上云等效成本」口径一致。
    - 零第三方依赖，纯算术，可随时 reset() 重计。
    """

    def __init__(self) -> None:
        """初始化空的会话统计。"""
        self.queries = 0            # 累计查询次数
        self.actual_tokens = 0      # 有代工层实际返回 token 累计
        self.baseline_tokens = 0    # 无代工层全表基线 token 累计

    def record(self, actual_tokens: int, baseline_tokens: int) -> None:
        """
        累计一次查询的 token 埋点。

        actual_tokens   ：本次返回内容的粗估 token。
        baseline_tokens ：本次全表扫描基线的粗估 token。
        baseline <= 0（如未开启 TOKEN_TRACE）时跳过，避免拉低整体节省比例。
        """
        if baseline_tokens <= 0:
            return
        self.queries += 1
        self.actual_tokens += actual_tokens
        self.baseline_tokens += baseline_tokens

    @property
    def saved_tokens(self) -> int:
        """累计节省的 token 数。"""
        return self.baseline_tokens - self.actual_tokens

    @property
    def saved_ratio(self) -> float:
        """节省比例（0~1）；无基线时为 0。"""
        if self.baseline_tokens <= 0:
            return 0.0
        return self.saved_tokens / self.baseline_tokens

    def snapshot(self) -> dict:
        """
        返回会话统计快照（含节省比例与等效成本），供 API / 报告直接序列化。
        """
        from .cost import cost_of  # 局部导入，避免模块级循环依赖

        actual_cost = cost_of(self.actual_tokens, 0)
        baseline_cost = cost_of(self.baseline_tokens, 0)
        return {
            "queries": self.queries,
            "actual_tokens": self.actual_tokens,
            "baseline_tokens": self.baseline_tokens,
            "saved_tokens": self.saved_tokens,
            "saved_ratio": round(self.saved_ratio, 4),
            "saved_ratio_pct": f"{self.saved_ratio * 100:.1f}%",
            "actual_cost": actual_cost,
            "baseline_cost": baseline_cost,
            "saved_cost": round(baseline_cost - actual_cost, 4),
        }

    def reset(self) -> None:
        """清空会话统计，供下一轮独立计量。"""
        self.queries = 0
        self.actual_tokens = 0
        self.baseline_tokens = 0
