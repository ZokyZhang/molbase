# -*- coding: utf-8 -*-
"""
TextRank 抽取式摘要 + 智能切块（P0，零模型，直接服务「降 token」目标）。

设计说明（对齐「对标 MySQL、自我运行」）：
- TextRank 就是 PageRank 在「句子相似度图」上的应用，零模型、零第三方依赖、确定性；
  作为代工层「最小上下文」的零模型压缩手段，介于 LLM 生成式摘要与原文截断之间。
- 智能切块按段落（空行）优先、句子兜底，把长文本切成可精准投喂的上下文块。
"""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """按中文/英文句末标点、分号、换行切分句子，过滤空句。"""
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def split_blocks(text: str, max_chars: int = 500) -> list[str]:
    """
    智能切块：段落（空行）是天然边界，每段一块；单段超长才按句子兜底切。

    返回的每个块都是尽量完整的语义单元，供代工层精准定位投喂。
    """
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            result.append(p)
            continue
        # 超长段落按句子兜底切分（句子间用句号分隔，保持可读）
        cur = ""
        for s in split_sentences(p):
            sep = "。" if cur else ""
            if len(cur) + len(sep) + len(s) <= max_chars:
                cur = cur + sep + s
            else:
                if cur:
                    result.append(cur)
                cur = s
        if cur:
            result.append(cur)
    return result


def _tokens(text: str) -> list[str]:
    """复用 index 模块的中英分词（中文 bigram + 英文小写单词）。"""
    from .index import tokenize

    return tokenize(text)


def _similarity(a: str, b: str) -> float:
    """句子相似度：词袋 Jaccard 相似度（零模型，确定性）。"""
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


def textrank_summarize(
    text: str,
    ratio: float = 0.3,
    min_sentences: int = 1,
    max_sentences: int = 5,
) -> str:
    """
    TextRank 抽取式摘要：句子为节点、句子相似度为边，幂迭代 PageRank 排序，
    取 top 句按原文顺序拼接为摘要。

    - 句子数过少时直接返回原文（无需摘要）。
    - 全程零模型、零第三方依赖、确定性。
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return ""
    top_n = max(min_sentences, min(max_sentences, int(round(n * ratio))))
    if n <= top_n:
        return text

    # 构建相似度矩阵（无向，对称）
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _similarity(sentences[i], sentences[j])
            sim[i][j] = sim[j][i] = s

    # 行归一化：得到转移概率矩阵 P(i -> j)
    for i in range(n):
        total = sum(sim[i])
        if total > 0:
            sim[i] = [x / total for x in sim[i]]

    # 幂迭代 PageRank（damping=0.85）
    d = 0.85
    r = [1.0 / n] * n
    for _ in range(100):
        r_new = [(1.0 - d) / n] * n
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += sim[j][i] * r[j]
            r_new[i] += d * acc
        if sum(abs(r_new[i] - r[i]) for i in range(n)) < 1e-6:
            r = r_new
            break
        r = r_new

    # 取 top 句，按原文顺序拼接
    idx = sorted(range(n), key=lambda i: r[i], reverse=True)[:top_n]
    idx.sort()
    return "。".join(sentences[i] for i in idx) + "。"
