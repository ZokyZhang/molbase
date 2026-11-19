# -*- coding: utf-8 -*-
"""
索引模块：倒排索引 + BM25F 字段加权召回（P1 升级）。

升级点：
1. 中文词典分词：内置词典 + 正向最大匹配（FMM）+ 停用词过滤，
   未登录词回退 bigram，召回与语义兼顾。
2. BM25F：标题 / 正文分字段加权（标题命中权重更高）。

设计说明：纯 Python 标准库实现，零第三方依赖。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..config import (
    BM25_B,
    BM25_FIELD_WEIGHT_CONTENT,
    BM25_FIELD_WEIGHT_TITLE,
    BM25_K1,
)
from ..store import Entry


# 英文 / 数字词正则
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# 内置中文词典（正向最大匹配 + 全切分用，可按领域扩展）
_LEXICON = {
    "数据库", "索引", "查询", "优化", "向量", "检索", "重排", "召回", "技能",
    "代码", "方法", "函数", "模型", "知识", "代工", "关联", "图谱", "摘要",
    "分词", "停用", "文档", "内容", "标题", "正文", "标签", "热度", "新鲜度",
    "反馈", "采纳", "引用", "命中", "自我", "进化", "闭环", "系统", "引擎",
    "存储", "倒排", "算法", "排序", "推荐", "语义", "压缩", "上下文", "中间件",
    "架构", "接口", "编排", "路由", "意图", "大模型", "本地", "离线", "增强",
    "降级", "哈希", "嵌入", "图检索", "调用", "编译", "反编译",
    "人脸", "识别", "检测", "图像", "屏幕", "时间", "控制", "策略",
    "作业", "管理", "家庭", "教育", "孩子", "家长", "学习",
    "数据库索引", "查询优化", "关联方法", "技能代管", "代码代工", "作业管理",
}

# 停用词表（过滤虚词 / 无意义词）
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "等", "这", "那", "我", "你",
    "他", "她", "它", "们", "我们", "你们", "一个", "一种", "这个", "那个",
    "进行", "以及", "但是", "因为", "所以", "如果", "然后", "可以", "需要",
    "通过", "对于", "关于", "根据", "其中", "如下", "以上", "什么", "怎么",
    "如何", "哪些", "the", "a", "an", "of", "to", "and", "or", "in", "on", "for",
}

# 词典最长词长度（用于 FMM 最大匹配）
_MAX_WORD_LEN = max(len(w) for w in _LEXICON)


def _fmm(s: str) -> list[str]:
    """正向最大匹配（FMM）分词：优先匹配词典中的最长词，未匹配到则切单字。"""
    result: list[str] = []
    i, n = 0, len(s)
    while i < n:
        matched = None
        for j in range(min(n, i + _MAX_WORD_LEN), i, -1):
            if s[i:j] in _LEXICON:
                matched = s[i:j]
                break
        if matched:
            result.append(matched)
            i += len(matched)
        else:
            result.append(s[i])
            i += 1
    return result


def _lexicon_words_in(s: str) -> list[str]:
    """全切分：返回 s 中所有出现在词典里的子串（长度 >= 2，去重），提升语义覆盖。"""
    found: set[str] = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 2, min(n, i + _MAX_WORD_LEN) + 1):
            w = s[i:j]
            if w in _LEXICON:
                found.add(w)
    return list(found)


def _bigrams(s: str) -> list[str]:
    """对连续中文字符串做字符 bigram 切分（未登录词召回兜底）。"""
    out: list[str] = []
    if len(s) == 1:
        if s not in _STOPWORDS:
            out.append(s)
    else:
        for i in range(len(s) - 1):
            bg = s[i : i + 2]
            if bg not in _STOPWORDS:
                out.append(bg)
    return out


def tokenize(text: str) -> list[str]:
    """
    中文词典分词 + 英文词提取：

    - 英文 / 数字词直接提取（转小写）。
    - 中文：词典全切分词（精准语义）+ bigram 兜底（未登录词召回）+ 单字。
    - 过滤停用词。
    """
    tokens: list[str] = []

    # 1. 英文 / 数字词
    for w in _WORD_RE.findall(text):
        w = w.lower()
        if w not in _STOPWORDS:
            tokens.append(w)

    # 2. 中文部分
    non_ascii = _WORD_RE.sub(" ", text)
    cjk = re.sub(r"[\s\W]+", "", non_ascii)
    if cjk:
        # 2.1 词典全切分词（精准语义）
        tokens.extend(_lexicon_words_in(cjk))
        # 2.2 bigram 兜底（召回）
        tokens.extend(_bigrams(cjk))
        # 2.3 单字（提升召回，过滤停用词）
        tokens.extend(ch for ch in cjk if ch not in _STOPWORDS)

    return tokens


@dataclass
class ScoredHit:
    """召回命中结果：条目 + 得分。"""

    entry: Entry
    score: float


class BM25Index:
    """倒排索引 + BM25F 字段加权打分。"""

    def __init__(
        self,
        k1: float = BM25_K1,
        b: float = BM25_B,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        """
        初始化索引。

        field_weights：字段权重映射，默认标题 3.0 / 正文 1.0（标题命中更关键）。
        """
        self.k1 = k1
        self.b = b
        self.field_weights = field_weights or {
            "title": BM25_FIELD_WEIGHT_TITLE,
            "content": BM25_FIELD_WEIGHT_CONTENT,
        }
        self.documents: list[Entry] = []
        self.fields: dict[str, dict] = {}   # field -> 字段级索引数据
        self.idf: dict[str, float] = {}     # 文档级 IDF

    def build(self, entries: list[Entry]) -> None:
        """根据条目列表构建字段级倒排索引。"""
        self.documents = entries
        n = max(len(entries), 1)
        self.fields = {}

        # 字段提取器
        extractors = {
            "title": lambda e: e.title,
            "content": lambda e: e.content,
        }

        # 逐字段构建索引
        for field, extractor in extractors.items():
            tokens_list = [tokenize(extractor(e)) for e in entries]
            doc_freq = [Counter(t) for t in tokens_list]
            doc_len = [len(t) for t in tokens_list]
            avg_len = sum(doc_len) / n
            postings: dict[str, list[int]] = {}
            for i, freq in enumerate(doc_freq):
                for term in freq:
                    postings.setdefault(term, []).append(i)
            self.fields[field] = {
                "tokens": tokens_list,
                "doc_freq": doc_freq,
                "doc_len": doc_len,
                "avg_len": avg_len,
                "postings": postings,
            }

        # 文档级 IDF：词出现在多少文档（合并字段）
        doc_terms: dict[str, set] = {}
        for fd in self.fields.values():
            for term, docs in fd["postings"].items():
                doc_terms.setdefault(term, set()).update(docs)
        for term, docs in doc_terms.items():
            df = len(docs)
            self.idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_n: int) -> list[ScoredHit]:
        """BM25F 检索：返回得分最高的 top_n 个条目。"""
        q_tokens = tokenize(query)
        if not q_tokens or not self.documents:
            return []

        scores: dict[int, float] = {}
        for term in set(q_tokens):
            if term not in self.idf:
                continue
            idf = self.idf[term]

            # 合并所有字段中命中该词的文档
            docs: set[int] = set()
            for fd in self.fields.values():
                docs.update(fd["postings"].get(term, []))

            for doc_idx in docs:
                # 加权字段内归一化 tf
                tf_norm = 0.0
                for field, w in self.field_weights.items():
                    fd = self.fields.get(field)
                    if fd is None:
                        continue
                    tf = fd["doc_freq"][doc_idx][term]
                    if tf == 0:
                        continue
                    dl = fd["doc_len"][doc_idx]
                    avgl = max(fd["avg_len"], 1.0)
                    tf_norm += w * (tf / (1 - self.b + self.b * (dl / avgl)))
                # BM25F 得分
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (
                    tf_norm / (self.k1 + tf_norm)
                )

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [ScoredHit(entry=self.documents[i], score=s) for i, s in ranked]
