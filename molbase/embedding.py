# -*- coding: utf-8 -*-
"""
可插拔向量化（embedding）抽象层。

核心设计（对齐用户诉求：对标 MySQL，杜绝「模型版本匹配不上就失效」）：

1. 默认后端 `HashEmbedding` 是**确定性哈希向量**（feature hashing），
   完全零模型、零第三方依赖，向量只由文本内容决定，永不失效、跨版本稳定。
2. 可选后端（SentenceTransformer / Ollama 等）是「增强插件」：
   - 只有当模型可用时才启用；
   - 模型不可用 / 版本不匹配时，**自动降级**回 HashEmbedding，绝不崩溃。
3. 向量维度与算法由后端自己声明，检索层只认抽象接口，不关心具体模型。

这样保证了：MolBase的「能跑起来」不依赖任何模型版本，像 MySQL 一样
装完即可自我运行；embedding 只是锦上添花的语义增强。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """向量化后端的抽象协议。"""

    name: str    # 后端名（如 hash / sentence-transformers / ollama）
    dim: int     # 向量维度

    def embed(self, text: str) -> list[float]:
        """把文本编码为 dim 维向量（需 L2 归一化）。"""
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化：使向量模长为 1，便于余弦相似度（点积）计算。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


class HashEmbedding:
    """
    确定性哈希向量（feature hashing）。

    - 零模型、零依赖：用标准库 hashlib 把每个 token 哈希到固定维度桶。
    - 确定性：相同文本永远得到相同向量（跨进程、跨版本稳定），绝不失效。
    - 语义近似：词袋叠加 + 哈希签名，适合作为「离线无大模型版」的基线语义分。

    这是「对标 MySQL 自我运行」的关键：即使没有任何模型，也能做轻量语义检索。
    """

    name = "hash"
    dim = 256  # 默认 256 维，足够中小知识库使用

    def __init__(self, dim: int = 256) -> None:
        """初始化哈希向量维度。"""
        self.dim = dim

    def _token_hashes(self, text: str) -> list[tuple[int, int]]:
        """把文本拆成 token，返回每个 token 的 (桶下标, 符号)。"""
        # 复用 index 模块的分词逻辑
        from .index import tokenize

        tokens = tokenize(text)
        pairs: list[tuple[int, int]] = []
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            # 用前 8 字节做桶下标，第 9 字节的奇偶做符号（±1）
            bucket = int.from_bytes(h[:8], "big") % self.dim
            sign = 1 if h[8] % 2 == 0 else -1
            pairs.append((bucket, sign))
        return pairs

    def embed(self, text: str) -> list[float]:
        """
        把文本编码为 dim 维哈希向量并 L2 归一化。

        算法：对每个 token 做 MD5 哈希 -> 桶下标 + 符号，累加后归一化。
        """
        vec = [0.0] * self.dim
        for bucket, sign in self._token_hashes(text):
            vec[bucket] += sign
        return _l2_normalize(vec)


class SentenceTransformerEmbedding:
    """
    可选的本地 sentence-transformers 模型后端（增强插件）。

    - 需要 pip install sentence-transformers 才能使用。
    - 若模型加载失败（版本不匹配 / 未安装），构造函数抛异常，
      由上层捕获后降级到 HashEmbedding。
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """加载本地 embedding 模型；失败抛异常（由上层降级处理）。"""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("未安装 sentence-transformers，请先 pip install") from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """用本地模型编码文本并归一化。"""
        import numpy as np  # type: ignore

        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in np.asarray(vec)]


class TfidfEmbedding:
    """
    TF-IDF 加权哈希向量（P2，零模型，确定性）。

    - 在 HashEmbedding 基础上，用 TF-IDF 词权重代替统一权重，
      常用词降权、罕见词升权，语义区分度更高。
    - 需要先 build(corpus) 统计 IDF；未 build 时 idf 为空，退化为 HashEmbedding 等价。
    - 零模型、零第三方依赖；语料不变则向量确定，不涉及模型版本问题。
    """

    name = "tfidf"

    def __init__(self, dim: int = 256) -> None:
        """初始化维度与空 IDF 表。"""
        self.dim = dim
        self.idf: dict[str, float] = {}

    def build(self, corpus: list[str]) -> None:
        """根据语料统计 IDF（idf = ln((N+1)/(df+1)) + 1）。"""
        from .index import tokenize

        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(tokenize(doc)):
                df[term] = df.get(term, 0) + 1
        n = max(len(corpus), 1)
        self.idf = {t: math.log((n + 1.0) / (c + 1.0)) + 1.0 for t, c in df.items()}

    def embed(self, text: str) -> list[float]:
        """TF-IDF 加权哈希向量（L2 归一化）。"""
        from .index import tokenize

        vec = [0.0] * self.dim
        for term in tokenize(text):
            w = self.idf.get(term, 1.0)  # 未登录词用默认权重
            h = hashlib.md5(term.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:8], "big") % self.dim
            sign = 1 if h[8] % 2 == 0 else -1
            vec[bucket] += w * sign
        return _l2_normalize(vec)


def simhash(text: str, dim: int = 64) -> int:
    """
    SimHash（P2）：把文本哈希成 dim 位指纹，用于近重复检测。

    对每个 token 做 MD5 哈希，按位累加（1 加 / 0 减），
    最后以每位的正负决定指纹。零模型、确定性。
    """
    from .index import tokenize

    v = [0] * dim
    for term in tokenize(text):
        h = hashlib.md5(term.encode("utf-8")).digest()
        for i in range(dim):
            bit = (h[i // 8] >> (i % 8)) & 1
            v[i] += 1 if bit else -1
    result = 0
    for i in range(dim):
        if v[i] > 0:
            result |= 1 << i
    return result


def hamming_distance(a: int, b: int) -> int:
    """两个整数指纹的海明距离（不同位数）。"""
    return bin(a ^ b).count("1")


def is_near_duplicate(text1: str, text2: str, threshold: int = 3, dim: int = 64) -> bool:
    """判断两段文本是否近重复（SimHash 海明距离 <= 阈值）。"""
    return hamming_distance(simhash(text1, dim), simhash(text2, dim)) <= threshold


def create_embedder(backend: str = "hash", model_name: str | None = None) -> EmbeddingProvider:
    """
    工厂函数：按后端名创建向量化器。

    - backend="hash"：零模型确定性哈希（默认，永不失效）。
    - backend="tfidf"：TF-IDF 加权哈希（需先 build(corpus)，零模型）。
    - backend="sentence-transformers"：本地模型，失败自动降级回 hash。

    任何异常都会降级回 HashEmbedding，保证「绝不崩溃」。
    """
    if backend == "tfidf":
        return TfidfEmbedding()
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedding(model_name or "all-MiniLM-L6-v2")
        except Exception:
            # 模型不可用/版本不匹配 -> 降级到零模型哈希向量
            return HashEmbedding()
    # 默认（含 "hash"）走零模型哈希
    return HashEmbedding()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（假设两向量已 L2 归一化，直接点积）。"""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
