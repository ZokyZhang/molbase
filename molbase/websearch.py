# -*- coding: utf-8 -*-
"""
网络搜索外部源适配器。

职责（代工层 = 查询优化器 + 执行计划的「外部召回源」）：
- 网络搜索作为内部知识库（BM25F + 关联图）之外的第三个召回源。
- 定位关系：内部知识库 = 权威沉淀（优先）；网络搜索 = 实时兜底（补充）。
  由代工层统一融合排序，大模型只吃融合后的结果，绝不自己搜。

设计原则（对齐「对标 MySQL、杜绝模型版本失效」）：
- 零第三方依赖：HTTP 用标准库 urllib，JSON 用标准库 json。
- 可插拔：SearchProvider 抽象协议，默认 NullSearchProvider（不启用，返回空），
  需要时注入 HttpSearchProvider 或自定义实现。
- 失效降级：任何网络异常 / 超时 / 解析失败都返回空列表，绝不崩溃。
  （网络搜索是增强，不是必需品——拔掉后系统照常自足运行。）

搜索后端建议（任选其一，均与 OpenAI 无关、可本地/自建）：
- SearXNG：本地/自建聚合搜索，JSON 接口（推荐，无需 API key）。
  默认地址 http://localhost:8888/search?q=...&format=json
- 其他：任意返回 {"results": [{title, content, url}]} 结构的 HTTP 接口均可适配。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import WEB_SEARCH_ENDPOINT, WEB_SEARCH_TIMEOUT, WEB_SEARCH_TOP_N


@dataclass
class SearchHit:
    """一条网络搜索结果（外部召回源的最小单元）。"""

    title: str
    content: str
    url: str
    score: float = 1.0   # 相关性分（1.0 最高，随搜索排序递减）


class SearchProvider:
    """
    搜索提供者抽象协议（可插拔）。

    实现方只需提供 search(query, top_n) -> list[SearchHit]。
    约定：任何异常都必须被吞掉并返回空列表，保证代工层「绝不崩溃」。
    """

    name = "abstract"

    def search(self, query: str, top_n: int = WEB_SEARCH_TOP_N) -> list[SearchHit]:
        """执行一次网络搜索，返回命中列表。"""
        raise NotImplementedError


class NullSearchProvider(SearchProvider):
    """空搜索提供者（默认）：不启用网络搜索，永远返回空列表。"""

    name = "null"

    def search(self, query: str, top_n: int = WEB_SEARCH_TOP_N) -> list[SearchHit]:
        """不联网，直接返回空。"""
        return []


class HttpSearchProvider(SearchProvider):
    """
    HTTP 搜索提供者：请求 SearXNG 兼容的 JSON 接口。

    - endpoint：SearXNG 地址（不含 /search 路径），默认 http://localhost:8888
    - 请求：GET {endpoint}/search?q={query}&format=json
    - 响应：{"results": [{"title":..., "content":..., "url":...}, ...]}
    - 任何异常（连接失败/超时/JSON 解析错误）都返回空列表，绝不抛错。
    """

    name = "http"

    def __init__(
        self,
        endpoint: str = WEB_SEARCH_ENDPOINT,
        timeout: float = WEB_SEARCH_TIMEOUT,
    ) -> None:
        """初始化 HTTP 搜索提供者。"""
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, top_n: int = WEB_SEARCH_TOP_N) -> list[SearchHit]:
        """请求搜索接口并解析结果，失败返回空列表。"""
        url = f"{self.endpoint}/search?q={quote(query)}&format=json"
        try:
            req = Request(url, headers={"User-Agent": "molbase/1.0"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # 网络不可用 / 超时 / 返回非 JSON -> 降级为空，绝不崩溃
            return []

        hits: list[SearchHit] = []
        results = data.get("results", []) if isinstance(data, dict) else []
        for i, item in enumerate(results[:top_n]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            url = str(item.get("url", "")).strip()
            if not title and not content:
                continue
            # 相关性分：按搜索排序位置线性递减（top-1 = 1.0，逐位 -0.1，下限 0.3）
            hits.append(
                SearchHit(
                    title=title,
                    content=content,
                    url=url,
                    score=max(0.3, 1.0 - i * 0.1),
                )
            )
        return hits


def create_search_provider(
    enabled: bool,
    endpoint: str = WEB_SEARCH_ENDPOINT,
    timeout: float = WEB_SEARCH_TIMEOUT,
) -> SearchProvider:
    """
    工厂函数：按开关创建搜索提供者。

    - enabled=False：返回 NullSearchProvider（默认，不联网）。
    - enabled=True ：返回 HttpSearchProvider（联网，失败自动降级为空结果）。

    任何异常都不会抛出，保证代工层自足运行。
    """
    if not enabled:
        return NullSearchProvider()
    try:
        return HttpSearchProvider(endpoint=endpoint, timeout=timeout)
    except Exception:
        # 配置错误等 -> 降级为空搜索，绝不崩溃
        return NullSearchProvider()
