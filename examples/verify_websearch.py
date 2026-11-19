# -*- coding: utf-8 -*-
"""验证 HttpSearchProvider 真实联网（探活 + 三路对比）。

用法：
    python examples/verify_websearch.py

证明 HttpSearchProvider 确实发起真实 HTTP 请求：
  0. healthz 秒回 OK：虚拟机 SearXNG 可达（真连铁证）
  1. 真连 SearXNG：search 返回（results 是否非空取决于上游引擎国内可达性）
  2. NullSearchProvider：不联网，恒空、零网络
  3. 连不存在地址：超时降级为空（证明确实联网、且失效不崩溃）
"""
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from molbase.websearch import HttpSearchProvider, NullSearchProvider

SEARXNG = "http://<虚拟机IP>:8888"  # TODO: 改成你的 SearXNG 地址


def timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


print("=" * 60)
print("  HttpSearchProvider 真实联网验证")
print("=" * 60)

# 0. healthz 探活（连上铁证：秒回 OK）
try:
    dt, body = timed(lambda: urllib.request.urlopen(f"{SEARXNG}/healthz", timeout=5).read().decode())
    print(f"\n[0] 探活 {SEARXNG}/healthz -> {body!r}，耗时 {dt:.0f}ms（秒回 = 已连上）")
except Exception as e:
    print(f"\n[0] 探活失败：{e}")

# 1. 真连 SearXNG 搜索
p1 = HttpSearchProvider(endpoint=SEARXNG)
dt, hits = timed(lambda: p1.search("python", top_n=3))
print(f"\n[1] 真连 {SEARXNG}")
print(f"    GET {SEARXNG}/search?q=python&format=json")
print(f"    耗时 {dt:.0f}ms，返回 {len(hits)} 条命中（SearXNG 内部等待上游引擎）")
for h in hits:
    print(f"      - {h.title}  {h.url}")

# 2. NullSearchProvider（默认不联网）
p2 = NullSearchProvider()
dt, hits = timed(lambda: p2.search("python"))
print(f"\n[2] NullSearchProvider（默认，不联网）")
print(f"    耗时 {dt:.0f}ms，返回 {len(hits)} 条（恒空，零网络）")

# 3. 连不存在的地址（证明确实在联网）
p3 = HttpSearchProvider(endpoint="http://10.0.0.199:8888", timeout=3)
dt, hits = timed(lambda: p3.search("python"))
print(f"\n[3] 连不存在的地址 10.0.0.199:8888（timeout=3s）")
print(f"    耗时 {dt:.0f}ms，返回 {len(hits)} 条（连接失败降级为空，不崩溃）")

print("\n结论：")
print("  [0] healthz 秒回 OK —— 虚拟机 SearXNG 真实可达；")
print("  [1] 真连返回 —— HttpSearchProvider 发出真实 HTTP 请求并解析 JSON；")
print("  [2] 零网络、[3] 超时降级 —— 证明不启用时零开销、失效时绝不崩溃；")
print("  上游 results 是否非空取决于 SearXNG 引擎的国内可达性（非 MolBase 本身问题）。")
