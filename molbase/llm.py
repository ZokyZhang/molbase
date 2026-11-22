# -*- coding: utf-8 -*-
"""
本地大模型（LLM）接入层 —— V2 接本地大模型的核心。

定位（对齐用户诉求）：
- V2 接的是本地大模型（如 Qwen3.5 9B 这类 LLM），不是向量模型；
  向量模型（embedding）只作为「参数化、可插拔」的普通依赖存在。
- 本地 LLM 在代工层承担「要语义理解、低 token」的活：
    意图路由（语义分类）、查询改写、重排打分、摘要压缩、技能生成。
- 确定性算法（BM25 / 图 / AST / 哈希向量）仍是主链路，LLM 只是增强插件。

设计原则（对齐「对标 MySQL、杜绝模型版本匹配不上就失效」）：
- 客户端用标准库 urllib 实现，零第三方依赖（不引入 openai 库）。
- 走 OpenAI 兼容 /v1/chat/completions 协议：Ollama / vLLM / llama.cpp 通用，
  用户只改 endpoint + 模型名即可。
- LLM 是「增强插件」：不可用 / 超时 / 版本不匹配时，工厂函数返回 None，
  上层据此自动降级回纯算法路径，绝不崩溃。
- 模型名、endpoint、超时、温度等全部参数化（硬编码在 config.py）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class LocalLLM:
    """
    本地大模型客户端（标准库实现，零第三方依赖），支持两种协议：

    - ollama-native：Ollama 原生 /api/chat，自动关思考（think=false），
      qwen 等思考模型必须走此协议（OpenAI 兼容端点对思考模型不兼容，
      关思考不生效，会导致正文为空或原样复制不压缩）。
    - openai：OpenAI 兼容 /v1/chat/completions，vLLM/llama.cpp 通用。
    """

    name = "local-llm"

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        protocol: str = "ollama-native",
    ) -> None:
        """
        初始化客户端（惰性：不联网，第一次 chat 才真正请求）。

        endpoint   ：基础 URL，如 "http://localhost:11434/v1"（含或不含 /v1）
        model      ：模型名（即版本指纹，如 "qwen3.5-9b"），换模型/版本只改这一处
        timeout    ：单次请求超时秒数
        temperature：采样温度，0.0 表示确定性输出（降幻觉）
        max_tokens ：单次生成上限
        protocol   ：协议名，ollama-native | openai
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.protocol = protocol
        # ollama-native：由 /v1 推导出 Ollama 原生 /api/chat 地址
        if protocol == "ollama-native":
            base = self.endpoint
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            self._chat_url = base + "/api/chat"
        else:
            self._chat_url = self.endpoint + "/chat/completions"

    def chat(self, system: str, user: str) -> str:
        """
        发送一次对话，返回助手文本。失败（未启动/超时/版本不匹配）抛异常，
        由上层捕获后降级，绝不在此处吞错。
        """
        if self.protocol == "ollama-native":
            return self._chat_ollama(system, user)
        return self._chat_openai(system, user)

    def _chat_ollama(self, system: str, user: str) -> str:
        """Ollama 原生 /api/chat：关思考（think=false），返回 message.content。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["message"]["content"]

    def _chat_openai(self, system: str, user: str) -> str:
        """OpenAI 兼容 /v1/chat/completions，返回 choices[0].message.content。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def complete_json(self, system: str, user: str) -> dict | None:
        """
        请求模型返回 JSON，并解析为 dict。

        容错：模型可能用 ```json ... ``` 代码块包裹输出，先剥离再解析。
        解析失败返回 None（上层据此降级）。
        """
        text = self.chat(system, user).strip()
        # 剥离可能的 markdown 代码块
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None


def create_llm(
    endpoint: str | None,
    model: str | None,
    timeout: float = 30.0,
    temperature: float = 0.0,
    max_tokens: int = 512,
    protocol: str = "ollama-native",
) -> LocalLLM | None:
    """
    工厂函数：创建本地大模型客户端。

    - endpoint 或 model 为空（未配置）→ 返回 None（表示「未启用」）。
    - 其余情况构造客户端，真正的可用性在首次 chat 时通过 try/except 暴露。
    """
    if not endpoint or not model:
        return None
    return LocalLLM(
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
        protocol=protocol,
    )
