# -*- coding: utf-8 -*-
"""
服务化 HTTP API 层：让 MolBase 部署后能供局域网内其他应用调用。

定位（对齐项目重定位「服务器部署后供其他应用调用」）：
- 用标准库 http.server 实现 REST/JSON 接口，零第三方依赖（不引入 Flask/FastAPI）。
- 任何语言/应用（Java/Python/Node/Shell/curl）都能通过 HTTP/JSON 调用代工层，
  不必与 molbase 同进程、同语言——这正是「轻应用、重知识」的落地形态。
- 默认监听 127.0.0.1（仅本机）；显式配置 SERVER_HOST=0.0.0.0 开放局域网。
- 可选 SERVER_AUTH_TOKEN 做 Bearer 鉴权（受信局域网可留空）。

运行：
    python -m molbase.server                 # 默认 127.0.0.1:8787
    python -m molbase.server --host 0.0.0.0 --port 8787   # 开放局域网

接口一览（均返回 JSON）：
    GET  /health              -> 存活 + 统计
    GET  /stats               -> 条目统计 + 会话级 token 节省统计
    GET  /token               -> 会话级 token 节省统计（节省比例 / 等效降本）
    GET  /metrics             -> 持久化埋点统计（累计节省 + 最近明细，供监控大屏）
    GET  /                    -> 监控大屏（配套服务，数据源自知识库 /metrics + /stats）
    GET  /dashboard           -> 同 /（监控大屏别名）
    POST /query               -> {"query":"...", "top_k":5} 检索最小上下文
    POST /feedback            -> {"entry_id":"...", "action":"hit|adopt|cite|reject"}
    POST /summarize           -> {"text":"...", "max_chars":200} 上下文压缩
    POST /skill               -> {"intent":"..."} 匹配技能包（工具执行见 P2 路线）
    POST /ingest/knowledge    -> {"entry_id","title","content","source","meta"}
    POST /harvest             -> {"session_id","transcript","source","meta"} 会话回写（提炼成核心资产沉淀）
    POST /tooltrace           -> {"session_id","intent","tool_calls":[...]} 工具调用轨迹沉淀（观察→固化→复用，达标自动生成 skill）
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import SANITIZE_ENABLED, SERVER_AUTH_TOKEN, SERVER_HOST, SERVER_PORT
from .orchestrator import Orchestrator, QueryResponse, QueryResult
from .sanitize import sanitize_text

# 监控大屏页面路径（随包分发，服务启动后一步到位直接托管）
_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------
def _result_to_dict(r: QueryResult) -> dict[str, Any]:
    """QueryResult -> JSON 可序列化 dict（输出前对 content 做 PII 脱敏）。"""
    content = sanitize_text(r.content) if SANITIZE_ENABLED else r.content
    return {
        "entry_id": r.entry_id,
        "entry_type": r.entry_type,
        "title": r.title,
        "content": content,
        "source": r.source,
        "score": r.score,
        "meta": r.meta,
    }


def _response_to_dict(resp: QueryResponse) -> dict[str, Any]:
    """QueryResponse -> JSON 可序列化 dict（含质量底线 confidence/abstain）。"""
    return {
        "route": resp.route,
        "query": resp.query,
        "rewritten": resp.rewritten,
        "results": [_result_to_dict(r) for r in resp.results],
        "web_results": [_result_to_dict(r) for r in resp.web_results],
        "token_estimate": resp.token_estimate,
        "baseline_tokens": resp.baseline_tokens,
        "content_chars": resp.content_chars,
        "confidence": resp.confidence,
        "abstain": resp.abstain,
    }


# ---------------------------------------------------------------------------
# HTTP 请求处理器
# ---------------------------------------------------------------------------
class KBRequestHandler(BaseHTTPRequestHandler):
    """
    请求处理器：解析 JSON 请求，路由到代工层，返回 JSON 响应。

    通过 create_server 用 type() 动态注入 orc 与 auth_token（类属性），
    避免全局可变状态，便于多实例隔离。
    """

    orc: Orchestrator | None = None   # 由 create_server 注入
    auth_token: str = ""              # 由 create_server 注入

    # ---- 基础收发 ----
    def _send_json(self, code: int, obj: dict[str, Any]) -> None:
        """统一 JSON 响应出口。"""
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any] | None:
        """读取并解析请求体 JSON，失败返回 None。"""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return None

    def _authed(self) -> bool:
        """校验 Bearer 令牌；未配置令牌时恒通过。"""
        if not self.auth_token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth in (f"Bearer {self.auth_token}", self.auth_token)

    def _send_html(self, html: str) -> None:
        """统一 HTML 响应出口（监控大屏页面）。"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ok(self, obj: dict[str, Any]) -> None:
        self._send_json(200, obj)

    def _err(self, code: int, msg: str) -> None:
        self._send_json(code, {"error": msg})

    # ---- HTTP 方法 ----
    def do_GET(self) -> None:
        """GET：/health、/stats、/token、/metrics、/（监控大屏）。"""
        if not self._authed():
            return self._err(401, "unauthorized")
        if self.path in ("/health", "/healthz"):
            return self._ok({"status": "ok", "stats": self.orc.stats()})
        if self.path == "/stats":
            return self._ok({
                "stats": self.orc.stats(),
                "token": self.orc.token_stats(),
            })
        if self.path == "/token":
            return self._ok({"token": self.orc.token_stats()})
        if self.path == "/metrics":
            return self._ok(self.orc.metrics_stats())
        if self.path in ("/", "/dashboard", "/index.html"):
            return self._serve_dashboard()
        return self._err(404, "not found")

    def _serve_dashboard(self) -> None:
        """
        返回监控大屏页面（配套服务，一步到位部署）。

        数据不在 HTML 内写死，而是由页面 JS 实时调用本服务的 /metrics 与
        /stats 接口从知识库拉取——大屏本质是「基于知识库做的应用 demo」。
        """
        try:
            html = _DASHBOARD_PATH.read_text(encoding="utf-8")
        except OSError:
            return self._err(500, "dashboard.html not found")
        return self._send_html(html)

    def do_POST(self) -> None:
        """POST：按路径路由到代工层各接口。"""
        if not self._authed():
            return self._err(401, "unauthorized")
        data = self._read_json()
        if data is None:
            return self._err(400, "invalid json body")

        path = self.path.rstrip("/")
        if path == "/query":
            return self._handle_query(data)
        if path == "/feedback":
            return self._handle_feedback(data)
        if path == "/summarize":
            return self._handle_summarize(data)
        if path == "/skill":
            return self._handle_skill(data)
        if path == "/ingest/knowledge":
            return self._handle_ingest_knowledge(data)
        if path == "/harvest":
            return self._handle_harvest(data)
        if path == "/tooltrace":
            return self._handle_tooltrace(data)
        return self._err(404, "not found")

    # ---- 各接口处理器 ----
    def _handle_query(self, data: dict[str, Any]) -> None:
        """检索最小上下文（可选 tags 按 meta.tags 过滤，AND 语义）。"""
        query = str(data.get("query", "")).strip()
        if not query:
            return self._err(400, "missing 'query'")
        top_k = int(data.get("top_k", 5))
        tags = data.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]
        resp = self.orc.query(query, top_k=top_k, tags=tags)
        self._ok(_response_to_dict(resp))

    def _handle_feedback(self, data: dict[str, Any]) -> None:
        """回写反馈信号（hit/adopt/cite/reject）。"""
        entry_id = str(data.get("entry_id", "")).strip()
        action = str(data.get("action", "hit")).strip()
        if not entry_id:
            return self._err(400, "missing 'entry_id'")
        self.orc.feedback(entry_id, action)
        self._ok({"ok": True, "entry_id": entry_id, "action": action})

    def _handle_summarize(self, data: dict[str, Any]) -> None:
        """上下文压缩。"""
        text = str(data.get("text", ""))
        if not text:
            return self._err(400, "missing 'text'")
        max_chars = int(data.get("max_chars", 200))
        summary = self.orc.summarize(text, max_chars=max_chars)
        self._ok({"summary": summary, "original_chars": len(text), "summary_chars": len(summary)})

    def _handle_skill(self, data: dict[str, Any]) -> None:
        """
        匹配技能包（P1 版）：按意图检索技能，返回技能内容与元信息。
        完整「执行工具步骤」见定位路线 P2。
        """
        intent = str(data.get("intent", "")).strip()
        if not intent:
            return self._err(400, "missing 'intent'")
        resp = self.orc.query(intent)
        skills = [r for r in resp.results if r.entry_type == "skill"]
        self._ok({
            "intent": intent,
            "route": resp.route,
            "skills": [_result_to_dict(s) for s in skills],
            "note": "工具执行能力见 P2 路线（REPOSITION.md 第九章）",
        })

    def _handle_ingest_knowledge(self, data: dict[str, Any]) -> None:
        """摄入一条知识条目（带冲突四态判定，返回 state）。"""
        entry_id = str(data.get("entry_id", "")).strip()
        title = str(data.get("title", "")).strip()
        content = str(data.get("content", "")).strip()
        if not entry_id or not title or not content:
            return self._err(400, "missing entry_id/title/content")
        source = str(data.get("source", ""))
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        result = self.orc.ingest_knowledge(entry_id, title, content, source=source, meta=meta)
        self._ok({
            "ok": True,
            "entry_id": result.entry_id,
            "state": result.state,
            "against_id": result.against_id,
            "reason": result.reason,
        })

    def _handle_harvest(self, data: dict[str, Any]) -> None:
        """会话回写：把整段会话提炼成一份核心资产沉淀回知识库。"""
        session_id = str(data.get("session_id", "")).strip()
        transcript = str(data.get("transcript", "")).strip()
        if not session_id or not transcript:
            return self._err(400, "missing session_id/transcript")
        source = str(data.get("source", ""))
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        result = self.orc.harvest_session(session_id, transcript, source=source, meta=meta)
        self._ok({
            "ok": result.state not in ("empty",),
            "entry_id": result.entry_id,
            "state": result.state,
            "engine": result.engine,
            "title": result.title,
            "reason": result.reason,
            "related_fingerprints": result.related_fingerprints,
        })

    def _handle_tooltrace(self, data: dict[str, Any]) -> None:
        """
        工具调用轨迹沉淀：观察大模型本次会话实际调用了哪些工具，累计操作指纹，
        达标自动生成可复用 skill（实现 N+1 -> N 的 token 收敛）。
        """
        session_id = str(data.get("session_id", "")).strip()
        intent = str(data.get("intent", "")).strip()
        tool_calls = data.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return self._err(400, "missing 'tool_calls' (list)")
        source = str(data.get("source", ""))
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        result = self.orc.record_tooltrace(session_id, intent, tool_calls, source=source, meta=meta)
        self._ok({
            "fingerprint": result.fingerprint,
            "count": result.count,
            "generated": result.generated,
            "skill_id": result.skill_id,
            "reason": result.reason,
        })

    def log_message(self, format: str, *args: Any) -> None:
        """简化访问日志，避免刷屏。"""
        print(f"[molbase] {self.address_string()} {format % args}")


# ---------------------------------------------------------------------------
# 服务创建与入口
# ---------------------------------------------------------------------------
def create_server(
    orc: Orchestrator,
    host: str | None = None,
    port: int | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """
    创建 HTTP 服务。

    orc        ：已初始化的代工层编排器。
    host/port  ：监听地址/端口；不传则用 config（可被外部配置覆盖）。
    auth_token ：访问令牌；不传则用 config.SERVER_AUTH_TOKEN。
    """
    host = host or SERVER_HOST
    port = port if port is not None else SERVER_PORT
    token = auth_token if auth_token is not None else SERVER_AUTH_TOKEN

    # 动态子类注入依赖，避免全局可变状态
    handler = type(
        "BoundKBRequestHandler",
        (KBRequestHandler,),
        {"orc": orc, "auth_token": token},
    )
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    """命令行入口：python -m molbase.server [--host H] [--port P]。"""
    import argparse

    parser = argparse.ArgumentParser(description="MolBase 服务化 HTTP API")
    parser.add_argument("--host", default=SERVER_HOST, help="监听地址（0.0.0.0 开放局域网）")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="监听端口")
    args = parser.parse_args()

    orc = Orchestrator()
    server = create_server(orc, host=args.host, port=args.port)
    print(f"[molbase] 服务已启动: http://{args.host}:{args.port}")
    print(f"[molbase] 监控大屏:   http://{args.host}:{args.port}/  (数据源自知识库 /metrics + /stats)")
    print(f"[molbase] 鉴权: {'已启用' if server.RequestHandlerClass.auth_token else '未启用'}")
    print(f"[molbase] 向量后端: {orc.embedder.name} | LLM: {orc.llm.name if orc.llm else '未启用'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[molbase] 收到中断，正在关闭…")
    finally:
        server.server_close()
        orc.close()


if __name__ == "__main__":
    main()
