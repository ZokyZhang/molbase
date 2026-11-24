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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import COMPACT_ENABLED, SANITIZE_ENABLED, SERVER_AUTH_TOKEN, SERVER_HOST, SERVER_PORT
from .orchestrator import Orchestrator, QueryResponse, QueryResult
from .sanitize import sanitize_text

# 监控大屏页面路径（随包分发，服务启动后一步到位直接托管）
_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


# ---------------------------------------------------------------------------
# 序列化辅助
# ---------------------------------------------------------------------------
def _result_to_dict(r: QueryResult) -> dict[str, Any]:
    """QueryResult -> JSON 可序列化 dict（默认返回语义保持摘要，原文挂 full_content）。"""
    content = sanitize_text(r.content) if SANITIZE_ENABLED else r.content
    summary = sanitize_text(r.summary) if (SANITIZE_ENABLED and r.summary) else r.summary
    d = {
        "entry_id": r.entry_id,
        "entry_type": r.entry_type,
        "title": r.title,
        "source": r.source,
        "score": r.score,
        "meta": r.meta,
    }
    if COMPACT_ENABLED and summary:
        d["content"] = summary
        d["full_content"] = content
    else:
        d["content"] = content
    return d


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
        # v3 场景锚定层新增字段：让 HTTP 调用方拿到场景/实体/短路状态（不再丢弃）
        "scene_id": resp.scene_id,
        "entities": resp.entities,
        "resolved_version_id": resp.resolved_version_id,
        "refs_miss": resp.refs_miss,
        "skills": list(resp.skills),
        # v9.1 原文层定位：命中条目出自哪一节、正文哪个字符区间，供调用方回溯原文
        "refs": list(resp.refs),
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
        # 解析路径与 query string（如 /metrics?include_test=1 切换是否含测试数据）
        parsed = urlparse(self.path)
        path = parsed.path
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        if path in ("/health", "/healthz"):
            return self._ok({"status": "ok", "stats": self.orc.stats()})
        if path == "/stats":
            return self._ok({
                "stats": self.orc.stats(),
                "token": self.orc.token_stats(),
            })
        if path == "/token":
            return self._ok({"token": self.orc.token_stats()})
        if path == "/metrics":
            include_test = qs.get("include_test", "0") in ("1", "true", "yes", "on")
            return self._ok(self.orc.metrics_stats(include_test=include_test))
        if path == "/overview":
            return self._ok(self.orc.overview())
        if path == "/gaps":
            limit = int(qs.get("limit", "20"))
            return self._ok({"gaps": self.orc.store.abstain_gaps(limit=limit)})
        if path == "/api/drafts":
            # v7.2 弃答补数：草稿统计与清单（status 为空查全部）
            status = qs.get("status", "pending")
            limit = min(int(qs.get("limit", "50")), 200)
            return self._ok({"stats": self.orc.store.draft_stats(), "drafts": self.orc.store.list_drafts(status, limit)})
        if path == "/writeback":
            limit = int(qs.get("limit", "10"))
            return self._ok({"writeback": self.orc.store.writeback_recent(limit=limit)})
        if path == "/entries":
            return self._handle_entries(qs)
        if path == "/api/detail":
            try:
                metric = qs.get("metric", "token")
                limit = min(int(qs.get("limit", "50")), 500)
                offset = max(int(qs.get("offset", "0")), 0)
                q = qs.get("q", "")
                sort = qs.get("sort", "ts")
                order = qs.get("order", "desc")
                # v6 全量明细：显式维度过滤参数
                scene = qs.get("scene", "")
                label = qs.get("label", "")
                abstain = qs.get("abstain", "")
                include_test = qs.get("include_test", "0") in ("1", "true", "yes", "on")
                return self._ok(self.orc.metric_detail(
                    metric, limit, offset, q, sort, order, scene, label, abstain, include_test
                ))
            except (ValueError, TypeError):
                return self._err(400, "Invalid parameters")
        if path == "/api/compare":
            try:
                group_by = qs.get("group_by", "scene_id")
                include_test = qs.get("include_test", "0") in ("1", "true", "yes", "on")
                limit = min(int(qs.get("limit", "50")), 200)
                return self._ok(self.orc.metric_compare(group_by, include_test, limit))
            except (ValueError, TypeError):
                return self._err(400, "Invalid parameters")
        if path == "/api/alerts":
            return self._ok(self.orc.alerts())
        if path == "/api/alert-rules":
            return self._ok(self.orc.alert_rules())
        if path == "/api/trend":
            try:
                metric = qs.get("metric", "queries")
                limit = min(int(qs.get("limit", "100")), 500)
                since = float(qs["since"]) if "since" in qs else None
                return self._ok({"metric": metric, "points": self.orc.store.list_snapshots(metric, limit, since)})
            except (ValueError, TypeError, KeyError):
                return self._err(400, "Invalid parameters")
        if path == "/api/snapshot":
            return self._ok({"ok": True, "values": self.orc.capture_snapshot()})
        if path in ("/", "/dashboard", "/index.html"):
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

    def _handle_entries(self, qs: dict[str, str]) -> None:
        """按 tags/type 列出知识条目（AND 过滤），供门户等应用做目录浏览。"""
        tags = [t.strip() for t in qs.get("tags", "").split(",") if t.strip()]
        entry_type = qs.get("type", "").strip()
        try:
            limit = int(qs.get("limit", "200"))
        except ValueError:
            limit = 200
        out = []
        for e in self.orc.store.all_entries():
            if entry_type and e.entry_type != entry_type:
                continue
            etags = e.meta.get("tags") or []
            if isinstance(etags, str):
                etags = [etags]
            etags = [str(t) for t in etags]
            if tags and not all(t in etags for t in tags):
                continue
            content = sanitize_text(e.content) if SANITIZE_ENABLED else e.content
            out.append({
                "entry_id": e.entry_id,
                "entry_type": e.entry_type,
                "title": e.title,
                "content": content,
                "source": e.source,
                "tags": etags,
                "heat": e.heat,
                "created_at": e.created_at,
                "last_access": e.last_access,
            })
        out.sort(key=lambda x: (-(x["heat"] or 0.0), -x["last_access"]))
        out = out[:limit]
        self._ok({"entries": out, "total": len(out)})

    def do_POST(self) -> None:
        """POST：按路径路由到代工层各接口。"""
        if not self._authed():
            return self._err(401, "unauthorized")
        data = self._read_json()
        if data is None:
            return self._err(400, "invalid json body")

        path = self.path.rstrip("/")
        # 记录接口调用（驱动「调用率」指标）：每次真实调用落库 api_calls
        if path == "/query":
            self.orc.record_api_call("query")
            return self._handle_query(data)
        if path == "/feedback":
            self.orc.record_api_call("feedback")
            return self._handle_feedback(data)
        if path == "/summarize":
            self.orc.record_api_call("summarize")
            return self._handle_summarize(data)
        if path == "/skill":
            self.orc.record_api_call("skill")
            return self._handle_skill(data)
        if path == "/ingest/knowledge":
            self.orc.record_api_call("ingest")
            return self._handle_ingest_knowledge(data)
        if path == "/ingest/files":
            self.orc.record_api_call("ingest_files")
            return self._handle_ingest_files(data)
        if path == "/harvest":
            self.orc.record_api_call("harvest")
            return self._handle_harvest(data)
        if path == "/tooltrace":
            self.orc.record_api_call("tooltrace")
            return self._handle_tooltrace(data)
        if path == "/expand":
            self.orc.record_api_call("expand")
            return self._handle_expand(data)
        if path == "/compact/backfill":
            self.orc.record_api_call("backfill")
            return self._handle_backfill(data)
        if path == "/evolve":
            self.orc.record_api_call("evolve")
            return self._ok(self.orc.evolve())
        if path == "/gap/generate":
            # v7.2 弃答补数：从弃答缺口生成草稿（只建草稿不入库）
            self.orc.record_api_call("gap_generate")
            limit = int(data.get("limit", 50))
            return self._ok(self.orc.generate_gap_drafts(limit=limit))
        if path == "/gap/confirm":
            # v7.2 弃答补数：显式确认草稿→入库→命中回验
            self.orc.record_api_call("gap_confirm")
            draft_id = str(data.get("draft_id", "")).strip()
            if not draft_id:
                return self._err(400, "missing 'draft_id'")
            title = str(data.get("title", "")).strip()
            content = str(data.get("content", "")).strip()
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else None
            return self._ok(self.orc.confirm_gap_draft(draft_id, title, content, meta))
        return self._err(404, "not found")

    # ---- 各接口处理器 ----
    def _handle_query(self, data: dict[str, Any]) -> None:
        """检索最小上下文（可选 tags 按 meta.tags 过滤，AND 语义；可选 session_id 归集会话；可选 device_mac 本机加权）。"""
        query = str(data.get("query", "")).strip()
        if not query:
            return self._err(400, "missing 'query'")
        top_k = int(data.get("top_k", 5))
        tags = data.get("tags")
        if tags is not None and not isinstance(tags, list):
            tags = [str(tags)]
        session_id = str(data.get("session_id", "")).strip()
        if not session_id:
            # HTTP 直连兜底：未显式传会话时按来源 IP 归集（无法区分同设备多会话；
            # 精确的会话归集由 MCP 层进程级 session_id 保证，这里仅让直连调用可见）。
            client_ip = self.client_address[0] if self.client_address else "unknown"
            session_id = f"http::{client_ip}"
        device_mac = str(data.get("device_mac", "")).strip()
        # v3 场景锚定层：解析 scene 与 refs 参数（三级短路 Level1 file-grounded）
        scene = str(data.get("scene", "")).strip()
        refs = data.get("refs")
        if refs is not None and not isinstance(refs, list):
            refs = [str(refs)]
        # v6 全量明细标记：只允许 golden / negative / real，其他值视为未标记
        label = str(data.get("label", "")).strip()
        if label not in ("golden", "negative", "real"):
            label = ""
        resp = self.orc.query(
            query, top_k=top_k, tags=tags, session_id=session_id,
            device_mac=device_mac, scene=scene, refs=refs, label=label,
        )
        self._ok(_response_to_dict(resp))

    def _handle_feedback(self, data: dict[str, Any]) -> None:
        """回写反馈信号（hit/adopt/cite/reject；可选 session_id 归集会话）。"""
        entry_id = str(data.get("entry_id", "")).strip()
        action = str(data.get("action", "hit")).strip()
        if not entry_id:
            return self._err(400, "missing 'entry_id'")
        session_id = str(data.get("session_id", "")).strip()
        if not session_id:
            client_ip = self.client_address[0] if self.client_address else "unknown"
            session_id = f"http::{client_ip}"
        # v7：可选来源标记，区分显式反馈与系统自动正反馈；非法值一律按 manual
        source = str(data.get("source", "manual")).strip()
        if source not in ("manual", "auto"):
            source = "manual"
        self.orc.feedback(entry_id, action, session_id=session_id, source=source)
        self._ok({"ok": True, "entry_id": entry_id, "action": action, "source": source})

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

    def _handle_ingest_files(self, data: dict[str, Any]) -> None:
        """批量摄入客户端上传的文件（三级 + 设备标识），返回统计。"""
        # 提取并清理 device_mac 字段
        device_mac = str(data.get("device_mac", "")).strip()
        # 校验 device_mac 是否为空
        if not device_mac:
            return self._err(400, "missing 'device_mac'")
        # 提取 files 字段并校验是否为非空列表
        files = data.get("files")
        if not isinstance(files, list) or len(files) == 0:
            return self._err(400, "missing 'files' (list)")
        # 调用编排器执行批量文件摄入
        result = self.orc.ingest_files(device_mac, files)
        # 返回成功响应（含编排器统计信息）
        self._ok({"ok": True, **result})

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

    def _handle_expand(self, data: dict[str, Any]) -> None:
        """按需展开原文：凭 entry_id 返回原文全文 + 摘要。"""
        entry_id = str(data.get("entry_id", "")).strip()
        if not entry_id:
            return self._err(400, "missing 'entry_id'")
        entry = self.orc.expand(entry_id)
        if entry is None:
            return self._err(404, "not found")
        content = sanitize_text(entry.content) if SANITIZE_ENABLED else entry.content
        summary = sanitize_text(entry.summary) if (SANITIZE_ENABLED and entry.summary) else entry.summary
        return self._ok({
            "entry_id": entry.entry_id,
            "title": entry.title,
            "content": content,
            "summary": summary,
        })

    def _handle_backfill(self, data: dict[str, Any]) -> None:
        """批量回填存量摘要。"""
        limit = int(data.get("limit", 50))
        result = self.orc.backfill_summaries(limit)
        return self._ok({"ok": True, **result})

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


def _snapshot_loop(orc, interval: int = 300) -> None:
    # 后台定时采集指标快照（daemon 线程调用），每 interval 秒一次，异常不影响主服务。
    while True:
        try:
            orc.capture_snapshot()
        except Exception as e:
            print(f"[molbase] 快照采集失败：{e}")
        time.sleep(interval)
# 在 main() 里服务启动后调用 threading.Thread(target=_snapshot_loop, args=(orc,), daemon=True).start() 即可启用

def main() -> None:
    """命令行入口：python -m molbase.server [--host H] [--port P]。"""
    import argparse

    parser = argparse.ArgumentParser(description="MolBase 服务化 HTTP API")
    parser.add_argument("--host", default=SERVER_HOST, help="监听地址（0.0.0.0 开放局域网）")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="监听端口")
    args = parser.parse_args()

    orc = Orchestrator()
    server = create_server(orc, host=args.host, port=args.port)
    # 启动后台快照线程（每 5 分钟采集一次趋势数据，v4 新增）
    threading.Thread(target=_snapshot_loop, args=(orc,), daemon=True).start()
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
