#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolBase MCP Server —— 零依赖 stdio MCP 桥接器（单文件，纯 Python 标准库）。

作用：
    把局域网/本机的 MolBase 知识库 HTTP API（默认 http://127.0.0.1:8787）翻译成
    MCP（Model Context Protocol）工具，供 WorkBuddy 等 MCP 客户端通过 stdio 调用。

设计原则（对齐用户约定）：
    - 零第三方依赖：仅用 json / sys / urllib 标准库，无需 pip install，随处可跑。
    - 单文件：本文件可独立拷贝到任意设备，只需 Python 3 即可运行。
    - 所有方法及代码块均有中文注释。

环境变量（硬编码默认值，可被覆盖）：
    MOLBASE_URL    知识库服务器地址，默认 http://127.0.0.1:8787
                   （局域网部署时指向中心机，如 http://10.0.0.200:8787）
    MOLBASE_TOKEN  可选 Bearer 令牌，留空则不鉴权（受信内网默认）

MCP 传输：stdio（NDJSON，每条 JSON-RPC 消息一行，以换行符分隔）。
调试日志统一输出到 stderr，绝不污染 stdout 协议流。

在 WorkBuddy 中的配置示例（~/.workbuddy/mcp.json）：
    {
      "mcpServers": {
        "molbase": {
          "command": "python",
          "args": ["/绝对路径/molbase_mcp.py"],
          "env": { "MOLBASE_URL": "http://10.0.0.200:8787" }
        }
      }
    }
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# 配置（硬编码默认值，可被环境变量覆盖）
# ---------------------------------------------------------------------------
# 知识库服务器地址：默认本机；局域网部署时用 MOLBASE_URL 指向中心机
SERVER_URL = os.environ.get("MOLBASE_URL", "http://127.0.0.1:8787").rstrip("/")
# 可选 Bearer 令牌：受信内网默认留空
AUTH_TOKEN = os.environ.get("MOLBASE_TOKEN", "")

# MCP 协议版本号（向后兼容）
MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "molbase"
SERVER_VERSION = "1.2.0"

# ---------------------------------------------------------------------------
# 会话感知：MCP 进程启动时生成「进程级会话 id」，调用方未显式传 session_id 时
# 自动注入，保证每一次 query/feedback/harvest/tooltrace 都能归集到「本 MCP 实例」。
# stdio MCP 进程通常与一个 WorkBuddy 会话一一对应（会话开始拉起、结束关闭），
# 因此进程级 id 可近似「当前会话」，让监控大屏实时看到「每个会话 MolBase 参与
# 了几次、命中/弃答、是否逃票」，不再依赖模型自觉传参（此前 session_id 100% 缺失）。
# ---------------------------------------------------------------------------
try:
    _HOSTNAME = socket.gethostname()
except Exception:
    _HOSTNAME = "unknown"
SESSION_ID = os.environ.get("MOLBASE_SESSION_ID") or (
    f"mcp::{_HOSTNAME}::{os.getpid()}::{int(time.time())}"
)
# 需要会话归集的工具：arguments 未传 session_id 时自动注入进程级会话 id
_SESSION_AWARE_TOOLS = {
    "molbase_query", "molbase_feedback", "molbase_harvest", "molbase_tooltrace",
}

# ---------------------------------------------------------------------------
# 本地兜底检索配置：中心库「没数据/不准」时，检索当前设备本地的知识藏身处。
# 理念：很多信息没纳入中心库，却躺在本机的技能文件/记忆/工作区日志里，
# 这些也「算是有」，只是没纳管。哪台设备调用就查哪台设备。
# ---------------------------------------------------------------------------
def _default_local_dirs() -> list[str]:
    """默认检索目录：当前 agent（WorkBuddy）在 C 盘的缓存空间里的知识藏身处。"""
    home = os.path.expanduser("~")
    dirs = [
        os.path.join(home, ".workbuddy", "skills"),   # 技能文件（SKILL.md）
        os.path.join(home, ".workbuddy", "memory"),   # 记忆目录
        os.path.join(home, "WorkBuddy"),              # 工作区（项目文档 + .workbuddy/memory 日志）
    ]
    # 用户级个人记忆目录（含 MEMORY.md），多用户用通配匹配
    try:
        dirs += glob.glob(os.path.join(home, ".workbuddy", "user-*-personal"))
    except Exception:
        pass
    return dirs

# 本地检索目录：可用 MOLBASE_LOCAL_SEARCH_DIRS 覆盖（分号或逗号分隔多个目录）
LOCAL_SEARCH_DIRS = [
    d for d in os.environ.get(
        "MOLBASE_LOCAL_SEARCH_DIRS",
        ";".join(_default_local_dirs()),
    ).replace(",", ";").split(";") if d.strip()
]
# 本地 qwen 服务（判相关性用）：无 Ollama 时自动降级为纯 grep 结果
LOCAL_QWEN_URL = os.environ.get("MOLBASE_QWEN_URL", "http://127.0.0.1:11434")
LOCAL_QWEN_MODEL = os.environ.get("MOLBASE_QWEN_MODEL", "qwen3.6-27b-q3")
# 本地兜底开关：MOLBASE_LOCAL_FALLBACK=0 可关闭本地检索
LOCAL_FALLBACK_ON = os.environ.get("MOLBASE_LOCAL_FALLBACK", "1") not in ("0", "false", "off", "no")


# ---------------------------------------------------------------------------
# HTTP 客户端：把 MCP 工具调用转发到知识库 HTTP API
# ---------------------------------------------------------------------------
def http_call(path: str, payload: dict | None = None) -> dict:
    """
    调用知识库 HTTP 接口（GET 无 body / POST 带 JSON body），返回解析后的 JSON dict。

    path    ：接口路径，如 "/query"、"/ingest/knowledge"、"/health"。
    payload ：POST 请求体；None 则发 GET。
    """
    url = SERVER_URL + path
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if AUTH_TOKEN:
        # 配置了令牌时附加 Bearer 鉴权头
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    if payload is None:
        req = urllib.request.Request(url, headers=headers, method="GET")
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 4xx/5xx：把错误信息转成 dict，避免上层抛异常导致 MCP 崩溃
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except Exception:
            detail = {"error": f"HTTP {e.code}"}
        return {"error": detail.get("error", f"HTTP {e.code}")}
    except Exception as e:
        # 网络不可达 / 超时：返回结构化错误
        return {"error": f"无法连接知识库服务器 {SERVER_URL}：{e}"}


# ---------------------------------------------------------------------------
# 工具定义（name + description + inputSchema，供 tools/list 返回）
# ---------------------------------------------------------------------------
def _tool(name: str, description: str, schema: dict, required: list[str] | None = None) -> dict:
    """构造单个工具的 MCP 定义对象。"""
    s = dict(schema)
    if required:
        s["required"] = required
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": s, **({"required": required} if required else {})},
    }


# 工具清单：与知识库 HTTP 接口一一对应
TOOLS = [
    _tool(
        "molbase_query",
        "检索知识库最小上下文（信息获取的唯一入口）。按意图召回最相关条目，"
        "返回经重排后的最小上下文块供模型推理。若中心库无高置信度命中（abstain），"
        "会自动检索当前设备本地（技能文件/记忆/工作区日志）兜底，命中即返回并自动纳管入库。",
        {
            "query": {"type": "string", "description": "检索意图（自然语言或关键词）"},
            "top_k": {"type": "integer", "description": "返回条目数，默认 5"},
            "tags": {"type": "array", "items": {"type": "string"},
                     "description": "可选分类过滤（AND 语义），如 [运维,环境,配置,架构,规范,设计,业务,敏感]"},
            "session_id": {"type": "string", "description": "可选会话标识，用于监控大屏的会话链路追踪"},
        },
        ["query"],
    ),
    _tool(
        "molbase_ingest_knowledge",
        "沉淀一条新知识到知识库（产出知识/部署信息/项目约定后必须回写）。含冲突四态判定："
        "duplicate（近重复跳过）/ conflict（同ID异源，待裁决）/ merge（同源覆盖）。",
        {
            "entry_id": {"type": "string", "description": "唯一条目 ID，建议 kb::<唯一标识>"},
            "title": {"type": "string", "description": "条目标题"},
            "content": {"type": "string", "description": "条目正文（Markdown）"},
            "source": {"type": "string", "description": "来源说明，可选"},
            "meta": {"type": "object", "description": "元信息，可选，如 {\"tags\":[\"运维\"]}"},
        },
        ["entry_id", "title", "content"],
    ),
    _tool(
        "molbase_feedback",
        "回写反馈信号，驱动知识库热度与排序（越用越准）。action 取值：hit/adopt/cite/reject。",
        {
            "entry_id": {"type": "string", "description": "目标条目 ID"},
            "action": {"type": "string", "description": "hit=命中 / adopt=采纳 / cite=引用 / reject=拒绝"},
            "session_id": {"type": "string", "description": "可选会话标识，用于监控大屏的会话链路追踪"},
        },
        ["entry_id", "action"],
    ),
    _tool(
        "molbase_harvest",
        "会话回写：把整段会话提炼成一份核心资产沉淀回知识库，供后续复用（每次会话收尾时调用）。",
        {
            "session_id": {"type": "string", "description": "会话唯一 ID"},
            "transcript": {"type": "string", "description": "本会话完整内容/摘要"},
            "source": {"type": "string", "description": "来源说明，可选"},
            "meta": {"type": "object", "description": "元信息，可选，如 {\"owner\":\"zoky\"}"},
        },
        ["session_id", "transcript"],
    ),
    _tool(
        "molbase_tooltrace",
        "工具调用轨迹沉淀：记录本次会话实际调用的工具序列，累计操作指纹，"
        "达标自动生成可复用 skill（实现 N+1→N 的 token 收敛）。",
        {
            "session_id": {"type": "string", "description": "会话唯一 ID"},
            "intent": {"type": "string", "description": "本次意图描述"},
            "tool_calls": {"type": "array",
                           "items": {"type": "object"},
                           "description": "工具调用序列，如 [{\"tool\":\"web_search\",\"args\":{\"query\":\"...\"}}]"},
        },
        ["session_id", "intent", "tool_calls"],
    ),
    _tool(
        "molbase_summarize",
        "上下文压缩：把长文本压缩成指定长度的摘要（降 token）。",
        {
            "text": {"type": "string", "description": "待压缩文本"},
            "max_chars": {"type": "integer", "description": "摘要最大字符数，默认 200"},
        },
        ["text"],
    ),
    _tool(
        "molbase_stats",
        "查询知识库条目统计与 token 节省统计。",
        {},
    ),
    _tool(
        "molbase_health",
        "知识库服务器存活探活，返回状态与条目统计。",
        {},
    ),
    _tool(
        "molbase_gaps",
        "拉取弃答缺口清单：最近被「宁可不答」拦下的查询及其归因"
        "（no_data 没数据 / weak_match 弱相关 / ambiguous 区分度不足），"
        "用于发现知识库还缺什么、该补什么，驱动「从无到有」的持续积累。",
        {
            "limit": {"type": "integer", "description": "返回缺口条数，默认 20"},
        },
    ),
    _tool(
        "molbase_evolve",
        "手动触发一次自我进化：基于共现建关联图 + 冷知识衰减 + 死知识归档。"
        "补完数据后调用，可让关联图更密、置信度更高、弃答率下降。",
        {},
    ),
]

# 工具名 -> (HTTP 方法路径, 参数字段名映射)
TOOL_ROUTES = {
    "molbase_query": ("/query", None),
    "molbase_ingest_knowledge": ("/ingest/knowledge", None),
    "molbase_feedback": ("/feedback", None),
    "molbase_harvest": ("/harvest", None),
    "molbase_tooltrace": ("/tooltrace", None),
    "molbase_summarize": ("/summarize", None),
    "molbase_stats": ("/stats", None),
    "molbase_health": ("/health", None),
    "molbase_gaps": ("/gaps", None),
    "molbase_evolve": ("/evolve", None),
}


# ---------------------------------------------------------------------------
# 结果整理：把 HTTP 返回整理成适合模型阅读的文本
# ---------------------------------------------------------------------------
def _format_query_result(data: dict) -> str:
    """把 /query 返回整理成最小上下文的可读文本。"""
    if data.get("abstain"):
        # 质量底线：置信度不足，明确告知模型不要硬答
        return ("[abstain] 知识库无高置信度命中（confidence=%.3f），"
                "请据实说明「不知道」，不要编造答案。\n原始响应：%s"
                % (data.get("confidence", 0.0), json.dumps(data, ensure_ascii=False)))

    lines = []
    results = data.get("results", [])
    if not results:
        return "[空] 知识库未检索到相关条目。\n原始响应：" + json.dumps(data, ensure_ascii=False)

    for i, r in enumerate(results, 1):
        lines.append(f"【{i}】{r.get('title', '')} (id={r.get('entry_id', '')}, score={r.get('score', 0):.3f})")
        lines.append(r.get("content", ""))
        lines.append("")
    return "\n".join(lines).strip()


def _format_generic_result(data: dict) -> str:
    """通用结果：直接序列化为 JSON 文本。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 本地兜底检索：中心库「没数据/不准」时，检索当前设备本地知识藏身处。
# 纯 Python 标准库实现（os.walk + 关键词匹配），零第三方依赖，任何设备可跑。
# ---------------------------------------------------------------------------
# 参与本地检索的文本文件后缀
_LOCAL_TEXT_EXTS = {
    ".md", ".txt", ".log", ".json", ".py", ".yml", ".yaml", ".js", ".ts",
    ".java", ".html", ".sh", ".ini", ".conf", ".xml", ".csv", ".sql", ".rst",
}
# 跳过的高噪音目录
_LOCAL_SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".venv", "venv", "binaries",
    "dist", "build", ".idea", ".vscode", "plugins", "blobs", "cache",
}
# 单个文件最大读取字节（跳过超大文件，避免拖慢检索）
_LOCAL_MAX_FILE_BYTES = 200_000


def _split_keywords(query: str) -> list[str]:
    """把查询拆成检索关键词：中文按常见分隔符切分，英文保留词。"""
    parts = re.split(r"[\s,，。、；;：:()（）\[\]【】\"'?!？!]+", query or "")
    kws = [p.strip() for p in parts if p.strip()]
    if not kws:
        kws = [query.strip()]
    # 过滤过短的无意义词（单字符英文/数字噪声），但保留单字中文
    kws = [k for k in kws if len(k) >= 2 or re.search(r"[\u4e00-\u9fff]", k)]
    return kws if kws else [query.strip()]


def _extract_snippet(text: str, keyword: str, width: int = 220) -> str:
    """提取关键词附近片段（去换行，压缩空白）。"""
    idx = text.find(keyword)
    if idx < 0:
        return text[:width]
    start = max(0, idx - width // 2)
    end = min(len(text), idx + width // 2)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _local_grep(query: str, limit: int = 8) -> list[tuple[str, str, int]]:
    """本机关键词检索：返回 [(文件路径, 命中片段, 命中关键词数)]，按命中数降序。"""
    keywords = _split_keywords(query)
    hits: list[tuple[str, str, int]] = []
    for d in LOCAL_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs_, files in os.walk(d):
            dirs_[:] = [
                x for x in dirs_
                if x not in _LOCAL_SKIP_DIRS and not x.startswith(".")
            ]
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in _LOCAL_TEXT_EXTS:
                    continue
                path = os.path.join(root, fn)
                try:
                    size = os.path.getsize(path)
                    if size > _LOCAL_MAX_FILE_BYTES:
                        continue
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception:
                    continue
                matched = [k for k in keywords if k in text]
                if matched:
                    hits.append((path, _extract_snippet(text, matched[0]), len(matched)))
            if len(hits) >= limit * 4:
                break
        if len(hits) >= limit * 4:
            break
    hits.sort(key=lambda x: -x[2])
    return hits[:limit]


def _qwen_rank(query: str, candidates: list[tuple[str, str, int]]) -> tuple[list[int] | None, str]:
    """调本地 qwen 判断候选相关性。返回 (相关候选下标列表, 精简答案)；失败返回 (None, "")。

    该模型为推理模型，会先输出 reasoning 再输出 content，故 max_tokens 需给足，
    否则 content 为空导致判断失效。"""
    if not candidates:
        return None, ""
    cand_text = "\n".join(
        f"[{i}] {path}: {snippet}"
        for i, (path, snippet, _) in enumerate(candidates, 1)
    )
    prompt = (
        f"查询：{query}\n\n"
        f"以下是本机文件检索到的候选片段（可能相关也可能不相关）：\n{cand_text}\n\n"
        f"请判断哪些片段与查询真正相关，只输出一个 JSON 对象（不要 markdown 代码块、不要多余文字），"
        f"格式：{{\"relevant\": [相关片段编号数组], \"answer\": \"中文精简答案 150 字内\"}}。"
        f"若都不相关，relevant 为空数组、answer 为空字符串。不要编造片段外的内容。"
    )
    try:
        body = {
            "model": LOCAL_QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 8192,
            "stream": False,
        }
        req = urllib.request.Request(
            LOCAL_QWEN_URL.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            out = json.loads(r.read().decode("utf-8"))
        content = out["choices"][0]["message"].get("content", "").strip()
        if not content:
            return None, ""
        # 剥离可能的 markdown 代码块包裹
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)
        # 解析相关编号（容错：接受 int 或数字字符串，1-based）
        relevant: list[int] = []
        for x in data.get("relevant", []) or []:
            try:
                idx = int(x) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and idx not in relevant:
                relevant.append(idx)
        answer = str(data.get("answer", "") or "").strip()
        if not relevant and not answer:
            return None, ""
        return relevant, answer
    except Exception:
        return None, ""


def _auto_ingest_local(query: str, candidates: list[tuple[str, str, int]]) -> int:
    """把本地命中的记录自动纳管到中心库（ingest），返回成功条数。"""
    n = 0
    for path, snippet, _ in candidates:
        entry_id = "local::" + hashlib.md5(path.encode("utf-8")).hexdigest()[:16]
        payload = {
            "entry_id": entry_id,
            "title": "本地记录：" + os.path.basename(path),
            "content": f"[本地文件 {path}]\n{snippet}",
            "source": "device-local-fallback",
            "meta": {"tags": ["本地", "未纳管"], "local_path": path},
        }
        try:
            res = http_call("/ingest/knowledge", payload)
            if isinstance(res, dict) and not res.get("error"):
                n += 1
        except Exception:
            pass
    return n


def _local_fallback(query: str) -> str | None:
    """本地兜底主入口：grep 命中 → qwen 判相关性 → 自动纳管 → 返回结果；无命中返回 None。"""
    if not LOCAL_FALLBACK_ON or not query:
        return None
    cands = _local_grep(query)
    if not cands:
        return None
    relevant, answer = _qwen_rank(query, cands)
    # qwen 未给出明确相关集（失败/判空）时，保守起见全部视为相关（纳管并展示）
    if relevant is None:
        relevant = list(range(len(cands)))
    ingested = _auto_ingest_local(query, [cands[i] for i in relevant])
    lines = [
        f"[本地兜底] 中心库无高置信度命中，已检索当前设备本地，找到 {len(cands)} 条本地记录"
        f"（判相关 {len(relevant)} 条，自动纳管入库 {ingested} 条）：",
        "",
    ]
    for i in relevant:
        path, snippet, cnt = cands[i]
        lines.append(f"【本地 {i + 1}】{path} (命中 {cnt} 词)")
        lines.append(f"  {snippet}")
        lines.append("")
    if answer:
        lines.append("【本地 qwen 相关性判断】" + answer)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 工具执行：MCP tools/call 的落点
# ---------------------------------------------------------------------------
def call_tool(name: str, arguments: dict) -> str:
    """执行单个工具，返回文本结果。"""
    route = TOOL_ROUTES.get(name)
    if not route:
        return f"未知工具: {name}"

    # 会话感知：会话相关工具未传 session_id 时自动注入进程级会话 id，
    # 保证调用被归集到「本 MCP 实例」，监控大屏可实时追踪每个会话的参与度。
    if name in _SESSION_AWARE_TOOLS and not str(arguments.get("session_id", "") or "").strip():
        arguments = dict(arguments)
        arguments["session_id"] = SESSION_ID

    path, _ = route
    # gaps 走 GET 带 query string（limit 可选）
    if name == "molbase_gaps":
        limit = int(arguments.get("limit", 20) or 20)
        data = http_call("/gaps?limit=" + str(limit), None)
    # 无参数的工具（stats/health）走 GET，其余走 POST
    elif name in ("molbase_stats", "molbase_health") and not arguments:
        data = http_call(path, None)
    else:
        data = http_call(path, arguments)

    # 返回结果中存在 error 字段时视为失败
    if isinstance(data, dict) and data.get("error"):
        return "[错误] " + str(data["error"])

    if name == "molbase_query":
        # 本地兜底：中心库 abstain（没数据/不准）或无结果时，检索当前设备本地
        if data.get("abstain") or not data.get("results"):
            local = _local_fallback(arguments.get("query", ""))
            if local:
                return local
        return _format_query_result(data)
    return _format_generic_result(data)


# ---------------------------------------------------------------------------
# MCP stdio 协议层（JSON-RPC 2.0，NDJSON 分帧）
# ---------------------------------------------------------------------------
def _read_message() -> dict | None:
    """
    从 stdin 读取一条 JSON-RPC 消息（单行 JSON）。
    返回 None 表示 EOF（客户端关闭）。
    """
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        # 非 JSON 行（极少见）：忽略，返回空对象避免崩溃
        return {}


def _write_message(msg: dict) -> None:
    """向 stdout 写出一条 JSON-RPC 消息（单行 JSON + 换行）。"""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(*args) -> None:
    """调试日志写 stderr，避免污染 stdout 协议流。"""
    print("[molbase-mcp]", *args, file=sys.stderr, flush=True)


def handle_message(msg: dict) -> dict | None:
    """
    处理一条 JSON-RPC 请求/通知，返回应回写的响应 dict（通知返回 None）。

    msg  : 客户端发来的 JSON-RPC 消息。
    """
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # ---- 握手：initialize ----
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    # ---- 列工具：tools/list ----
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    # ---- 调工具：tools/call ----
    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        try:
            text = call_tool(name, arguments)
            is_error = text.startswith("[错误]")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
            }
        except Exception as e:
            # 任何未预期异常都返回给客户端，不崩进程
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": f"[错误] 工具执行异常：{e}"}], "isError": True},
            }

    # ---- 探活：ping ----
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # ---- 通知类（无 id，不响应）----
    if msg_id is None:
        return None

    # ---- 未知方法 ----
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main() -> None:
    """stdio 主循环：逐行读取请求，处理后回写响应。"""
    _log(f"启动，后端={SERVER_URL}，鉴权={'启用' if AUTH_TOKEN else '关闭'}")
    while True:
        msg = _read_message()
        if msg is None:
            _log("stdin EOF，退出")
            break
        if not msg:
            continue
        resp = handle_message(msg)
        if resp is not None:
            _write_message(resp)


if __name__ == "__main__":
    main()
