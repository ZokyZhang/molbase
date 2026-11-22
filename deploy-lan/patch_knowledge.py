# -*- coding: utf-8 -*-
"""
派工脚本：把「缺失/不准」的 4 类知识，交由本地 qwen3.6-27b 生成正文，
再由本脚本 ingest 落盘到虚拟机 MolBase 中心库（10.0.0.200:8787）。

本脚本属「编排落盘」层（机械操作）：派工单 -> 调 qwen -> 拿回正文 -> ingest。
知识条目正文由 qwen 生成，非本脚本撰写。
"""
import json
import urllib.request

QWEN_URL = "http://127.0.0.1:11434/v1/chat/completions"
KB_URL = "http://10.0.0.200:8787/ingest/knowledge"

# ---------- 派工单：事实材料（编排者提供，qwen 负责组织成文） ----------
MATERIALS = {
    "qwen-worker": {
        "entry_id": "kb::qwen-worker-local-inference",
        "title": "qwen-worker 本地推理协议（外部 agent 分工 + qwen3.6-27b 执行）",
        "facts": (
            "分工硬约束（2026-11-21 起生效，取代已作废的 AutoClaw 编码委派 PIT-07-02）："
            "外部/云端 agent 只做分析、规划、评审、编排调度；一切实际产出（写代码、写文档/报告、翻译、内容生成）"
            "必须交由本地 qwen 执行。"
            "本地执行者：Ollama 服务，地址 http://127.0.0.1:11434（本机直连，硬编码不加密）；"
            "主模型 qwen3.6-27b-q3（26.9B，Q3_K_M 量化，ctx 8192）；备用模型 qwen3.5-9b；"
            "接口为 OpenAI 兼容 POST /v1/chat/completions；embedding 用 bge-m3（仅检索/向量，非生成）。"
            "调用参数：写代码/事实类 temperature=0.2，文档/报告 0.4，内容创作 0.7，max_tokens=4096，stream=false。"
            "约束：单次派工合计约 8000 token 以内，大任务拆段派工；外部 agent 不得自己写实现代码、写文档正文，首稿即交 qwen。"
        ),
        "tags": ["架构", "规范", "配置"],
    },
    "autoclaw-deprecated": {
        "entry_id": "kb::autoclaw-deprecated-migration",
        "title": "AutoClaw 编码委派约束已作废（迁移到 qwen-worker）",
        "facts": (
            "AutoClaw 编码委派约束（编号 PIT-07-02）自 2026-11-21 起作废，不再委派 AutoClaw 承接编码。"
            "历史背景：edu-family 项目的软件设计、编码和测试曾由 AutoClaw 承接，按风险分级推进——"
            "纯样式小改可直接实施，涉及接口、删除或合规的高风险改动须先确认。"
            "替代方案：新规则改为所有外部/云端 agent 只做分析、规划、评审、编排调度，一切实际产出"
            "（写代码、写文档/报告、翻译、内容生成）必须交由本地 Ollama 上的 qwen3.6-27b 执行，"
            "完整协议见 skill qwen-worker（~/.workbuddy/skills/qwen-worker/SKILL.md）。"
        ),
        "tags": ["架构", "规范"],
    },
    "molbase-source-structure": {
        "entry_id": "kb::molbase-source-structure",
        "title": "MolBase 源码内部结构（20 个 Python 模块职责）",
        "facts": (
            "MolBase 项目位于开发机 WorkBuddy/2026-11-17-22-17-16/molbase，纯标准库零第三方依赖，约 4154 行。"
            "核心模块：orchestrator.py(1466行,代工层编排器统一入口:意图路由/召回/重排/反馈)；"
            "store/__init__.py(869行,SQLite 存储底座,管理 knowledge/code/skill 三类条目与热度新鲜度信号)；"
            "config.py(336行,全局配置,所有参数硬编码不占位符不加密)；server.py(386行,HTTP API 层,标准库 http.server)。"
            "子包：retrieval(混合召回 BM25+关联图,三信号重排)；index(倒排索引+BM25F 字段加权,中文词典分词+FMM)；"
            "graph(关联图,纯 Python)；code(AST 符号索引+PageRank 关联方法定位)；skills(SKILL.md 解析注册)。"
            "辅助模块：embedding.py(HashEmbedding 确定性哈希向量+可选增强插件)；evolution.py(自我进化:热度/共现建边/冷衰减)；"
            "llm.py(本地 LLM 接入层:意图路由/改写/重排/摘要/技能生成)；summarize.py(TextRank 抽取式摘要)；"
            "tokens.py(token 粗估)；cost.py(token 折算人民币成本)；sanitize.py(PII 正则脱敏)；"
            "profiler.py(逐环节 token/成本/耗时剖析)；websearch.py(网络搜索外部源适配器)。"
        ),
        "tags": ["架构", "设计", "运维"],
    },
    "molbase-mcp-config": {
        "entry_id": "kb::molbase-mcp-config",
        "title": "MolBase MCP Server 配置（stdio + mcp.json + 环境变量）",
        "facts": (
            "MCP Server 为单文件 molbase_mcp.py（零依赖，Python 标准库实现 stdio 协议），"
            "内部 HTTP 转发到中心知识库。工具集（9 个）：molbase_query/molbase_ingest_knowledge/"
            "molbase_feedback/molbase_harvest/molbase_tooltrace/molbase_summarize/molbase_stats/"
            "molbase_health/molbase_gaps/molbase_evolve。"
            "WorkBuddy 接入：在 ~/.workbuddy/mcp.json 的 mcpServers 加 molbase 段，"
            "command=python，args=[molbase_mcp.py 的绝对路径]，env.MOLBASE_URL=http://10.0.0.200:8787。"
            "中心服务器：虚拟机 10.0.0.200:8787（监听 0.0.0.0，systemd 常驻 molbase-lan.service），受信内网不鉴权。"
            "接入后需在 WorkBuddy 连接器管理页对 molbase 点「信任」才生效。"
        ),
        "tags": ["配置", "运维", "环境"],
    },
}


def call_qwen(single: dict) -> dict:
    """派工：让 qwen 基于单条事实材料生成 1 条知识条目（严格 JSON 对象）。"""
    prompt = (
        "你是知识库内容整理助手。请根据下面的「事实材料」生成 1 条知识条目。\n"
        "要求：\n"
        "1. 只做忠实转写与通顺组织，不虚构、不补充材料外的事实；\n"
        "2. content 用中文，250~400 字，保留材料里的所有硬参数（地址/端口/模型名/编号）；\n"
        "3. 输出一个严格 JSON 对象（不要 markdown 代码块包裹，不要多余文字），"
        "字段为 entry_id、title、content、tags（tags 为字符串数组）。\n\n"
        "事实材料如下：\n" + json.dumps(single, ensure_ascii=False, indent=1)
    )
    body = {
        "model": "qwen3.6-27b-q3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 4096,
        "stream": False,
    }
    req = urllib.request.Request(
        QWEN_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"]


def parse_entry(raw: str) -> dict:
    """解析 qwen 返回的单个 JSON 对象（剥离可能的 markdown 包裹）。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def ingest(entry: dict) -> dict:
    """机械落盘：把 qwen 生成的条目 ingest 到虚拟机中心库。"""
    payload = {
        "entry_id": entry["entry_id"],
        "title": entry["title"],
        "content": entry["content"],
        "source": "qwen-worker 生成 2026-11-21",
        "meta": {"tags": entry.get("tags", [])},
    }
    req = urllib.request.Request(
        KB_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    for key, single in MATERIALS.items():
        print(f"[{key}] 派工 qwen 生成正文...")
        try:
            raw = call_qwen(single)
            entry = parse_entry(raw)
            res = ingest(entry)
            print(f"  ✓ {entry['entry_id']} -> {res.get('state', res)}")
        except Exception as ex:
            print(f"  ✗ [{key}] 失败: {ex}")
            print("    原始返回:", (raw[:200] if 'raw' in dir() else ''))


if __name__ == "__main__":
    main()
