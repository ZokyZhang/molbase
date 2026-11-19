# -*- coding: utf-8 -*-
"""
WorkBuddy 会话记录沉淀脚本：把各项目工作区的核心知识文档摄入 MolBase，自动分类打标签。

标签体系（meta.tags）：
    敏感 / 运维 / 环境 / 配置 / 架构 / 规范 / 设计 / 业务 / 测试

设计要点：
- 扫描 `C:/Users/Administrator/WorkBuddy/2026-*/` 下的 md/txt（会话产出物）。
- 排除依赖/构建产物/内部工作目录（node_modules、90-脚本与验证产物、.zwork、代码目录等）。
- 每个文档自动打 1..N 个标签（路径关键词 + 内容关键词），敏感信息识别后加「敏感」标签。
- entry_id = `wb::<项目>::<相对路径去扩展名>`，幂等 upsert，可重复运行。
- meta = {"tags": [...], "project": "...", "path": "...", "sensitive": bool}。

用法：
    python ingest_workbuddy.py            # 增量摄入（upsert）
    python ingest_workbuddy.py --fresh    # 清空 molbase.db 后重新摄入全部语料
    python ingest_workbuddy.py --dry-run  # 只统计不写入

注意：敏感信息采用「明文 + 敏感标签」策略（用户 2026-11-18 拍板），
      检索 `tags=["敏感"]` 可列出全部含密码/密钥/token 的条目，便于后续统一治理。
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from molbase.config import DB_PATH
from molbase.orchestrator import Orchestrator
from molbase.store import Entry

WB_ROOT = Path("C:/Users/Administrator/WorkBuddy")

# ---------------------------------------------------------------------------
# 排除规则：黑名单路径片段（含任一即跳过）
# ---------------------------------------------------------------------------
EXCLUDE_PARTS = [
    ".zwork", ".workbuddy", ".git", "node_modules",
    "_逆向资料/开源参考",
    "90-脚本与验证产物", "90-新版本", "_backup", "99-归档",
    "80-代码-后端", "81-代码-前端", "81-代码-移动端",
    "82-代码-算法服务", "82-代码-设备端",
    "demo", "run", "_tools", "build", "dist", "target",
    "edu-family-snapshot.zip",
]

# 排除文件模式（依赖自带的 README/CHANGELOG/LICENSE 等噪音）
EXCLUDE_FILE_RE = re.compile(
    r"(?i)(^|[/\\])(package[-_]lock|CHANGELOG|LICENSE|NOTICE|AUTHORS|\.npmignore|"
    r"npm-(child|parent)\.log|.*\.min\.md|.*\.map$)"
)

# MolBase 项目自身（已由 ingest.py 摄入，此处跳过）
SKIP_PROJECTS = {"molbase", "2026-11-17-22-17-16"}

# ---------------------------------------------------------------------------
# 标签规则：命中任一关键词即打该标签（对「相对路径 + 内容」匹配）
# 说明：关键词收紧到「主题级」，避免「报告/验收」等通用词导致全命中。
# ---------------------------------------------------------------------------
TAG_RULES: list[tuple[str, list[str]]] = [
    ("运维", ["部署", "运维", "排障", "systemd", "nginx", "docker", "备份",
              "恢复", "重启", "守护", "服务启动", "启动脚本"]),
    ("环境", ["虚拟机", "开发环境", "开发机", "桥接", "DHCP", "内网", "局域网",
              "SSH", "端口映射", "网络拓扑", "路由器", "路由", "OpenWrt", "网关"]),
    ("配置", ["application.yml", "application.properties", "config", "配置项",
              "端口号", "硬编码", "环境变量", "参数化", "密钥管理", "电源方案", "电源"]),
    ("架构", ["系统架构", "架构设计", "模块化", "技术栈", "微服务", "单体",
              "分层", "中间件", "选型", "RAG", "检索架构"]),
    ("规范", ["工作流", "规范", "约定", "红线", "编码规范", "纪律", "职责定义",
              "留痕", "模板", "WF-", "欠账", "注释规范"]),
    ("设计", ["详细设计", "设计文档", "设计契约", "实现说明", "DD-", "ARCH-REV"]),
    ("业务", ["需求提出", "业务规则", "产品", "构思", "积分", "错题", "作业管理",
              "批改", "档案", "兑换", "家校", "家长端", "学生端"]),
]

# ---------------------------------------------------------------------------
# 敏感信息识别：真实密码/密钥/token 赋值模式（值非占位符）
# ---------------------------------------------------------------------------
SENSITIVE_ASSIGN_RE = re.compile(
    r'(?i)(password|passwd|pwd|secret|access[_-]?token|refresh[_-]?token|'
    r'api[_-]?key|appsecret|app[_-]?secret|jwt[_-]?secret|密码|口令|密钥)\s*[:=]\s*["\']?([^"\'\s,;，；]+)["\']?'
)
# 视为「占位符/非真实值」的模式（命中则不判敏感）
PLACEHOLDER_RE = re.compile(
    r'(?i)^(empty|null|none|待填|占位|xxxx+|xxx|example|your_|<>|<|>|\$|'
    r'<redacted>|redacted|\[redacted\]|\[.*\]|.*\$\{.*\}|待定|未填|留空)$'
)
# 文件名即含敏感（ssh 脚本、token 文件、acl 凭据等）
SENSITIVE_FILE_RE = re.compile(r"(?i)(ssh|uptoken|token|_acl|credentials|密钥|凭据)")


def infer_project(rel_parts: list[str], fname: str) -> str:
    """从相对路径推断项目名。"""
    if len(rel_parts) >= 2:
        seg = rel_parts[1]
        # 第二段是目录（非文件）→ 大概率是项目目录名
        if not seg.lower().endswith((".md", ".txt")):
            return seg
    # 第二段是文件 → 从文件名前缀推断
    for prefix in ("family-edu", "family-kb", "mesh-rag", "edu-family", "周易"):
        if fname.startswith(prefix):
            return prefix
    # 时间戳目录下的零散文档 → 用其上级目录做项目
    return rel_parts[0] if rel_parts else "misc"


def classify_tags(rel_path: str, text: str) -> list[str]:
    """按路径 + 内容关键词自动打标签。"""
    haystack = rel_path + "\n" + text
    tags = []
    for tag, kws in TAG_RULES:
        if any(kw.lower() in haystack.lower() for kw in kws):
            tags.append(tag)
    # 「测试」按目录特征判断：路径含「NN-测试」产物目录（如 05-测试）。
    # 只用路径、不看正文，避免正文里引用「05-测试」目录名导致过度命中。
    if any(p.endswith("测试") for p in rel_path.split("/")):
        tags.append("测试")
    return tags


def detect_sensitive(rel_path: str, text: str) -> bool:
    """识别是否含真实敏感信息（文件名或赋值模式命中且值非占位符）。"""
    if SENSITIVE_FILE_RE.search(rel_path):
        return True
    for m in SENSITIVE_ASSIGN_RE.finditer(text):
        value = m.group(2).strip()
        if not value:
            continue
        if PLACEHOLDER_RE.match(value):
            continue
        # 值看起来是真实凭据（长度合理、非纯占位）
        if 2 <= len(value) <= 128:
            return True
    return False


def iter_docs():
    """遍历全部候选文档，产出 (rel_path, project, title, content)。"""
    for wb in sorted(WB_ROOT.glob("2026-*")):
        if wb.name in SKIP_PROJECTS:
            continue
        for md in sorted(wb.rglob("*")):
            if md.suffix.lower() not in (".md", ".txt"):
                continue
            if md.is_dir():
                continue
            rel = md.relative_to(WB_ROOT)
            rel_str = str(rel).replace("\\", "/")
            # 排除黑名单路径片段
            if any(part in rel_str for part in EXCLUDE_PARTS):
                continue
            # 排除依赖噪音文件
            if EXCLUDE_FILE_RE.search(md.name):
                continue
            # MolBase 项目自身（molbase/ 子目录）
            if rel_str.startswith("2026-11-17-22-17-16/"):
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue
            # 标题：首个 # 标题，否则用文件名（去扩展名）
            title = md.stem
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break
            rel_parts = rel_str.split("/")
            project = infer_project(rel_parts, md.name)
            yield rel_str, project, title, text


def main() -> None:
    fresh = "--fresh" in sys.argv
    dry = "--dry-run" in sys.argv

    if fresh:
        try:
            Path(DB_PATH).unlink(missing_ok=True)
            print(f"[fresh] 已删除旧库 {DB_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 删除旧库失败（可能被占用）：{exc}")

    orc = Orchestrator()
    tag_counter: Counter = Counter()
    proj_counter: Counter = Counter()
    sensitive_ids: list[str] = []
    entries: list[Entry] = []
    n = 0

    for rel_str, project, title, text in iter_docs():
        tags = classify_tags(rel_str, text)
        sensitive = detect_sensitive(rel_str, text)
        if sensitive:
            if "敏感" not in tags:
                tags.append("敏感")
        if not tags:
            tags = ["未分类"]
        # 稳定 entry_id：wb::<项目>::<相对路径去扩展名>
        stem = rel_str.rsplit(".", 1)[0].replace("/", "-")
        entry_id = f"wb::{project}::{stem}"
        meta = {
            "tags": tags,
            "project": project,
            "path": rel_str,
            "sensitive": sensitive,
        }
        entries.append(
            Entry(
                entry_type="knowledge",
                entry_id=entry_id,
                title=title,
                content=text,
                source=str(WB_ROOT / rel_str),
                meta=meta,
            )
        )
        n += 1
        for t in tags:
            tag_counter[t] += 1
        proj_counter[project] += 1
        if sensitive:
            sensitive_ids.append(entry_id)

    if dry:
        orc.close()
        print(f"[dry-run] 共扫描 {n} 篇候选文档（未写入）")
    else:
        # 批量 upsert + 一次重建索引（避免逐条 _rebuild 的 O(n²) 开销）
        for e in entries:
            orc.store.upsert(e)
        orc._rebuild()
        print(f"[ok] 已摄入 {n} 篇文档")
        print(f"[stats] {orc.stats()}")
        orc.close()

    print("\n=== 标签分布 ===")
    for tag, cnt in tag_counter.most_common():
        print(f"  {tag}: {cnt}")
    print("\n=== 项目分布 ===")
    for proj, cnt in proj_counter.most_common():
        print(f"  {proj}: {cnt}")
    print(f"\n=== 敏感条目 {len(sensitive_ids)} 条 ===")
    for sid in sensitive_ids:
        print(f"  {sid}")


if __name__ == "__main__":
    main()
