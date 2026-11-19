# -*- coding: utf-8 -*-
"""
技能代管模块：SKILL.md 解析与注册。

职责：
- 解析 agentskills.io 标准的 SKILL.md（YAML frontmatter + Markdown 正文）。
- 把技能注册为 skill 类型条目，由知识库代管。
- 新场景产生的新 skill 优先生成写入知识库（代管）。

设计说明：
- frontmatter 采用手写解析器解析 YAML 的 name/description 两个必填字段，
  避免引入 PyYAML 重依赖；如需完整 YAML 可替换为 yaml.safe_load。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..store import Entry


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


@dataclass
class SkillMeta:
    """SKILL.md 的元数据。"""

    name: str
    description: str
    body: str          # Markdown 正文
    raw: str           # 原始内容


def parse_skill_md(text: str) -> SkillMeta:
    """
    解析 SKILL.md 文本：
    - 提取 YAML frontmatter 中的 name 和 description；
    - 其余部分作为 body。
    """
    m = _FRONTMATTER_RE.match(text.strip())
    if not m:
        raise ValueError("SKILL.md 缺少 YAML frontmatter（--- 包裹的 name/description）")

    frontmatter, body = m.group(1), m.group(2)
    name = ""
    description = ""
    for line in frontmatter.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[len("name:") :].strip().strip("\"'")
        elif line.startswith("description:"):
            description = line[len("description:") :].strip().strip("\"'")

    if not name:
        raise ValueError("SKILL.md frontmatter 缺少必填字段 name")
    return SkillMeta(name=name, description=description, body=body.strip(), raw=text)


class SkillRegistry:
    """技能注册表：负责技能包的发现、解析、入库代管。"""

    def __init__(self, skills_dir: Path | str) -> None:
        """初始化，指定技能包目录。"""
        self.skills_dir = Path(skills_dir)

    def discover(self) -> list[Path]:
        """发现目录下所有 SKILL.md 文件。"""
        return sorted(self.skills_dir.rglob("SKILL.md"))

    def load_entry(self, skill_path: Path | str) -> Entry:
        """解析单个 SKILL.md 并转成 skill 类型条目。"""
        path = Path(skill_path)
        text = path.read_text(encoding="utf-8")
        meta = parse_skill_md(text)
        # 用技能名做稳定 ID，避免路径变动导致重复
        entry_id = f"skill::{meta.name}"
        return Entry(
            entry_type="skill",
            entry_id=entry_id,
            title=meta.name,
            content=f"{meta.description}\n\n{meta.body}",
            source=str(path),
            meta={"skill_name": meta.name, "description": meta.description},
        )

    def load_all(self) -> list[Entry]:
        """发现并加载全部技能条目。"""
        return [self.load_entry(p) for p in self.discover()]
