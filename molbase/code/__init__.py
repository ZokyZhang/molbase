# -*- coding: utf-8 -*-
"""
代码代工模块：AST 符号索引 + PageRank 关联方法定位。

职责（代工层 = 编译/反编译代工）：
- 用 Python 标准库 ast 解析源码，抽取「符号」（函数/类/方法）及其调用关系。
- 构建符号索引：给定符号名，可精准定位源码片段（反编译代工）。
- 构建调用图，供关联方法推荐。

设计说明：
- 复用标准库 ast，零外部依赖（生产可换 tree-sitter 支持多语言）。
- 每个符号生成一个 code 类型的 Entry，内容 = 该符号的源码片段。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..store import Entry


@dataclass
class Symbol:
    """一个代码符号（函数/类/方法）。"""

    name: str               # 符号名
    qualified_name: str     # 完整限定名（含类名）
    source: str             # 源码片段
    file_path: str          # 所在文件
    lineno: int             # 起始行号
    calls: list[str]        # 被调用的符号名列表
    entry_id: str           # 生成的知识库条目 ID


class CodeIndexer:
    """Python 源码 AST 符号索引器。"""

    def __init__(self, repo_dir: Path | str) -> None:
        """初始化，指定代码仓库目录。"""
        self.repo_dir = Path(repo_dir)

    def index_file(self, file_path: Path | str) -> list[Symbol]:
        """解析单个 Python 文件，返回其中全部顶层符号。"""
        path = Path(file_path)
        source_text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return []  # 无法解析则跳过

        symbols: list[Symbol] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym = self._symbol_from_func(node, source_text, str(path), None)
                if sym:
                    symbols.append(sym)
            elif isinstance(node, ast.ClassDef):
                # 类本身 + 类内方法
                symbols.append(
                    Symbol(
                        name=node.name,
                        qualified_name=node.name,
                        source=ast.get_source_segment(source_text, node) or "",
                        file_path=str(path),
                        lineno=node.lineno,
                        calls=[],
                        entry_id=f"code::{path.name}::{node.name}",
                    )
                )
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sym = self._symbol_from_func(
                            sub, source_text, str(path), node.name
                        )
                        if sym:
                            symbols.append(sym)
        return symbols

    def _symbol_from_func(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_text: str,
        file_path: str,
        class_name: Optional[str],
    ) -> Optional[Symbol]:
        """从函数节点抽取 Symbol。"""
        qualified = f"{class_name}.{node.name}" if class_name else node.name
        calls: list[str] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    calls.append(sub.func.attr)
        return Symbol(
            name=node.name,
            qualified_name=qualified,
            source=ast.get_source_segment(source_text, node) or "",
            file_path=file_path,
            lineno=node.lineno,
            calls=calls,
            entry_id=f"code::{Path(file_path).name}::{qualified}",
        )

    def to_entries(self, symbols: list[Symbol]) -> list[Entry]:
        """把符号列表转成知识库 code 类型条目。"""
        entries: list[Entry] = []
        for sym in symbols:
            entries.append(
                Entry(
                    entry_type="code",
                    entry_id=sym.entry_id,
                    title=sym.qualified_name,
                    content=f"# {sym.qualified_name}\n{sym.source}",
                    source=f"{sym.file_path}:{sym.lineno}",
                    meta={"calls": sym.calls, "file": sym.file_path},
                )
            )
        return entries

    def build_call_graph(self, symbols: list[Symbol]) -> list[tuple[str, str]]:
        """
        构建调用图边列表：返回 (caller_entry_id, callee_entry_id) 二元组。
        用 qualified_name 反查 symbol 得到 entry_id。
        """
        name_to_id: dict[str, str] = {}
        for sym in symbols:
            name_to_id[sym.qualified_name] = sym.entry_id
            name_to_id[sym.name] = sym.entry_id  # 允许短名匹配

        edges: list[tuple[str, str]] = []
        for sym in symbols:
            for callee in sym.calls:
                callee_id = name_to_id.get(callee)
                if callee_id and callee_id != sym.entry_id:
                    edges.append((sym.entry_id, callee_id))
        return edges
