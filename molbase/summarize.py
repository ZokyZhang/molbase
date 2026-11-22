# -*- coding: utf-8 -*-
"""
TextRank 抽取式摘要 + 智能切块（P0，零模型，直接服务「降 token」目标）。

设计说明（对齐「对标 MySQL、自我运行」）：
- TextRank 就是 PageRank 在「句子相似度图」上的应用，零模型、零第三方依赖、确定性；
  作为代工层「最小上下文」的零模型压缩手段，介于 LLM 生成式摘要与原文截断之间。
- 智能切块按段落（空行）优先、句子兜底，把长文本切成可精准投喂的上下文块。
"""

from __future__ import annotations

import re

from .config import LLM_MAX_CONTEXT_CHARS


def split_sentences(text: str) -> list[str]:
    """按中文/英文句末标点、分号、换行切分句子，过滤空句。"""
    parts = re.split(r"[。！？!?；;\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def split_blocks(text: str, max_chars: int = 500) -> list[str]:
    """
    智能切块：段落（空行）是天然边界，每段一块；单段超长才按句子兜底切。

    返回的每个块都是尽量完整的语义单元，供代工层精准定位投喂。
    """
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            result.append(p)
            continue
        # 超长段落按句子兜底切分（句子间用句号分隔，保持可读）
        cur = ""
        for s in split_sentences(p):
            sep = "。" if cur else ""
            if len(cur) + len(sep) + len(s) <= max_chars:
                cur = cur + sep + s
            else:
                if cur:
                    result.append(cur)
                cur = s
        if cur:
            result.append(cur)
    return result


def _tokens(text: str) -> list[str]:
    """复用 index 模块的中英分词（中文 bigram + 英文小写单词）。"""
    from .index import tokenize

    return tokenize(text)


def _similarity(a: str, b: str) -> float:
    """句子相似度：词袋 Jaccard 相似度（零模型，确定性）。"""
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    union = len(ta | tb)
    return len(ta & tb) / union if union else 0.0


def textrank_summarize(
    text: str,
    ratio: float = 0.3,
    min_sentences: int = 1,
    max_sentences: int = 5,
) -> str:
    """
    TextRank 抽取式摘要：句子为节点、句子相似度为边，幂迭代 PageRank 排序，
    取 top 句按原文顺序拼接为摘要。

    - 句子数过少时直接返回原文（无需摘要）。
    - 全程零模型、零第三方依赖、确定性。
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return ""
    top_n = max(min_sentences, min(max_sentences, int(round(n * ratio))))
    if n <= top_n:
        return text

    # 构建相似度矩阵（无向，对称）
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _similarity(sentences[i], sentences[j])
            sim[i][j] = sim[j][i] = s

    # 行归一化：得到转移概率矩阵 P(i -> j)
    for i in range(n):
        total = sum(sim[i])
        if total > 0:
            sim[i] = [x / total for x in sim[i]]

    # 幂迭代 PageRank（damping=0.85）
    d = 0.85
    r = [1.0 / n] * n
    for _ in range(100):
        r_new = [(1.0 - d) / n] * n
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += sim[j][i] * r[j]
            r_new[i] += d * acc
        if sum(abs(r_new[i] - r[i]) for i in range(n)) < 1e-6:
            r = r_new
            break
        r = r_new

    # 取 top 句，按原文顺序拼接
    idx = sorted(range(n), key=lambda i: r[i], reverse=True)[:top_n]
    idx.sort()
    return "。".join(sentences[i] for i in idx) + "。"


# 2. 函数 dedup_noise(text: str) -> str：L1 无损去冗余
def dedup_noise(text: str) -> str:
    """
    L1 无损去冗余：
    - 连续 3 个以上空行压成 1 个空行；
    - 每行首尾空白去除；
    - 重复段落（按空行切段，SimHash 相似度 >= 0.90 视为重复只留首个）；
    - 装饰性分隔线（连续 --- === *** 行压成 1 行）。
    不删任何实质内容。
    """
    lines = text.splitlines()
    cleaned_lines = []
    sep_lines = []
    current_sep_count = 0
    current_sep_line = ""

    def is_sep_line(line: str) -> bool:
        return bool(re.match(r"^([\-_=*]{3,})\s*$", line.strip()))

    def tokenize(text: str) -> list[str]:
        # 修正：匹配中英文及数字，解决中文去重失效问题
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text)

    def is_similar(a: str, b: str) -> bool:
        # 简易 SimHash 模拟：分词后集合相似度
        ta = set(tokenize(a))
        tb = set(tokenize(b))
        if not ta or not tb:
            return False
        union = len(ta | tb)
        if union == 0:
            return False
        return len(ta & tb) / union >= 0.90

    def is_duplicate_paragraph(a: str, b: str) -> bool:
        # 按空行切段后的段落比较
        pa = re.split(r"\n\s*\n", a)
        pb = re.split(r"\n\s*\n", b)
        # 取最大长度段比较
        max_pa = max(pa, key=len) if pa else ""
        max_pb = max(pb, key=len) if pb else ""
        return is_similar(max_pa, max_pb)

    # 第一遍：处理空行和分隔线
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            # 处理连续空行
            if current_sep_count >= 3:
                # 已经压过了，跳过
                pass
            elif current_sep_count == 2:
                # 前面已经有 2 个空行，当前是第 3 个，压成 1 个
                if cleaned_lines and cleaned_lines[-1] == "":
                    pass
                else:
                    cleaned_lines.append("")
                current_sep_count = 0
            else:
                current_sep_count += 1
                if current_sep_count == 1:
                    cleaned_lines.append("")
            i += 1
            continue

        # 处理分隔线
        if is_sep_line(stripped):
            if current_sep_count > 0:
                # 连续分隔线，压成一行
                if not current_sep_line:
                    current_sep_line = stripped
                else:
                    current_sep_line += " " + stripped
            else:
                # 单行分隔线，直接加入
                current_sep_line = stripped
            current_sep_count += 1
            i += 1
            continue

        # 非空非分隔线
        if current_sep_count > 0:
            # 分隔线后接内容，结束分隔线序列
            if current_sep_line:
                cleaned_lines.append(current_sep_line)
                current_sep_line = ""
            current_sep_count = 0

        # 去除首尾空白
        cleaned_lines.append(stripped)
        i += 1

    # 处理末尾可能的分隔线
    if current_sep_line:
        cleaned_lines.append(current_sep_line)

    # 第二遍：去重段落
    result_lines = []
    seen_paragraphs = []
    current_para = []

    for line in cleaned_lines:
        if line == "":
            # 段落结束
            if current_para:
                para_text = "\n".join(current_para)
                if not seen_paragraphs or not is_duplicate_paragraph(para_text, seen_paragraphs[-1]):
                    seen_paragraphs.append(para_text)
                    result_lines.extend(current_para)
                current_para = []
            else:
                result_lines.append(line)
        else:
            current_para.append(line)

    # 处理最后一段
    if current_para:
        para_text = "\n".join(current_para)
        if not seen_paragraphs or not is_duplicate_paragraph(para_text, seen_paragraphs[-1]):
            seen_paragraphs.append(para_text)
            result_lines.extend(current_para)

    return "\n".join(result_lines)


_COMPACT_SYSTEM = (
    "你是 MolBase 的语义保持摘要器。把下方文本压缩成紧凑的结构化摘要，硬约束：\n"
    "1. 硬事实逐字保留（IP、端口、命令、版本号、路径、账号、数字、代码块、表格、key:value 配置），"
    "一个字都不许改、漏、缩写；\n"
    "2. 只压缩软修辞：过渡句、解释性文字、客套话压成一句要点；完全相同或近义的重复句只保留一句；\n"
    "3. 保留 Markdown 标题层级、表格和代码块框架；\n"
    "4. 输出总字数必须严格不超过 {max_chars} 字。\n"
    "只输出摘要本身，不要任何解释、前后缀或客套话。"
)


def extract_facts(text: str) -> list[str]:
    """
    L2 硬事实抽取：用正则抽出 4 类明确硬事实返回去重列表。
    类型包括：IPv4、反引号命令、三反引号代码块、版本号。
    不再抽取端口、路径、key:value 等易误匹配项。
    """
    facts = set()

    # 1. IPv4
    ipv4_pattern = r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
    for m in re.finditer(ipv4_pattern, text):
        facts.add(m.group())

    # 2. 反引号命令（含反引号）
    cmd_pattern = r"`[^`]+`"
    for m in re.finditer(cmd_pattern, text):
        facts.add(m.group())

    # 3. 三反引号代码块
    code_block_pattern = r"```[\s\S]*?```"
    for m in re.finditer(code_block_pattern, text):
        facts.add(m.group())

    # 4. 版本号
    version_pattern = r"\b\d+\.\d+(?:\.\d+)?\b"
    for m in re.finditer(version_pattern, text):
        facts.add(m.group())

    return sorted(list(facts))


def semantic_preserve(text: str, max_chars: int = 500, entry_id: str = "", llm=None) -> str:
    """
    主入口：语义保持压缩。
    逻辑：
    - 先 dedup_noise(text)，若结果长度 <= max_chars 直接返回；
    - 若 llm 非 None，调用 LLM 生成摘要；
    - 校验硬事实：若摘要中无缺失硬事实，返回摘要；
    - 兜底：若 LLM 不可用、摘要为空或存在缺失硬事实，直接返回去冗余原文（不截断、不追加碎片、不追加 textrank）。
    """
    # 1. 去冗余
    cleaned_text = dedup_noise(text)

    # 2. 长度检查
    if len(cleaned_text) <= max_chars:
        if entry_id:
            return f"{cleaned_text}\n[原文: {entry_id}]"
        return cleaned_text

    # 3. LLM 摘要（超长文本跳过 LLM，避免超上下文窗口被截断丢语义）
    summary = ""
    if llm is not None and len(cleaned_text) <= LLM_MAX_CONTEXT_CHARS:
        try:
            system_prompt = _COMPACT_SYSTEM.format(max_chars=max_chars)
            summary = llm.chat(system_prompt, cleaned_text)
            if not summary:
                summary = ""
        except Exception:
            summary = ""

    # 4. 硬事实校验与返回
    if summary:
        # 摘要膨胀守卫：摘要不比去冗余原文明显更短（>=90%），视为压缩失败，回退原文
        if len(summary) >= len(cleaned_text) * 0.9:
            summary = ""
        else:
            facts = extract_facts(cleaned_text)
            missing = [f for f in facts if f not in summary]
            if not missing:
                # 关键硬事实零丢失，直接返回摘要
                if entry_id:
                    return f"{summary}\n[原文: {entry_id}]"
                return summary

    # 5. 兜底逻辑：返回去冗余原文，保证语义 100% 不变
    if entry_id:
        return f"{cleaned_text}\n[原文: {entry_id}]"
    return cleaned_text
