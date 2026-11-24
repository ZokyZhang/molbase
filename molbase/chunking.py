import hashlib
import re


def split_segments(text: str) -> list[tuple]:
    """
    按 Markdown 标题把正文切成「章节段」，返回 [(start_line, end_line, heading, path)]。

    path 是该标题在标题栈中的快照（如 "A/B/C"），用于后续生成可回溯的层级定位；
    代码围栏（``` / ~~~）内的 # 不算标题，避免把注释里的井号误识别成章节。
    """
    if not text:
        return [(0, 1, "", "")]
    
    level_stack = []
    title_stack = []
    lines = text.split("\n")
    total_lines = len(lines)
    
    fence = False
    heading_list = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_fence_toggle = stripped.startswith("```") or stripped.startswith("~~~")
        
        if is_fence_toggle:
            fence = not fence
        
        in_fence = fence
        
        if not in_fence:
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                
                while level_stack and level_stack[-1] >= level:
                    level_stack.pop()
                    title_stack.pop()
                
                level_stack.append(level)
                title_stack.append(title)
                
                snapshot = "/".join(title_stack)
                heading_list.append((i, title, snapshot))
    
    if not heading_list:
        return [(0, total_lines, "", "")]
    
    result = []
    
    if heading_list[0][0] > 0:
        result.append((0, heading_list[0][0], "", ""))
    
    for i in range(len(heading_list)):
        start = heading_list[i][0]
        end = heading_list[i + 1][0] if i + 1 < len(heading_list) else total_lines
        heading = heading_list[i][1]
        path = heading_list[i][2]
        result.append((start, end, heading, path))
    
    return result


def chunk_markdown(text: str, max_chars: int = 1200, overlap: int = 100) -> list[dict]:
    """
    把 Markdown 正文切成带层级定位的检索块，供「原文层」引用回溯。

    切块策略：先按标题分章节，再在章节内按空行分段做贪心装箱（不超过 max_chars）；
    单段超过 max_chars 时硬切并保留 overlap 字符重叠，避免跨块语义被截断。
    章节边界不跨块，保证每个块的 path 唯一。
    """
    if not text or text.strip() == "":
        return []
    
    lines = text.split("\n")
    offsets = [0] * len(lines)
    for i in range(1, len(lines)):
        offsets[i] = offsets[i-1] + len(lines[i-1]) + 1

    chunks = []
    ordinal = 0

    def emit(bs, be, seg_start, seg_text, heading, path):
        nonlocal ordinal
        block = seg_text[bs:be]
        content_hash = hashlib.md5(block.encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
        chunk_id = hashlib.md5((content_hash + ":" + str(ordinal)).encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
        chunks.append({
            'chunk_id': chunk_id,
            'ordinal': ordinal,
            'heading': heading,
            'path': path,
            'text': block,
            'char_start': seg_start + bs,
            'char_end': seg_start + be,
            'content_hash': content_hash
        })
        ordinal += 1

    for (sl, el, heading, path) in split_segments(text):
        seg_start = offsets[sl]
        seg_end = offsets[el-1] + len(lines[el-1]) if el > sl else seg_start
        seg_text = text[seg_start:seg_end]
        if seg_text == "":
            continue
        if len(seg_text) <= max_chars:
            emit(0, len(seg_text), seg_start, seg_text, heading, path)
            continue

        # 1) 构造段落区间：分隔符是连续两个以上换行（可夹空白）
        parts = []
        pos = 0
        for m in re.finditer(r"\n[ \t]*\n+", seg_text):
            if m.start() > pos:
                parts.append((pos, m.start()))
            pos = m.end()
        if pos < len(seg_text):
            parts.append((pos, len(seg_text)))
        parts = [p for p in parts if seg_text[p[0]:p[1]].strip()]

        # 2) 贪心装箱 + 超长段落硬切
        cur_s = None
        cur_e = None
        for (ps, pe) in parts:
            if pe - ps > max_chars:
                if cur_s is not None:
                    emit(cur_s, cur_e, seg_start, seg_text, heading, path)
                    cur_s = cur_e = None
                step = max(1, max_chars - overlap)
                start = ps
                while start < pe:
                    end = min(start + max_chars, pe)
                    emit(start, end, seg_start, seg_text, heading, path)
                    start += step
            else:
                if cur_s is None:
                    cur_s, cur_e = ps, pe
                elif pe - cur_s <= max_chars:
                    cur_e = pe
                else:
                    emit(cur_s, cur_e, seg_start, seg_text, heading, path)
                    cur_s, cur_e = ps, pe
        if cur_s is not None:
            emit(cur_s, cur_e, seg_start, seg_text, heading, path)

    return chunks
