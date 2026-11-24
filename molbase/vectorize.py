import hashlib
import math

def model_fingerprint(embedder) -> str:
    """
    返回该 embedding 后端的稳定指纹字符串，格式为 "<name>:<model>:<dim>"。
    """
    name = getattr(embedder, "name", "unknown")
    model = getattr(embedder, "model", "")
    dim = getattr(embedder, "dim", 0)
    return f"{name}:{model}:{dim}"


def semantic_text(title: str, content: str, summary: str = "", max_chars: int = 300) -> str:
    """
    构造用于向量化的语义文本，总长度不超过 max_chars。

    约定：title 与 summary 优先完整保留，只有剩余额度才分配给 content。
    默认 300 与 SEMANTIC_SIM_CONTENT_CHARS 对齐——三条调用路径（预计算入库、
    现场精排、预计算重排）必须传同一口径，否则同一对查询-条目会算出不同余弦。
    """
    if content is None:
        content = ""
    
    parts = []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if content:
        parts.append(content)
    
    # 计算 title 和 summary 占用的长度（包括换行符）
    fixed_length = 0
    if parts:
        # 构建固定部分：title + "\n" + summary (if exists) + "\n"
        # 注意：如果只有 title，没有 summary 和 content，则不需要最后的换行
        # 但为了统一逻辑，我们先构建前缀
        
        prefix_parts = []
        if title:
            prefix_parts.append(title)
        if summary:
            prefix_parts.append(summary)
            
        prefix_str = "\n".join(prefix_parts)
        fixed_length = len(prefix_str)
        
        # 如果还有 content，需要预留一个换行符
        if content:
            fixed_length += 1 # for the newline before content

    # 计算 content 允许的最大长度
    available_for_content = max_chars - fixed_length
    
    truncated_content = ""
    if content and available_for_content > 0:
        if len(content) > available_for_content:
            truncated_content = content[:available_for_content]
        else:
            truncated_content = content
            
    # 重新组装
    final_parts = []
    if title:
        final_parts.append(title)
    if summary:
        final_parts.append(summary)
    if truncated_content:
        final_parts.append(truncated_content)
        
    return "\n".join(final_parts)


def ensure_entry_vectors(store, embedder, kind: str = "entry", force: bool = False,
                         batch_size: int = 16, entry_ids: list | None = None) -> dict:
    """
    保证条目向量已预计算（增量：只补 text_hash 变化或缺失的条目）。

    entry_ids 给定时只处理这些条目（入库单条后调用，避免全库扫描）；
    为 None 时处理全库（离线回填 / 启动自愈）。

    返回 {"fp","total","computed","skipped","failed"}，供调用方观测覆盖率。
    """
    from .config import SEMANTIC_SIM_CONTENT_CHARS

    fp = model_fingerprint(embedder)
    if entry_ids is None:
        entries = store.all_entries()
    else:
        wanted = set(entry_ids)
        entries = [e for e in store.all_entries() if e.entry_id in wanted]

    existing = {} if force else store.get_vector_hashes(fp, kind)
    to_compute = []
    skipped = 0
    for entry in entries:
        # 语义文本窗口统一取 SEMANTIC_SIM_CONTENT_CHARS（300）：
        # 实测 1500 字窗口会让"内容泛化的大文档"胜过"精准的小文档"（真实问题被挤下 top1），
        # 长文档的精准命中交给切块级检索（chunking）后续版本解决。
        text = semantic_text(entry.title, entry.content or "", entry.summary or "",
                             max_chars=SEMANTIC_SIM_CONTENT_CHARS)
        text_hash = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        if not force and existing.get(entry.entry_id) == text_hash:
            skipped += 1
            continue
        to_compute.append((entry.entry_id, text, text_hash))

    computed = 0
    failed = 0
    for i in range(0, len(to_compute), batch_size):
        batch = to_compute[i:i + batch_size]
        try:
            vectors = embedder.embed_batch([b[1] for b in batch])
            # 整批一次性写入（单事务提交），避免每条向量一次 commit 拖慢入库
            rows = []
            for j, (entry_id, _t, text_hash) in enumerate(batch):
                if j < len(vectors) and vectors[j]:
                    rows.append((entry_id, kind, fp, vectors[j], text_hash))
            computed += store.upsert_vectors(rows)
        except Exception:
            failed += len(batch)

    return {"fp": fp, "total": len(entries), "computed": computed, "skipped": skipped, "failed": failed}

def cosine_topn(query_vec: list, vectors: dict, top_n: int = 10) -> list:
    """
    在已归一化向量集合里做余弦检索。
    """
    if not query_vec or not vectors:
        return []
    
    query_len = len(query_vec)
    results = []
    
    for owner_id, vec in vectors.items():
        if len(vec) != query_len:
            continue
        
        # 计算点积 (假设已 L2 归一化，点积即余弦相似度)
        score = sum(q * v for q, v in zip(query_vec, vec))
        results.append((owner_id, score))
    
    # 按 score 降序排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_n]