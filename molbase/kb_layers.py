"""
知识库四层模型的落地工具（v9.1）。

M olBase 的知识库不是「一堆文档块」，而是四层互相咬合的结构：

    树（topic_tree）   —— 导航层：我从哪个主题找到你
    网（cooccur/PMI）  —— 关联层：你和谁一起出现，权重有多可信
    标签（tag_vocab）  —— 路由层：受控词表，把自由标签收敛成可统计的概念
    原文（chunks）     —— 溯源层：一句话答案能精确还原到正文第几段

本模块只放「把条目加工成这四层」的纯函数与加工入口，
不做检索决策——检索与闸门仍在 orchestrator，避免职责混淆。
"""

import hashlib

from .chunking import chunk_markdown
from .config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS

# 标签同义词归一表（硬编码，不做外部依赖）：
# 左值是实际出现过的写法，右值是受控词表的规范名。
TAG_ALIAS = {
    "部署": "运维",
    "上线": "运维",
    "发布": "运维",
    "环境": "运维",
    "配置": "运维",
    "数据库": "数据",
    "存储": "数据",
    "检索": "RAG",
    "召回": "RAG",
    "向量": "RAG",
    "精排": "RAG",
    "重排": "RAG",
    "大模型": "LLM",
    "模型": "LLM",
    "提示词": "LLM",
    "backend": "后端",
    "frontend": "前端",
}


def normalize_tag(tag: str) -> str:
    """把自由标签收敛成受控标签：去空白 + 同义词映射 + 统一小写（中文不受影响）。"""
    key = str(tag or "").strip()
    if not key:
        return ""
    return TAG_ALIAS.get(key, key).lower()


def normalize_tags(tags) -> list[str]:
    """批量归一化并去重，保持首次出现顺序（顺序会影响路由提示词的可读性）。"""
    if isinstance(tags, str):
        tags = [tags]
    out: list[str] = []
    seen = set()
    for t in tags or []:
        n = normalize_tag(t)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _chunk_id(entry_id: str, ordinal: int, text_hash: str) -> str:
    """
    生成稳定 chunk_id。

    用「条目 id + 序号 + 内容哈希」三者派生：内容变了 id 就变，
    旧的块向量自然失效，不会出现「新正文配旧向量」的错位。
    """
    raw = f"{entry_id}:{ordinal}:{text_hash}"
    return "chunk::" + hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def build_entry_chunks(entry, max_chars: int = CHUNK_MAX_CHARS,
                       overlap: int = CHUNK_OVERLAP_CHARS) -> list[dict]:
    """
    把条目正文切成可直接落库的块列表。

    返回每项含 chunk_id / ordinal / heading / path / char_start / char_end /
    text_len / body / text_hash。正文为空时返回空列表（不产生零长块）。
    """
    body = entry.content or ""
    if not body.strip():
        return []
    raw_chunks = chunk_markdown(body, max_chars=max_chars, overlap=overlap)
    out = []
    for c in raw_chunks:
        text = c.get("text") or c.get("body") or ""
        if not text.strip():
            continue
        text_hash = hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        ordinal = int(c.get("ordinal", 0))
        out.append({
            "chunk_id": _chunk_id(entry.entry_id, ordinal, text_hash),
            "ordinal": ordinal,
            "heading": c.get("heading") or "",
            "path": c.get("path") or "",
            "char_start": int(c.get("char_start", 0)),
            "char_end": int(c.get("char_end", 0)),
            "text_len": len(text),
            "body": text,
            "text_hash": text_hash,
        })
    return out


def index_entry_chunks(store, embedder, entry, max_chars: int = CHUNK_MAX_CHARS,
                       overlap: int = CHUNK_OVERLAP_CHARS, with_vectors: bool = True) -> dict:
    """
    加工单个条目的原文层与块向量，返回统计。

    流程：切块 -> 落 chunks 表 -> 逐块算向量写入 vector_index(owner_kind="chunk")。
    块向量用于「长文档精准命中」：条目级只取正文前 N 字，长文档的答案常在正文后段，
    块级向量能把后段的段落单独捞出来比对。
    """
    from .vectorize import model_fingerprint

    chunks = build_entry_chunks(entry, max_chars=max_chars, overlap=overlap)
    saved = store.save_chunks(entry.entry_id, chunks)

    vectors = 0
    if with_vectors and chunks and embedder is not None:
        try:
            fp = model_fingerprint(embedder)
            # 块向量按「标题 + 块正文」构造：只给块正文会丢掉它属于哪篇文档的语境
            texts = [f"{entry.title}\n{c['body']}" for c in chunks]
            vecs = embedder.embed_batch(texts)
            rows = []
            for c, v in zip(chunks, vecs):
                if v:
                    rows.append((c["chunk_id"], "chunk", fp, v, c["text_hash"]))
            vectors = store.upsert_vectors(rows)
        except Exception:
            vectors = 0

    return {"entry_id": entry.entry_id, "chunks": saved, "vectors": vectors}


def chunk_vectors_for(store, embedder, entry_ids: list) -> dict:
    """
    取一批条目的块向量，返回 {entry_id: [(chunk_id, vec)]}。

    只对正文较长的条目取块向量：短文档的条目级向量本就覆盖全文，
    再算块级没有增量信息，纯属浪费比对开销。
    """
    from .vectorize import model_fingerprint

    if not entry_ids or embedder is None:
        return {}
    try:
        fp = model_fingerprint(embedder)
    except Exception:
        return {}

    out: dict[str, list] = {}
    for entry_id in entry_ids:
        chunks = store.get_chunks(entry_id)
        if not chunks:
            continue
        ids = [c["chunk_id"] for c in chunks]
        vecs = store.get_vectors(fp, "chunk", owner_ids=ids)
        pairs = [(cid, vecs[cid]) for cid in ids if cid in vecs]
        if pairs:
            out[entry_id] = pairs
    return out


def max_chunk_similarity(q_vec: list, pairs: list) -> float:
    """
    算查询向量与某条目所有块的最高余弦（归一化后取最大值）。

    为什么取 max 而不是 mean：一条命中只需要「有一个段落答得上」，
    取均值会被同一文档里的无关段落稀释，反而惩罚了长文档。
    """
    if not q_vec or not pairs:
        return 0.0
    q_norm = sum(a * a for a in q_vec) ** 0.5 or 1.0
    best = 0.0
    for _cid, vec in pairs:
        if not vec or len(vec) != len(q_vec):
            continue
        v_norm = sum(b * b for b in vec) ** 0.5 or 1.0
        sim = sum(a * b for a, b in zip(q_vec, vec)) / (q_norm * v_norm)
        if sim > best:
            best = sim
    return max(0.0, min(1.0, best))


def build_topic_tree_nodes(entries, max_children: int = 200) -> list[dict]:
    """
    从条目集生成两层主题树节点：根层 = 受控标签，子层 = 该标签下的条目标题。

    只做两层是刻意的选择：三层以上需要真实聚类依据（现在没有），
    硬凑出来的层级会让导航变成噪音。标签层已经能回答「这类知识我都存了什么」。
    无标签的条目归入「未分类」，保证树能覆盖全库、不留孤儿。
    """
    by_tag: dict[str, list] = {}
    for e in entries:
        tags = normalize_tags(e.meta.get("tags") or [])
        if not tags:
            tags = ["未分类"]
        for t in tags:
            by_tag.setdefault(t, []).append(e.title or e.entry_id)

    nodes = []
    for tag in sorted(by_tag.keys()):
        nodes.append({"topic_name": tag, "parent_name": "", "level": 1, "scene_id": ""})
        titles = sorted(set(by_tag[tag]))[:max_children]
        for title in titles:
            nodes.append({"topic_name": title, "parent_name": tag, "level": 2, "scene_id": ""})
    return nodes


def rebuild_layers(store, embedder=None, entries=None, with_vectors: bool = False,
                   batch_size: int = 64) -> dict:
    """
    一次性重建四层（标签/树/原文），返回各层规模。

    幂等：标签与树全量重建，原文层 clear 后全量重写。
    写入路径刻意做成「先全部算好、再一次性落库」：逐条目 commit 在 730 条目规模下
    会把 12000+ 块的写入拖到数分钟（每次 commit 一次 fsync），批量后只需一次提交。
    with_vectors=False 时只切块不算向量（大库可先建结构，向量另跑离线回填）。
    """
    from .vectorize import model_fingerprint

    all_entries = entries if entries is not None else store.all_entries()

    # 1) 标签词表：受控归一化 + 文档频次
    vocab = store.rebuild_tag_vocab(TAG_ALIAS)

    # 2) 主题树：两层（标签 -> 标题）
    tree_nodes = build_topic_tree_nodes(all_entries)
    tree_count = store.rebuild_topic_tree(tree_nodes)

    # 3) 原文层：全部条目切块，收集后一次落库
    all_chunks: list[dict] = []
    fp = ""
    if with_vectors and embedder is not None:
        try:
            fp = model_fingerprint(embedder)
        except Exception:
            fp = ""

    for e in all_entries:
        chunks = build_entry_chunks(e)
        for c in chunks:
            all_chunks.append({**c, "entry_id": e.entry_id})

    chunk_total = store.save_all_chunks(all_chunks, clear=True)

    # 4) 块向量：分批 embed，边算边写（单批失败只跳过该批，不拖垮整轮重建）
    if fp and all_chunks:
        title_map = {e.entry_id: (e.title or "") for e in all_entries}
        pending: list[tuple] = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            try:
                texts = [f"{title_map.get(c['entry_id'], '')}\n{c['body']}" for c in batch]
                vecs = embedder.embed_batch(texts)
                for c, v in zip(batch, vecs):
                    if v:
                        pending.append((c["chunk_id"], "chunk", fp, v, c["text_hash"]))
            except Exception:
                # 单批 embed 失败（超时/维度异常）只丢这一批，其余批继续
                continue
            if len(pending) >= 2000:
                store.upsert_vectors(pending)
                pending = []
        if pending:
            store.upsert_vectors(pending)

    return {
        "tags": vocab["tags"],
        "tree_nodes": tree_count,
        "chunks": chunk_total,
        "chunk_vectors": store.count_vectors(fp, "chunk") if fp else 0,
    }
