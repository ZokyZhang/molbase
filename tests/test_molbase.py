# -*- coding: utf-8 -*-
"""MolBase单元测试（使用标准库 unittest，无需 pytest）。"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from molbase.config import ensure_dirs
from molbase.index import tokenize, BM25Index
from molbase.skills import parse_skill_md
from molbase.store import Entry, Store
from molbase.code import CodeIndexer
from molbase.embedding import HashEmbedding, create_embedder, cosine_similarity
from molbase.llm import create_llm
from molbase.orchestrator import Orchestrator
from molbase.tokens import TokenLedger, estimate_tokens
from molbase.websearch import NullSearchProvider, SearchHit, create_search_provider


class TestTokenize(unittest.TestCase):
    """分词测试。"""

    def test_english_tokens(self):
        """英文单词应被提取并转小写。"""
        tokens = tokenize("Hello World Python")
        self.assertIn("hello", tokens)
        self.assertIn("python", tokens)

    def test_chinese_bigram(self):
        """中文应产生 bigram 与单字 token。"""
        tokens = tokenize("作业管理")
        self.assertIn("作业", tokens)
        self.assertIn("业管", tokens)


class TestBM25(unittest.TestCase):
    """BM25 召回测试。"""

    def setUp(self):
        """准备测试文档。"""
        self.entries = [
            Entry(entry_type="knowledge", entry_id="d1", title="苹果", content="苹果是一种水果"),
            Entry(entry_type="knowledge", entry_id="d2", title="香蕉", content="香蕉也是水果"),
            Entry(entry_type="knowledge", entry_id="d3", title="汽车", content="汽车是交通工具"),
        ]
        self.idx = BM25Index()
        self.idx.build(self.entries)

    def test_search_hits_relevant(self):
        """查询'水果'应命中 d1/d2 而非 d3。"""
        hits = self.idx.search("水果", top_n=3)
        ids = {h.entry.entry_id for h in hits}
        self.assertIn("d1", ids)
        self.assertIn("d2", ids)


class TestSkillParse(unittest.TestCase):
    """SKILL.md 解析测试。"""

    def test_parse_valid(self):
        """合法 frontmatter 应解析出 name 与 description。"""
        text = "---\nname: pdf-extract\ndescription: 提取PDF字段\n---\n\n# 步骤\n1. 读取"
        meta = parse_skill_md(text)
        self.assertEqual(meta.name, "pdf-extract")
        self.assertIn("提取PDF字段", meta.description)

    def test_parse_missing_name(self):
        """缺少 name 字段应抛异常。"""
        text = "---\ndescription: 无name\n---\n正文"
        with self.assertRaises(ValueError):
            parse_skill_md(text)


class TestStore(unittest.TestCase):
    """存储底座测试。"""

    def test_upsert_and_get(self):
        """插入后可读取，反馈后热度增加。"""
        store = Store(db_path=Path(ensure_dirs() or ".") / "_test.db")
        e = Entry(entry_type="knowledge", entry_id="t1", title="测试", content="内容")
        store.upsert(e)
        got = store.get("t1")
        self.assertIsNotNone(got)
        self.assertEqual(got.title, "测试")

        store.add_heat("t1", 3.0)
        got2 = store.get("t1")
        self.assertAlmostEqual(got2.heat, 3.0)
        store.close()
        # 清理测试库
        (Path(ensure_dirs() or ".") / "_test.db").unlink(missing_ok=True)


class TestCodeIndexer(unittest.TestCase):
    """代码代工（AST 符号索引）测试。"""

    def test_index_functions(self):
        """应抽取函数与调用关系。"""
        repo = Path(__file__).resolve().parent.parent / "code_repo"
        indexer = CodeIndexer(repo)
        symbols = []
        for py in repo.rglob("*.py"):
            symbols.extend(indexer.index_file(py))
        names = {s.qualified_name for s in symbols}
        self.assertIn("detect_face", names)
        # recognize_person 调用了 load_image / detect_face
        recog = [s for s in symbols if s.qualified_name == "recognize_person"]
        if recog:
            self.assertTrue(any(c in recog[0].calls for c in ("load_image", "detect_face")))


class TestEmbedding(unittest.TestCase):
    """向量化抽象层测试（零模型基线 + 降级机制）。"""

    def test_hash_deterministic(self):
        """同一文本两次哈希向量应完全一致（跨版本稳定）。"""
        emb = HashEmbedding()
        self.assertEqual(emb.embed("人脸识别"), emb.embed("人脸识别"))

    def test_hash_semantic(self):
        """相似文本相似度应高于无关文本。"""
        emb = HashEmbedding()
        a = emb.embed("人脸识别的代码方法")
        b = emb.embed("人脸检测的函数")
        c = emb.embed("屏幕时间控制策略")
        self.assertGreater(cosine_similarity(a, b), cosine_similarity(a, c))

    def test_fallback_to_hash(self):
        """未安装的模型后端应自动降级为 hash，绝不抛错。"""
        emb = create_embedder("sentence-transformers", "all-MiniLM-L6-v2")
        self.assertEqual(emb.name, "hash")
        # 不存在的后端也应降级
        self.assertEqual(create_embedder("not-exist").name, "hash")

    def test_hash_vector_normalized(self):
        """哈希向量应 L2 归一化（模长约 1）。"""
        emb = HashEmbedding()
        v = emb.embed("测试文本内容")
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=4)


class _FakeLLM:
    """模拟可用的本地大模型：语义分类返回 code，摘要返回固定文本。"""

    name = "fake"

    def complete_json(self, system, user):
        """返回语义路由结果。"""
        return {"route": "code", "entities": ["人脸"]}

    def chat(self, system, user):
        """返回固定摘要。"""
        return "这是摘要。"


class _BrokenLLM:
    """模拟不可用的本地大模型：任何调用都抛异常。"""

    name = "broken"

    def complete_json(self, system, user):
        raise RuntimeError("model offline")

    def chat(self, system, user):
        raise RuntimeError("model offline")


class _RewriteLLM:
    """模拟可改写查询的本地大模型：chat 根据提示词区分改写/摘要。"""

    name = "rewrite"

    def complete_json(self, system, user):
        return {"route": "knowledge", "entities": []}

    def chat(self, system, user):
        # 改写提示词 -> 返回检索友好关键词
        if "改写" in system:
            return "数据库索引 查询优化"
        return "摘要"


class TestLLM(unittest.TestCase):
    """本地大模型接入层测试（V2 增强 + 降级）。"""

    def test_create_llm_none_when_unconfigured(self):
        """未配置 endpoint/model 时应返回 None（未启用）。"""
        self.assertIsNone(create_llm(None, None))
        self.assertIsNone(create_llm("", "qwen3.5-9b"))

    def test_create_llm_object(self):
        """配置了 endpoint/model 时应创建客户端，模型名即版本指纹。"""
        llm = create_llm("http://localhost:11434/v1", "qwen3.5-9b")
        self.assertIsNotNone(llm)
        self.assertEqual(llm.model, "qwen3.5-9b")

    def test_route_llm_semantic(self):
        """LLM 可用时用语义分类：规则会判 knowledge，但 LLM 判 code。"""
        orc = Orchestrator(llm=_FakeLLM())
        self.addCleanup(orc.close)
        self.assertEqual(orc._route("帮我看看识别脸的那个函数"), "code")

    def test_route_fallback_on_llm_error(self):
        """LLM 抛异常时应降级回规则路由，绝不崩溃。"""
        orc = Orchestrator(llm=_BrokenLLM())
        self.addCleanup(orc.close)
        self.assertEqual(orc._route("人脸识别的代码方法"), "code")

    def test_summarize_with_llm(self):
        """LLM 可用时应返回生成式摘要。"""
        orc = Orchestrator(llm=_FakeLLM())
        self.addCleanup(orc.close)
        self.assertEqual(orc.summarize("任意内容"), "这是摘要。")

    def test_summarize_fallback_truncate(self):
        """无 LLM 时应降级为按长度截断原文。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        out = orc.summarize("x" * 300, max_chars=50)
        self.assertLessEqual(len(out), 51)  # 50 字符 + 省略号


class TestEvolution(unittest.TestCase):
    """自我进化闭环测试（零模型，全自动）。"""

    def _make_store(self):
        """创建临时数据库用于单元测试。"""
        db = Path(ensure_dirs() or ".") / "_test_evo.db"
        return Store(db_path=db), db

    def test_cooccur_auto_link(self):
        """共现达到阈值后应自动建立关联边（自我发现关联）。"""
        from molbase.evolution import EvolutionEngine
        store, db = self._make_store()
        try:
            store.upsert(Entry(entry_type="knowledge", entry_id="a", title="A", content="内容A"))
            store.upsert(Entry(entry_type="knowledge", entry_id="b", title="B", content="内容B"))
            evo = EvolutionEngine(store)
            for _ in range(3):  # 阈值 3
                evo.record_query(["a", "b"])
            stats = evo.learn()
            self.assertEqual(stats["edges"], 1)
            a = store.get("a")
            self.assertIn("b", a.meta.get("relations", []))
        finally:
            store.close()
            db.unlink(missing_ok=True)

    def test_cooccur_same_type_only(self):
        """类型感知：跨类型共现不应建边，同类型才建边。"""
        from molbase.evolution import EvolutionEngine
        store, db = self._make_store()
        try:
            store.upsert(Entry(entry_type="knowledge", entry_id="k1", title="K1", content="知识"))
            store.upsert(Entry(entry_type="skill", entry_id="s1", title="S1", content="技能"))
            evo = EvolutionEngine(store)
            for _ in range(3):  # 阈值 3
                evo.record_query(["k1", "s1"])  # 跨类型共现
            stats = evo.learn()
            self.assertEqual(stats["edges"], 0)  # 跨类型不建边
            k1 = store.get("k1")
            self.assertNotIn("s1", k1.meta.get("relations", []))
        finally:
            store.close()
            db.unlink(missing_ok=True)

    def test_cold_decay(self):
        """长期未访问的冷知识应自动衰减热度。"""
        import time
        from molbase.evolution import EvolutionEngine
        store, db = self._make_store()
        try:
            e = Entry(entry_type="knowledge", entry_id="c", title="冷知识", content="内容")
            e.heat = 10.0
            e.last_access = time.time() - 100 * 86400  # 100 天前
            store.upsert(e)
            evo = EvolutionEngine(store)
            stats = evo.learn()
            self.assertGreaterEqual(stats["cold_decayed"], 1)
            self.assertLess(store.get("c").heat, 10.0)
        finally:
            store.close()
            db.unlink(missing_ok=True)


class TestOrchestratorEvolution(unittest.TestCase):
    """编排器自我进化集成测试（query 自动命中热度 + 闭环反哺）。"""

    def setUp(self):
        """每个用例用独立临时库，避免共享生产库的历史共现数据干扰断言。"""
        self._tmpdir = tempfile.TemporaryDirectory()
        # 先注册目录清理，再注册 orc.close —— addCleanup 按 LIFO 执行，
        # 确保先关闭 SQLite 连接、再删临时目录（避免 Windows 文件占用报错）。
        self.addCleanup(self._tmpdir.cleanup)
        self._db = Path(self._tmpdir.name) / "test.db"

    def test_query_auto_heat(self):
        """query 命中应自动累加热度（用得多自然热）。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("evo_heat", "热度条目", "关于自我进化热度的独特内容")
        before = orc.store.get("evo_heat").heat
        orc.query("自我进化热度")
        after = orc.store.get("evo_heat").heat
        self.assertGreater(after, before)

    def test_evolve_closed_loop(self):
        """共现学习后，关联边应进入关联图（闭环反哺检索）。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("evo_a", "知识A", "关于数据库索引")
        orc.ingest_knowledge("evo_b", "知识B", "关于查询优化")
        for _ in range(3):
            orc.evolution.record_query(["evo_a", "evo_b"])
        stats = orc.evolve()
        self.assertEqual(stats["edges"], 1)
        neighbors = orc.graph.neighbors("evo_a", resolver=orc._resolver)
        self.assertTrue(any(n.entry.entry_id == "evo_b" for n in neighbors))


class TestQueryRewrite(unittest.TestCase):
    """查询改写测试（本地大模型第三个任务 + 降级）。"""

    def test_rewrite_with_llm(self):
        """LLM 可用时应改写口语化查询为检索友好关键词。"""
        orc = Orchestrator(llm=_RewriteLLM())
        self.addCleanup(orc.close)
        rw = orc._rewrite("我想了解一下数据库索引是怎么加速查询的呀")
        self.assertEqual(rw, "数据库索引 查询优化")

    def test_rewrite_fallback_no_llm(self):
        """无 LLM 时应降级返回原查询。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        q = "数据库索引"
        self.assertEqual(orc._rewrite(q), q)

    def test_query_uses_rewrite(self):
        """query 应使用改写后的查询检索，并在响应中带回 rewritten。"""
        orc = Orchestrator(llm=_RewriteLLM())
        self.addCleanup(orc.close)
        orc.ingest_knowledge("rw_a", "数据库索引", "关于数据库索引的知识")
        resp = orc.query("我想了解数据库索引的知识")
        self.assertEqual(resp.rewritten, "数据库索引 查询优化")


class TestSummarize(unittest.TestCase):
    """TextRank 摘要与智能切块测试（P0，零模型）。"""

    def test_split_sentences(self):
        """应按中文标点切分句子。"""
        from molbase.summarize import split_sentences
        s = split_sentences("第一句。第二句！第三句？")
        self.assertEqual(len(s), 3)

    def test_split_blocks(self):
        """应按段落（空行）切块。"""
        from molbase.summarize import split_blocks
        text = "段落一\n\n段落二\n\n段落三"
        blocks = split_blocks(text, max_chars=100)
        self.assertEqual(len(blocks), 3)

    def test_textrank_summarize(self):
        """TextRank 应输出比原文更短的摘要。"""
        from molbase.summarize import textrank_summarize
        text = (
            "数据库索引能加速查询。索引是B树结构。B树能快速定位数据。"
            "查询优化器会选择合适的索引。没有索引就会全表扫描。"
        )
        summary = textrank_summarize(text, ratio=0.4, max_sentences=3)
        self.assertTrue(0 < len(summary) < len(text))


class TestP1(unittest.TestCase):
    """P1 升级测试：词典分词 + BM25F 字段加权 + 负反馈。"""

    def test_lexicon_tokenize(self):
        """词典词应被完整切分出来（精准语义）。"""
        tokens = tokenize("数据库索引")
        self.assertIn("数据库", tokens)
        self.assertIn("索引", tokens)

    def test_stopword_filter(self):
        """停用词应被过滤。"""
        self.assertNotIn("的", tokenize("的"))

    def test_bm25f_title_boost(self):
        """标题命中应比正文命中得分更高。"""
        entries = [
            Entry(entry_type="knowledge", entry_id="t1", title="数据库索引", content="一些无关内容"),
            Entry(entry_type="knowledge", entry_id="t2", title="无关标题", content="这里讲数据库索引的用法"),
        ]
        idx = BM25Index()
        idx.build(entries)
        hits = idx.search("数据库索引", top_n=2)
        ids = [h.entry.entry_id for h in hits]
        self.assertEqual(ids[0], "t1")  # 标题命中排前

    def test_reject_negative_feedback(self):
        """reject 负反馈应使热度下降。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("rej_1", "测试", "内容")
        orc.feedback("rej_1", "adopt")  # +3
        before = orc.store.get("rej_1").heat
        orc.feedback("rej_1", "reject")  # -3
        after = orc.store.get("rej_1").heat
        self.assertLess(after, before)


class TestP2(unittest.TestCase):
    """P2 升级测试：TF-IDF 向量 + Personalized PageRank + SimHash 去重。"""

    def test_personalized_pagerank(self):
        """Personalized PageRank 应以种子为 teleport 找关联节点。"""
        from molbase.graph import RelationGraph
        g = RelationGraph()
        g.add_edge("a", "b")
        g.add_edge("a", "c")
        g.add_edge("b", "c")
        hits = g.personalized_pagerank(["a"], top_n=5)
        self.assertTrue(len(hits) > 0)
        ids = {h.entry.entry_id for h in hits}
        self.assertIn("b", ids)
        self.assertIn("c", ids)

    def test_simhash_near_duplicate(self):
        """相同文本应判为近重复，不同文本不应判为近重复。"""
        from molbase.embedding import is_near_duplicate
        self.assertTrue(is_near_duplicate("数据库索引能加速查询", "数据库索引能加速查询"))
        self.assertFalse(is_near_duplicate("数据库索引能加速查询", "今天天气很好适合出门游玩"))

    def test_tfidf_embedding(self):
        """TF-IDF 向量下，相似文本相似度应高于无关文本。"""
        from molbase.embedding import TfidfEmbedding, cosine_similarity
        emb = TfidfEmbedding()
        emb.build(["苹果是一种水果", "汽车是交通工具", "水果富含维生素"])
        a = emb.embed("水果")
        b = emb.embed("水果苹果")
        c = emb.embed("汽车")
        self.assertGreater(cosine_similarity(a, b), cosine_similarity(a, c))


class _FakeSearchProvider:
    """模拟可用的网络搜索提供者：返回固定外部结果。"""

    name = "fake"

    def search(self, query, top_n=3):
        return [
            SearchHit(title="外部知识", content="来自网络的内容", url="http://example.com", score=1.0)
        ]


class TestWebSearch(unittest.TestCase):
    """网络搜索外部源适配器测试（兜底 + 采纳沉淀 + 降级）。"""

    def setUp(self):
        """每个用例用独立临时库，避免共享 DB 的残留数据干扰召回判断。"""
        self._tmpdir = tempfile.TemporaryDirectory()
        # 先注册目录清理，后注册 orc.close —— addCleanup 按 LIFO 执行，
        # 确保先关闭 SQLite 连接、再删临时目录（避免 Windows 文件占用报错）。
        self.addCleanup(self._tmpdir.cleanup)
        self._db = Path(self._tmpdir.name) / "test.db"

    def test_null_provider_returns_empty(self):
        """空搜索提供者（默认关闭）应返回空列表。"""
        self.assertEqual(NullSearchProvider().search("任意"), [])

    def test_create_provider_disabled(self):
        """开关关闭时应创建空提供者。"""
        p = create_search_provider(enabled=False)
        self.assertEqual(p.name, "null")

    def test_should_web_search_off_by_default(self):
        """默认未启用网络搜索时，不应触发网络兜底。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        self.assertFalse(orc._should_web_search([]))

    def test_web_search_fallback_when_internal_weak(self):
        """内部召回弱时应触发网络搜索兜底，web_results 非空。"""
        orc = Orchestrator(search_provider=_FakeSearchProvider(), db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("ws_a", "苹果", "苹果是水果")  # 与查询无关，内部命中弱
        resp = orc.query("量子计算的最新进展")
        self.assertTrue(len(resp.web_results) > 0)
        self.assertEqual(resp.web_results[0].entry_type, "web")

    def test_no_web_search_when_internal_good(self):
        """内部命中良好时不应触发网络搜索。"""
        orc = Orchestrator(search_provider=_FakeSearchProvider(), db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("ws_b", "数据库索引", "数据库索引能加速查询")
        resp = orc.query("数据库索引")
        self.assertEqual(resp.web_results, [])

    def test_adopt_web_result_precipitates(self):
        """采纳网络结果应沉淀为内部知识条目（搜索→沉淀→复用）。"""
        orc = Orchestrator(search_provider=_FakeSearchProvider(), db_path=self._db)
        self.addCleanup(orc.close)
        web = orc._web_search("量子计算", 3)[0]
        new_id = orc.adopt_web_result(web)
        entry = orc.store.get(new_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.entry_type, "knowledge")
        self.assertEqual(entry.meta.get("origin"), "web")


class TestTokenTrace(unittest.TestCase):
    """Token 埋点测试（粗估）。"""

    def test_estimate_tokens_chinese(self):
        """中文应约 1 字 = 1 token。"""
        self.assertEqual(estimate_tokens("数据库索引"), 5)

    def test_estimate_tokens_english(self):
        """英文应约 4 字符 = 1 token。"""
        self.assertEqual(estimate_tokens("database"), 2)  # 8 字符 / 4 = 2

    def test_query_token_trace(self):
        """query 应附带 token 埋点，且基线（全表扫描）≥ 返回内容。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("tk_a", "数据库索引", "数据库索引能加速查询速度")
        resp = orc.query("数据库索引")
        self.assertGreater(resp.token_estimate, 0)
        self.assertGreaterEqual(resp.baseline_tokens, resp.token_estimate)
        self.assertGreater(resp.content_chars, 0)


class TestTokenLedger(unittest.TestCase):
    """会话级 token 台账测试（节省比例 + 等效降本）。"""

    def test_ledger_accumulates_and_ratio(self):
        """多次查询应累计，节省比例 = (基线 - 实际) / 基线。"""
        ledger = TokenLedger()
        ledger.record(actual_tokens=100, baseline_tokens=1000)
        ledger.record(actual_tokens=50, baseline_tokens=500)
        self.assertEqual(ledger.queries, 2)
        self.assertEqual(ledger.actual_tokens, 150)
        self.assertEqual(ledger.baseline_tokens, 1500)
        self.assertEqual(ledger.saved_tokens, 1350)
        self.assertAlmostEqual(ledger.saved_ratio, 0.9)

    def test_ledger_skips_zero_baseline(self):
        """baseline<=0（未开启埋点）应跳过，不污染比例。"""
        ledger = TokenLedger()
        ledger.record(0, 0)
        self.assertEqual(ledger.queries, 0)
        self.assertEqual(ledger.saved_ratio, 0.0)

    def test_ledger_snapshot(self):
        """snapshot 应包含节省比例与等效成本字段。"""
        ledger = TokenLedger()
        ledger.record(1000, 10000)
        snap = ledger.snapshot()
        for key in ("saved_ratio", "saved_ratio_pct", "saved_tokens", "saved_cost"):
            self.assertIn(key, snap)
        self.assertGreater(snap["saved_tokens"], 0)
        self.assertGreater(snap["saved_ratio"], 0.0)

    def test_orchestrator_token_stats(self):
        """orchestrator.token_stats() 应返回会话级累计统计。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("led_a", "数据库索引", "数据库索引能加速查询")
        orc.ingest_knowledge("led_b", "查询优化", "查询优化避免全表扫描")
        orc.query("数据库索引")
        orc.query("查询优化")
        snap = orc.token_stats()
        self.assertEqual(snap["queries"], 2)
        self.assertGreater(snap["baseline_tokens"], 0)
        self.assertGreaterEqual(snap["saved_tokens"], 0)
        self.assertGreaterEqual(snap["saved_ratio"], 0.0)


class TestMetricsPersistence(unittest.TestCase):
    """持久化埋点测试（metrics 表，部署后实时监控的数据源）。"""

    def _make_orc(self):
        """每个用例用独立临时库，避免共享 DB 残留干扰。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "metrics_test.db"
        orc = Orchestrator(db_path=db)
        self.addCleanup(orc.close)
        return orc

    def test_query_persists_metric(self):
        """每次 query 应写入一条 metrics 埋点记录。"""
        orc = self._make_orc()
        orc.ingest_knowledge("mt_a", "数据库索引", "数据库索引能加速查询速度")
        orc.query("数据库索引")
        rows = orc.store.list_metrics()
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["route"], "knowledge")

    def test_metric_aggregate_saved_ratio(self):
        """聚合应正确计算累计节省比例（基线 > 实际）。"""
        orc = self._make_orc()
        orc.ingest_knowledge("mt_b", "数据库索引", "数据库索引能加速查询速度")
        orc.query("数据库索引")
        orc.query("数据库索引")
        agg = orc.store.metric_aggregate()
        self.assertEqual(agg["queries"], 2)
        self.assertGreater(agg["baseline_tokens"], 0)
        self.assertGreaterEqual(agg["saved_tokens"], 0)
        self.assertGreaterEqual(agg["saved_ratio"], 0.0)

    def test_metrics_stats_shape(self):
        """metrics_stats 应返回 aggregate + recent + ledger 三段结构。"""
        orc = self._make_orc()
        orc.ingest_knowledge("mt_c", "查询优化", "查询优化避免全表扫描")
        orc.query("查询优化")
        out = orc.metrics_stats()
        for key in ("aggregate", "recent", "ledger"):
            self.assertIn(key, out)
        self.assertIn("saved_ratio_pct", out["aggregate"])
        self.assertGreaterEqual(len(out["recent"]), 1)


class TestHarvestSession(unittest.TestCase):
    """会话回写测试：把会话提炼成核心资产沉淀回知识库。"""

    def _make_orc(self):
        """每个用例用独立临时库，避免共享 DB 残留干扰。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "harvest_test.db"
        orc = Orchestrator(db_path=db)
        self.addCleanup(orc.close)
        return orc

    def test_harvest_textrank_fallback(self):
        """未启用 LLM 时应降级 TextRank，落库一份结构化资产。"""
        orc = self._make_orc()
        transcript = (
            "今天讨论了数据库索引。结论是：索引能加速查询。"
            "决定下周上线查询优化。待办是补充索引文档。"
        )
        r = orc.harvest_session("sess-001", transcript)
        self.assertEqual(r.state, "new")
        self.assertEqual(r.engine, "textrank")
        self.assertEqual(r.entry_id, "session::sess-001")
        entry = orc.store.get("session::sess-001")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.entry_type, "knowledge")
        self.assertEqual(entry.meta["session_id"], "sess-001")
        self.assertEqual(entry.meta["engine"], "textrank")
        self.assertIn("结论", entry.content)

    def test_harvest_empty_transcript(self):
        """空会话应返回 state=empty，不落库。"""
        orc = self._make_orc()
        r = orc.harvest_session("sess-empty", "   ")
        self.assertEqual(r.state, "empty")
        self.assertEqual(r.engine, "none")
        self.assertIsNone(orc.store.get("session::sess-empty"))

    def test_harvest_dedup(self):
        """同 ID 同源重复回写应判 duplicate 跳过。"""
        orc = self._make_orc()
        transcript = "讨论了数据库索引，索引能加速查询，决定上线查询优化。"
        r1 = orc.harvest_session("sess-dup", transcript, source="agent:abc")
        r2 = orc.harvest_session("sess-dup", transcript, source="agent:abc")
        self.assertEqual(r1.state, "new")
        self.assertEqual(r2.state, "duplicate")

    def test_harvest_retrievable(self):
        """回写后的核心资产应能被 query 召回（重复利用 + 供他人使用）。"""
        orc = self._make_orc()
        transcript = "讨论了家庭教育物联网项目，决定采用 MQTT 控制设备。"
        orc.harvest_session("sess-retr", transcript)
        resp = orc.query("MQTT 设备控制")
        ids = [r.entry_id for r in resp.results]
        self.assertIn("session::sess-retr", ids)

    def test_harvest_persists_metric(self):
        """回写应写入 route=harvest 的埋点，供监控大屏统计。"""
        orc = self._make_orc()
        orc.harvest_session("sess-metric", "讨论了数据库索引，结论是索引能加速查询。")
        rows = orc.store.list_metrics()
        self.assertTrue(any(row["route"] == "harvest" for row in rows))

    def test_harvest_links_tooltrace(self):
        """harvest 应把同会话的操作指纹关联进 meta（结论↔过程，不靠反推）。"""
        orc = self._make_orc()
        # 先喂工具轨迹（会话收尾前，上游已把「怎么干的」沉淀进来）
        orc.record_tooltrace(
            "sess-link",
            "检索并更新文档",
            [
                {"tool": "web_search", "args": {"query": "数据库索引"}},
                {"tool": "file_read", "args": {"path": "/docs/index.md"}},
                {"tool": "file_edit", "args": {"path": "/docs/index.md", "content": "更新"}},
            ],
        )
        # 再回写结论，应自动关联同会话指纹
        r = orc.harvest_session("sess-link", "讨论了数据库索引，结论是索引能加速查询。")
        self.assertGreater(len(r.related_fingerprints), 0)
        entry = orc.store.get("session::sess-link")
        self.assertIn("related_fingerprints", entry.meta)
        self.assertEqual(entry.meta["related_fingerprints"], r.related_fingerprints)

    def test_harvest_no_tooltrace_no_links(self):
        """无工具轨迹时，harvest 的 related_fingerprints 应为空，不影响回写。"""
        orc = self._make_orc()
        r = orc.harvest_session("sess-nolink", "讨论了数据库索引，结论是索引能加速查询。")
        self.assertEqual(r.related_fingerprints, [])
        self.assertIsNotNone(orc.store.get("session::sess-nolink"))


class TestToolTrace(unittest.TestCase):
    """工具调用轨迹沉淀测试：观察→聚类→固化，实现 N+1 -> N 的 token 收敛。"""

    # 一个可复用的多工具操作序列（3 个工具，参数键代表模式）
    TOOL_CALLS = [
        {"tool": "web_search", "args": {"query": "数据库索引"}},
        {"tool": "file_read", "args": {"path": "/docs/index.md"}},
        {"tool": "file_edit", "args": {"path": "/docs/index.md", "content": "更新"}},
    ]

    def _make_orc(self):
        """每个用例用独立临时库 + 临时技能目录，避免污染真实 skills_repo。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        db = Path(tmp.name) / "tooltrace_test.db"
        skills_dir = Path(tmp.name) / "skills"
        orc = Orchestrator(db_path=db, skills_dir=skills_dir)
        self.addCleanup(orc.close)
        return orc

    def test_tooltrace_empty_calls(self):
        """空工具调用序列应返回空指纹，不落库。"""
        orc = self._make_orc()
        r = orc.record_tooltrace("sess-1", "意图", [])
        self.assertEqual(r.fingerprint, "")
        self.assertEqual(r.reason, "工具调用序列为空，跳过")

    def test_tooltrace_below_threshold_no_generate(self):
        """未达阈值时应只观察、不生成 skill。"""
        orc = self._make_orc()
        r = orc.record_tooltrace("sess-1", "检索文档", self.TOOL_CALLS)
        self.assertEqual(r.count, 1)
        self.assertFalse(r.generated)
        self.assertIn("未达阈值", r.reason)

    def test_tooltrace_threshold_generates_skill(self):
        """同一指纹累计达阈值后应自动生成 skill（模板兜底），落库 entry_type=skill。"""
        orc = self._make_orc()
        for i in range(3):
            r = orc.record_tooltrace(f"sess-{i}", "检索并更新文档", self.TOOL_CALLS)
        self.assertTrue(r.generated)
        self.assertTrue(r.skill_id.startswith("skill::auto-"))
        entry = orc.store.get(r.skill_id)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.entry_type, "skill")
        self.assertTrue(entry.meta.get("auto_generated"))
        # 生成的 SKILL.md 应写回 skills_repo 磁盘
        skill_path = orc.skills_dir / r.skill_id.split("::", 1)[1] / "SKILL.md"
        self.assertTrue(skill_path.exists())

    def test_tooltrace_no_duplicate_generate(self):
        """已生成过 skill 后，再次命中应直接复用、不重复生成。"""
        orc = self._make_orc()
        for i in range(3):
            orc.record_tooltrace(f"sess-{i}", "检索并更新文档", self.TOOL_CALLS)
        r4 = orc.record_tooltrace("sess-4", "检索并更新文档", self.TOOL_CALLS)
        self.assertFalse(r4.generated)
        self.assertIn("已存在", r4.reason)

    def test_tooltrace_single_tool_no_generate(self):
        """单工具调用（少于最小流程长度）达阈值也不固化成流程。"""
        orc = self._make_orc()
        single = [{"tool": "web_search", "args": {"query": "x"}}]
        for i in range(3):
            r = orc.record_tooltrace(f"sess-{i}", "搜索", single)
        self.assertFalse(r.generated)
        self.assertIn("序列过短", r.reason)

    def test_tooltrace_generated_skill_retrievable(self):
        """自动生成的 skill 应能被 /skill 语义召回（实现「复用」闭环）。"""
        orc = self._make_orc()
        for i in range(3):
            orc.record_tooltrace(f"sess-{i}", "检索并更新文档", self.TOOL_CALLS)
        resp = orc.query("检索并更新文档")
        skills = [x for x in resp.results if x.entry_type == "skill"]
        self.assertTrue(any(s.meta.get("auto_generated") for s in skills))

    def test_tooltrace_aggregate(self):
        """轨迹应持久化，aggregate 能反映操作模式收敛情况。"""
        orc = self._make_orc()
        for i in range(2):
            orc.record_tooltrace(f"sess-{i}", "检索并更新文档", self.TOOL_CALLS)
        agg = orc.store.tooltrace_aggregate()
        self.assertEqual(agg["traces"], 2)
        self.assertEqual(agg["fingerprints"], 1)
        self.assertEqual(agg["top"][0]["count"], 2)


class TestKnowledgeDirIngest(unittest.TestCase):
    """知识目录批量摄入测试。"""
    def test_ingest_knowledge_dir(self):
        """应批量摄入 markdown，标题取首个 # 标题，且可被 query 检索。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "部署-虚拟机.md"
            md.write_text("# 虚拟机环境\n虚拟机 IP 是 10.0.0.200", encoding="utf-8")
            n = orc.ingest_knowledge_dir(tmp)
            self.assertEqual(n, 1)
            resp = orc.query("虚拟机 IP")
            self.assertTrue(any("10.0.0.200" in r.content for r in resp.results))


class TestTagFilter(unittest.TestCase):
    """按标签过滤检索测试（meta.tags）。"""

    def test_query_filter_by_single_tag(self):
        """query(tags=[...]) 应只返回含该标签的条目。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("tg_ops", "部署说明", "systemd 部署服务", meta={"tags": ["运维"]})
        orc.ingest_knowledge("tg_arch", "架构设计", "模块化单体架构", meta={"tags": ["架构"]})
        resp = orc.query("部署", tags=["运维"])
        ids = [r.entry_id for r in resp.results]
        self.assertIn("tg_ops", ids)
        self.assertNotIn("tg_arch", ids)

    def test_query_tags_and_semantics(self):
        """多标签应为 AND：只返回同时含全部标签的条目。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("tg_env", "虚拟机环境", "虚拟机 IP 10.0.0.200 含密码", meta={"tags": ["环境", "敏感"]})
        orc.ingest_knowledge("tg_ops", "部署", "systemd 部署", meta={"tags": ["运维"]})
        resp = orc.query("虚拟机", tags=["环境", "敏感"])
        ids = [r.entry_id for r in resp.results]
        self.assertIn("tg_env", ids)
        self.assertNotIn("tg_ops", ids)

    def test_query_without_tags_unfiltered(self):
        """不带 tags 时应正常返回（不过滤）。"""
        orc = Orchestrator()
        self.addCleanup(orc.close)
        orc.ingest_knowledge("tg_a", "知识A", "关于数据库", meta={"tags": ["运维"]})
        resp = orc.query("数据库")
        self.assertTrue(any(r.entry_id == "tg_a" for r in resp.results))


class TestQualityGuardrail(unittest.TestCase):
    """质量底线测试：输出 abstain 闸门 + 输入冲突闸门 + PII 脱敏 + 负反馈隔离。"""

    def setUp(self):
        """每个用例用独立临时库。"""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._db = Path(self._tmpdir.name) / "test.db"

    def test_query_abstain_on_unknown(self):
        """无关查询应 abstain=True（宁可不答也不错答）。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("qg_apple", "苹果", "苹果是一种水果")
        resp = orc.query("量子力学的薛定谔方程")
        self.assertTrue(resp.abstain)
        self.assertLess(resp.confidence, 0.45)

    def test_query_confidence_on_hit(self):
        """强相关查询应 abstain=False 且置信度为正。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("qg_db", "数据库索引", "数据库索引能加速查询速度")
        resp = orc.query("数据库索引")
        self.assertFalse(resp.abstain)
        self.assertGreater(resp.confidence, 0.0)

    def test_ingest_duplicate_skipped(self):
        """同源重复摄入相同内容应判 duplicate 并跳过。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        r1 = orc.ingest_knowledge("qg_dup", "标题", "数据库索引能加速查询", source="s1")
        r2 = orc.ingest_knowledge("qg_dup", "标题", "数据库索引能加速查询", source="s1")
        self.assertEqual(r1.state, "new")
        self.assertEqual(r2.state, "duplicate")

    def test_ingest_conflict_recorded(self):
        """同 ID 不同来源应判 conflict 并记录待裁决（宁可少存不错存）。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        r1 = orc.ingest_knowledge("qg_c", "支出", "本月支出五千元", source="爸爸记账")
        r2 = orc.ingest_knowledge("qg_c", "支出", "本月支出八千元", source="妈妈记账")
        self.assertEqual(r1.state, "new")
        self.assertEqual(r2.state, "conflict")
        conflicts = orc.store.list_conflicts()
        self.assertEqual(len(conflicts), 1)

    def test_ingest_pinned_rejected(self):
        """人工固定(pinned)的条目应拒绝覆盖。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("qg_pin", "固定", "原始内容", source="s1")
        orc.store.update_meta("qg_pin", {"pinned": True})
        r = orc.ingest_knowledge("qg_pin", "固定", "新内容", source="s1")
        self.assertEqual(r.state, "pinned")
        self.assertEqual(orc.store.get("qg_pin").content, "原始内容")

    def test_reject_quarantine(self):
        """reject 累计达阈值应隔离，不再参与召回。"""
        orc = Orchestrator(db_path=self._db)
        self.addCleanup(orc.close)
        orc.ingest_knowledge("qg_rej", "坏知识", "关于错误信息的独特内容")
        for _ in range(3):
            orc.feedback("qg_rej", "reject")
        entry = orc.store.get("qg_rej")
        self.assertTrue(entry.meta.get("quarantined"))
        resp = orc.query("错误信息")
        self.assertNotIn("qg_rej", [r.entry_id for r in resp.results])

    def test_sanitize_pii(self):
        """PII 脱敏：手机号/邮箱/密码应被打码。"""
        from molbase.sanitize import sanitize_text
        out = sanitize_text("联系 13800138000 邮箱 test@example.com 密码 password=abc123")
        self.assertNotIn("13800138000", out)
        self.assertNotIn("test@example.com", out)
        self.assertNotIn("abc123", out)
        self.assertIn("****", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
