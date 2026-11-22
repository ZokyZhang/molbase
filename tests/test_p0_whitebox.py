import unittest
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch
from molbase import config
from molbase.orchestrator import Orchestrator
from molbase.embedding import HashEmbedding


class TestP0WhiteBox(unittest.TestCase):
    """MolBase P0 整改白盒测试"""

    def setUp(self):
        self.orc = Orchestrator()

    def tearDown(self):
        self.orc.close()

    def test_config_v3_constants(self):
        """验证配置常量及权重和"""
        self.assertEqual(config.EMBEDDING_BACKEND, "ollama")
        self.assertEqual(config.OLLAMA_EMBED_MODEL, "bge-m3")
        self.assertEqual(config.ABSTAIN_CONFIDENCE, 0.55)
        self.assertEqual(config.SCENE_DEFAULT, "global")
        self.assertEqual(config.TAG_FILTER_MODE, "or_boost")
        self.assertEqual(config.SCENE_RELAX_MIN, 2)
        
        total = (config.CONF_W_STRENGTH + config.CONF_W_MARGIN + 
                 config.CONF_W_TAG + config.CONF_W_SCENE + config.CONF_W_ENTITY)
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_embedder_is_ollama_bge_m3(self):
        """验证 Ollama 嵌入器初始化及向量属性"""
        p = self.orc._init_embedder(None, "ollama")
        self.assertEqual(p.name, "ollama")
        self.assertEqual(p.dim, 1024)
        
        vec = p.embed("test")
        self.assertEqual(len(vec), 1024)
        l2 = sum(x*x for x in vec)**0.5
        self.assertAlmostEqual(l2, 1.0, places=2)
        
        vec2 = p.embed("test")
        self.assertIs(vec, vec2)

    def test_route_rules_strong_signal(self):
        """验证路由规则强信号匹配"""
        self.assertEqual(self.orc._route_rules("这个函数怎么调用"), "knowledge")
        self.assertEqual(self.orc._route_rules("查看源码实现"), "code")
        self.assertEqual(self.orc._route_rules("def foo():"), "code")
        self.assertEqual(self.orc._route_rules("用技能包执行"), "skill")
        self.assertEqual(self.orc._route_rules("如何摄入知识条目"), "knowledge")

    def test_scene_resolution_priority(self):
        """验证场景解析优先级"""
        self.assertEqual(self.orc.resolve_scene(scene_hint="ops"), "ops")
        self.assertEqual(self.orc.resolve_scene(query="生产环境怎么部署"), "production")
        self.assertEqual(self.orc.resolve_scene(), "global")

    def test_compute_confidence_dynamic_norm(self):
        """验证置信度动态归一化计算"""
        c1, s1, m1 = self.orc._compute_confidence([], "q")
        self.assertEqual(c1, 0.0)
        
        h1 = SimpleNamespace(entry=SimpleNamespace(meta={"scene": "global"}), score=0.9)
        h2 = SimpleNamespace(entry=SimpleNamespace(meta={"scene": "global"}), score=0.5)
        ranked = [h1, h2]
        
        c_no_tag, _, _ = self.orc._compute_confidence(ranked, "q")
        c_tag, _, _ = self.orc._compute_confidence(ranked, "q", query_tags=["nonexistent"])
        self.assertAlmostEqual(c_no_tag, c_tag, places=2)
        
        c_g, _, _ = self.orc._compute_confidence(ranked, "q", scene_id="global")
        c_o, _, _ = self.orc._compute_confidence(ranked, "q", scene_id="ops")
        self.assertTrue(0 <= c_g <= 1)
        self.assertTrue(0 <= c_o <= 1)

    def test_filter_scene_relax(self):
        """验证场景过滤松弛机制"""
        hits = [
            SimpleNamespace(entry=SimpleNamespace(meta={"scene": "global"})),
            SimpleNamespace(entry=SimpleNamespace(meta={"scene": "global"})),
            SimpleNamespace(entry=SimpleNamespace(meta={"scene": "global"})),
            SimpleNamespace(entry=SimpleNamespace(meta={"scene": "ops"})),
            SimpleNamespace(entry=SimpleNamespace(meta={"scene": "ops"})),
        ]
        
        res_g = self.orc._filter_by_scene(hits, "global")
        self.assertEqual(len(res_g), 5)
        
        res_o = self.orc._filter_by_scene(hits, "ops")
        self.assertTrue(len(res_o) > 0)
        self.assertTrue(any(h.entry.meta["scene"] == "ops" for h in res_o))

    def test_filter_refs(self):
        """验证引用过滤逻辑"""
        hits = [
            SimpleNamespace(entry=SimpleNamespace(entry_id="kb::target::1", source="kb::target")),
            SimpleNamespace(entry=SimpleNamespace(entry_id="kb::other::2", source="kb::other")),
            SimpleNamespace(entry=SimpleNamespace(entry_id="random::3", source="random")),
        ]
        
        self.assertEqual(len(self.orc._filter_by_refs(hits, ["kb::target"])), 1)
        self.assertEqual(len(self.orc._filter_by_refs(hits, [])), 3)

    def test_db_schema_tables(self):
        """验证数据库表结构及数据存在性"""
        conn = sqlite3.connect("data/molbase.db")
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        
        expected = {"scenes", "scene_entries", "knowledge_versions", "entity_index", "topic_tree", "semantic_cache"}
        self.assertTrue(expected.issubset(tables))
        
        cur.execute("SELECT COUNT(*) FROM entries")
        self.assertGreater(cur.fetchone()[0], 0)
        conn.close()

    def test_resolve_version_priority(self):
        """验证版本解析优先级逻辑"""
        mem_conn = sqlite3.connect(":memory:")
        # resolve_version 内部用 row["scene_id"] 字典式访问，必须设置行工厂
        mem_conn.row_factory = sqlite3.Row
        mem_conn.execute("""CREATE TABLE knowledge_versions(
            knowledge_id TEXT, scene_id TEXT, status TEXT, 
            authority REAL, confidence REAL, valid_from TEXT, version_id TEXT)""")
        
        mem_conn.execute("INSERT INTO knowledge_versions VALUES ('k1','ops','active',0.5,0.8,'2026-01-01','v_exact')")
        mem_conn.execute("INSERT INTO knowledge_versions VALUES ('k1','ops/prod','active',1.0,1.0,'2026-12-01','v_child')")
        mem_conn.execute("INSERT INTO knowledge_versions VALUES ('k1','global','active',1.0,1.0,'2026-12-31','v_global')")
        mem_conn.execute("INSERT INTO knowledge_versions VALUES ('k1','xx','active',1.0,1.0,'2026-12-31','v_other')")
        
        with patch.object(self.orc, "store", SimpleNamespace(_conn=mem_conn)):
            self.assertEqual(self.orc.resolve_version("k1", "ops")["version_id"], "v_exact")
            self.assertEqual(self.orc.resolve_version("k1", "prod")["version_id"], "v_global")
        
        mem_conn.close()

    def test_realtime_query_fields(self):
        """验证实时查询响应字段完整性"""
        resp = self.orc.query("MolBase 质量底线设计 四道防线 两道闸门")
        self.assertTrue(len(resp.results) > 0)
        self.assertGreater(resp.confidence, 0)
        self.assertTrue(resp.scene_id)
        self.assertTrue(hasattr(resp, "resolved_version_id"))
        self.assertTrue(hasattr(resp, "refs_miss"))

    def test_hash_embedding_deterministic(self):
        """验证 HashEmbedding 确定性"""
        emb = HashEmbedding(64)
        self.assertEqual(emb.embed("同一句话"), emb.embed("同一句话"))


if __name__ == "__main__":
    unittest.main()