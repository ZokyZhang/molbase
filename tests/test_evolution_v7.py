import sys
import os
import tempfile
import unittest

# 插入项目根路径
sys.path.insert(0, r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase")

from molbase.store import Entry, Store
from molbase.orchestrator import Orchestrator

class TestEvolutionV7Whitebox(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_molbase.db")
        # 初始化 Store 并写入基础数据
        self.store = Store(self.db_path)
        self.store.upsert(Entry("knowledge", "k1", "标题", "正文内容", source="test"))
        self.store.close()
        # 创建 Orchestrator 读取数据
        self.orc = Orchestrator(embedding_backend='hash', db_path=self.db_path)

    def tearDown(self):
        self.orc.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_feedback_log_source_column(self):
        """PRAGMA table_info(feedback_log) 取列名，断言 'source' in 列名"""
        conn = self.orc.store._conn
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(feedback_log)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn('source', columns)

    def test_record_feedback_auto_source(self):
        """orc.store.record_feedback(..., source="auto") 后 feedback_aggregate()["by_source"]["auto"] == 1"""
        self.orc.store.record_feedback("s1", "k1", "hit", source="auto")
        agg = self.orc.store.feedback_aggregate()
        self.assertEqual(agg["by_source"]["auto"], 1)

    def test_recent_returns_dict_list(self):
        """record_feedback + record_evolve 后，feedback_recent(3) 与 evolve_recent(3) 都是 list，且首项是 dict"""
        self.orc.store.record_feedback("s1", "k1", "hit", source="manual")
        self.orc.store.record_evolve(added_edges=1, decayed=0, archived=0)
        
        fb_recent = self.orc.store.feedback_recent(limit=3)
        self.assertIsInstance(fb_recent, list)
        self.assertTrue(len(fb_recent) > 0)
        self.assertIsInstance(fb_recent[0], dict)
        self.assertIn('source', fb_recent[0])

        ev_recent = self.orc.store.evolve_recent(limit=3)
        self.assertIsInstance(ev_recent, list)
        self.assertTrue(len(ev_recent) > 0)
        self.assertIsInstance(ev_recent[0], dict)
        self.assertIn('added_edges', ev_recent[0])

    def test_evolution_health_empty_safe(self):
        """新建空 db 的 Store.evolution_health() 返回 dict，含 12 键，空库比率均为 0.0"""
        empty_db_path = os.path.join(self.tmpdir, "empty_test.db")
        empty_store = Store(empty_db_path)
        health = empty_store.evolution_health()
        empty_store.close()
        os.remove(empty_db_path)

        expected_keys = {
            'queries', 'feedback_total', 'by_action', 'by_source', 'feedback_rate',
            'auto_feedback', 'auto_feedback_ratio', 'evolve', 'gap_total',
            'gap_no_data', 'tooltrace_count', 'session_complete_rate'
        }
        self.assertEqual(set(health.keys()), expected_keys)
        self.assertEqual(health['feedback_rate'], 0.0)
        self.assertEqual(health['auto_feedback_ratio'], 0.0)
        self.assertEqual(health['session_complete_rate'], 0.0)

    def test_auto_feedback_threshold(self):
        """orc._auto_feedback("k1","s",0.9) 后 auto 计数为 1；再 orc._auto_feedback("k1","s",0.2) 后仍为 1"""
        self.orc._auto_feedback("k1", "s", 0.9)
        agg1 = self.orc.store.feedback_aggregate()
        self.assertEqual(agg1["by_source"]["auto"], 1)

        self.orc._auto_feedback("k1", "s", 0.2)
        agg2 = self.orc.store.feedback_aggregate()
        self.assertEqual(agg2["by_source"]["auto"], 1)

    def test_feedback_evolve_trigger(self):
        """连续 5 次 orc.feedback("k1","adopt") 后 evolve_aggregate()["steps"] == 1"""
        for _ in range(5):
            self.orc.feedback("k1", action="adopt", session_id="s1", source="manual")
        
        agg = self.orc.store.evolve_aggregate()
        self.assertEqual(agg["steps"], 1)

    def test_metric_detail_feedback_evolve(self):
        """record_feedback 后 orc.metric_detail("feedback")["total"] >= 1；orc.metric_detail("evolve") 是 dict 且含 'total'"""
        self.orc.store.record_feedback("s1", "k1", "hit", source="manual")
        
        fb_detail = self.orc.metric_detail("feedback", limit=50)
        self.assertGreaterEqual(fb_detail["total"], 1)

        ev_detail = self.orc.metric_detail("evolve", limit=50)
        self.assertIsInstance(ev_detail, dict)
        self.assertIn('total', ev_detail)

    def test_overview_health(self):
        """orc.overview()["evolution_health"] 含 'tooltrace_count' 且 isinstance(int)"""
        overview = self.orc.overview()
        health = overview["evolution_health"]
        self.assertIn('tooltrace_count', health)
        self.assertIsInstance(health['tooltrace_count'], int)

if __name__ == '__main__':
    unittest.main()