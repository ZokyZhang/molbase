import sys
import os
import tempfile
import shutil
import unittest

# 严格遵循契约：sys.path 插入
sys.path.insert(0, r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase")

from molbase.store import Store, Entry
from molbase.orchestrator import Orchestrator


class TestV72SceneIndex(unittest.TestCase):
    """测试 MolBase v7.2 场景索引与实体版本管理功能"""

    def setUp(self):
        """设置测试环境：创建临时数据库路径并初始化 Store 和 Orchestrator"""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.store = Store(self.db_path)
        self.orc = Orchestrator(embedding_backend='hash', db_path=self.db_path)

    def tearDown(self):
        """清理测试环境：关闭连接并删除临时目录"""
        self.orc.close()
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_scene_asset_reflow(self):
        """测试场景资产重流：插入知识条目并查询，验证场景统计正确"""
        # 用 ingest_knowledge 插入（内部会重建索引，确保 query 能召回新条目）
        self.orc.ingest_knowledge('k1', '数据库索引优化', '数据库索引查询优化的B树哈希全文方法')

        # 执行查询
        self.orc.query('数据库索引 查询优化', scene='ops')
        
        # 获取场景统计（用 orc.store 同一连接，确保写入对统计可见）
        stats = self.orc.store.scene_stats()

        # 断言场景数量至少为1，场景条目数量至少为1
        self.assertGreaterEqual(stats['scenes'], 1)
        self.assertGreaterEqual(stats['scene_entries'], 1)

    def test_scene_stats_keys(self):
        """测试空库场景统计：验证返回字典包含4个键且值均为整数"""
        stats = self.store.scene_stats()
        
        # 验证键的数量为4
        self.assertEqual(len(stats), 4)
        
        # 验证所有值都是整数
        for value in stats.values():
            self.assertIsInstance(value, int)

    def test_entity_upsert_idempotent(self):
        """测试实体插入幂等性：两次插入同一实体，获取列表长度应为1"""
        # 先插入一个条目以便关联实体
        self.store.upsert(Entry('knowledge', 'k1', '测试条目', '内容'))
        
        # 两次插入同一实体
        self.store.upsert_entity('实体A', 'k1')
        self.store.upsert_entity('实体A', 'k1')
        
        # 获取关联实体
        entities = self.store.get_entities_for_entry('k1')
        
        # 断言实体列表长度为1（去重）
        self.assertEqual(len(entities), 1)

    def test_version_upsert_ok(self):
        """测试版本插入：两次插入同一版本，场景统计中知识版本数应为1"""
        # 先插入一个条目以便关联版本
        self.store.upsert(Entry('knowledge', 'k1', '测试条目', '内容'))
        
        # 两次插入同一版本
        self.store.upsert_version('k1', 'v1', 'ops', 'k1', facets_match={'a': 1})
        self.store.upsert_version('k1', 'v1', 'ops', 'k1', facets_match={'a': 1})
        
        # 获取场景统计
        stats = self.store.scene_stats()
        
        # 断言知识版本数为1
        self.assertEqual(stats['knowledge_versions'], 1)


class TestV72SkillReuse(unittest.TestCase):
    """测试 MolBase v7.2 技能复用与自动匹配功能"""

    def setUp(self):
        """设置测试环境：创建临时数据库路径并初始化 Store 和 Orchestrator"""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.store = Store(self.db_path)
        self.orc = Orchestrator(embedding_backend='hash', db_path=self.db_path)

    def tearDown(self):
        """清理测试环境：关闭连接并删除临时目录"""
        self.orc.close()
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_auto_skill_matched(self):
        """测试自动技能匹配：插入自动生成的技能，查询应返回该技能"""
        # 插入自动生成的技能
        self.store.upsert(Entry(
            'skill', 
            'skill::csv-audit', 
            'csv-audit', 
            'CSV 审计 重复ID 缺失字段',
            meta={'auto_generated': True, 'description': '审计CSV导出文件重复ID'}
        ))
        
        # 插入一条知识兜底，确保查询有上下文
        self.store.upsert(Entry('knowledge', 'k1', 'CSV处理', 'CSV文件处理相关'))
        
        # 执行查询
        resp = self.orc.query('CSV 审计 重复ID')
        
        # 断言技能列表非空
        self.assertTrue(len(resp.skills) > 0)
        
        # 断言特定技能ID在返回的技能列表中
        skill_ids = [s['entry_id'] for s in resp.skills]
        self.assertIn('skill::csv-audit', skill_ids)

    def test_unrelated_query_no_skill(self):
        """测试无关查询无技能：查询无关内容，技能列表应为空"""
        # 插入自动生成的技能
        self.store.upsert(Entry(
            'skill', 
            'skill::csv-audit', 
            'csv-audit', 
            'CSV 审计 重复ID 缺失字段',
            meta={'auto_generated': True, 'description': '审计CSV导出文件重复ID'}
        ))
        
        # 插入一条知识兜底
        self.store.upsert(Entry('knowledge', 'k1', 'CSV处理', 'CSV文件处理相关'))
        
        # 执行无关查询
        resp = self.orc.query('量子物理黑洞')
        
        # 断言技能列表为空
        self.assertEqual(resp.skills, [])

    def test_skill_not_auto_hidden(self):
        """测试非自动技能隐藏：普通技能不应被自动匹配逻辑选中"""
        # 插入一条普通技能（meta 无 auto_generated）
        self.store.upsert(Entry(
            'skill', 
            'skill::manual', 
            'manual-skill', 
            '手动技能描述',
            meta={'description': '这是一个手动创建的技能'}
        ))
        
        # 执行自动技能匹配
        matched_skills = self.orc._match_auto_skills('任意文本')
        
        # 断言匹配结果中不包含该普通技能
        skill_ids = [s['entry_id'] for s in matched_skills]
        self.assertNotIn('skill::manual', skill_ids)


class TestV72GapDraft(unittest.TestCase):
    """测试 MolBase v7.2 差距草稿生成与确认功能"""

    def setUp(self):
        """设置测试环境：创建临时数据库路径并初始化 Store 和 Orchestrator"""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.store = Store(self.db_path)
        self.orc = Orchestrator(embedding_backend='hash', db_path=self.db_path)

    def tearDown(self):
        """清理测试环境：关闭连接并删除临时目录"""
        self.orc.close()
        self.store.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generate_drafts_from_abstain(self):
        """测试从弃权记录生成草稿：记录两次弃权，生成草稿应创建1个，跳过0个，待处理1个"""
        # 记录弃权
        self.store.record_abstain('问题A', 'knowledge', 'no_data', 'empty_recall')
        self.store.record_abstain('问题B', 'knowledge', 'ambiguous', 'low_margin')
        
        # 生成差距草稿
        r = self.orc.generate_gap_drafts()
        
        # 断言结果
        self.assertEqual(r['created'], 1)
        self.assertEqual(r['skipped'], 0)
        self.assertEqual(r['pending'], 1)

    def test_generate_idempotent(self):
        """测试生成草稿幂等性：两次生成，第二次应跳过已存在的草稿"""
        # 记录弃权
        self.store.record_abstain('问题A', 'knowledge', 'no_data', 'empty_recall')
        
        # 第一次生成
        r1 = self.orc.generate_gap_drafts()
        
        # 第二次生成
        r2 = self.orc.generate_gap_drafts()
        
        # 断言第二次生成结果
        self.assertGreaterEqual(r2['skipped'], 1)
        self.assertEqual(r2['created'], 0)

    def test_confirm_draft_closes_loop(self):
        """测试确认草稿闭环：记录弃权，生成草稿，确认草稿，验证状态更新"""
        # 记录弃权
        self.store.record_abstain('量子纠缠实验验证方法', 'knowledge', 'no_data', 'empty_recall')
        
        # 生成草稿
        r = self.orc.generate_gap_drafts()
        
        # 获取待处理草稿ID（pending 是计数，明细需从 list_drafts 取）
        pending = self.orc.store.list_drafts('pending')
        self.assertGreaterEqual(len(pending), 1)
        draft_id = pending[0]['draft_id']
        
        # 确认草稿
        out = self.orc.confirm_gap_draft(
            draft_id, 
            title='量子纠缠实验验证', 
            content='量子纠缠实验验证方法是贝尔不等式检验与纠缠光子对测量'
        )
        
        # 断言确认结果
        self.assertTrue(out['ok'])
        self.assertTrue(out['confirmed'])
        
        # 验证草稿状态已更新为 confirmed
        draft = self.store.get_draft(draft_id)
        self.assertEqual(draft['status'], 'confirmed')

    def test_confirm_missing_draft(self):
        """测试确认不存在草稿：确认不存在的草稿ID，应返回失败"""
        # 尝试确认不存在的草稿
        out = self.orc.confirm_gap_draft('draft::none')
        
        # 断言返回失败
        self.assertFalse(out['ok'])


if __name__ == '__main__':
    unittest.main()
