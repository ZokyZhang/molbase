import unittest
import json
import time
import subprocess
import sys
import os
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8799"
PROJECT_ROOT = r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase"

class TestP0BlackBoxHTTP(unittest.TestCase):
    proc = None

    @classmethod
    def setUpClass(cls):
        cls.proc = subprocess.Popen(
            [sys.executable, "-m", "molbase.server", "--host", "127.0.0.1", "--port", "8799"],
            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(40):
            time.sleep(0.5)
            try:
                req = urllib.request.Request(f"{BASE_URL}/health")
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    break
            except Exception:
                pass
        else:
            cls.fail("服务启动超时")

    @classmethod
    def tearDownClass(cls):
        if cls.proc:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.proc.kill()

    def post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    def get(self, path):
        req = urllib.request.Request(f"{BASE_URL}{path}")
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    def test_health_stats_metrics(self):
        """健康检查与指标接口验证"""
        h = self.get("/health")
        self.assertEqual(h["status"], "ok")
        for p in ["/stats", "/metrics", "/overview"]:
            d = self.get(p)
            self.assertIsInstance(d, dict)
            self.assertTrue(len(d) > 0)

    def test_real_questions_should_not_abstain(self):
        """真实业务问题不应拒答"""
        golden = [
            ("MolBase 质量底线设计 四道防线 两道闸门", "质量底线"),
            ("SearXNG 部署 网络搜索兜底后端", "SearXNG"),
            ("虚拟机环境与访问", "虚拟机"),
            ("WF-13 Skill 强制嵌入规范 三道卡口", "WF-13"),
            ("CSV 导出文件中的重复 ID 缺失必填字段 审计", "csv-audit"),
            ("提取 PDF 文档中的表单字段与文本", "pdf-extract"),
            ("qwen-worker 分工约束 本地模型代工", "qwen-worker"),
            ("家庭教育物联网系统 一期 AI 批改 模型选型", "AI 批改"),
        ]
        for q, kw in golden:
            with self.subTest(query=q):
                r = self.post("/query", {"query": q, "top_k": 5, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
                self.assertFalse(r["abstain"])
                self.assertTrue(len(r["results"]) > 0)
                self.assertGreater(r["confidence"], 0)
                self.assertIn(kw, r["results"][0]["title"])

    def test_negative_questions_should_abstain(self):
        """无关问题应拒答"""
        for q in ["今天天气怎么样", "给我讲个冷笑话", "量子纠缠如何在MySQL里配置"]:
            with self.subTest(query=q):
                r = self.post("/query", {"query": q, "top_k": 5, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
                self.assertTrue(r["abstain"] or len(r["results"]) == 0)

    def test_response_fields_complete(self):
        """响应字段完整性检查"""
        r = self.post("/query", {"query": "MolBase 架构", "top_k": 1, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
        for f in ["scene_id", "entities", "resolved_version_id", "refs_miss"]:
            self.assertIn(f, r)
        self.assertIsInstance(r["refs_miss"], bool)

    def test_scene_same_question_different_scene(self):
        """不同场景路由验证"""
        q = "怎么配置"
        r1 = self.post("/query", {"query": q, "top_k": 1, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
        r2 = self.post("/query", {"query": q, "top_k": 1, "scene": "family-edu", "refs": [], "tags": [], "session_id": "s1"})
        self.assertEqual(r1["scene_id"], "ops")
        self.assertEqual(r2["scene_id"], "family-edu")

    def test_refs_hit_and_miss(self):
        """引用命中与未命中验证"""
        r1 = self.post("/query", {"query": "测试", "top_k": 1, "scene": "ops", "refs": ["kb::not-exist-xyz"], "tags": [], "session_id": "s1"})
        self.assertTrue(r1["refs_miss"])
        r2 = self.post("/query", {"query": "MolBase 服务化部署", "top_k": 1, "scene": "ops", "refs": ["kb::MolBase部署"], "tags": [], "session_id": "s1"})
        self.assertFalse(r2["refs_miss"])
        self.assertTrue(len(r2["results"]) > 0)

    def test_ingest_then_query(self):
        """知识注入与检索闭环"""
        eid = f"kb::p0-blackbox-test-{int(time.time())}"
        try:
            self.post("/ingest/knowledge", {"entry_id": eid, "title": "P0黑盒测试专用知识点 分布式锁 Redis 实现", "content": "Redis 分布式锁实现细节", "source": "test", "meta": {}})
            time.sleep(1)
            r = self.post("/query", {"query": "P0黑盒测试专用知识点 分布式锁", "top_k": 5, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
            found = any(x["entry_id"] == eid for x in r["results"])
            self.assertTrue(found or r.get("state") in ["duplicate", "merge"])
        finally:
            pass

    def test_feedback_endpoint(self):
        """反馈接口验证"""
        r = self.post("/query", {"query": "MolBase 架构", "top_k": 1, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
        eid = r["results"][0]["entry_id"]
        fb = self.post("/feedback", {"entry_id": eid, "action": "hit"})
        self.assertTrue(fb["ok"])

    def test_bad_request_400(self):
        """空查询应返回 400"""
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/query", {"query": "", "top_k": 1, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
        self.assertEqual(cm.exception.code, 400)

    def test_abstain_rate_realistic(self):
        """拒答率合理性检查"""
        golden = [
            ("MolBase 质量底线设计 四道防线 两道闸门", "质量底线"),
            ("SearXNG 部署 网络搜索兜底后端", "SearXNG"),
            ("虚拟机环境与访问", "虚拟机"),
            ("WF-13 Skill 强制嵌入规范 三道卡口", "WF-13"),
            ("CSV 导出文件中的重复 ID 缺失必填字段 审计", "csv-audit"),
            ("提取 PDF 文档中的表单字段与文本", "pdf-extract"),
            ("qwen-worker 分工约束 本地模型代工", "qwen-worker"),
            ("家庭教育物联网系统 一期 AI 批改 模型选型", "AI 批改"),
        ]
        cnt = 0
        for q, _ in golden:
            with self.subTest(query=q):
                r = self.post("/query", {"query": q, "top_k": 5, "scene": "ops", "refs": [], "tags": [], "session_id": "s1"})
                if r["abstain"]:
                    cnt += 1
        self.assertLessEqual(cnt, 2)