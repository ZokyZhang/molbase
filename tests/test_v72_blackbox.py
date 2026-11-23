# -*- coding: utf-8 -*-
"""MolBase v7.2 黑盒测试：HTTP 层验证三方向新接口（标准库 unittest）。"""

import unittest
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json

PROJECT_ROOT = r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase"


class TestV72Blackbox(unittest.TestCase):
    """黑盒：拉起本地服务，验证 /query.skills、/api/drafts、/gap/generate、/gap/confirm。"""

    server_process = None
    base_url = "http://127.0.0.1:8802"

    @classmethod
    def setUpClass(cls):
        cls.server_process = subprocess.Popen(
            [sys.executable, "-m", "molbase.server", "--host", "127.0.0.1", "--port", "8802"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        start = time.time()
        while time.time() - start < 30:
            try:
                with urllib.request.urlopen(f"{cls.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError):
                time.sleep(1)
        raise RuntimeError("server failed to start")

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path):
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_01_query_has_skills_field(self):
        """测试 /query 响应含 skills 字段（v7.2 隐式复用）。"""
        data = self._post("/query", {"query": "数据库索引 查询优化", "scene": "ops"})
        self.assertIn("skills", data)
        self.assertIsInstance(data["skills"], list)
        # 原有字段一个不少
        for k in ("route", "results", "confidence", "abstain", "scene_id", "entities"):
            self.assertIn(k, data)

    def test_02_get_drafts(self):
        """测试 GET /api/drafts 返回 stats + drafts。"""
        data = self._get("/api/drafts?status=pending")
        self.assertIn("stats", data)
        self.assertIn("drafts", data)
        self.assertIn("total", data["stats"])
        self.assertIsInstance(data["drafts"], list)

    def test_03_gap_generate_then_confirm(self):
        """测试弃答补数闭环：generate→drafts→confirm。"""
        # 先造一个弃答（无关查询触发 abstain）
        self._post("/query", {"query": "zzz无关查询qqq罕见词xyzzy", "scene": "global"})
        gen = self._post("/gap/generate", {"limit": 50})
        self.assertIn("created", gen)
        self.assertIn("skipped", gen)
        self.assertIn("pending", gen)
        # 取一个 pending 草稿确认（若有）
        drafts = self._get("/api/drafts?status=pending")["drafts"]
        if drafts:
            did = drafts[0]["draft_id"]
            out = self._post("/gap/confirm", {
                "draft_id": did,
                "title": "补数测试条目",
                "content": "这是一条通过弃答补数流程确认入库的测试知识条目。",
            })
            self.assertTrue(out.get("ok"), out)
            self.assertIn("state", out)
            self.assertIn("verified", out)
            self.assertIn("confidence", out)
        else:
            self.skipTest("no pending draft to confirm")

    def test_04_gap_confirm_missing_id(self):
        """测试 /gap/confirm 缺 draft_id 返回 400。"""
        req = urllib.request.Request(
            f"{self.base_url}/gap/confirm",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
