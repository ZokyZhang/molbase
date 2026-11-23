import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.parse

PROJECT_ROOT = r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase"

class TestMonitorV6Blackbox(unittest.TestCase):
    server_process = None
    base_url = "http://127.0.0.1:8801"

    @classmethod
    def setUpClass(cls):
        """
        启动服务器进程，等待健康检查通过。
        """
        # 启动服务器
        cmd = [sys.executable, "-m", "molbase.server", "--host", "127.0.0.1", "--port", "8801"]
        cls.server_process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        
        # 等待健康检查
        max_wait = 20
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                req = urllib.request.Request(f"{cls.base_url}/health")
                with urllib.request.urlopen(req, timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.5)
        
        if time.time() - start_time >= max_wait:
            cls.server_process.terminate()
            raise RuntimeError("Server failed to start within timeout")

    @classmethod
    def tearDownClass(cls):
        """
        终止服务器进程。
        """
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()

    def _get(self, path, params=None):
        """
        辅助函数：发送 GET 请求。
        """
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode('utf-8'))

    def _post(self, path, data):
        """
        辅助函数：发送 POST 请求。
        """
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode('utf-8'))

    def test_detail_api_full_fields(self):
        """
        测试 1: test_detail_api_full_fields
        GET /api/detail?metric=token&limit=5，断言返回 dict、items 为 list、首行包含关键字段。
        """
        status, data = self._get("/api/detail", {"metric": "token", "limit": 5})
        self.assertEqual(status, 200)
        self.assertIsInstance(data, dict)
        self.assertIn("items", data)
        self.assertIsInstance(data["items"], list)
        
        if data["items"]:
            first_item = data["items"][0]
            required_fields = ["label", "scene_id", "abstain", "strength", "latency_total_ms", "hits_detail"]
            for field in required_fields:
                self.assertIn(field, first_item, f"Missing field: {field}")

    def test_compare_api(self):
        """
        测试 2: test_compare_api
        GET /api/compare?group_by=label，断言 group_by='label'、groups 是 list、available_group_by 含 'scene_id' 与 'day'。
        """
        status, data = self._get("/api/compare", {"group_by": "label"})
        self.assertEqual(status, 200)
        self.assertEqual(data.get("group_by"), "label")
        self.assertIsInstance(data.get("groups"), list)
        self.assertIn("scene_id", data.get("available_group_by", []))
        self.assertIn("day", data.get("available_group_by", []))

    def test_query_label_e2e(self):
        """
        测试 3: test_query_label_e2e
        POST /query {"query":"edu-family 项目接口","label":"golden","top_k":3}，断言 HTTP 200 且响应 dict。
        再 GET /api/detail?metric=token&label=golden&limit=5，断言 total>=1。
        """
        # 1. Query
        status, data = self._post("/query", {
            "query": "edu-family 项目接口",
            "label": "golden",
            "top_k": 3
        })
        self.assertEqual(status, 200)
        self.assertIsInstance(data, dict)
        
        # 2. Detail check
        status, detail_data = self._get("/api/detail", {"metric": "token", "label": "golden", "limit": 5})
        self.assertEqual(status, 200)
        self.assertGreaterEqual(detail_data.get("total", 0), 1)

    def test_detail_filters(self):
        """
        测试 4: test_detail_filters
        GET /api/detail?metric=token&abstain=0&limit=5，断言返回 200 且 items 中每行 abstain 为 0（若有数据）。
        """
        status, data = self._get("/api/detail", {"metric": "token", "abstain": "0", "limit": 5})
        self.assertEqual(status, 200)
        
        if data.get("items"):
            for item in data["items"]:
                self.assertEqual(item.get("abstain"), 0, f"Expected abstain=0, got {item.get('abstain')}")
        else:
            # 如果没有数据，断言通过（空列表也符合 abstain=0 的条件）
            pass