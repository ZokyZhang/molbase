import unittest
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json
import tempfile
import os

PROJECT_ROOT = r"C:\Users\Administrator\WorkBuddy\2026-11-17-22-17-16\molbase"

class TestEvolutionV7Blackbox(unittest.TestCase):
    server_process = None
    base_url = "http://127.0.0.1:8801"

    @classmethod
    def setUpClass(cls):
        # Start server
        cls.server_process = subprocess.Popen(
            [sys.executable, "-m", "molbase.server", "--host", "127.0.0.1", "--port", "8801"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Poll /health
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                req = urllib.request.Request(f"{cls.base_url}/health")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError):
                time.sleep(1)
        raise RuntimeError("Server failed to start within timeout")

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()

    def test_01_get_overview_evolution_health(self):
        url = f"{self.base_url}/overview"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode('utf-8'))
            self.assertIn("evolution_health", data)

    def test_02_post_feedback_auto_source(self):
        # Post feedback
        payload = json.dumps({"entry_id": "bb_test_01", "action": "adopt", "source": "auto"}).encode('utf-8')
        req = urllib.request.Request(f"{self.base_url}/feedback", data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
        
        # Check detail
        url = f"{self.base_url}/api/detail?metric=feedback&limit=5"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            items = data.get("items", [])
            # At least one item should have source='auto'
            found_auto = any(item.get("source") == "auto" for item in items)
            self.assertTrue(found_auto)

    def test_03_get_api_detail_evolve(self):
        url = f"{self.base_url}/api/detail?metric=evolve&limit=5"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read().decode('utf-8'))
            # items should be a list
            self.assertIsInstance(data.get("items"), list)

    def test_04_post_feedback_illegal_source(self):
        # Send illegal source 'hack', expect it to be normalized to 'manual' or handled gracefully
        # Spec: "返回 source='manual'" implies the response or the stored data reflects manual.
        # Usually APIs normalize input. We check the response or subsequent read.
        payload = json.dumps({"entry_id": "bb_test_02", "action": "adopt", "source": "hack"}).encode('utf-8')
        req = urllib.request.Request(f"{self.base_url}/feedback", data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            # Check if the system treats it as manual (either in response or by re-reading)
            # Assuming the API response might echo the normalized source or we check via detail
            data = json.loads(response.read().decode('utf-8'))
            # If the API returns the processed source
            if "source" in data:
                self.assertEqual(data["source"], "manual")
            else:
                # Fallback: check via detail API
                detail_url = f"{self.base_url}/api/detail?metric=feedback&limit=1"
                req_d = urllib.request.Request(detail_url)
                with urllib.request.urlopen(req_d, timeout=10) as resp:
                    detail_data = json.loads(resp.read().decode('utf-8'))
                    if detail_data.get("items"):
                        self.assertEqual(detail_data["items"][0].get("source"), "manual")

    def test_05_get_dashboard_markers(self):
        url = f"{self.base_url}/dashboard"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
            self.assertIn("feedback", content)
            self.assertIn("evolve", content)
            self.assertIn("renderFlywheelPanels", content)

if __name__ == '__main__':
    unittest.main()