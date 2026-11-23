# -*- coding: utf-8 -*-
"""
v6 全量明细监控白盒测试。

覆盖：metrics 新列幂等迁移、detail 落库、测试数据隔离、分组对比、
query 端到端打标、metric_detail 多维过滤。
仅使用标准库，不依赖外部网络 / Ollama（使用 hash embedding）。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from molbase.orchestrator import Orchestrator
from molbase.store import Entry, Store


EXPECTED_COLUMNS = {
    "scene_id", "label", "tags", "query_tags", "entities", "refs", "rewritten",
    "resolved_version_id", "refs_miss", "abstain", "abstain_reason",
    "abstain_subreason", "strength", "margin", "top1_score", "top1_entry_id",
    "top1_title", "hits_detail", "web_results", "embed_model",
    "embedding_backend", "rewrite_used", "latency_total_ms",
    "latency_recall_ms", "latency_rerank_ms", "confidence_threshold",
}


class TestMonitorV6(unittest.TestCase):
    """MolBase v6 全量明细监控白盒测试。"""

    def setUp(self) -> None:
        """每个用例使用独立临时库。"""
        self.tmp = tempfile.mkdtemp(prefix="molbase-v6-")
        self.db_path = os.path.join(self.tmp, "metrics.db")
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        """先关闭连接再清理临时文件，规避 Windows 文件占用。"""
        try:
            self.store.close()
        except Exception:
            pass
        for name in os.listdir(self.tmp):
            try:
                os.remove(os.path.join(self.tmp, name))
            except OSError:
                pass
        try:
            os.rmdir(self.tmp)
        except OSError:
            pass

    def _metric(self, query: str, label: str = "", is_test: int = 0, **detail) -> None:
        """写入一条带全维度 detail 的埋点（统一走真实签名）。"""
        payload = {"label": label}
        payload.update(detail)
        self.store.record_metric(
            query=query,
            route="knowledge",
            hits=1,
            actual_tokens=100,
            baseline_tokens=200,
            confidence=0.7,
            is_test=is_test,
            recall_candidates=4,
            recall_tokens=200,
            session_id="whitebox",
            detail=payload,
        )

    def test_01_schema_full_columns(self) -> None:
        """metrics 表包含全部 26 个新列，且重复初始化幂等。"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(metrics)")}
        conn.close()
        self.assertTrue(EXPECTED_COLUMNS.issubset(columns), EXPECTED_COLUMNS - columns)
        second = Store(self.db_path)
        second.close()

    def test_02_record_metric_detail_json(self) -> None:
        """detail 中的 JSON 字段可正确序列化并原样读回。"""
        self._metric(
            "edu-family 接口",
            label="golden",
            scene_id="edu-family",
            tags=["运维", "环境"],
            entities=["edu-family"],
            hits_detail=[{"entry_id": "k1", "title": "接口文档", "score": 0.8}],
            strength=0.72,
            latency_total_ms=123,
        )
        row = self.store.list_metrics(limit=1)[0]
        self.assertEqual(row["label"], "golden")
        self.assertEqual(row["scene_id"], "edu-family")
        self.assertEqual(json.loads(row["tags"]), ["运维", "环境"])
        self.assertEqual(json.loads(row["hits_detail"])[0]["entry_id"], "k1")
        self.assertEqual(row["latency_total_ms"], 123)

    def test_03_list_metrics_test_filter(self) -> None:
        """默认排除 is_test=1；include_test=True 时全部返回。"""
        self._metric("real", label="real", is_test=0)
        self._metric("demo", label="real", is_test=1)
        self.assertEqual(len(self.store.list_metrics(limit=10)), 1)
        self.assertEqual(len(self.store.list_metrics(limit=10, include_test=True)), 2)

    def test_04_metric_compare_groups(self) -> None:
        """按 label 分组对比，非法分组维度回退 scene_id。"""
        self._metric("good", label="golden", strength=0.8)
        self._metric("bad", label="negative", strength=0.2, abstain=1)
        result = self.store.metric_compare("label")
        groups = {g["g"]: g for g in result["groups"]}
        self.assertIn("golden", groups)
        self.assertIn("negative", groups)
        for key in ("conf_avg", "strength_avg", "intercept_rate", "abstain_rate", "saved_ratio"):
            self.assertIn(key, groups["golden"])
        self.assertEqual(self.store.metric_compare("__bad__")["group_by"], "scene_id")

    def test_05_query_detail_end_to_end(self) -> None:
        """query 打标后全维度落库（先建条目再建 Orchestrator，避免索引未建）。"""
        init = Orchestrator(embedding_backend="hash", db_path=self.db_path)
        init.store.upsert(Entry(
            "knowledge", "k1", "edu-family 项目接口",
            "edu-family 家庭教育项目 API 接口文档 登录注册 档案查询", source="test",
        ))
        init.close()
        self.store.close()
        orc = Orchestrator(embedding_backend="hash", db_path=self.db_path)
        try:
            resp = orc.query("edu-family 项目接口", label="golden", session_id="whitebox")
            self.assertTrue(resp.results or resp.abstain)
            row = orc.store.list_metrics(limit=1)[0]
            self.assertEqual(row["label"], "golden")
            self.assertTrue(row["scene_id"])
            self.assertIn(row["abstain"], (0, 1))
            self.assertGreaterEqual(row["latency_total_ms"], 0)
            self.assertIsInstance(json.loads(row["hits_detail"]), list)
        finally:
            orc.close()
            self.store = Store(self.db_path)

    def test_06_metric_detail_filters(self) -> None:
        """metric_detail 的 label / abstain 过滤生效。"""
        self._metric("good", label="golden", abstain=0)
        self._metric("bad", label="negative", abstain=1)
        orc = Orchestrator(embedding_backend="hash", db_path=self.db_path)
        try:
            only_golden = orc.metric_detail("token", label="golden")
            self.assertEqual(only_golden["total"], 1)
            self.assertEqual(only_golden["items"][0]["label"], "golden")
            only_abstain = orc.metric_detail("token", abstain="1")
            self.assertEqual(only_abstain["total"], 1)
            self.assertEqual(only_abstain["items"][0]["abstain"], 1)
        finally:
            orc.close()


if __name__ == "__main__":
    unittest.main()
