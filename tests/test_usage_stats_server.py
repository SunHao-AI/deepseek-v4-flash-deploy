#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""usage_stats_server 单元测试（标准库 unittest，零依赖）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from usage_stats_server import build_payload, calc_cost, parse_metrics  # noqa: E402
from usage_stats_server import UsageCollector, UsageHandler  # noqa: E402


class FakeResponse:
    """模拟 urllib.request.urlopen 的返回值（支持 with 上下文与 read）。"""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


class TestParseMetrics(unittest.TestCase):
    def test_old_metric_names(self):
        text = (
            "# HELP prompt_tokens_total Number of tokens evaluated\n"
            'prompt_tokens_total{t="all"} 823456\n'
            'tokens_predicted_total{t="all"} 411111\n'
            'prompt_tokens_seconds{t="all"} 800\n'
            'predicted_tokens_seconds{t="all"} 55.2\n'
        )
        m = parse_metrics(text)
        self.assertEqual(m["prompt_total"], 823456.0)
        self.assertEqual(m["predicted_total"], 411111.0)
        self.assertEqual(m["prompt_rate"], 800.0)
        self.assertAlmostEqual(m["predicted_rate"], 55.2)

    def test_new_llamacpp_prefix(self):
        # llama.cpp b10298+（DeepSeek-V4 部署版本）实际命名
        text = (
            'llamacpp:prompt_tokens_total 10\n'
            'llamacpp:tokens_predicted_total 20\n'
            'llamacpp:prompt_tokens_seconds 300\n'
            'llamacpp:predicted_tokens_seconds 60\n'
        )
        m = parse_metrics(text)
        self.assertEqual(m["prompt_total"], 10.0)
        self.assertEqual(m["predicted_total"], 20.0)
        self.assertEqual(m["prompt_rate"], 300.0)
        self.assertAlmostEqual(m["predicted_rate"], 60.0)

    def test_old_llamacpp_prefix(self):
        # llama.cpp 2024-2025 中期旧命名（tokens_evaluated_total），仍需兼容
        text = (
            'llamacpp:tokens_evaluated_total{t="all"} 10\n'
            'llamacpp:tokens_predicted_total{t="all"} 20\n'
            'llamacpp:prompt_tokens_seconds{t="all"} 300\n'
            'llamacpp:tokens_predicted_seconds{t="all"} 60\n'
        )
        m = parse_metrics(text)
        self.assertEqual(m["prompt_total"], 10.0)
        self.assertEqual(m["predicted_total"], 20.0)
        self.assertEqual(m["prompt_rate"], 300.0)
        self.assertAlmostEqual(m["predicted_rate"], 60.0)

    def test_b10298_full_metrics(self):
        # 官方 llama.cpp b10298 /metrics 完整输出（含 HELP/TYPE 注释行）
        text = (
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n"
            "# TYPE llamacpp:prompt_tokens_total counter\n"
            "llamacpp:prompt_tokens_total 73163\n"
            "# HELP llamacpp:prompt_seconds_total Prompt process time\n"
            "# TYPE llamacpp:prompt_seconds_total counter\n"
            "llamacpp:prompt_seconds_total 83.483\n"
            "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n"
            "# TYPE llamacpp:tokens_predicted_total counter\n"
            "llamacpp:tokens_predicted_total 7637\n"
            "# HELP llamacpp:tokens_predicted_seconds_total Predict process time\n"
            "# TYPE llamacpp:tokens_predicted_seconds_total counter\n"
            "llamacpp:tokens_predicted_seconds_total 160.979\n"
            "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.\n"
            "# TYPE llamacpp:prompt_tokens_seconds gauge\n"
            "llamacpp:prompt_tokens_seconds 876.382\n"
            "# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.\n"
            "# TYPE llamacpp:predicted_tokens_seconds gauge\n"
            "llamacpp:predicted_tokens_seconds 47.441\n"
        )
        m = parse_metrics(text)
        self.assertEqual(m["prompt_total"], 73163.0)
        self.assertEqual(m["predicted_total"], 7637.0)
        self.assertAlmostEqual(m["prompt_rate"], 876.382)
        self.assertAlmostEqual(m["predicted_rate"], 47.441)

    def test_empty_text(self):
        m = parse_metrics("")
        self.assertEqual(m["prompt_total"], 0.0)
        self.assertEqual(m["predicted_total"], 0.0)
        self.assertEqual(m["prompt_rate"], 0.0)
        self.assertEqual(m["predicted_rate"], 0.0)


class TestCalcCost(unittest.TestCase):
    def test_cost(self):
        self.assertAlmostEqual(calc_cost(1_000_000, 1_000_000, 1.0, 2.0), 3.0)
        self.assertAlmostEqual(calc_cost(823_456, 411_111, 1.0, 2.0), 0.823456 + 0.822222, places=4)


class TestBuildPayload(unittest.TestCase):
    def test_ok_payload_with_budget(self):
        snap = {"ok": True, "error": None, "prompt_total": 500_000, "predicted_total": 250_000, "prompt_rate": 0.0, "predicted_rate": 40.0}
        p = build_payload(snap, 1.0, 2.0, budget=10.0)
        self.assertTrue(p["isValid"])
        self.assertAlmostEqual(p["used"], 1.0)
        self.assertEqual(p["total"], 10.0)
        self.assertAlmostEqual(p["remaining"], 9.0)
        self.assertEqual(p["unit"], "CNY")
        self.assertIn("40.0 tok/s", p["extra"])
        self.assertIn("750,000", p["extra"])

    def test_ok_payload_no_budget(self):
        snap = {"ok": True, "error": None, "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        p = build_payload(snap, 1.0, 2.0, budget=None)
        self.assertIsNone(p["total"])
        self.assertIsNone(p["remaining"])

    def test_failure_payload(self):
        snap = {"ok": False, "error": "connection refused", "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        p = build_payload(snap, 1.0, 2.0, budget=None)
        self.assertFalse(p["isValid"])
        self.assertIn("connection refused", p["invalidMessage"])


class TestUsageCollector(unittest.TestCase):
    def test_poll_once_success(self):
        metrics_text = (
            'llamacpp:prompt_tokens_total 100\n'
            'llamacpp:tokens_predicted_total 200\n'
            'llamacpp:prompt_tokens_seconds 300\n'
            'llamacpp:predicted_tokens_seconds 50\n'
        )

        def fake_urlopen(request, timeout: float = 0):  # noqa: ARG001
            self.assertIsNone(request.get_header("Authorization"))
            return FakeResponse(metrics_text.encode("utf-8"))

        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            collector._poll_once()
        snap = collector.snapshot()
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["prompt_total"], 100.0)
        self.assertEqual(snap["predicted_total"], 200.0)
        self.assertAlmostEqual(snap["predicted_rate"], 50.0)

    def test_poll_once_with_api_key(self):
        def fake_urlopen(request, timeout: float = 0):  # noqa: ARG001
            self.assertEqual(request.get_header("Authorization"), "Bearer secret")
            return FakeResponse(b"")

        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key="secret")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            collector._poll_once()
        # 空 body 解析全 0，仍算成功（ok=True）
        self.assertTrue(collector.snapshot()["ok"])

    def test_poll_once_failure(self):
        def fake_urlopen(request, timeout: float = 0):  # noqa: ARG001
            raise ConnectionError("connection refused")

        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            collector._poll_once()
        snap = collector.snapshot()
        self.assertFalse(snap["ok"])
        self.assertIn("connection refused", snap["error"])

    def test_on_demand_has_no_background_thread(self):
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None, mode="on-demand")
        self.assertIsNone(collector._thread)
        # start() 不应抛错（无后台线程）
        collector.start()
        collector.stop()

    def test_poll_mode_has_background_thread(self):
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None, mode="poll")
        self.assertIsNotNone(collector._thread)

    def test_get_snapshot_on_demand_fetches_fresh(self):
        # on-demand 模式：get_snapshot() 每次同步拉取最新指标
        metrics_text = (
            'llamacpp:prompt_tokens_total 100\n'
            'llamacpp:tokens_predicted_total 200\n'
            'llamacpp:prompt_tokens_seconds 300\n'
            'llamacpp:predicted_tokens_seconds 50\n'
        )
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None, mode="on-demand")
        with mock.patch("urllib.request.urlopen", side_effect=lambda req, timeout=0: FakeResponse(metrics_text.encode("utf-8"))):
            snap = collector.get_snapshot()
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["prompt_total"], 100.0)
        self.assertEqual(snap["predicted_total"], 200.0)
        self.assertAlmostEqual(snap["predicted_rate"], 50.0)

    def test_get_snapshot_poll_returns_cached(self):
        # poll 模式：get_snapshot() 不触发网络请求，直接返回缓存快照
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None, mode="poll")
        collector._snapshot = {"ok": True, "error": None, "prompt_total": 7.0, "predicted_total": 8.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("poll 模式不应发起网络请求")):
            snap = collector.get_snapshot()
        self.assertEqual(snap["prompt_total"], 7.0)

    def test_rate_fallback_by_delta(self):
        # 无速率 gauge 时，用轮询差值计算速率；mock time.monotonic 冻结时钟保证跨平台稳定
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        text1 = 'llamacpp:tokens_predicted_total{t="all"} 100\n'
        text2 = 'llamacpp:tokens_predicted_total{t="all"} 150\n'
        with mock.patch("usage_stats_server.time.monotonic", return_value=100.0):
            with mock.patch("urllib.request.urlopen", side_effect=lambda req, timeout=0: FakeResponse(text1.encode("utf-8"))):
                collector._poll_once()
        # 第二次轮询，模拟 2 秒间隔（monotonic 返回 102.0）
        with mock.patch("usage_stats_server.time.monotonic", return_value=102.0):
            with mock.patch("urllib.request.urlopen", side_effect=lambda req, timeout=0: FakeResponse(text2.encode("utf-8"))):
                collector._poll_once()
        snap = collector.snapshot()
        self.assertAlmostEqual(snap["predicted_rate"], 25.0)  # (150-100)/2


class TestUsageHandler(unittest.TestCase):
    def _start_server(self, collector, budget):
        server = ThreadingHTTPServer(("127.0.0.1", 0), UsageHandler)
        UsageHandler.config = {"collector": collector, "price_in": 1.0, "price_out": 2.0, "budget": budget}
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_api_usage_ok(self):
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        collector._snapshot = {"ok": True, "error": None, "prompt_total": 500_000, "predicted_total": 250_000, "prompt_rate": 0.0, "predicted_rate": 40.0}
        server = self._start_server(collector, budget=10.0)
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/usage", timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(body["isValid"])
            self.assertAlmostEqual(body["used"], 1.0)
            self.assertEqual(body["total"], 10.0)
            self.assertIn("40.0 tok/s", body["extra"])
        finally:
            server.shutdown()
            server.server_close()

    def test_api_usage_failure(self):
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        collector._snapshot = {"ok": False, "error": "boom", "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        server = self._start_server(collector, budget=None)
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/usage", timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            self.assertFalse(body["isValid"])
            self.assertIn("boom", body["invalidMessage"])
        finally:
            server.shutdown()
            server.server_close()

    def test_unknown_path_404(self):
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        server = self._start_server(collector, budget=None)
        try:
            import urllib.request

            try:
                urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/other", timeout=5)
                self.fail("应返回 404")
            except urllib.error.HTTPError as error:
                self.assertEqual(error.code, 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
