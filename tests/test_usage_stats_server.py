#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""usage_stats_server 单元测试（标准库 unittest，零依赖）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from usage_stats_server import build_payload, calc_cost, parse_metrics  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
