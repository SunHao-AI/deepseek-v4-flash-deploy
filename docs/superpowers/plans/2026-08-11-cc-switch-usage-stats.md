# cc-switch 用量统计服务 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个轻量统计服务，轮询 llama-server `/metrics`，按 DeepSeek-V4-Flash 官方价格折算累计费用，暴露 `/api/usage` 供 cc-switch 用量查询展示。

**Architecture:** 单文件 Python 标准库常驻服务（`script/usage_stats_server.py`），后台线程每 5s 轮询 llama-server Prometheus 端点，内存态保存用量快照；HTTP 层提供 `/api/usage` JSON 接口。llama-server 启动命令追加 `--metrics`，nginx 新增一条精确匹配 location 将 cc-switch 的用量查询请求转发到统计服务。API 主流量路径不变。

**Tech Stack:** Python 3.10+（标准库：http.server / urllib / threading / re / unittest），零第三方依赖。

## Global Constraints

- 运行期零第三方依赖：统计服务与测试仅使用 Python 标准库
- Python 版本要求 `>=3.10`（pyproject.toml）
- 配置优先级：命令行参数 > 环境变量 > `.env` > 脚本内置默认值
- 统计服务端口必须用 5002（5001 已被 nginx `location /` 占用）
- llama-server 地址：`http://192.168.77.210:18888`
- 费用单价：输入 1.0 元/M、输出 2.0 元/M（DeepSeek-V4-Flash 官方价，.env 可配置）
- 所有新增/修改文件提交到 git，提交信息遵循仓库现有风格（`feat:` / `docs:` / `test:` 前缀，中文描述）
- 测试运行命令（项目根目录）：`python -m unittest tests/test_usage_stats_server.py -v`（Windows 本机），若用 uv 则 `uv run python -m unittest tests/test_usage_stats_server.py -v`

---

### Task 1: 统计服务核心纯函数（指标解析 + 费用计算 + 响应构造）

**Files:**
- Create: `script/usage_stats_server.py`（本任务写入模块头、常量、`parse_metrics`、`calc_cost`、`build_payload`、`load_env`；Task 2 追加 Collector 与 HTTP 层）
- Create: `tests/test_usage_stats_server.py`（本任务先写解析/费用/响应三个测试类）

**Interfaces:**
- Consumes: 无（本任务为最底层，无前置依赖）
- Produces:
  - `parse_metrics(text: str) -> dict[str, float]` —— 键为 `prompt_total` / `predicted_total` / `prompt_rate` / `predicted_rate`
  - `calc_cost(prompt_total: float, predicted_total: float, price_in: float, price_out: float) -> float`
  - `build_payload(snap: dict, price_in: float, price_out: float, budget: float | None) -> dict`
  - `load_env(env_path: Path | None = None) -> Path`
  - 模块级常量 `METRIC_NAMES: dict[str, list[str]]`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m unittest tests/test_usage_stats_server.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'usage_stats_server'`）

- [ ] **Step 3: 写最小实现（模块骨架 + 纯函数）**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================
# @File   : script/usage_stats_server.py
# @Desc   : cc-switch 用量统计服务 —— 轮询 llama-server /metrics，折算费用，暴露 /api/usage
# ===============================================================================
"""cc-switch 用量统计服务。

轮询 llama-server 的 /metrics（Prometheus）端点，聚合自启动以来累计的
输入/输出 tokens 与生成速率，按 DeepSeek-V4-Flash 官方价格折算累计费用，
并暴露 /api/usage 供 cc-switch 的「用量查询 → 自定义」配置消费。

纯标准库实现，零第三方依赖。运行：
    python3 script/usage_stats_server.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Prometheus 指标名（按 llama.cpp 新旧版本前缀做容错匹配，取第一个命中的）
METRIC_NAMES = {
    "prompt_total": ["llamacpp:tokens_evaluated_total", "llama_tokens_evaluated_total", "prompt_tokens_total"],
    "predicted_total": ["llamacpp:tokens_predicted_total", "llama_tokens_predicted_total", "tokens_predicted_total"],
    "prompt_rate": ["llamacpp:prompt_tokens_seconds", "prompt_tokens_seconds"],
    "predicted_rate": ["llamacpp:tokens_predicted_seconds", "llamacpp:predicted_tokens_seconds", "predicted_tokens_seconds"],
}


def _fmt_int(value: float) -> str:
    """千分位格式化整数。"""
    return f"{int(round(value)):,}"


def parse_metrics(text: str) -> dict[str, float]:
    """解析 Prometheus 文本，返回 {prompt_total, predicted_total, prompt_rate, predicted_rate}。

    缺失的指标返回 0.0；速率为 gauge 值（tok/s）。
    """
    result = {"prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
    for key, names in METRIC_NAMES.items():
        for name in names:
            pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.MULTILINE)
            m = pattern.search(text)
            if m:
                try:
                    result[key] = float(m.group(1))
                except ValueError:
                    pass
                break
    return result


def calc_cost(prompt_total: float, predicted_total: float, price_in: float, price_out: float) -> float:
    """按元/M tokens 单价折算累计费用（元）。"""
    return prompt_total / 1e6 * price_in + predicted_total / 1e6 * price_out


def load_env(env_path: Path | None = None) -> Path:
    """加载 .env（已存在的环境变量优先，不覆盖），返回实际检查的 .env 路径。"""
    path = env_path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return path
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return path
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return path


def build_payload(snap: dict, price_in: float, price_out: float, budget: float | None) -> dict:
    """由用量快照构造 cc-switch 可识别的 /api/usage 响应。

    snap 键：ok(bool)、error(str|None)、prompt_total、predicted_total、prompt_rate、predicted_rate。
    """
    if not snap["ok"]:
        return {"isValid": False, "invalidMessage": f"llama-server 不可用：{snap['error'] or '未知错误'}"}
    prompt = snap["prompt_total"]
    predicted = snap["predicted_total"]
    used = round(calc_cost(prompt, predicted, price_in, price_out), 2)
    payload = {
        "isValid": True,
        "used": used,
        "unit": "CNY",
        "planName": "DeepSeek-V4-Flash 本地部署",
        "extra": (
            f"累计 {_fmt_int(prompt + predicted)} tokens"
            f"（输入 {_fmt_int(prompt)} / 输出 {_fmt_int(predicted)}）"
            f"| 生成速率 {snap['predicted_rate']:.1f} tok/s"
        ),
    }
    if budget is not None:
        payload["total"] = budget
        payload["remaining"] = round(max(budget - used, 0.0), 2)
    else:
        payload["total"] = None
        payload["remaining"] = None
    return payload
```

（注：文件顶部已 import 后续 Task 用到的模块——argparse/json/threading/time/urllib.request/http.server——让测试文件在 Task 1 结束即可被 import；这些模块在本任务中暂未使用，属于 Task 2 的前置 import，不算死代码。）

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m unittest tests/test_usage_stats_server.py -v`
Expected: PASS（9 个用例全过）

- [ ] **Step 5: 提交**

```bash
git add script/usage_stats_server.py tests/test_usage_stats_server.py
git commit -m "feat: 新增 cc-switch 用量统计服务（指标解析/费用计算/响应构造）"
```

---

### Task 2: 轮询采集器与 HTTP 服务（/api/usage）

**Files:**
- Modify: `script/usage_stats_server.py`（追加 `UsageCollector` 类、`UsageHandler` 类、`main()` 与 `__main__` 入口）
- Modify: `tests/test_usage_stats_server.py`（追加 Collector 与 HTTP 层测试类）

**Interfaces:**
- Consumes: Task 1 的 `parse_metrics`、`calc_cost`、`build_payload`、`load_env`、`METRIC_NAMES`
- Produces:
  - `UsageCollector.__init__(base_url: str, poll_interval: float, api_key: str | None)`
  - `UsageCollector.start() / stop() / snapshot() -> dict`、`UsageCollector._poll_once()`
  - `UsageHandler`（http.server handler，类属性 `config: dict`，键 `collector` / `price_in` / `price_out` / `budget`）
  - `main()` 入口（读取环境变量 `USAGE_PORT` / `USAGE_LLAMA_BASE` / `USAGE_POLL_INTERVAL` / `USAGE_PRICE_IN` / `USAGE_PRICE_OUT` / `USAGE_BUDGET` / `LLAMA_API_KEY`）

- [ ] **Step 1: 追加失败测试**

**第一步，在顶部 import 区（`sys.path.insert` 之后）补全本任务所需的 import：**

```python
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

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
```

**第二步，在文件末尾（`if __name__ == "__main__":` 之前）追加测试类：**

```python
class TestUsageCollector(unittest.TestCase):
    def test_poll_once_success(self):
        metrics_text = (
            'llamacpp:tokens_evaluated_total{t="all"} 100\n'
            'llamacpp:tokens_predicted_total{t="all"} 200\n'
            'llamacpp:prompt_tokens_seconds{t="all"} 300\n'
            'llamacpp:tokens_predicted_seconds{t="all"} 50\n'
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

    def test_rate_fallback_by_delta(self):
        # 无速率 gauge 时，用轮询差值计算速率
        collector = UsageCollector("http://127.0.0.1:18888", 5, api_key=None)
        text1 = 'llamacpp:tokens_predicted_total{t="all"} 100\n'
        text2 = 'llamacpp:tokens_predicted_total{t="all"} 150\n'
        with mock.patch("urllib.request.urlopen", side_effect=lambda req, timeout=0: FakeResponse(text1.encode("utf-8"))):
            collector._poll_once()
        # 手动前移 last 时间，模拟 2 秒间隔
        collector._last["time"] -= 2.0
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
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m unittest tests/test_usage_stats_server.py -v`
Expected: FAIL（`ImportError: cannot import name 'UsageCollector'`）

- [ ] **Step 3: 实现 Collector 与 HTTP 层**

在 `build_payload` 之后、文件末尾追加：

```python
class UsageCollector:
    """后台轮询 llama-server /metrics，维护最近一次用量快照。"""

    def __init__(self, base_url: str, poll_interval: float, api_key: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.api_key = api_key
        self._lock = threading.Lock()
        self._snapshot = {"ok": False, "error": None, "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        self._last = {"time": None, "predicted_total": 0.0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.poll_interval)

    def _poll_once(self) -> None:
        url = f"{self.base_url}/metrics"
        try:
            request = urllib.request.Request(url)
            if self.api_key:
                request.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(request, timeout=10) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            metrics = parse_metrics(body)
            now = time.monotonic()
            rate = metrics["predicted_rate"]
            if rate <= 0.0 and self._last["time"] is not None:
                delta_t = now - self._last["time"]
                delta_tokens = metrics["predicted_total"] - self._last["predicted_total"]
                if delta_t > 0:
                    rate = delta_tokens / delta_t
            with self._lock:
                self._snapshot = {
                    "ok": True,
                    "error": None,
                    "prompt_total": metrics["prompt_total"],
                    "predicted_total": metrics["predicted_total"],
                    "prompt_rate": metrics["prompt_rate"],
                    "predicted_rate": rate,
                }
            self._last = {"time": now, "predicted_total": metrics["predicted_total"]}
        except Exception as error:  # noqa: BLE001 —— 轮询失败仅记录，不中断服务
            with self._lock:
                self._snapshot = {"ok": False, "error": str(error), "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)


class UsageHandler(BaseHTTPRequestHandler):
    config: dict = {}  # 由 main() 注入 {"collector", "price_in", "price_out", "budget"}

    def do_GET(self) -> None:  # noqa: N802 —— http.server 命名约定
        if self.path.rstrip("/") == "/api/usage":
            cfg = self.config
            payload = build_payload(cfg["collector"].snapshot(), cfg["price_in"], cfg["price_out"], cfg["budget"])
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 —— 抑制默认请求日志
        pass


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="cc-switch 用量统计服务（轮询 llama-server /metrics 并暴露 /api/usage）")
    parser.add_argument("--port", type=int, default=int(os.environ.get("USAGE_PORT", "5002")))
    parser.add_argument("--llama-base", default=os.environ.get("USAGE_LLAMA_BASE", "http://192.168.77.210:18888"))
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("USAGE_POLL_INTERVAL", "5")))
    parser.add_argument("--price-in", type=float, default=float(os.environ.get("USAGE_PRICE_IN", "1.0")))
    parser.add_argument("--price-out", type=float, default=float(os.environ.get("USAGE_PRICE_OUT", "2.0")))
    parser.add_argument("--budget", type=float, default=(float(os.environ["USAGE_BUDGET"]) if os.environ.get("USAGE_BUDGET") else None))
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY") or None)
    args = parser.parse_args()

    collector = UsageCollector(args.llama_base, args.poll_interval, args.api_key)
    collector.start()

    UsageHandler.config = {"collector": collector, "price_in": args.price_in, "price_out": args.price_out, "budget": args.budget}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), UsageHandler)
    print(f"cc-switch 用量统计服务运行于 http://127.0.0.1:{args.port}/api/usage（轮询 {args.llama_base}/metrics）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m unittest tests/test_usage_stats_server.py -v`
Expected: PASS（原有 9 个 + 新增 8 个 = 17 个用例全过）

- [ ] **Step 5: 提交**

```bash
git add script/usage_stats_server.py tests/test_usage_stats_server.py
git commit -m "feat: 完成用量统计服务的轮询采集与 /api/usage HTTP 接口"
```

---

### Task 3: llama-server 启动脚本启用 /metrics

**Files:**
- Modify: `script/start_v4_flash_gguf.py:405-430`（启动命令列表追加 `--metrics`）
- Modify: `script/start_v4_flash_background.sh:51`（ARGS 追加 `--metrics`）

**Interfaces:**
- Consumes: 无
- Produces: llama-server 启动后暴露 `http://192.168.77.210:18888/metrics`（统计服务数据源）

- [ ] **Step 1: 修改 Python 启动脚本**

在 `command` 列表（`"# ---- 服务 ----"` 后第一个命令块）中，于 `"--flash-attn", "on",` 之后追加 `"--metrics",`，使 `command` 变为：

```python
    command = [
        str(server),
        "--model",
        str(model_path.resolve()),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--parallel",
        str(args.parallel),
        "--n-gpu-layers",
        "999",
        "--split-mode",
        "layer",
        "--tensor-split",
        gpu_split,
        "--jinja",
        "--reasoning",
        args.reasoning,
        "--reasoning-format",
        args.reasoning_format,
        "--flash-attn",
        "on",
        "--metrics",
    ]
```

- [ ] **Step 2: 修改后台启动脚本**

`start_v4_flash_background.sh` 中 `ARGS=(...)` 行追加 `--metrics`：

```bash
ARGS=(--model "$MODEL" --skip-build --no-console --port "$PORT" --parallel "$PARALLEL" --metrics)
```

- [ ] **Step 3: 静态验证改动**

Run: `git diff script/start_v4_flash_gguf.py script/start_v4_flash_background.sh`
Expected: 两处均只新增 `--metrics` 参数，无其他改动

- [ ] **Step 4: 提交**

```bash
git add script/start_v4_flash_gguf.py script/start_v4_flash_background.sh
git commit -m "feat: llama-server 启动命令启用 --metrics 监控端点"
```

---

### Task 4: .env.example 新增用量统计配置段

**Files:**
- Modify: `.env.example`（文件末尾追加配置段）

**Interfaces:**
- Consumes: Task 2 的 `main()` 环境变量约定
- Produces: 配置文档，供用户复制到 `.env` 使用

- [ ] **Step 1: 追加配置段**

在 `.env.example` 末尾追加：

```bash
# ---------- cc-switch 用量统计服务（usage_stats_server.py） ----------
# 统计服务监听端口（注意：5001 已被 nginx location / 占用，勿改）
USAGE_PORT=5002
# llama-server 地址（/metrics 数据源）
USAGE_LLAMA_BASE=http://192.168.77.210:18888
# 轮询间隔（秒）
USAGE_POLL_INTERVAL=5
# DeepSeek-V4-Flash 官方价（元/M tokens；官方调价后在此更新）
USAGE_PRICE_IN=1.0
USAGE_PRICE_OUT=2.0
# 预算（元）；留空则不显示 total/remaining，仅显示已用费用
#USAGE_BUDGET=100
# llama-server /metrics 若要求鉴权，填 API 密钥（Bearer 透传）；留空则不携带
#LLAMA_API_KEY=root123456
```

- [ ] **Step 2: 验证文件可被脚本解析**

Run: `python -c "from pathlib import Path; p=Path('.env.example'); assert 'USAGE_PORT' in p.read_text(encoding='utf-8'); print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 3: 提交**

```bash
git add .env.example
git commit -m "docs: .env.example 新增 cc-switch 用量统计服务配置段"
```

---

### Task 5: 服务器侧接线与验证清单（nginx + cc-switch，无代码改动）

**Files:** 无（交付物为下述配置片段与手动验证步骤，用户在其服务器/客户端执行）

**Interfaces:**
- Consumes: Task 1-4 的产物（统计服务、/metrics、.env 配置）
- Produces: nginx location 与 cc-switch 配置文本

- [ ] **Step 1: nginx 新增精确匹配 location**

编辑服务器上 `/etc/nginx/sites-enabled/myflaskapp`，在 `location ~ ^/210/llm/(.*)$` **之前**插入：

```nginx
    # ---- 210 LLM 用量统计（cc-switch 用量查询）----
    location ~ ^/210/llm/v1/api/usage$ {
        proxy_pass http://127.0.0.1:5002/api/usage;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
```

执行 `nginx -t && systemctl reload nginx`（或 `nginx -s reload`）使配置生效。

- [ ] **Step 2: 重启 llama-server 启用 /metrics**

在节点 210 上重启服务（`bash script/start_v4_flash_background.sh`，或先 kill 再启动），然后验证：

Run: `curl http://127.0.0.1:18888/metrics | grep -E "tokens_(evaluated|predicted)|predicted_tokens" | head`
Expected: 输出包含累计 tokens 与速率的指标行

- [ ] **Step 3: 启动统计服务并验证接口**

在节点 210 上启动统计服务（需先 `cp .env.example .env` 并按需修改，再执行）：

Run: `nohup python3 script/usage_stats_server.py >> /raid5/sh/logs/usage-stats.log 2>&1 &`
然后验证本机与 nginx 链路：

Run: `curl http://127.0.0.1:5002/api/usage`
Expected: JSON，`isValid: true`，含 `used` / `extra` / `unit: "CNY"`

Run: `curl http://127.0.0.1:5000/210/llm/v1/api/usage`
Expected: 同一 JSON（nginx 转发链路通）

- [ ] **Step 4: cc-switch 用量查询配置**

在 cc-switch 中打开该供应商卡片的「用量查询」开关，查询方式选「自定义」，粘贴：

```js
({
  request: {
    url: "{{baseUrl}}/api/usage",
    method: "GET",
    headers: { "Authorization": "Bearer {{apiKey}}", "User-Agent": "cc-switch/1.0" }
  },
  extractor: function(response) {
    if (!response || response.error || response.isValid === false) {
      return { isValid: false, invalidMessage: (response && (response.invalidMessage || (response.error && response.error.message))) || "接口调用失败" };
    }
    return {
      isValid: true,
      used: response.used,
      remaining: response.remaining,
      total: response.total,
      unit: response.unit || "CNY",
      planName: response.planName,
      extra: response.extra
    };
  }
})
```

点击「刷新」，卡片底部应显示累计费用（CNY）与 `extra` 中的 tokens 总量和生成速率。

- [ ] **Step 5: 故障演练验证错误处理**

停止 llama-server（或仅停统计服务），再在 cc-switch 点「刷新」，卡片应显示红色失效提示（`llama-server 不可用：...`），而非崩溃。验证后恢复服务。

---

## Self-Review

**1. Spec 覆盖：**
- llama-server `--metrics` → Task 3 ✓
- 统计服务轮询/解析/费用/`/api/usage` → Task 1+2 ✓
- nginx 精确匹配 location（端口 5002、位于 `^/210/llm/(.*)$` 之前）→ Task 5 Step 1 ✓
- cc-switch extractor 配置 → Task 5 Step 4 ✓
- `.env` 配置段（7 个变量）→ Task 4 ✓
- 错误处理（llama 不可用/首次全 0/鉴权透传/指标容错）→ Task 1 `build_payload`、Task 2 `UsageCollector` ✓
- 速率兜底（gauge 缺失用差值）→ Task 2 `_poll_once` + 测试 `test_rate_fallback_by_delta` ✓

**2. 占位符扫描：** 所有步骤均含完整代码或精确文本，无 TBD/TODO ✓

**3. 类型一致性：** `snapshot()` 键名 `ok/error/prompt_total/predicted_total/prompt_rate/predicted_rate` 在 Task 1 `build_payload` 与 Task 2 `UsageCollector` 中一致；`METRIC_NAMES` 键名与 `parse_metrics` 返回键一致；环境变量名在 Task 2 `main()` 与 Task 4 `.env.example` 中一致 ✓
