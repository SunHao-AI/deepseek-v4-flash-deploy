#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================
# @File   : script/usage_stats_server.py
# @Desc   : cc-switch 用量统计服务 —— 轮询 llama-server /metrics，折算费用，暴露 /api/usage
# ===============================================================================
"""cc-switch 用量统计服务。

聚合 llama-server 的 /metrics（Prometheus）端点，自启动以来累计的
输入/输出 tokens 与生成速率，按 DeepSeek-V4-Flash 官方价格折算累计费用，
并暴露 /api/usage 供 cc-switch 的「用量查询 → 自定义」配置消费。

支持两种数据获取模式（--mode / USAGE_MODE）：
- poll（默认）：后台线程按 USAGE_POLL_INTERVAL 定时轮询 /metrics，
  /api/usage 返回最近一次缓存快照，响应快、对 llama-server 压力恒定。
- on-demand（主动获取）：不启动后台线程，由 cc-switch 轮询触发，
  每次 /api/usage 请求时同步拉取一次最新 /metrics，数据实时性最好。

纯标准库实现，零第三方依赖。运行：
    python3 script/usage_stats_server.py
    python3 script/usage_stats_server.py --mode on-demand
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

# Prometheus 指标名（按 llama.cpp 新旧版本命名做容错匹配，取第一个命中的）
# 官方 llama.cpp b10298（DeepSeek-V4 部署版本）实测命名：
#   输入计数 llamacpp:prompt_tokens_total、输出计数 llamacpp:tokens_predicted_total
#   速率 gauge llamacpp:prompt_tokens_seconds / llamacpp:predicted_tokens_seconds
METRIC_NAMES = {
    "prompt_total": [
        "llamacpp:prompt_tokens_total",     # llama.cpp b10298+（当前部署版本）
        "llamacpp:tokens_evaluated_total",  # llama.cpp 2024-2025 中期旧命名
        "llama_tokens_evaluated_total",     # llama.cpp 2024 早期
        "prompt_tokens_total",              # llama.cpp 2023 无前缀
    ],
    "predicted_total": [
        "llamacpp:tokens_predicted_total",  # llama.cpp 2024+（b10298 亦为此名）
        "llamacpp:predicted_tokens_total",  # 防御性变体（部分 fork/版本顺序不同）
        "llama_tokens_predicted_total",     # llama.cpp 2024 早期
        "tokens_predicted_total",           # llama.cpp 2023 无前缀
    ],
    "prompt_rate": ["llamacpp:prompt_tokens_seconds", "prompt_tokens_seconds"],
    "predicted_rate": ["llamacpp:predicted_tokens_seconds", "llamacpp:tokens_predicted_seconds", "predicted_tokens_seconds"],
}

# 预编译的指标匹配模式（避免每轮轮询重复编译正则）
METRIC_PATTERNS = {
    key: [
        re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.MULTILINE)
        for name in names
    ]
    for key, names in METRIC_NAMES.items()
}


def _fmt_int(value: float) -> str:
    """千分位格式化整数。"""
    return f"{int(round(value)):,}"


def parse_metrics(text: str) -> dict[str, float]:
    """解析 Prometheus 文本，返回 {prompt_total, predicted_total, prompt_rate, predicted_rate}。

    缺失的指标返回 0.0；速率为 gauge 值（tok/s）。
    """
    result = {"prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
    for key, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
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


class UsageCollector:
    """聚合 llama-server /metrics 用量。

    mode="poll"：后台线程定时轮询，维护最近一次缓存快照；
    mode="on-demand"：不启动后台线程，由 get_snapshot() 在每次请求时同步拉取。
    """

    def __init__(self, base_url: str, poll_interval: float, api_key: str | None, mode: str = "poll") -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.api_key = api_key
        self.mode = mode
        self._lock = threading.Lock()
        self._snapshot = {"ok": False, "error": None, "prompt_total": 0.0, "predicted_total": 0.0, "prompt_rate": 0.0, "predicted_rate": 0.0}
        self._last = {"time": None, "predicted_total": 0.0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True) if mode == "poll" else None

    def start(self) -> None:
        if self._thread is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_snapshot(self) -> dict:
        """返回用量快照。

        poll 模式返回最近一次缓存快照；on-demand 模式先同步拉取一次最新指标再返回。
        """
        if self.mode == "on-demand":
            self._poll_once()
        return self.snapshot()

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
                    rate = max(delta_tokens / delta_t, 0.0)
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
            payload = build_payload(cfg["collector"].get_snapshot(), cfg["price_in"], cfg["price_out"], cfg["budget"])
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
    parser.add_argument("--host", default=os.environ.get("USAGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("USAGE_PORT", "5002")))
    parser.add_argument("--llama-base", default=os.environ.get("USAGE_LLAMA_BASE", "http://192.168.77.210:18888"))
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("USAGE_POLL_INTERVAL", "5")))
    parser.add_argument(
        "--mode",
        choices=("poll", "on-demand"),
        default=os.environ.get("USAGE_MODE", "poll"),
        help="数据获取模式：poll=后台定时轮询（默认）；on-demand=由 cc-switch 轮询触发、每次请求同步拉取",
    )
    parser.add_argument("--price-in", type=float, default=float(os.environ.get("USAGE_PRICE_IN", "1.0")))
    parser.add_argument("--price-out", type=float, default=float(os.environ.get("USAGE_PRICE_OUT", "2.0")))
    parser.add_argument("--budget", type=float, default=(float(os.environ["USAGE_BUDGET"]) if os.environ.get("USAGE_BUDGET") else None))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LLAMA_API_KEY") or os.environ.get("API_KEY") or None,
        help="轮询 /metrics 时携带的 Bearer token；默认取 LLAMA_API_KEY，未设置则回退复用 API_KEY",
    )
    args = parser.parse_args()

    collector = UsageCollector(args.llama_base, args.poll_interval, args.api_key, mode=args.mode)
    collector.start()

    UsageHandler.config = {"collector": collector, "price_in": args.price_in, "price_out": args.price_out, "budget": args.budget}
    server = ThreadingHTTPServer((args.host, args.port), UsageHandler)
    mode_desc = "由 cc-switch 轮询触发、每次请求同步拉取" if args.mode == "on-demand" else f"后台每 {args.poll_interval:g}s 轮询"
    print(f"cc-switch 用量统计服务运行于 http://{args.host}:{args.port}/api/usage（{mode_desc} {args.llama_base}/metrics）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        collector.stop()
        server.server_close()


if __name__ == "__main__":
    main()
