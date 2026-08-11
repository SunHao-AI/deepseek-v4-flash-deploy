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
