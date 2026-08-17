#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/base.py — 引擎适配器抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.capabilities import Capabilities
from core.profile import Profile


class RequirementError(RuntimeError):
    """硬性条件不满足，拒绝启动。"""


class EngineAdapter(ABC):
    def __init__(self, profile: Profile, caps: Capabilities):
        self.profile = profile
        self.caps = caps
        self.warnings: list[str] = []

    @abstractmethod
    def build_command(self) -> tuple[list[str], dict[str, str]]:
        """返回 (启动命令, 需注入的环境变量)。"""

    @abstractmethod
    def check_requirements(self) -> None:
        """校验硬件/配置门槛；可降级的写 self.warnings，硬性不满足抛 RequirementError。"""

    @abstractmethod
    def metrics_mapping(self) -> dict[str, list[str]] | None:
        """Prometheus 指标名映射；None 表示该引擎不支持精确统计。"""

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}/health"

    def pre_start(self) -> None:
        """启动前钩子（下载/编译/pull）。"""

    def post_start(self) -> None:
        """启动后钩子（如 ollama 预加载模型）。"""

    def stop_patterns(self) -> list[str]:
        return []

    def api_key_args(self) -> list[str]:
        return ["--api-key", self.profile.api_key] if self.profile.api_key else []
