#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/__init__.py — 引擎注册表。"""
from __future__ import annotations

from modelctl.core.profile import ProfileError
from modelctl.engines.base import EngineAdapter
from modelctl.engines.llamacpp import LlamaCppAdapter
from modelctl.engines.ollama import OllamaAdapter
from modelctl.engines.sglang import SglangAdapter
from modelctl.engines.vllm import VllmAdapter

_REGISTRY: dict[str, type[EngineAdapter]] = {
    "llamacpp": LlamaCppAdapter,
    "ollama": OllamaAdapter,
    "vllm": VllmAdapter,
    "sglang": SglangAdapter,
}


def get_adapter(engine: str) -> type[EngineAdapter]:
    try:
        return _REGISTRY[engine]
    except KeyError:
        raise ProfileError(f"引擎未实现：{engine}（已实现：{sorted(_REGISTRY)}）") from None
