#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/__init__.py — 引擎注册表。"""
from __future__ import annotations

from core.profile import ProfileError
from engines.base import EngineAdapter
from engines.llamacpp import LlamaCppAdapter
from engines.ollama import OllamaAdapter
from engines.sglang import SglangAdapter
from engines.vllm import VllmAdapter

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
