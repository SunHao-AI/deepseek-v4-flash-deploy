"""engines/_persist.py — 将下载后的本地 model 路径写回 profile YAML。"""

from __future__ import annotations

from pathlib import Path

import yaml


def persist_model_path(profile_path: Path, engine: str, model_path: str) -> None:
    """仅更新 YAML 中 <engine>.model 字段，写回前备份原文件为 .yaml.bak。

    下载成功后才调用；失败方保留原 YAML 不变。
    """
    original = profile_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(original)
    if not isinstance(raw, dict):
        raise ValueError(f"{profile_path} 顶层必须是映射")

    backup = profile_path.with_name(profile_path.name + ".bak")
    backup.write_text(original, encoding="utf-8")

    engine_config = raw.setdefault(engine, {})
    if not isinstance(engine_config, dict):
        raise ValueError(f"{profile_path} 中 {engine} 段必须是映射")
    engine_config["model"] = model_path

    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
