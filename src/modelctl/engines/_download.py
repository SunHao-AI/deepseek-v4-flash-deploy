"""engines/_download.py — 统一的 ModelScope 下载工具。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from loguru import logger


def ensure_modelscope() -> None:
    """确保 modelscope 已安装，否则自动安装。"""
    if importlib.util.find_spec("modelscope") is None:
        logger.info("未安装 modelscope，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "modelscope"], check=True)


def download_repo(modelscope_id: str, local_root: Path) -> Path:
    """下载 ModelScope 仓库到 local_root/<repo_last_part>，返回本地目录。"""
    ensure_modelscope()
    from modelscope import snapshot_download  # type: ignore[import-not-found]

    destination = local_root / modelscope_id.rsplit("/", 1)[-1]
    destination.mkdir(parents=True, exist_ok=True)
    logger.info(f"从 ModelScope 下载 {modelscope_id} 到 {destination}")
    snapshot_download(
        model_id=modelscope_id,
        local_dir=str(destination),
    )
    return destination
