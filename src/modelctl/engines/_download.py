"""engines/_download.py — 统一的 ModelScope 下载工具。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from loguru import logger

# 模块级可 patch 的引用（方案 D）：测试直接 monkeypatch 本模块属性即可；
# 未安装 modelscope 时保持 None，调用前由 ensure_modelscope() 安装并延迟重导入。
try:  # pragma: no cover - 真实环境由 ensure_modelscope 安装
    from modelscope import snapshot_download  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    snapshot_download = None  # type: ignore[assignment]


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
