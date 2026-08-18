"""tests/test_engines_download.py — ModelScope 下载工具测试。"""

import sys
import types
from pathlib import Path

import modelctl.engines._download as dl


def test_download_repo_uses_modelscope(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(model_id, local_dir, **kwargs):
        calls.append((model_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    # 注入假 modelscope 模块，避免依赖真实安装；并禁用自动安装。
    fake_modelscope = types.ModuleType("modelscope")
    fake_modelscope.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
    monkeypatch.setattr(dl, "ensure_modelscope", lambda: None)

    result = dl.download_repo("unsloth/Qwen3.8-27B-GGUF", tmp_path)
    assert calls == [("unsloth/Qwen3.8-27B-GGUF", str(tmp_path / "Qwen3.8-27B-GGUF"))]
    assert result == tmp_path / "Qwen3.8-27B-GGUF"
