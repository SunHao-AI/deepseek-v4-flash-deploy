"""tests/test_engines_unsloth.py — Unsloth 适配器测试。"""

import pytest

from modelctl.core.capabilities import Capabilities
from modelctl.core.profile import load_profile
from modelctl.engines import get_adapter
from modelctl.engines.base import RequirementError
from modelctl.engines.unsloth import UnslothAdapter

CAPS8 = Capabilities(gpu_count=8, compute_capability="8.9", binaries={"unsloth": True})


def _write(tmp_path, text, name="u.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return load_profile(name[:-5], tmp_path)


def test_unsloth_registered():
    assert get_adapter("unsloth") is UnslothAdapter


def test_unsloth_requirements_rejects_without_binary(tmp_path):
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\napi_key: k\nunsloth:\n  model: m\n")
    a = get_adapter("unsloth")(p, Capabilities(gpu_count=8, binaries={"unsloth": False}))
    with pytest.raises(RequirementError, match="unsloth"):
        a.check_requirements()


def test_unsloth_requirements_requires_api_key(tmp_path):
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\nunsloth:\n  model: m\n")
    a = get_adapter("unsloth")(p, CAPS8)
    with pytest.raises(RequirementError, match="api_key"):
        a.check_requirements()


def test_unsloth_requirements_allow_download_only(tmp_path):
    p = _write(
        tmp_path,
        "name: u\nengine: unsloth\nport: 30000\napi_key: k\n"
        "unsloth:\n  model: ''\n  download:\n    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF\n"
        "    quant: UD-Q8_K_XL\n",
    )
    a = get_adapter("unsloth")(p, CAPS8)
    a.check_requirements()  # model 为空但有 download 段时不应报错


def test_unsloth_tensor_parallel_requires_2_gpus(tmp_path):
    p = _write(
        tmp_path,
        "name: u\nengine: unsloth\nport: 30000\napi_key: k\n"
        "unsloth:\n  model: m\n  tensor_parallel: true\n",
    )
    a = get_adapter("unsloth")(p, Capabilities(gpu_count=1, binaries={"unsloth": True}))
    with pytest.raises(RequirementError, match="2 块 GPU"):
        a.check_requirements()


def test_unsloth_build_command(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_API_KEY", "sk-test")
    p = _write(
        tmp_path,
        "name: u\nengine: unsloth\nport: 30000\napi_key: ${UNSLOTH_API_KEY}\n"
        "unsloth:\n  model: unsloth/Test-GGUF\n  gguf_variant: UD-Q4_K_XL\n  context_length: 32768\n",
    )
    a = get_adapter("unsloth")(p, CAPS8)
    cmd, _env = a.build_command()
    assert cmd[:3] == ["unsloth", "studio", "--api-only"]
    assert cmd[cmd.index("-p") + 1] == "30000"
    assert cmd[cmd.index("--model") + 1] == "unsloth/Test-GGUF:UD-Q4_K_XL"
    assert cmd[cmd.index("--context-length") + 1] == "32768"
    assert cmd[cmd.index("--api-key") + 1] == "sk-test"


def test_unsloth_build_command_local_path_ignores_variant(tmp_path):
    p = _write(
        tmp_path,
        f"name: u\nengine: unsloth\nport: 30000\nunsloth:\n  model: {tmp_path}/model.gguf\n"
        f"  gguf_variant: UD-Q4_K_XL\n",
    )
    (tmp_path / "model.gguf").write_text("x", encoding="utf-8")
    a = get_adapter("unsloth")(p, CAPS8)
    cmd, _env = a.build_command()
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "model.gguf")


def test_unsloth_health_url_and_metrics(tmp_path):
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\nunsloth:\n  model: m\n")
    a = get_adapter("unsloth")(p, CAPS8)
    assert a.health_url() == "http://127.0.0.1:30000/v1/models"
    assert a.metrics_mapping() is None
    assert a.stop_patterns() == ["unsloth"]
