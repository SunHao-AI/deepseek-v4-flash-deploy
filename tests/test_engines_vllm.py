import pytest

from modelctl.core.capabilities import Capabilities
from modelctl.core.profile import load_profile
from modelctl.engines import get_adapter
from modelctl.engines.base import RequirementError

CAPS8 = Capabilities(gpu_count=8, compute_capability="8.9", binaries={"vllm": True, "sglang": True})


def _write(tmp_path, text, name="m.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return load_profile(name[:-5], tmp_path)


def test_vllm_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/raid5/sh/model/huggingface")
    p = _write(
        tmp_path,
        "name: q\nengine: vllm\nport: 8000\nvllm:\n"
        "  model: Qwen/Qwen3-32B\n  tensor_parallel_size: 2\n  max_model_len: 32768\n"
        '  extra_args: "--enable-prefix-caching"\n',
    )
    a = get_adapter("vllm")(p, CAPS8)
    a.check_requirements()
    cmd, env = a.build_command()
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-32B"]
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--max-model-len") + 1] == "32768"
    assert "--enable-prefix-caching" in cmd
    assert env["HF_HOME"] == "/raid5/sh/model/huggingface"


def test_vllm_fp8_cc_check(tmp_path):
    p = _write(tmp_path, "name: q\nengine: vllm\nport: 8000\nvllm:\n  model: m\n  quantization: fp8\n")
    a = get_adapter("vllm")(p, Capabilities(gpu_count=8, compute_capability="7.5", binaries={"vllm": True}))
    with pytest.raises(RequirementError, match="8.9"):
        a.check_requirements()


def test_vllm_tp_exceeds(tmp_path):
    p = _write(tmp_path, "name: q\nengine: vllm\nport: 8000\nvllm:\n  model: m\n  tensor_parallel_size: 16\n")
    a = get_adapter("vllm")(p, CAPS8)
    with pytest.raises(RequirementError, match="GPU"):
        a.check_requirements()


def test_sglang_command(tmp_path):
    p = _write(
        tmp_path, "name: s\nengine: sglang\nport: 30000\nsglang:\n  model: Qwen/Qwen3-32B\n  tensor_parallel_size: 4\n"
    )
    a = get_adapter("sglang")(p, CAPS8)
    a.check_requirements()
    cmd, _ = a.build_command()
    assert "sglang.launch_server" in cmd
    assert cmd[cmd.index("--tp") + 1] == "4"


def test_vllm_metrics(tmp_path):
    p = _write(tmp_path, "name: q\nengine: vllm\nport: 8000\nvllm:\n  model: m\n")
    a = get_adapter("vllm")(p, CAPS8)
    assert a.metrics_mapping()["prompt_total"] == ["vllm:prompt_tokens_total"]
    assert a.metrics_mapping()["predicted_total"] == ["vllm:generation_tokens_total"]
