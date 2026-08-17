# -*- coding: utf-8 -*-
import pytest

from modelctl.core.capabilities import probe
from modelctl.core.profile import ProfileError, load_profile
from modelctl.engines import get_adapter
from modelctl.engines.base import RequirementError

SMI = "\n".join(["RTX 5880 Ada Generation, 49140, 48000, 580.65.05, 8.9"] * 8)


def _profile(tmp_path, extra=""):
    (tmp_path / "m.gguf").write_bytes(b"0" * 1024)
    (tmp_path / "dspark-x.gguf").write_bytes(b"0" * 512)
    yaml_text = f"""
name: ds
engine: llamacpp
port: 18888
llamacpp:
  model: {tmp_path}/m.gguf
  parallel: 2
  gpu_count: 8
{extra}"""
    (tmp_path / "ds.yaml").write_text(yaml_text, encoding="utf-8")
    return load_profile("ds", tmp_path)


def test_build_command(tmp_path):
    caps = probe(nvidia_smi_output=SMI)
    adapter = get_adapter("llamacpp")(_profile(tmp_path), caps)
    adapter.check_requirements()
    cmd, env = adapter.build_command()
    assert "--model" in cmd and "18888" in cmd
    assert cmd[cmd.index("--ctx-size") + 1] == str(2 * 1048576)
    assert "--model-draft" in cmd
    assert "--cache-type-k" in cmd
    assert "--metrics" in cmd


def test_dspark_disabled_when_no_draft(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"0" * 1024)
    (tmp_path / "ds.yaml").write_text(f"name: ds\nengine: llamacpp\nport: 18888\nllamacpp:\n  model: {tmp_path}/m.gguf\n  gpu_count: 8\n", encoding="utf-8")
    caps = probe(nvidia_smi_output=SMI)
    adapter = get_adapter("llamacpp")(load_profile("ds", tmp_path), caps)
    adapter.check_requirements()
    cmd, _ = adapter.build_command()
    assert "--model-draft" not in cmd
    assert any("park" in w.lower() for w in adapter.warnings)


def test_gpu_count_exceeds_hw(tmp_path):
    caps = probe(nvidia_smi_output="\n".join(["RTX 5880 Ada Generation, 49140, 48000, 580.65.05, 8.9"] * 2))
    adapter = get_adapter("llamacpp")(_profile(tmp_path), caps)
    with pytest.raises(RequirementError, match="GPU"):
        adapter.check_requirements()


def test_metrics_mapping_keys(tmp_path):
    caps = probe(nvidia_smi_output=SMI)
    adapter = get_adapter("llamacpp")(_profile(tmp_path), caps)
    m = adapter.metrics_mapping()
    assert "llamacpp:prompt_tokens_total" in m["prompt_total"]
    assert "llamacpp:tokens_predicted_total" in m["predicted_total"]


def test_unknown_engine():
    with pytest.raises(ProfileError):
        get_adapter("tensorrt")
