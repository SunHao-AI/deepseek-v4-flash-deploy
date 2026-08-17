# -*- coding: utf-8 -*-
import pytest

from modelctl.core.capabilities import Capabilities
from modelctl.core.profile import load_profile
from modelctl.engines import get_adapter
from modelctl.engines.base import RequirementError


def _profile(tmp_path):
    (tmp_path / "qwen3-ollama.yaml").write_text("name: qwen3-ollama\nengine: ollama\nport: 11434\n" "ollama:\n  model: qwen3:32b\n  num_parallel: 2\n  context_length: 32768\n", encoding="utf-8")
    return load_profile("qwen3-ollama", tmp_path)


def test_build_command(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", "/raid5/sh/model/ollama-models")
    caps = Capabilities(binaries={"ollama": True})
    a = get_adapter("ollama")(_profile(tmp_path), caps)
    cmd, env = a.build_command()
    assert cmd == ["ollama", "serve"]
    assert env["OLLAMA_HOST"] == "0.0.0.0:11434"
    assert env["OLLAMA_MODELS"] == "/raid5/sh/model/ollama-models"
    assert env["OLLAMA_NUM_PARALLEL"] == "2"
    assert env["OLLAMA_CONTEXT_LENGTH"] == "32768"


def test_missing_binary(tmp_path):
    caps = Capabilities(binaries={"ollama": False})
    a = get_adapter("ollama")(_profile(tmp_path), caps)
    with pytest.raises(RequirementError, match="ollama"):
        a.check_requirements()


def test_metrics_mapping_none(tmp_path):
    caps = Capabilities(binaries={"ollama": True})
    a = get_adapter("ollama")(_profile(tmp_path), caps)
    assert a.metrics_mapping() is None


def test_health_url_root(tmp_path):
    caps = Capabilities(binaries={"ollama": True})
    a = get_adapter("ollama")(_profile(tmp_path), caps)
    assert a.health_url() == "http://127.0.0.1:11434/"
