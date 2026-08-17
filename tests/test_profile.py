# -*- coding: utf-8 -*-
"""tests/test_profile.py — Profile 加载、插值与校验测试。"""
import pytest

from modelctl.core.profile import Profile, ProfileError, list_profiles, load_profile

YAML = """
name: demo
engine: llamacpp
port: 18888
api_key: ${TEST_KEY}
llamacpp:
  model: /models/x.gguf
  parallel: 2
usage:
  price_in: 1.0
"""


def _write(tmp_path, text, name="demo.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_load_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    d = _write(tmp_path, YAML)
    p = load_profile("demo", d)
    assert isinstance(p, Profile)
    assert p.name == "demo" and p.engine == "llamacpp" and p.port == 18888
    assert p.api_key == "secret"
    assert p.engine_config == {"model": "/models/x.gguf", "parallel": 2}
    assert p.usage == {"price_in": 1.0}


def test_missing_required_field(tmp_path):
    d = _write(tmp_path, "name: demo\nengine: llamacpp\n")
    with pytest.raises(ProfileError, match="port"):
        load_profile("demo", d)


def test_unknown_engine(tmp_path):
    d = _write(tmp_path, "name: demo\nengine: tensorrt\nport: 1\n")
    with pytest.raises(ProfileError, match="tensorrt"):
        load_profile("demo", d)


def test_interpolate_missing_var(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    d = _write(tmp_path, "name: demo\nengine: vllm\nport: 8000\napi_key: ${NOPE_VAR}\n")
    with pytest.raises(ProfileError, match="NOPE_VAR"):
        load_profile("demo", d)


def test_nested_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOT", "/raid5/sh/model")
    d = _write(tmp_path, "name: demo\nengine: ollama\nport: 11434\nollama:\n  model: ${ROOT}/x\n")
    p = load_profile("demo", d)
    assert p.engine_config["model"] == "/raid5/sh/model/x"


def test_list_profiles_sorted(tmp_path):
    _write(tmp_path, "name: b\nengine: vllm\nport: 1\n", "b.yaml")
    _write(tmp_path, "name: a\nengine: vllm\nport: 2\n", "a.yaml")
    assert [p.name for p in list_profiles(tmp_path)] == ["a", "b"]


def test_missing_file(tmp_path):
    with pytest.raises(ProfileError, match="不存在"):
        load_profile("ghost", tmp_path)
