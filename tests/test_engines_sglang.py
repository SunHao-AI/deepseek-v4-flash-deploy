"""tests/test_engines_sglang.py — SGLang 适配器下载/persist 测试。"""

from modelctl.core.capabilities import Capabilities
from modelctl.core.profile import load_profile
from modelctl.engines import get_adapter

CAPS8 = Capabilities(gpu_count=8, compute_capability="8.9", binaries={"vllm": True, "sglang": True})


def _write(tmp_path, text, name="m.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return load_profile(name[:-5], tmp_path)


def test_sglang_requirements_allow_download_only(tmp_path):
    p = _write(
        tmp_path,
        "name: s\nengine: sglang\nport: 30000\nsglang:\n  model: ''\n  download:\n    modelscope_id: Qwen/Qwen3-32B\n",
    )
    a = get_adapter("sglang")(p, CAPS8)
    a.check_requirements()  # model 为空但有 download 段时不应报错


def test_sglang_pre_start_downloads_and_persists(tmp_path, monkeypatch):
    p = _write(
        tmp_path,
        "name: s\nengine: sglang\nport: 30000\nsglang:\n  model: ''\n  download:\n    modelscope_id: Qwen/Qwen3-32B\n",
    )
    a = get_adapter("sglang")(p, CAPS8)

    downloaded = tmp_path / "model-hf" / "Qwen3-32B"
    # import 位于 sglang 模块顶部，monkeypatch 模块属性即可生效。
    monkeypatch.setattr("modelctl.engines.sglang.download_repo", lambda mid, root: downloaded)
    # 使用真实 persist_model_path，同时验证 YAML 被写回。

    a.pre_start()
    assert p.engine_config["model"] == str(downloaded.resolve())
    content = p.path.read_text(encoding="utf-8")
    assert f"model: {downloaded.resolve()}" in content
    assert (tmp_path / "m.yaml.bak").is_file()


def test_sglang_pre_start_skips_when_model_exists(tmp_path, monkeypatch):
    p = _write(
        tmp_path,
        f"name: s\nengine: sglang\nport: 30000\nsglang:\n  model: {tmp_path}/model-hf/Qwen3-32B\n",
    )
    (tmp_path / "model-hf" / "Qwen3-32B").mkdir(parents=True)
    a = get_adapter("sglang")(p, CAPS8)

    calls = []

    def _fail(*args, **kwargs):  # 不应被调用
        calls.append("called")
        return tmp_path

    monkeypatch.setattr("modelctl.engines.sglang.download_repo", _fail)
    monkeypatch.setattr("modelctl.engines.sglang.persist_model_path", _fail)

    a.pre_start()  # model 路径已存在，直接返回
    assert calls == []
