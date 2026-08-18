# modelctl 新增 Unsloth 引擎实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 modelctl 新增 `unsloth` 推理引擎（无头 API 模式），支持 Unsloth 动态量化 GGUF 模型的部署、下载、健康检查与用量统计降级。

**Architecture:** 遵循现有 `EngineAdapter` 插件模式：新增 `src/modelctl/engines/unsloth.py` 实现核心钩子，在引擎注册表与能力探测中注册 `unsloth`，模型下载复用 llamacpp 的 `download_gguf`（ModelScope 分片下载），对外暴露 OpenAI 兼容 API。

**Tech Stack:** Python 3.12+、PyYAML、loguru、pytest（测试）、ruff（lint）、mypy（类型检查）

## Global Constraints

- 不新增 Python 包依赖（unsloth 为外部二进制，独立环境安装）
- 遵循现有引擎适配器模式：`check_requirements` / `pre_start` / `build_command` / `health_url` / `post_start` / `metrics_mapping` / `stop_patterns` 钩子语义与 llamacpp/vllm/sglang 一致
- unsloth 无头服务的具体 flag 以目标机器 `unsloth --help` 实测为准；本计划按官方文档（`unsloth studio --api-only -H 0.0.0.0 -p <port> --model <id>[:<variant>] --context-length <n>`）实现，flag 集中为模块常量，便于调整
- 测试 mock 模式参照 `tests/test_engines_sglang.py` / `tests/test_engines_vllm.py`（monkeypatch 模块属性）
- ruff line-length 120；全部用户可见文案与代码注释使用中文
- 健康检查依赖 `wait_health(url, timeout, profile.api_key)`（自动携带 Bearer 头）；unsloth 引擎 profile **必须配置 api_key**

---

### Task 1: UnslothAdapter 核心实现与引擎注册

**Files:**
- Create: `src/modelctl/engines/unsloth.py`
- Modify: `src/modelctl/engines/__init__.py:8-18`
- Modify: `src/modelctl/core/capabilities.py:10`
- Modify: `src/modelctl/core/profile.py:17`
- Modify: `src/modelctl/cli.py:195`
- Test: `tests/test_engines_unsloth.py`

**Interfaces:**
- Consumes: `EngineAdapter`（`profile` / `caps` / `api_key_args()`）、`RequirementError`、`Capabilities.binaries/gpu_count`、`free_vram_total_mb(caps)`、`profile.engine_config/port/api_key`。
- Produces: `UnslothAdapter`；`get_adapter("unsloth")` 返回该类；`ENGINE_BINARIES` 含 `"unsloth"`；`KNOWN_ENGINES` 含 `"unsloth"`。

- [ ] **Step 1: 编写失败测试**

创建 `tests/test_engines_unsloth.py`：

```python
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
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\napi_key: k\nunsloth:\n  model: m\n  tensor_parallel: true\n")
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
        f"name: u\nengine: unsloth\nport: 30000\nunsloth:\n  model: {tmp_path}/model.gguf\n  gguf_variant: UD-Q4_K_XL\n",
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_engines_unsloth.py -v`
Expected: FAIL — `ModuleNotFoundError: modelctl.engines.unsloth`（模块不存在）

- [ ] **Step 3: 实现最小代码**

创建 `src/modelctl/engines/unsloth.py`：

```python
#!/usr/bin/env python3
"""engines/unsloth.py — Unsloth 无头服务（unsloth studio --api-only）适配器。"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from modelctl.core.capabilities import free_vram_total_mb
from modelctl.engines.base import EngineAdapter, RequirementError

# Unsloth 无头服务固定参数。
# 注意：具体 flag 需在目标机器上以 `unsloth --help` / `unsloth start --no-launch`
# 实测确认；如与文档不一致，仅需调整本文件常量，不影响其他引擎。
UNSLOTH_BIN = "unsloth"
STUDIO_ARGS = ["studio", "--api-only", "-H", "0.0.0.0"]


class UnslothAdapter(EngineAdapter):
    def check_requirements(self) -> None:
        if not self.caps.binaries.get("unsloth"):
            raise RequirementError("未安装 unsloth（PATH 中找不到 unsloth 命令）")
        cfg = self.profile.engine_config
        if not cfg.get("model") and not cfg.get("download"):
            raise RequirementError(f"{self.profile.name}：unsloth.model 必填（或配置 download 段自动下载）")
        if not self.profile.api_key:
            raise RequirementError(
                f"{self.profile.name}：unsloth 引擎必须配置 api_key（健康检查 /v1/models 依赖 Bearer 认证）"
            )
        if cfg.get("tensor_parallel") and self.caps.gpu_count < 2:
            raise RequirementError(f"tensor_parallel 需要至少 2 块 GPU，当前 {self.caps.gpu_count}")
        self._check_vram(cfg)
        # 用量统计降级提示：无头 API 模式的 /metrics 端点尚未验证
        self.warnings.append("unsloth 引擎暂未验证 /metrics 端点，用量统计降级为'不支持精确统计'")

    def _check_vram(self, cfg: dict) -> None:
        """GGUF 本地文件存在时按文件大小做显存预检。"""
        model = str(cfg.get("model") or "")
        if not model:
            return
        p = Path(model).expanduser()
        if not p.is_file():
            return
        need_mb = p.stat().st_size / 1024 / 1024 * 1.1
        free_mb = free_vram_total_mb(self.caps)
        if need_mb > free_mb:
            raise RequirementError(f"剩余显存不足：模型约需 {need_mb:.0f}MB（×1.1），剩余 {free_mb}MB")

    def _model_ref(self, cfg: dict) -> str:
        """构造 --model 参数：本地路径原样；HF ID 追加 :<gguf_variant>。"""
        model = str(cfg["model"])
        if Path(model).expanduser().is_file() or Path(model).expanduser().is_dir():
            return model
        variant = cfg.get("gguf_variant")
        return f"{model}:{variant}" if variant else model

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        cmd = [UNSLOTH_BIN, *STUDIO_ARGS, "-p", str(self.profile.port)]
        cmd += ["--model", self._model_ref(cfg)]
        if cfg.get("context_length"):
            cmd += ["--context-length", str(cfg["context_length"])]
        if cfg.get("tensor_parallel"):
            cmd += ["--tensor-parallel"]
        if cfg.get("load_in_4bit"):
            cmd += ["--load-in-4bit"]
        cmd += self.api_key_args()
        if cfg.get("extra_args"):
            cmd += shlex.split(str(cfg["extra_args"]))
        env = {"HF_HOME": os.environ["HF_HOME"]} if os.environ.get("HF_HOME") else {}
        return cmd, env

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}/v1/models"

    def metrics_mapping(self) -> None:
        return None

    def stop_patterns(self) -> list[str]:
        return ["unsloth"]
```

修改 `src/modelctl/engines/__init__.py`：

```python
from modelctl.engines.unsloth import UnslothAdapter

_REGISTRY: dict[str, type[EngineAdapter]] = {
    "llamacpp": LlamaCppAdapter,
    "ollama": OllamaAdapter,
    "vllm": VllmAdapter,
    "sglang": SglangAdapter,
    "unsloth": UnslothAdapter,
}
```

修改 `src/modelctl/core/capabilities.py` 第 10 行：

```python
ENGINE_BINARIES = ["ollama", "vllm", "sglang", "unsloth"]  # llamacpp 由源码编译，不在此列
```

修改 `src/modelctl/core/profile.py` 第 17 行：

```python
KNOWN_ENGINES = {"llamacpp", "ollama", "vllm", "sglang", "unsloth"}
```

修改 `src/modelctl/cli.py` 第 195 行（probe 子命令引擎列表）：

```python
    for name in ("ollama", "vllm", "sglang", "unsloth"):
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_engines_unsloth.py -v`
Expected: PASS（13 个测试）

- [ ] **Step 5: 运行全量测试确认无回归**

Run: `uv run pytest`
Expected: PASS（现有测试不因注册表/能力探测变更而失败）

- [ ] **Step 6: Commit**

```bash
git add src/modelctl/engines/unsloth.py src/modelctl/engines/__init__.py src/modelctl/core/capabilities.py src/modelctl/core/profile.py src/modelctl/cli.py tests/test_engines_unsloth.py
git commit -m "feat(engines): 新增 unsloth 引擎适配器与注册"
```

---

### Task 2: pre_start 自动下载与持久化 + 环境变量 + 示例 profile

**Files:**
- Modify: `src/modelctl/engines/unsloth.py`（顶部 import 与 `pre_start`）
- Modify: `.env.example`
- Create: `models/unsloth/deepseek-v4-unsloth.yaml`
- Test: 扩展 `tests/test_engines_unsloth.py`

**Interfaces:**
- Consumes: `download_gguf(modelscope_id, model_root, quant, want_dspark) -> (Path, Path | None)`（来自 `modelctl.engines.llamacpp`）、`persist_model_path(profile_path, engine, model_path)`（来自 `modelctl.engines._persist`）、`PROJECT_ROOT`（来自 `modelctl.core.envfile`）。
- Produces: `pre_start()` 下载后更新 `profile.engine_config["model"]` 为本地路径并写回 YAML。

- [ ] **Step 1: 编写失败测试**

追加到 `tests/test_engines_unsloth.py`：

```python
def test_unsloth_pre_start_downloads_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "model-gguf"))
    p = _write(
        tmp_path,
        "name: u\nengine: unsloth\nport: 30000\n"
        "unsloth:\n  model: ''\n  download:\n"
        "    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF\n    quant: UD-Q8_K_XL\n",
    )
    a = get_adapter("unsloth")(p, CAPS8)

    downloaded = tmp_path / "model-gguf" / "DeepSeek-V4-Flash-0731-GGUF" / "UD-Q8_K_XL" / "model.gguf"
    monkeypatch.setattr("modelctl.engines.unsloth.download_gguf", lambda mid, root, quant, want: (downloaded, None))

    a.pre_start()
    assert p.engine_config["model"] == str(downloaded.resolve())
    content = p.path.read_text(encoding="utf-8")
    assert f"model: {downloaded.resolve()}" in content
    assert (tmp_path / "u.yaml.bak").is_file()


def test_unsloth_pre_start_skips_when_model_exists(tmp_path, monkeypatch):
    p = _write(
        tmp_path,
        f"name: u\nengine: unsloth\nport: 30000\nunsloth:\n  model: {tmp_path}/model.gguf\n",
    )
    (tmp_path / "model.gguf").write_text("x", encoding="utf-8")
    a = get_adapter("unsloth")(p, CAPS8)
    calls = []

    def _fail(*args, **kwargs):  # 不应被调用
        calls.append("called")
        return tmp_path

    monkeypatch.setattr("modelctl.engines.unsloth.download_gguf", _fail)
    monkeypatch.setattr("modelctl.engines.unsloth.persist_model_path", _fail)

    a.pre_start()
    assert calls == []


def test_unsloth_pre_start_download_failure_hints_hf(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "model-gguf"))
    p = _write(
        tmp_path,
        "name: u\nengine: unsloth\nport: 30000\n"
        "unsloth:\n  model: ''\n  download:\n"
        "    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF\n    quant: UD-Q8_K_XL\n",
    )
    a = get_adapter("unsloth")(p, CAPS8)

    def _fail(*args, **kwargs):
        raise RequirementError("ModelScope 下载失败")

    monkeypatch.setattr("modelctl.engines.unsloth.download_gguf", _fail)
    with pytest.raises(RequirementError, match="HF_ENDPOINT"):
        a.pre_start()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_engines_unsloth.py::test_unsloth_pre_start_downloads_and_persists tests/test_engines_unsloth.py::test_unsloth_pre_start_skips_when_model_exists tests/test_engines_unsloth.py::test_unsloth_pre_start_download_failure_hints_hf -v`
Expected: FAIL — `AttributeError: 'UnslothAdapter' object has no attribute 'pre_start'`（base 中 `pre_start` 返回 None，未下载）

- [ ] **Step 3: 实现 pre_start**

在 `src/modelctl/engines/unsloth.py` 顶部 import 增加：

```python
from modelctl.core.envfile import PROJECT_ROOT
from modelctl.engines._persist import persist_model_path
from modelctl.engines.llamacpp import download_gguf
```

在类中新增方法（放在 `build_command` 之前）：

```python
    def pre_start(self) -> None:
        cfg = self.profile.engine_config
        model = str(cfg.get("model") or "")
        if model and (Path(model).expanduser().is_file() or Path(model).expanduser().is_dir()):
            return
        if not cfg.get("download"):
            return
        dl = cfg["download"]
        model_root = Path(os.environ.get("MODEL_ROOT") or PROJECT_ROOT.parent / "model-gguf")
        try:
            model_match, _draft = download_gguf(
                dl["modelscope_id"], model_root, dl.get("quant", "UD-Q8_K_XL"), want_dspark=False
            )
        except RequirementError as error:
            raise RequirementError(
                f"{self.profile.name}：ModelScope 下载失败。\n{error}\n"
                "可配置 HF_ENDPOINT=https://hf-mirror.com 后从 Hugging Face 手动下载 "
                "unsloth GGUF 仓库，并将本地路径填入 unsloth.model。"
            ) from error
        persist_model_path(self.profile.path, "unsloth", str(model_match.resolve()))
        cfg["model"] = str(model_match.resolve())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_engines_unsloth.py -v`
Expected: PASS（16 个测试）

- [ ] **Step 5: 更新 `.env.example`**

在"模型存储目录"段之后追加：

```bash
# ---------- Unsloth 引擎 ----------
# Unsloth API key（unsloth 引擎必填；健康检查 /v1/models 依赖 Bearer 认证）
UNSLOTH_API_KEY=
# 远程 Unsloth 服务器地址（可选；不填则本地无头启动）
UNSLOTH_STUDIO_URL=
# HuggingFace 镜像（unsloth 模型 HF 兜底下载用，如 https://hf-mirror.com）
HF_ENDPOINT=
```

- [ ] **Step 6: 创建示例 profile**

创建 `models/unsloth/deepseek-v4-unsloth.yaml`：

```yaml
# deepseek-v4-unsloth.yaml —— DeepSeek-V4-Flash（Unsloth 引擎，无头 API）
# 命名避开根目录已有的 deepseek-v4.yaml（llamacpp 引擎，根目录优先）
name: deepseek-v4-unsloth
engine: unsloth
port: 8000
api_key: ${UNSLOTH_API_KEY}   # 必填：健康检查 /v1/models 依赖 Bearer 认证

unsloth:
  # HF 模型 ID 或本地 GGUF 路径；留空则通过 download 段自动下载
  model: unsloth/DeepSeek-V4-Flash-0731-GGUF
  # Unsloth 动态量化后缀；model 为本地路径时忽略
  gguf_variant: UD-Q8_K_XL
  context_length: 131072      # 请求的上下文长度
  tensor_parallel: false      # 多卡 GGUF tensor-parallel 加载
  load_in_4bit: false         # 非 GGUF HF 模型的 4bit 加载
  # ModelScope 自动下载配置（model 不存在本地路径时触发）
  download:
    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF
    quant: UD-Q8_K_XL
  extra_args: ""              # 透传其他 unsloth 参数

usage:
  price_in: 0.5
  price_out: 1.0
```

- [ ] **Step 7: 验证 profile 可加载**

Run: `uv run modelctl list --models-dir models`
Expected: 输出中包含 `deepseek-v4-unsloth unsloth 8000`（若本机未配置 `.env` 插值变量，可临时 `UV_NO_SYNC=1` 或设置 `UNSLOTH_API_KEY` 环境变量后执行）

- [ ] **Step 8: Commit**

```bash
git add src/modelctl/engines/unsloth.py .env.example models/unsloth/deepseek-v4-unsloth.yaml tests/test_engines_unsloth.py
git commit -m "feat(engines): unsloth 模型自动下载持久化与环境变量/示例 profile"
```

---

### Task 3: post_start 预热 + 文档更新 + 全量回归

**Files:**
- Modify: `src/modelctl/engines/unsloth.py`（顶部 import 与 `post_start`）
- Modify: `README.md`
- Modify: `docs/DeepSeek-V4-Flash后台启动指南.md`
- Test: 扩展 `tests/test_engines_unsloth.py`

**Interfaces:**
- Consumes: `profile.port`、`profile.api_key`。
- Produces: `post_start()` 对 `/v1/chat/completions` 发最小请求预热，失败静默。

- [ ] **Step 1: 编写失败测试**

追加到 `tests/test_engines_unsloth.py`：

```python
import json as _json  # 追加到文件顶部 import 区


def test_unsloth_post_start_sends_chat_request(tmp_path, monkeypatch):
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\napi_key: k\nunsloth:\n  model: m\n")
    a = get_adapter("unsloth")(p, CAPS8)
    seen = {}

    class _Resp:
        def read(self):
            return b"{}"

    def _fake(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = _json.loads(req.data)
        return _Resp()

    monkeypatch.setattr("modelctl.engines.unsloth.urllib.request.urlopen", _fake)

    a.post_start()
    assert seen["url"] == "http://127.0.0.1:30000/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["messages"][0]["content"] == "ping"


def test_unsloth_post_start_ignores_errors(tmp_path, monkeypatch):
    p = _write(tmp_path, "name: u\nengine: unsloth\nport: 30000\napi_key: k\nunsloth:\n  model: m\n")
    a = get_adapter("unsloth")(p, CAPS8)

    def _boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("modelctl.engines.unsloth.urllib.request.urlopen", _boom)
    a.post_start()  # 预热失败不应抛异常
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_engines_unsloth.py::test_unsloth_post_start_sends_chat_request tests/test_engines_unsloth.py::test_unsloth_post_start_ignores_errors -v`
Expected: FAIL — base 的 `post_start` 为空实现，无 `urllib.request` 请求发出（断言 seen 为空）

- [ ] **Step 3: 实现 post_start**

在 `src/modelctl/engines/unsloth.py` 顶部 import 增加：

```python
import json
import urllib.request
```

在类中新增方法（放在 `stop_patterns` 之前）：

```python
    def post_start(self) -> None:
        """预热：向 OpenAI 兼容端点发一个最小请求，降低首个请求冷启动延迟；失败不阻塞启动。"""
        body = json.dumps({"model": "default", "messages": [{"role": "user", "content": "ping"}]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.profile.port}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.profile.api_key}",
            },
        )
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except OSError:
            pass  # 预热失败不影响启动结果
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_engines_unsloth.py -v`
Expected: PASS（18 个测试）

- [ ] **Step 5: 更新 README.md**

按以下位置修改：

1. 第 3 行：`（llamacpp / ollama / vllm / sglang）` → `（llamacpp / ollama / vllm / sglang / unsloth）`
2. 第 7 行特性列表：`- **多引擎支持**：llamacpp（官方 llama.cpp + DSpark 投机解码）、ollama、vllm、sglang` → `- **多引擎支持**：llamacpp（官方 llama.cpp + DSpark 投机解码）、ollama、vllm、sglang、unsloth（无头 API 服务，Unsloth 动态量化 GGUF）`
3. 目录结构 models 段（第 33-37 行）追加：
   ```text
   │   ├── unsloth/                    # unsloth 引擎 profile 子目录
   │   │   └── deepseek-v4-unsloth.yaml
   ```
4. 第 74-77 行环境变量表追加一行：
   ```text
   | `deepseek-v4-unsloth` | `UNSLOTH_API_KEY`（必填）、`HF_ENDPOINT`（HF 兜底镜像）、`MODEL_ROOT`（ModelScope 下载） |
   ```
5. 第 108-111 行启动示例段追加：
   ```bash
   # 启动 DeepSeek-V4-Flash（unsloth 无头 API，Unsloth 动态量化 GGUF）
   # 模型从 ModelScope 下载并写回 profile；api_key 取 .env 中 UNSLOTH_API_KEY
   bash script/modelctl.sh start deepseek-v4-unsloth
   ```
6. 第 122-125 行验证段追加：
   ```bash
   curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $UNSLOTH_API_KEY"   # deepseek-v4-unsloth
   ```

- [ ] **Step 6: 更新后台启动指南**

在 `docs/DeepSeek-V4-Flash后台启动指南.md` 末尾追加章节：

```markdown
## Unsloth 引擎（实验性）

基于 Unsloth 无头 API 服务（`unsloth studio --api-only`）部署 Unsloth 动态量化 GGUF 模型。

### 前置条件

- 在目标服务器安装 Unsloth：`curl -fsSL https://unsloth.ai/install.sh | sh`（或独立 venv 安装，避免重依赖污染项目环境）
- `.env` 配置 `UNSLOTH_API_KEY`（必填，健康检查依赖）、可选 `HF_ENDPOINT`（HF 兜底镜像）、复用 `MODEL_ROOT`/`MODELSCOPE_CACHE`
- 启动前用 `unsloth --help` 核实无头服务 flag（`--api-only`、`--model`、`-p` 等），与本工具内置常量不一致时需调整 `engines/unsloth.py`

### 使用

```bash
bash script/modelctl.sh start deepseek-v4-unsloth   # 首次自动从 ModelScope 下载并写回 profile
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $UNSLOTH_API_KEY"
bash script/modelctl.sh status
```

### 已知限制

- 用量统计暂不支持精确统计（`/metrics` 端点未验证，`modelctl stats` 对该模型返回"不支持精确统计"）
- 健康检查使用 `/v1/models`（需认证），非 `/health`
```

- [ ] **Step 7: 全量回归（测试 + lint + 类型）**

Run:
```bash
uv run pytest
uv run ruff check src tests
uv run mypy src/modelctl
```
Expected: 全部 PASS（ruff 无告警，mypy 无错误）

- [ ] **Step 8: Commit**

```bash
git add src/modelctl/engines/unsloth.py tests/test_engines_unsloth.py README.md "docs/DeepSeek-V4-Flash后台启动指南.md"
git commit -m "feat(engines): unsloth 启动预热与文档更新"
```

---

## 自审结果

- **Spec 覆盖**：核心钩子（Task 1）、下载持久化与环境变量（Task 2）、预热/文档/回归（Task 3）；能力探测、注册表、KNOWN_ENGINES、probe 输出（Task 1）全部落实。
- **Placeholder 扫描**：所有步骤含完整代码与断言，无 TBD/TODO。
- **类型一致性**：`pre_start` 复用 `download_gguf` 的 `(Path, Path | None)` 返回签名（与 llamacpp 一致）；`persist_model_path(profile.path, "unsloth", ...)` 与现有引擎调用一致；`api_key_args()` 为 base 继承，无新增签名。

## 说明

- 未在计划中加入真实 GPU 集成测试步骤（依赖目标服务器与 Unsloth 安装），已列入文档"部署测试"章节作为后续人工验证项。
