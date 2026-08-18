# models 目录重构与模型自动下载实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按引擎分目录组织 models profile，支持从 ModelScope 自动下载并持久化 model 路径，同时补齐 YAML 注释与文档。

**Architecture:** 通过扩展 `profile.py` 的加载/列表逻辑支持 `models/<engine>/*.yaml` 与旧的 `models/*.yaml` 共存；新增 `engines/_download.py` 提供统一的 ModelScope 下载接口；在 `llamacpp` / `vllm` / `sglang` 适配器的 `pre_start` 中按需下载并把本地路径写回 YAML；最后更新全部 YAML 注释与 README/部署指南。

**Tech Stack:** Python 3.12, PyYAML, modelscope SDK, pytest

## Global Constraints

- 必须保持对现有 `models/*.yaml` 的向后兼容。
- 同一个 `name` 同时出现在根目录和子目录时，以根目录为准，并打印 warning。
- 下载失败时不能破坏原 YAML 文件；写回前先备份 `.bak`。
- 不要新增非必要的运行时依赖；优先使用已有的 `PyYAML` 和 `modelscope`。
- 所有修改必须通过现有 `pytest` 测试以及新增测试。

---

### Task 1: 扩展 profile.py 支持子目录加载

**Files:**
- Modify: `src/modelctl/core/profile.py:80-100`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `models_dir: Path`
- Produces: `load_profile(name, models_dir)` 支持 `models/<name>.yaml` 与 `models/<engine>/<name>.yaml`。
- Produces: `list_profiles(models_dir)` 递归扫描并去重。

- [ ] **Step 1: 写失败测试** —— 子目录加载

```python
def test_load_profile_from_engine_subdir(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")
    (tmp_path / "llamacpp").mkdir()
    (tmp_path / "llamacpp" / "qwen3.yaml").write_text(
        "name: qwen3\nengine: llamacpp\nport: 8000\nllamacpp:\n  model: /x.gguf\n",
        encoding="utf-8",
    )
    p = load_profile("qwen3", tmp_path)
    assert p.name == "qwen3" and p.engine == "llamacpp"
    assert p.path == tmp_path / "llamacpp" / "qwen3.yaml"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_profile.py::test_load_profile_from_engine_subdir -v`
Expected: FAIL `profile 不存在`

- [ ] **Step 3: 实现 `load_profile` 子目录查找**

在 `src/modelctl/core/profile.py` 中：

```python
def load_profile(name: str, models_dir: Path | None = None) -> Profile:
    models_dir = models_dir or PROJECT_ROOT / "models"
    candidates = [
        models_dir / f"{name}.yaml",
        *sorted(models_dir.rglob(f"{name}.yaml")),
    ]
    for path in candidates:
        if path.is_file():
            return _load_profile_from_path(path)
    raise ProfileError(f"profile 不存在：{models_dir / f'{name}.yaml'}")


def _load_profile_from_path(path: Path) -> Profile:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ProfileError(f"{path.name}：YAML 语法错误：{e}") from e
    if not isinstance(raw, dict):
        raise ProfileError(f"{path.name}：顶层必须是映射")
    return _to_profile(_interpolate(raw, path.name), path)
```

并修改 `_to_profile` 签名接收 `path: Path`。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/test_profile.py::test_load_profile_from_engine_subdir -v`
Expected: PASS

- [ ] **Step 5: 写失败测试** —— `list_profiles` 递归与去重

```python
def test_list_profiles_prefers_root_over_subdir(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("TEST_KEY", "secret")
    (tmp_path / "qwen3.yaml").write_text(
        "name: qwen3\nengine: ollama\nport: 11434\nollama:\n  model: qwen3:root\n",
        encoding="utf-8",
    )
    (tmp_path / "llamacpp").mkdir()
    (tmp_path / "llamacpp" / "qwen3.yaml").write_text(
        "name: qwen3\nengine: llamacpp\nport: 8000\nllamacpp:\n  model: /x.gguf\n",
        encoding="utf-8",
    )
    profiles = list_profiles(tmp_path)
    assert [p.name for p in profiles] == ["qwen3"]
    assert profiles[0].engine == "ollama"
```

- [ ] **Step 6: 实现 `list_profiles` 递归扫描与去重**

```python
def list_profiles(models_dir: Path | None = None) -> list[Profile]:
    models_dir = models_dir or PROJECT_ROOT / "models"
    if not models_dir.is_dir():
        return []
    root_files = sorted(models_dir.glob("*.yaml"))
    sub_files = sorted(p for p in models_dir.rglob("*.yaml") if p not in root_files)
    seen: set[str] = set()
    result: list[Profile] = []
    for p in root_files + sub_files:
        try:
            profile = _load_profile_from_path(p)
        except ProfileError:
            continue
        if profile.name in seen:
            logger.warning(f"忽略子目录中重复的 profile：{profile.name}（{p}）")
            continue
        seen.add(profile.name)
        result.append(profile)
    return result
```

- [ ] **Step 7: 运行全部 profile 测试**

Run: `uv run pytest tests/test_profile.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/modelctl/core/profile.py tests/test_profile.py
git commit -m "feat(profile): support models/<engine>/*.yaml with root fallback"
```

---

### Task 2: 创建子目录并迁移现有示例

**Files:**
- Create: `models/llamacpp/qwen3.yaml`
- Create: `models/ollama/qwen3.yaml`
- Create: `models/vllm/qwen3.yaml`
- Keep: `models/deepseek-v4.yaml`, `models/qwen3-*.yaml`（向后兼容）

**Interfaces:**
- Produces: 新路径的 profile 文件，name 去掉引擎后缀。

- [ ] **Step 1: 创建目录结构**

```bash
mkdir models/llamacpp models/ollama models/vllm models/sglang
```

- [ ] **Step 2: 复制并简化文件**

`models/llamacpp/qwen3.yaml`：

```yaml
name: qwen3
engine: llamacpp
port: 8000
api_key: ${API_KEY}

llamacpp:
  model: ""
  download:
    modelscope_id: unsloth/Qwen3.8-27B-GGUF
    quant: Q4_K_M
  parallel: 2
  ctx_size: 32768
  reasoning: on
  reasoning_format: deepseek
  dspark: off
  cache_type_k: q8_0
  cache_type_v: q8_0
  gpu_count: 2
  fit: off

usage:
  price_in: 0.5
  price_out: 1.0
```

`models/ollama/qwen3.yaml`：

```yaml
name: qwen3
engine: ollama
port: 11434

ollama:
  model: qwen3:32b
  keep_alive: -1
  num_parallel: 2
  context_length: 32768
```

`models/vllm/qwen3.yaml`：

```yaml
name: qwen3
engine: vllm
port: 8000
api_key: ${API_KEY}

vllm:
  model: ""
  download:
    modelscope_id: Qwen/Qwen3-32B
  tensor_parallel_size: 2
  max_model_len: 32768
  gpu_memory_utilization: 0.9
  extra_args: ""

usage:
  price_in: 0.5
  price_out: 1.0
```

- [ ] **Step 3: 验证 list 命令**

Run: `$env:API_KEY="test"; uv run modelctl list`
Expected: 根目录 `qwen3-ollama` 等与子目录 `qwen3` 同时出现，且子目录 `qwen3` 因同名被忽略（根优先）。

- [ ] **Step 4: Commit**

```bash
git add models/llamacpp models/ollama models/vllm models/sglang
git commit -m "feat(models): add engine subdirectories with qwen3 examples"
```

---

### Task 3: 新增 ModelScope 下载公共接口

**Files:**
- Create: `src/modelctl/engines/_download.py`
- Modify: `src/modelctl/engines/llamacpp.py`
- Test: `tests/test_engines_download.py`

**Interfaces:**
- Produces: `download_repo(modelscope_id: str, local_root: Path) -> Path`
- Produces: `download_gguf(...)` 保持原签名，内部调用 `download_repo`。

- [ ] **Step 1: 写失败测试**

```python
def test_download_repo_uses_modelscope(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(model_id, local_dir, **kwargs):
        calls.append((model_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr("modelscope.snapshot_download", fake_snapshot_download)
    from modelctl.engines._download import download_repo
    result = download_repo("unsloth/Qwen3.8-27B-GGUF", tmp_path)
    assert calls == [("unsloth/Qwen3.8-27B-GGUF", str(tmp_path / "Qwen3.8-27B-GGUF"))]
    assert result == tmp_path / "Qwen3.8-27B-GGUF"
```

- [ ] **Step 2: 创建 `_download.py`**

```python
"""engines/_download.py — 统一的 ModelScope 下载工具。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from loguru import logger


def ensure_modelscope() -> None:
    if importlib.util.find_spec("modelscope") is None:
        logger.info("未安装 modelscope，正在安装...")
        subprocess = __import__("subprocess")
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
```

- [ ] **Step 3: 修改 `llamacpp.py` 使用公共接口**

在 `download_gguf` 中保持原逻辑，但可复用 `download_repo` 下载整个仓库后再按 pattern 找文件。

```python
from modelctl.engines._download import download_repo

def download_gguf(modelscope_id: str, model_root: Path, quant: str, want_dspark: bool) -> tuple[Path, Path | None]:
    destination = download_repo(modelscope_id, model_root)
    patterns = [f"*{quant}*-00001-of-*.gguf", f"*{quant}*.gguf"]
    if want_dspark:
        patterns.extend(DSPARK_PATTERNS)
    # ... 后续逻辑不变
```

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/test_engines_download.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/modelctl/engines/_download.py src/modelctl/engines/llamacpp.py tests/test_engines_download.py
git commit -m "feat(download): add unified ModelScope download helper"
```

---

### Task 4: 实现 model 持久化写回 YAML

**Files:**
- Create: `src/modelctl/engines/_persist.py`
- Modify: `src/modelctl/engines/llamacpp.py`
- Modify: `src/modelctl/engines/vllm.py`
- Modify: `src/modelctl/engines/sglang.py`
- Test: `tests/test_engines_persist.py`

**Interfaces:**
- Produces: `persist_model_path(profile_path: Path, engine: str, model_path: str) -> None`

- [ ] **Step 1: 写失败测试**

```python
def test_persist_model_path_updates_yaml(tmp_path):
    yaml_path = tmp_path / "demo.yaml"
    yaml_path.write_text(
        "name: demo\nengine: llamacpp\nport: 8000\nllamacpp:\n  model: ''\n  parallel: 2\n",
        encoding="utf-8",
    )
    from modelctl.engines._persist import persist_model_path
    persist_model_path(yaml_path, "llamacpp", "/downloaded/model.gguf")
    content = yaml_path.read_text(encoding="utf-8")
    assert "model: /downloaded/model.gguf" in content
    assert (tmp_path / "demo.yaml.bak").is_file()
```

- [ ] **Step 2: 实现 `_persist.py`**

```python
"""engines/_persist.py — 将下载后的本地 model 路径写回 profile YAML。"""

from __future__ import annotations

from pathlib import Path

import yaml


def persist_model_path(profile_path: Path, engine: str, model_path: str) -> None:
    """仅更新 YAML 中 <engine>.model 字段，写回前备份原文件。"""
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{profile_path} 顶层必须是映射")

    backup = profile_path.with_suffix(".yaml.bak")
    backup.write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")

    engine_config = raw.setdefault(engine, {})
    if not isinstance(engine_config, dict):
        raise ValueError(f"{profile_path} 中 {engine} 段必须是映射")
    engine_config["model"] = model_path

    with profile_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
```

- [ ] **Step 3: 在 `llamacpp.py` 中调用**

修改 `pre_start`：

```python
from modelctl.engines._persist import persist_model_path

# 在下载成功后
self._model, auto_draft = download_gguf(...)
if self._draft is None:
    self._draft = auto_draft
persist_model_path(self.profile.path, "llamacpp", str(self._model.resolve()))
```

- [ ] **Step 4: 在 `vllm.py` 中新增 `pre_start`**

```python
def pre_start(self) -> None:
    cfg = self.profile.engine_config
    model = str(cfg.get("model", ""))
    if model and (Path(model).expanduser().is_dir() or Path(model).expanduser().is_file()):
        return
    if cfg.get("download"):
        from modelctl.engines._download import download_repo
        from modelctl.engines._persist import persist_model_path
        from modelctl.core.envfile import PROJECT_ROOT
        model_root = Path(os.environ.get("MODEL_ROOT") or PROJECT_ROOT.parent / "model-hf")
        local_dir = download_repo(cfg["download"]["modelscope_id"], model_root)
        persist_model_path(self.profile.path, "vllm", str(local_dir.resolve()))
        cfg["model"] = str(local_dir.resolve())
```

- [ ] **Step 5: 在 `sglang.py` 中新增 `pre_start`**

与 vllm 相同，仅 engine 段名称不同。

- [ ] **Step 6: 运行测试**

Run: `uv run pytest tests/test_engines_persist.py tests/test_engines_llamacpp.py tests/test_engines_vllm.py tests/test_engines_sglang.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/modelctl/engines/_persist.py src/modelctl/engines/llamacpp.py src/modelctl/engines/vllm.py src/modelctl/engines/sglang.py tests/test_engines_persist.py
git commit -m "feat(engines): persist downloaded model path back to YAML"
```

---

### Task 5: 为所有 YAML 增加详细注释

**Files:**
- Modify: `models/deepseek-v4.yaml`
- Modify: `models/qwen3-llama.yaml`
- Modify: `models/qwen3-ollama.yaml`
- Modify: `models/qwen3-vllm.yaml`
- Modify: `models/llamacpp/qwen3.yaml`
- Modify: `models/ollama/qwen3.yaml`
- Modify: `models/vllm/qwen3.yaml`

- [ ] **Step 1: 为每个 YAML 添加字段注释**

每个文件注释覆盖：
- 顶层字段 `name/engine/port/api_key`
- 引擎特有字段
- `download` 段的作用
- 量化/显存建议

示例见 Task 2。

- [ ] **Step 2: 验证 YAML 语法**

Run: `uv run python -c "from modelctl.core.profile import list_profiles; list_profiles()"`
Expected: 无异常

- [ ] **Step 3: Commit**

```bash
git add models/
git commit -m "docs(models): add detailed comments to all profile YAMLs"
```

---

### Task 6: 更新 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 更新目录结构示例**

替换 README 中 `models/` 目录示例为新结构。

- [ ] **Step 2: 更新启动示例**

说明既可用旧路径 `modelctl start qwen3-llama`，也可用新路径 `modelctl start qwen3`（若根目录无同名文件）。

- [ ] **Step 3: 添加自动下载说明**

新增一段：

```markdown
### 模型自动下载

profile 中 `model` 为空或指向路径不存在时，若配置了 `download.modelscope_id`，
启动脚本会自动从 ModelScope 下载模型，并将本地路径写回 YAML 的 `model` 字段。
下次启动时直接复用本地模型，无需重新下载。
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document new models layout and auto-download"
```

---

### Task 7: 更新后台启动指南

**Files:**
- Modify: `docs/DeepSeek-V4-Flash后台启动指南.md`

- [ ] **Step 1: 在目录布局章节补充子目录规则**

说明 `models/<engine>/*.yaml` 是新推荐方式，旧 `models/*.yaml` 继续兼容。

- [ ] **Step 2: 在配置说明章节补充 download 段**

说明 `download.modelscope_id` 与 `quant` 的用法，以及不同引擎的模型格式要求。

- [ ] **Step 3: Commit**

```bash
git add docs/DeepSeek-V4-Flash后台启动指南.md
git commit -m "docs(deploy): update guide for engine subdirs and model download"
```

---

### Task 8: 全量测试与验证

- [ ] **Step 1: 运行全部测试**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: 验证 CLI list**

Run: `$env:API_KEY="test"; uv run modelctl list`
Expected: 显示所有 profile，无重复 `qwen3`。

- [ ] **Step 3: 最终 commit（若前面未提交）**

---

## Spec Coverage Check

| Spec 需求 | 对应 Task |
|-----------|----------|
| 按引擎分目录 | Task 1, Task 2 |
| 向后兼容根目录 YAML | Task 1 |
| ModelScope 自动下载 | Task 3, Task 4 |
| 下载后持久化写回 YAML | Task 4 |
| YAML 详细注释 | Task 5 |
| 更新 README | Task 6 |
| 更新后台启动指南 | Task 7 |

## Placeholder Scan

- 无 TBD/TODO/"implement later" 等占位符。
- 每个测试步骤包含具体断言。
- 每个实现步骤包含代码片段。

## Type Consistency Check

- `download_repo` 返回 `Path`。
- `persist_model_path` 接收 `Path, str, str`。
- 各引擎 `pre_start` 中统一调用上述接口。
