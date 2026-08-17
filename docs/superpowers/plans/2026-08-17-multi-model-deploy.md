# 多模型部署启动器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单一 DeepSeek-V4 启动脚本改造为引擎插件式多模型启动器（modelctl），支持 llamacpp/ollama/vllm/sglang，每模型一个 YAML profile。

**Architecture:** 统一 CLI（modelctl.py）按 profile 的 `engine` 字段分发到引擎适配器（engines/*.py），共用核心模块：profile 加载（core/profile.py）、硬件能力探测（core/capabilities.py）、进程生命周期（core/process.py）、用量统计（core/stats.py）。现有 `start_v4_flash_gguf.py` 迁移为 llamacpp 适配器，`usage_stats_server.py` 迁移为 core/stats.py。

**Tech Stack:** Python 3.12+、PyYAML、标准库（http.server / subprocess / urllib）、pytest。

## Global Constraints

- 目标部署机：Linux，8× RTX 5880 Ada（CC 8.9）；开发机为 Windows，**测试必须可在 Windows 上运行**（所有 subprocess/nvidia-smi 调用必须可 mock，禁止在模块导入期执行外部命令）。
- `requires-python = ">=3.12"`；运行期依赖仅 `PyYAML>=6.0`；测试依赖 pytest。
- 配置优先级：profile YAML > `.env`（`${VAR}` 插值来源）> 代码默认值。
- 模型存储默认值（`.env` 可覆盖）：`MODEL_ROOT=/raid5/sh/model/model-gguf`、`MODELSCOPE_CACHE=/raid5/sh/model/modelscope`、`OLLAMA_MODELS=/raid5/sh/model/ollama-models`、`HF_HOME=/raid5/sh/model/huggingface`。
- 用量统计对外 `/api/usage` 输出格式与现版一致（cc-switch 无感）。
- 不支持的功能**自动降级 + warning**，不直接崩溃（除非显存/硬件硬性不满足，此时拒绝启动并说明原因）。
- 代码注释用中文。

## 文件结构总览

| 文件 | 职责 |
|---|---|
| `script/modelctl.py` | CLI 入口：start/stop/restart/status/list/probe |
| `script/modelctl.sh` | bash 薄封装（替代 start_v4_flash_background.sh） |
| `script/core/envfile.py` | .env 解析与注入（从 start_v4_flash_gguf.py 抽出） |
| `script/core/profile.py` | YAML profile 加载、${VAR} 插值、校验 |
| `script/core/capabilities.py` | nvidia-smi/二进制探测、CC 比较、显存预检 |
| `script/core/process.py` | 后台启动、PID 文件、stop/status、健康检查 |
| `script/core/stats.py` | 用量统计服务（多引擎指标映射） |
| `script/engines/__init__.py` | 引擎注册表 `get_adapter(engine) -> type[EngineAdapter]` |
| `script/engines/base.py` | EngineAdapter 抽象基类 |
| `script/engines/llamacpp.py` | llama.cpp 适配器（含编译/下载/DSpark） |
| `script/engines/ollama.py` | ollama 适配器 |
| `script/engines/vllm.py` | vllm 适配器 |
| `script/engines/sglang.py` | sglang 适配器 |
| `models/*.yaml` | 模型 profile |
| `tests/` | 各模块单元测试 |

---

### Task 1: 项目脚手架与 core/envfile.py

**Files:**
- Modify: `pyproject.toml`
- Create: `script/core/__init__.py`（空）、`script/engines/__init__.py`（本任务先空）、`script/core/envfile.py`
- Test: `tests/test_envfile.py`

**Interfaces:**
- Produces:
  - `load_env(env_path: Path | None = None) -> Path` — 把 .env 注入 os.environ（不覆盖已有变量），返回检查过的路径
  - `parse_env_file(path: Path) -> dict[str, str]`
  - `PROJECT_ROOT: Path` — `Path(__file__).resolve().parents[2]`

- [ ] **Step 1: 更新 pyproject.toml**

```toml
[project]
name = "modelctl"
version = "0.2.0"
description = "多模型部署启动器（llamacpp / ollama / vllm / sglang）"
requires-python = ">=3.12"
dependencies = ["PyYAML>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 写失败测试 `tests/test_envfile.py`**

```python
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.envfile import parse_env_file, load_env  # noqa: E402


def test_parse_env_file_basic(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# 注释\nFOO=bar\nEMPTY=\nQUOTED="a b"\nSINGLE=\'x\'\n', encoding="utf-8")
    assert parse_env_file(p) == {"FOO": "bar", "EMPTY": "", "QUOTED": "a b", "SINGLE": "x"}


def test_parse_env_file_missing(tmp_path):
    assert parse_env_file(tmp_path / "nope.env") == {}


def test_load_env_no_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("KEEP=from_file\nNEW=new_value\n", encoding="utf-8")
    monkeypatch.setenv("KEEP", "from_env")
    monkeypatch.delenv("NEW", raising=False)
    load_env(p)
    assert os.environ["KEEP"] == "from_env"
    assert os.environ["NEW"] == "new_value"
```

- [ ] **Step 3: 运行确认失败**

Run: `python -m pytest tests/test_envfile.py -v`
Expected: FAIL（ModuleNotFoundError: core.envfile）

- [ ] **Step 4: 实现 `script/core/envfile.py`（并创建两个空 `__init__.py`）**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/envfile.py — .env 解析与注入（优先级：已存在环境变量 > .env）。"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def load_env(env_path: Path | None = None) -> Path:
    path = env_path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return path
    for key, value in parse_env_file(path).items():
        os.environ.setdefault(key, value)
    return path
```

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_envfile.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml script/core/__init__.py script/engines/__init__.py script/core/envfile.py tests/test_envfile.py
git commit -m "feat(core): 项目脚手架升级 Python 3.12 + PyYAML，新增 envfile 模块"
```

---

### Task 2: core/profile.py — YAML 加载、插值、校验

**Files:**
- Create: `script/core/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `core.envfile.load_env`、`PROJECT_ROOT`
- Produces:
  - `@dataclass Profile`: `name: str`、`engine: str`、`port: int`、`api_key: str | None`、`engine_config: dict[str, Any]`、`usage: dict[str, Any]`、`path: Path | None`
  - `load_profile(name: str, models_dir: Path | None = None) -> Profile`（缺省 `PROJECT_ROOT / "models"`）
  - `list_profiles(models_dir: Path | None = None) -> list[Profile]`（按文件名排序）
  - `ProfileError(ValueError)`；`KNOWN_ENGINES = {"llamacpp", "ollama", "vllm", "sglang"}`

插值规则：字符串值中 `${VAR}` 从 `os.environ` 取（调用方先 `load_env()`）；变量未定义或为空 → `ProfileError`。dict/list 递归插值。`engine_config` 为与 `engine` 同名的段，缺失为空 dict。

- [ ] **Step 1: 写失败测试 `tests/test_profile.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.profile import Profile, ProfileError, list_profiles, load_profile  # noqa: E402

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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_profile.py -v`
Expected: FAIL（ModuleNotFoundError: core.profile）

- [ ] **Step 3: 实现 `script/core/profile.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/profile.py — 模型 profile（models/<name>.yaml）加载、${VAR} 插值与校验。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.envfile import PROJECT_ROOT

KNOWN_ENGINES = {"llamacpp", "ollama", "vllm", "sglang"}
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProfileError(ValueError):
    """profile 校验或插值失败。"""


@dataclass
class Profile:
    name: str
    engine: str
    port: int
    api_key: str | None = None
    engine_config: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None


def _interpolate(value: Any, source: str) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            var = m.group(1)
            env_val = os.environ.get(var)
            if env_val is None or env_val == "":
                raise ProfileError(f"{source}：插值变量 {var} 未在环境变量/.env 中定义")
            return env_val
        return _VAR_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, source) for v in value]
    return value


def _to_profile(raw: dict[str, Any], path: Path) -> Profile:
    src = path.name
    for key in ("name", "engine", "port"):
        if key not in raw or raw[key] in (None, ""):
            raise ProfileError(f"{src}：缺少必填字段 {key}")
    engine = str(raw["engine"])
    if engine not in KNOWN_ENGINES:
        raise ProfileError(f"{src}：未知引擎 {engine}（支持：{sorted(KNOWN_ENGINES)}）")
    port = int(raw["port"])
    if not 1 <= port <= 65535:
        raise ProfileError(f"{src}：port 必须在 1-65535，当前 {port}")
    engine_config = raw.get(engine) or {}
    if not isinstance(engine_config, dict):
        raise ProfileError(f"{src}：{engine} 段必须是映射")
    return Profile(
        name=str(raw["name"]), engine=engine, port=port,
        api_key=raw.get("api_key") or None,
        engine_config=engine_config, usage=raw.get("usage") or {}, path=path,
    )


def load_profile(name: str, models_dir: Path | None = None) -> Profile:
    models_dir = models_dir or PROJECT_ROOT / "models"
    path = models_dir / f"{name}.yaml"
    if not path.is_file():
        raise ProfileError(f"profile 不存在：{path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ProfileError(f"{path.name}：YAML 语法错误：{e}") from e
    if not isinstance(raw, dict):
        raise ProfileError(f"{path.name}：顶层必须是映射")
    return _to_profile(_interpolate(raw, path.name), path)


def list_profiles(models_dir: Path | None = None) -> list[Profile]:
    models_dir = models_dir or PROJECT_ROOT / "models"
    if not models_dir.is_dir():
        return []
    return [load_profile(p.stem, models_dir) for p in sorted(models_dir.glob("*.yaml"))]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_profile.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add script/core/profile.py tests/test_profile.py
git commit -m "feat(core): profile YAML 加载、插值与校验"
```

---

### Task 3: core/capabilities.py — 硬件能力探测

**Files:**
- Create: `script/core/capabilities.py`
- Test: `tests/test_capabilities.py`

**Interfaces:**
- Produces:
  - `@dataclass Capabilities`: `gpu_count: int = 0`、`gpu_name: str = ""`、`vram_total_mb: int = 0`（单卡）、`vram_free_mb: list[int]`、`cuda_driver: str = ""`、`compute_capability: str = ""`、`binaries: dict[str, bool]`
  - `probe(nvidia_smi_output: str | None = None) -> Capabilities` — 传入字符串时仅解析（测试用）；为 None 时实际调用 nvidia-smi，失败返回 gpu_count=0 兜底对象
  - `cc_at_least(cc: str, major: int, minor: int) -> bool`
  - `free_vram_total_mb(caps: Capabilities) -> int`
  - `which_binaries(names: list[str]) -> dict[str, bool]`
  - `ENGINE_BINARIES = ["ollama", "vllm", "sglang"]`

nvidia-smi 查询命令：
`nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits`

- [ ] **Step 1: 写失败测试 `tests/test_capabilities.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.capabilities import cc_at_least, free_vram_total_mb, probe  # noqa: E402

SMI_5880 = "\n".join(["RTX 5880 Ada Generation, 49140, 48000, 580.65.05, 8.9"] * 8)


def test_probe_5880():
    caps = probe(nvidia_smi_output=SMI_5880)
    assert caps.gpu_count == 8
    assert caps.gpu_name == "RTX 5880 Ada Generation"
    assert caps.vram_total_mb == 49140
    assert caps.vram_free_mb == [48000] * 8
    assert caps.compute_capability == "8.9"
    assert caps.cuda_driver == "580.65.05"


def test_probe_failure_returns_empty():
    caps = probe(nvidia_smi_output="")
    assert caps.gpu_count == 0
    assert caps.compute_capability == ""


def test_cc_at_least():
    assert cc_at_least("8.9", 8, 9)
    assert cc_at_least("8.9", 8, 0)
    assert not cc_at_least("8.9", 9, 0)
    assert not cc_at_least("", 8, 0)


def test_free_vram_total():
    caps = probe(nvidia_smi_output=SMI_5880)
    assert free_vram_total_mb(caps) == 48000 * 8
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_capabilities.py -v`
Expected: FAIL（ModuleNotFoundError: core.capabilities）

- [ ] **Step 3: 实现 `script/core/capabilities.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/capabilities.py — 启动前硬件/环境能力探测（GPU、CC、引擎二进制）。"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

ENGINE_BINARIES = ["ollama", "vllm", "sglang"]  # llamacpp 由源码编译，不在此列


@dataclass
class Capabilities:
    gpu_count: int = 0
    gpu_name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: list[int] = field(default_factory=list)
    cuda_driver: str = ""
    compute_capability: str = ""
    binaries: dict[str, bool] = field(default_factory=dict)


def which_binaries(names: list[str]) -> dict[str, bool]:
    return {n: shutil.which(n) is not None for n in names}


def _run_nvidia_smi() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    return out.stdout if out.returncode == 0 else ""


def _safe_smi() -> str:
    try:
        return _run_nvidia_smi()
    except (OSError, subprocess.SubprocessError):
        return ""


def probe(nvidia_smi_output: str | None = None) -> Capabilities:
    text = nvidia_smi_output if nvidia_smi_output is not None else _safe_smi()
    caps = Capabilities(binaries=which_binaries(ENGINE_BINARIES))
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if not rows:
        return caps
    frees: list[int] = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 5:
            continue
        if not caps.gpu_name:
            caps.gpu_name = parts[0]
            caps.cuda_driver = parts[3]
            caps.compute_capability = parts[4]
            try:
                caps.vram_total_mb = int(parts[1])
            except ValueError:
                pass
        try:
            frees.append(int(parts[2]))
        except ValueError:
            frees.append(0)
    caps.vram_free_mb = frees
    caps.gpu_count = len(frees)
    return caps


def cc_at_least(cc: str, major: int, minor: int) -> bool:
    try:
        hi, lo = (int(x) for x in cc.split(".", 1))
    except (ValueError, AttributeError):
        return False
    return (hi, lo) >= (major, minor)


def free_vram_total_mb(caps: Capabilities) -> int:
    return sum(caps.vram_free_mb)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_capabilities.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add script/core/capabilities.py tests/test_capabilities.py
git commit -m "feat(core): 硬件能力探测（GPU/CC/显存/引擎二进制）"
```

---

### Task 4: core/process.py — 进程生命周期管理

**Files:**
- Create: `script/core/process.py`
- Test: `tests/test_process.py`

**Interfaces:**
- Consumes: `core.envfile.PROJECT_ROOT`
- Produces:
  - `log_dir() -> Path` — `os.environ["LOG_DIR"]` 或 `PROJECT_ROOT.parent / "logs"`，自动 mkdir
  - `pid_file(name: str) -> Path` — `log_dir() / f"{name}.pid"`
  - `launch_log(name: str) -> Path | None` — 最新 `launch-{name}-*.log`
  - `start_detached(name: str, command: list[str], extra_env: dict[str, str]) -> int` — 后台启动（POSIX `start_new_session=True`），输出重定向到 `launch-{name}-{时间戳}.log`，写 PID 文件，返回 PID
  - `stop_instance(name: str, port: int, patterns: list[str]) -> bool` — PID terminate → 超时(10s) kill → fuser -k 端口 → pkill -f patterns；best-effort
  - `is_running(name: str) -> bool`
  - `wait_health(url: str, timeout: float, api_key: str | None = None) -> bool` — 2s 轮询 GET，2xx 成功；api_key 非空带 Bearer
  - `tail_file(path: Path, lines: int) -> str`

Windows 兼容：is_running 用 tasklist，stop_instance 用 taskkill；fuser/pkill 仅 POSIX。

- [ ] **Step 1: 写失败测试 `tests/test_process.py`**

```python
# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core import process  # noqa: E402


def test_pid_file_path(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    assert process.pid_file("demo") == tmp_path / "demo.pid"


def test_start_and_is_running(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    pid = process.start_detached("sleeper", [sys.executable, "-c", "import time; time.sleep(60)"], {})
    assert pid > 0
    assert process.is_running("sleeper")
    process.stop_instance("sleeper", port=1, patterns=[])
    deadline = time.time() + 5
    while process.is_running("sleeper") and time.time() < deadline:
        time.sleep(0.2)
    assert not process.is_running("sleeper")


def test_is_running_no_pidfile(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    assert not process.is_running("ghost")


def test_launch_log_created(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    process.start_detached("echoer", [sys.executable, "-c", "print('hello-log')"], {})
    time.sleep(1)
    log = process.launch_log("echoer")
    assert log is not None and "hello-log" in log.read_text(encoding="utf-8", errors="replace")


def test_tail_file(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
    assert process.tail_file(f, 3).splitlines() == ["line97", "line98", "line99"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_process.py -v`
Expected: FAIL（ModuleNotFoundError: core.process）

- [ ] **Step 3: 实现 `script/core/process.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/process.py — 引擎无关的进程生命周期：后台启动、PID、停止、健康检查。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from core.envfile import PROJECT_ROOT


def log_dir() -> Path:
    d = Path(os.environ.get("LOG_DIR") or PROJECT_ROOT.parent / "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file(name: str) -> Path:
    return log_dir() / f"{name}.pid"


def launch_log(name: str) -> Path | None:
    logs = sorted(log_dir().glob(f"launch-{name}-*.log"))
    return logs[-1] if logs else None


def start_detached(name: str, command: list[str], extra_env: dict[str, str]) -> int:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir() / f"launch-{name}-{stamp}.log"
    env = {**os.environ, **extra_env}
    fp = open(log_path, "a", encoding="utf-8")
    kwargs: dict = {"stdout": fp, "stderr": subprocess.STDOUT, "env": env, "stdin": subprocess.DEVNULL}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True  # nohup 语义：SSH 断开不影响
    proc = subprocess.Popen(command, **kwargs)
    pid_file(name).write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def is_running(name: str) -> bool:
    pf = pid_file(name)
    if not pf.is_file():
        return False
    try:
        pid = int(pf.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    if sys.platform == "win32":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_instance(name: str, port: int, patterns: list[str]) -> bool:
    """先 PID 优雅终止，再按端口/进程名兜底。返回是否有进程被终止。"""
    stopped = False
    pf = pid_file(name)
    if pf.is_file():
        try:
            pid = int(pf.read_text(encoding="utf-8").strip())
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.time() + 10
                while time.time() < deadline:
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.5)
                    except OSError:
                        break
                else:
                    os.killpg(pid, signal.SIGKILL)
            stopped = True
        except (ValueError, OSError):
            pass
        pf.unlink(missing_ok=True)
    if sys.platform != "win32":
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
        for pat in patterns:
            subprocess.run(["pkill", "-f", pat], capture_output=True)
    return stopped


def wait_health(url: str, timeout: float, api_key: str | None = None) -> bool:
    deadline = time.time() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def tail_file(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_process.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add script/core/process.py tests/test_process.py
git commit -m "feat(core): 进程生命周期管理（后台启动/PID/停止/健康检查）"
```

---

### Task 5: engines/base.py + 注册表 + llamacpp 适配器

**Files:**
- Create: `script/engines/base.py`、`script/engines/llamacpp.py`
- Modify: `script/engines/__init__.py`（填注册表）
- Test: `tests/test_engines_llamacpp.py`

**Interfaces:**
- Consumes: `core.profile.Profile / ProfileError`、`core.capabilities.Capabilities / probe / free_vram_total_mb`
- Produces（base.py）：
  - `class RequirementError(RuntimeError)`
  - `class EngineAdapter(ABC)`：`__init__(self, profile: Profile, caps: Capabilities)`，存 `self.profile / self.caps / self.warnings: list[str]`
  - 抽象方法：`build_command() -> tuple[list[str], dict[str, str]]`、`check_requirements() -> None`（降级写 warnings，硬性不满足抛 RequirementError）、`metrics_mapping() -> dict[str, list[str]] | None`
  - 可覆写：`health_url() -> str`（默认 `http://127.0.0.1:{port}/health`）、`pre_start()`、`post_start()`、`stop_patterns() -> list[str]`（默认 []）、`api_key_args() -> list[str]`（有 api_key 则 `["--api-key", key]`）
- Produces（`engines/__init__.py`）：`get_adapter(engine: str) -> type[EngineAdapter]`，未知引擎抛 `ProfileError`
- Produces（llamacpp.py）：`class LlamaCppAdapter(EngineAdapter)`，模块常量 `OFFICIAL_URL`、`CTX_PER_SLOT = 1_048_576`、`DSPARK_PATTERNS = ["*dspark*"]`

llamacpp 行为（迁移自 start_v4_flash_gguf.py，参数语义不变）：
- engine_config 键：`model`（必填）、`draft`、`parallel`（默认 2）、`ctx_size`（空 = parallel × 1M）、`reasoning`（on）、`reasoning_format`（deepseek）、`dspark`（on）、`spec_type`（draft-dspark）、`spec_draft_n_max`（3）、`n_gpu_layers_draft`（999）、`cache_type_k/v`（q8_0）、`gpu_count`（8）、`fit`（off）、`repeat_penalty`（空不传）、`source_dir`（空 = env `LLAMACPP_SOURCE_DIR` 或 `PROJECT_ROOT.parent/"llama.cpp"`）、`download`（dict：`modelscope_id`/`quant`）
- `check_requirements`：`caps.gpu_count == 0` → RequirementError；`gpu_count` 配置 > 实际 → RequirementError("GPU"字样)；model 文件不存在且无 download 段 → RequirementError；dspark=on 但找不到草稿 → warning + 关闭 dspark；模型文件大小 × 1.1 > 剩余显存总量 → RequirementError
- `build_command` 参数与原脚本一致：`[server, --model m, --host 0.0.0.0, --port p, --ctx-size c, --parallel n, --n-gpu-layers 999, --split-mode layer, --tensor-split "1,1,...", --jinja, --reasoning on, --reasoning-format deepseek, --flash-attn on, --metrics]` + api_key + repeat_penalty + dspark 段 + `--fit off` + cache-type
- `metrics_mapping` 四组：prompt_total=[`llamacpp:prompt_tokens_total`, `llamacpp:tokens_evaluated_total`, `llama_tokens_evaluated_total`, `prompt_tokens_total`]，predicted_total=[`llamacpp:tokens_predicted_total`, `llamacpp:predicted_tokens_total`, `llama_tokens_predicted_total`, `tokens_predicted_total`]，prompt_rate=[`llamacpp:prompt_tokens_seconds`, `prompt_tokens_seconds`]，predicted_rate=[`llamacpp:predicted_tokens_seconds`, `llamacpp:tokens_predicted_seconds`, `predicted_tokens_seconds`]
- `pre_start`：source_dir 不存在 → `git clone --depth 1 OFFICIAL_URL`；未编译 → cmake `-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release` 构建；download 段存在且 model 不存在 → modelscope `snapshot_download`（patterns 同原脚本）
- `stop_patterns`：`["llama-server"]`
- 从原脚本原样搬运的辅助函数：`download_gguf`、`find_server`、`_find_first`、`require`、`run`

- [ ] **Step 1: 写失败测试 `tests/test_engines_llamacpp.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.capabilities import probe  # noqa: E402
from core.profile import ProfileError, load_profile  # noqa: E402
from engines import get_adapter  # noqa: E402
from engines.base import RequirementError  # noqa: E402

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
    (tmp_path / "ds.yaml").write_text(
        f"name: ds\nengine: llamacpp\nport: 18888\nllamacpp:\n  model: {tmp_path}/m.gguf\n  gpu_count: 8\n",
        encoding="utf-8")
    caps = probe(nvidia_smi_output=SMI)
    adapter = get_adapter("llamacpp")(load_profile("ds", tmp_path), caps)
    adapter.check_requirements()
    cmd, _ = adapter.build_command()
    assert "--model-draft" not in cmd
    assert any("park" in w.lower() for w in adapter.warnings)


def test_gpu_count_exceeds_hw(tmp_path):
    caps = probe(nvidia_smi_output="\n".join(
        ["RTX 5880 Ada Generation, 49140, 48000, 580.65.05, 8.9"] * 2))
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
```

注意：`test_dspark_disabled_when_no_draft` 中草稿查找范围是 model 同目录及上级——测试目录无 dspark 文件故降级。`test_build_command` 依赖 tmp_path 下的 `dspark-x.gguf` 被自动发现。

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_engines_llamacpp.py -v`
Expected: FAIL（ImportError: engines.base）

- [ ] **Step 3: 实现 `script/engines/base.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/base.py — 引擎适配器抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.capabilities import Capabilities
from core.profile import Profile


class RequirementError(RuntimeError):
    """硬性条件不满足，拒绝启动。"""


class EngineAdapter(ABC):
    def __init__(self, profile: Profile, caps: Capabilities):
        self.profile = profile
        self.caps = caps
        self.warnings: list[str] = []

    @abstractmethod
    def build_command(self) -> tuple[list[str], dict[str, str]]:
        """返回 (启动命令, 需注入的环境变量)。"""

    @abstractmethod
    def check_requirements(self) -> None:
        """校验硬件/配置门槛；可降级的写 self.warnings，硬性不满足抛 RequirementError。"""

    @abstractmethod
    def metrics_mapping(self) -> dict[str, list[str]] | None:
        """Prometheus 指标名映射；None 表示该引擎不支持精确统计。"""

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}/health"

    def pre_start(self) -> None:
        """启动前钩子（下载/编译/pull）。"""

    def post_start(self) -> None:
        """启动后钩子（如 ollama 预加载模型）。"""

    def stop_patterns(self) -> list[str]:
        return []

    def api_key_args(self) -> list[str]:
        return ["--api-key", self.profile.api_key] if self.profile.api_key else []
```

- [ ] **Step 4: 实现 `script/engines/llamacpp.py`**

结构如下（辅助函数从 `script/start_v4_flash_gguf.py` 原样搬运）：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/llamacpp.py — 官方 llama.cpp (GGUF) 适配器，含编译/下载/DSpark。"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from core.capabilities import free_vram_total_mb
from core.envfile import PROJECT_ROOT
from engines.base import EngineAdapter, RequirementError

OFFICIAL_URL = "https://github.com/ggml-org/llama.cpp.git"
DSPARK_PATTERNS = ["*dspark*"]
CTX_PER_SLOT = 1_048_576


# --- 原样搬运自 start_v4_flash_gguf.py ---
# def run(command, *, cwd=None): ...
# def require(name): ...
# def find_server(source) -> Path: ...
# def _find_first(destination, patterns): ...
# def download_gguf(modelscope_id, model_root, quant, want_dspark): ...


class LlamaCppAdapter(EngineAdapter):
    def __init__(self, profile, caps):
        super().__init__(profile, caps)
        self._dspark = False
        self._draft: Path | None = None
        self._model: Path | None = None

    def check_requirements(self) -> None:
        cfg = self.profile.engine_config
        if self.caps.gpu_count == 0:
            raise RequirementError("未探测到 GPU（nvidia-smi 失败或无 GPU）")
        gpu_count = int(cfg.get("gpu_count", 8))
        if gpu_count > self.caps.gpu_count:
            raise RequirementError(f"profile gpu_count={gpu_count} 超过实际 GPU 数 {self.caps.gpu_count}")
        model = cfg.get("model")
        if not model:
            raise RequirementError(f"{self.profile.name}：llamacpp.model 必填")
        self._model = Path(model).expanduser()
        if not self._model.is_file() and not cfg.get("download"):
            raise RequirementError(f"找不到 GGUF 模型：{self._model}（且未配置 download 段）")
        # DSpark 草稿发现与显存降级
        if str(cfg.get("dspark", "on")).lower() in ("on", "true", "1"):
            self._draft = self._find_draft(cfg)
            if self._draft is None:
                self.warnings.append("未找到 DSpark 草稿模型，已自动关闭 DSpark")
            elif free_vram_total_mb(self.caps) < 11 * 1024:
                self.warnings.append("剩余显存不足 ~11GB，已自动关闭 DSpark")
                self._draft = None
            else:
                self._dspark = True
        # 显存预检：模型文件大小 × 1.1
        if self._model.is_file():
            need_mb = self._model.stat().st_size / 1024 / 1024 * 1.1
            if need_mb > free_vram_total_mb(self.caps):
                raise RequirementError(
                    f"剩余显存不足：模型约需 {need_mb:.0f}MB（×1.1），剩余 {free_vram_total_mb(self.caps)}MB")

    def _find_draft(self, cfg: dict) -> Path | None:
        if cfg.get("draft"):
            p = Path(cfg["draft"]).expanduser()
            return p if p.is_file() else None
        for base in (self._model.parent, *self._model.parents):
            if not base.exists():
                continue
            for name in ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf",
                         "dspark-DeepSeek-V4-Flash-0731-BF16.gguf"):
                if (base / name).is_file():
                    return base / name
            found = sorted(p for p in base.glob("*dspark*.gguf") if p.is_file())
            if found:
                return found[0]
        return None

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        source = Path(cfg.get("source_dir") or os.environ.get("LLAMACPP_SOURCE_DIR")
                      or PROJECT_ROOT.parent / "llama.cpp").expanduser().resolve()
        server = str(find_server(source))
        parallel = int(cfg.get("parallel", 2))
        ctx = int(cfg["ctx_size"]) if cfg.get("ctx_size") else parallel * CTX_PER_SLOT
        gpu_split = ",".join(["1"] * int(cfg.get("gpu_count", 8)))
        cmd = [server, "--model", str(self._model.resolve()),
               "--host", "0.0.0.0", "--port", str(self.profile.port),
               "--ctx-size", str(ctx), "--parallel", str(parallel),
               "--n-gpu-layers", "999", "--split-mode", "layer",
               "--tensor-split", gpu_split, "--jinja",
               "--reasoning", str(cfg.get("reasoning", "on")),
               "--reasoning-format", str(cfg.get("reasoning_format", "deepseek")),
               "--flash-attn", "on", "--metrics"]
        cmd += self.api_key_args()
        if cfg.get("repeat_penalty"):
            cmd += ["--repeat-penalty", str(cfg["repeat_penalty"])]
        if self._dspark and self._draft is not None:
            cmd += ["--model-draft", str(self._draft),
                    "--spec-type", str(cfg.get("spec_type", "draft-dspark")),
                    "--spec-draft-n-max", str(cfg.get("spec_draft_n_max", 3)),
                    "--n-gpu-layers-draft", str(cfg.get("n_gpu_layers_draft", 999))]
        cmd += ["--fit", str(cfg.get("fit", "off"))]
        if cfg.get("cache_type_k", "q8_0"):
            cmd += ["--cache-type-k", str(cfg.get("cache_type_k", "q8_0"))]
        if cfg.get("cache_type_v", "q8_0"):
            cmd += ["--cache-type-v", str(cfg.get("cache_type_v", "q8_0"))]
        env = {"MODELSCOPE_CACHE": os.environ["MODELSCOPE_CACHE"]} if os.environ.get("MODELSCOPE_CACHE") else {}
        return cmd, env

    def pre_start(self) -> None:
        cfg = self.profile.engine_config
        source = Path(cfg.get("source_dir") or os.environ.get("LLAMACPP_SOURCE_DIR")
                      or PROJECT_ROOT.parent / "llama.cpp").expanduser().resolve()
        if cfg.get("download") and not self._model.is_file():
            dl = cfg["download"]
            model_root = Path(os.environ.get("MODEL_ROOT") or PROJECT_ROOT.parent / "model-gguf")
            self._model, auto_draft = download_gguf(dl["modelscope_id"], model_root,
                                                    dl.get("quant", "UD-Q8_K_XL"), self._dspark)
            if self._draft is None:
                self._draft = auto_draft
        require("git"); require("cmake")
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", OFFICIAL_URL, str(source)])
        if not (source / "build" / "bin" / "llama-server").is_file():
            run(["cmake", "-S", str(source), "-B", str(source / "build"),
                 "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"])
            run(["cmake", "--build", str(source / "build"), "--config", "Release", "-j"])

    def metrics_mapping(self) -> dict[str, list[str]]:
        return {
            "prompt_total": ["llamacpp:prompt_tokens_total", "llamacpp:tokens_evaluated_total",
                             "llama_tokens_evaluated_total", "prompt_tokens_total"],
            "predicted_total": ["llamacpp:tokens_predicted_total", "llamacpp:predicted_tokens_total",
                                "llama_tokens_predicted_total", "tokens_predicted_total"],
            "prompt_rate": ["llamacpp:prompt_tokens_seconds", "prompt_tokens_seconds"],
            "predicted_rate": ["llamacpp:predicted_tokens_seconds",
                               "llamacpp:tokens_predicted_seconds", "predicted_tokens_seconds"],
        }

    def stop_patterns(self) -> list[str]:
        return ["llama-server"]
```

注意：测试用 fake 的 `find_server` 不存在二进制——`build_command` 中 `find_server` 会 SystemExit。为避免测试依赖真实编译产物，`find_server` 增加行为：找不到时返回 `source / "build" / "bin" / "llama-server"` 预期路径（仅当 `check_requirements` 已通过时 build_command 才被调用；pre_start 会真正编译）。测试因此无需真实二进制。

- [ ] **Step 5: 实现注册表 `script/engines/__init__.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/__init__.py — 引擎注册表。"""
from __future__ import annotations

from core.profile import ProfileError
from engines.base import EngineAdapter
from engines.llamacpp import LlamaCppAdapter

_REGISTRY: dict[str, type[EngineAdapter]] = {
    "llamacpp": LlamaCppAdapter,
    # ollama / vllm / sglang 在后续任务注册
}


def get_adapter(engine: str) -> type[EngineAdapter]:
    try:
        return _REGISTRY[engine]
    except KeyError:
        raise ProfileError(f"引擎未实现：{engine}（已实现：{sorted(_REGISTRY)}）") from None
```

- [ ] **Step 6: 运行确认通过**

Run: `python -m pytest tests/test_engines_llamacpp.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add script/engines/ tests/test_engines_llamacpp.py
git commit -m "feat(engines): 适配器基类与 llamacpp 适配器（迁移原启动脚本）"
```

---

### Task 6: ollama 适配器

**Files:**
- Create: `script/engines/ollama.py`
- Modify: `script/engines/__init__.py`（注册 ollama）
- Test: `tests/test_engines_ollama.py`

**Interfaces:**
- Consumes: `EngineAdapter`、`Capabilities`
- Produces: `class OllamaAdapter(EngineAdapter)`，额外公开方法 `unload_model() -> None`

行为：
- engine_config 键：`model`（必填，如 qwen3:32b）、`keep_alive`（默认 -1）、`num_parallel`（默认 2）、`context_length`（可选）
- `build_command`：`["ollama", "serve"]`；env：`OLLAMA_HOST=0.0.0.0:{port}`、`OLLAMA_MODELS`（os.environ 有则透传）、`OLLAMA_NUM_PARALLEL`、有值时 `OLLAMA_CONTEXT_LENGTH`
- `health_url`：`http://127.0.0.1:{port}/`（根路径返回 200）
- `check_requirements`：`caps.binaries.get("ollama")` False → RequirementError("ollama")；model 缺失 → RequirementError
- `pre_start`：`ollama list` 不含模型名 → `ollama pull <model>`
- `post_start`：`POST /api/generate` body `{"model": model, "keep_alive": keep_alive}` 预加载（urllib，timeout 600）
- `unload_model`：`POST /api/generate` body `{"model": model, "keep_alive": 0}`，异常静默
- `metrics_mapping`：None（无 Prometheus 指标）
- `stop_patterns`：`[]`（serve 为共享常驻服务，不由 patterns 杀；停止语义由 modelctl 对 ollama 引擎特判）

- [ ] **Step 1: 写失败测试 `tests/test_engines_ollama.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.capabilities import Capabilities  # noqa: E402
from core.profile import load_profile  # noqa: E402
from engines import get_adapter  # noqa: E402
from engines.base import RequirementError  # noqa: E402


def _profile(tmp_path):
    (tmp_path / "qwen3-ollama.yaml").write_text(
        "name: qwen3-ollama\nengine: ollama\nport: 11434\n"
        "ollama:\n  model: qwen3:32b\n  num_parallel: 2\n  context_length: 32768\n",
        encoding="utf-8")
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_engines_ollama.py -v`
Expected: FAIL（ProfileError: 引擎未实现：ollama）

- [ ] **Step 3: 实现 `script/engines/ollama.py` 并注册**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/ollama.py — ollama 适配器（serve 常驻 + 模型按需加载/卸载）。"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request

from engines.base import EngineAdapter, RequirementError


class OllamaAdapter(EngineAdapter):
    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        env = {"OLLAMA_HOST": f"0.0.0.0:{self.profile.port}"}
        if os.environ.get("OLLAMA_MODELS"):
            env["OLLAMA_MODELS"] = os.environ["OLLAMA_MODELS"]
        env["OLLAMA_NUM_PARALLEL"] = str(cfg.get("num_parallel", 2))
        if cfg.get("context_length"):
            env["OLLAMA_CONTEXT_LENGTH"] = str(cfg["context_length"])
        return ["ollama", "serve"], env

    def check_requirements(self) -> None:
        if not self.caps.binaries.get("ollama"):
            raise RequirementError("未安装 ollama（PATH 中找不到 ollama 命令）")
        if not self.profile.engine_config.get("model"):
            raise RequirementError(f"{self.profile.name}：ollama.model 必填（如 qwen3:32b）")

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.profile.port}/"

    def pre_start(self) -> None:
        model = str(self.profile.engine_config["model"])
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model.split(":")[0] not in out.stdout:
            subprocess.run(["ollama", "pull", model], check=True)

    def post_start(self) -> None:
        self._call_generate(self.profile.engine_config.get("keep_alive", -1))

    def unload_model(self) -> None:
        """stop 时卸载模型而非杀 serve（多模型共享服务）。"""
        try:
            self._call_generate(0)
        except OSError:
            pass

    def _call_generate(self, keep_alive) -> None:
        body = json.dumps({"model": self.profile.engine_config["model"],
                           "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.profile.port}/api/generate",
            data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()

    def metrics_mapping(self) -> None:
        return None
```

`engines/__init__.py`：import `OllamaAdapter` 并注册 `"ollama": OllamaAdapter`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_engines_ollama.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add script/engines/ollama.py script/engines/__init__.py tests/test_engines_ollama.py
git commit -m "feat(engines): ollama 适配器（serve 共享、模型预加载/卸载）"
```

---

### Task 7: vllm 与 sglang 适配器

**Files:**
- Create: `script/engines/vllm.py`、`script/engines/sglang.py`
- Modify: `script/engines/__init__.py`
- Test: `tests/test_engines_vllm.py`

**Interfaces:**
- Produces: `class VllmAdapter(EngineAdapter)`、`class SglangAdapter(EngineAdapter)`

**VllmAdapter**：
- engine_config 键：`model`（必填）、`tensor_parallel_size`（默认 1）、`max_model_len`（可选）、`gpu_memory_utilization`（默认 0.9）、`quantization`（可选）、`extra_args`（可选，shlex 拆分）
- `check_requirements`：无 vllm 二进制 → RequirementError；tp > caps.gpu_count（>0 时）→ RequirementError("GPU")；`quantization == "fp8"` 且非 `cc_at_least(cc, 8, 9)` → RequirementError("8.9")
- `build_command`：`["vllm", "serve", model, "--host", "0.0.0.0", "--port", port, "--tensor-parallel-size", tp, "--gpu-memory-utilization", gmu]` + 可选项 + `api_key_args()` + extra_args；env：`HF_HOME` 透传
- `metrics_mapping`：prompt_total=["vllm:prompt_tokens_total"]、predicted_total=["vllm:generation_tokens_total"]、两个 rate 为 []
- `stop_patterns`：`["vllm"]`

**SglangAdapter**：
- engine_config 键：`model`（必填）、`tensor_parallel_size`（默认 1）、`context_length`（可选）、`mem_fraction_static`（可选）、`extra_args`（可选）
- `check_requirements`：二进制（`sglang` 或 `python -m sglang --help` 可用性以 `caps.binaries["sglang"]` 为准）与 TP 校验
- `build_command`：`[sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--host", "0.0.0.0", "--port", port, "--tp", tp]` + 可选项（`--context-length`、`--mem-fraction-static`）+ extra_args；env：`HF_HOME` 透传
- `metrics_mapping`：prompt_total=["sglang:prompt_tokens_total"]、predicted_total=["sglang:generation_tokens_total"]、两个 rate 为 []
- `stop_patterns`：`["sglang"]`

- [ ] **Step 1: 写失败测试 `tests/test_engines_vllm.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.capabilities import Capabilities  # noqa: E402
from core.profile import load_profile  # noqa: E402
from engines import get_adapter  # noqa: E402
from engines.base import RequirementError  # noqa: E402

CAPS8 = Capabilities(gpu_count=8, compute_capability="8.9", binaries={"vllm": True, "sglang": True})


def _write(tmp_path, text, name="m.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return load_profile(name[:-5], tmp_path)


def test_vllm_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/raid5/sh/model/huggingface")
    p = _write(tmp_path, "name: q\nengine: vllm\nport: 8000\nvllm:\n"
               "  model: Qwen/Qwen3-32B\n  tensor_parallel_size: 2\n  max_model_len: 32768\n"
               "  extra_args: \"--enable-prefix-caching\"\n")
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
    a = get_adapter("vllm")(p, Capabilities(gpu_count=8, compute_capability="7.5",
                                            binaries={"vllm": True}))
    with pytest.raises(RequirementError, match="8.9"):
        a.check_requirements()


def test_vllm_tp_exceeds(tmp_path):
    p = _write(tmp_path, "name: q\nengine: vllm\nport: 8000\nvllm:\n  model: m\n  tensor_parallel_size: 16\n")
    a = get_adapter("vllm")(p, CAPS8)
    with pytest.raises(RequirementError, match="GPU"):
        a.check_requirements()


def test_sglang_command(tmp_path):
    p = _write(tmp_path, "name: s\nengine: sglang\nport: 30000\nsglang:\n"
               "  model: Qwen/Qwen3-32B\n  tensor_parallel_size: 4\n")
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_engines_vllm.py -v`
Expected: FAIL（ProfileError: 引擎未实现：vllm）

- [ ] **Step 3: 实现 `script/engines/vllm.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/vllm.py — vLLM 适配器。"""
from __future__ import annotations

import os
import shlex

from core.capabilities import cc_at_least
from engines.base import EngineAdapter, RequirementError


class VllmAdapter(EngineAdapter):
    def check_requirements(self) -> None:
        if not self.caps.binaries.get("vllm"):
            raise RequirementError("未安装 vllm（PATH 中找不到 vllm 命令）")
        cfg = self.profile.engine_config
        if not cfg.get("model"):
            raise RequirementError(f"{self.profile.name}：vllm.model 必填")
        tp = int(cfg.get("tensor_parallel_size", 1))
        if self.caps.gpu_count and tp > self.caps.gpu_count:
            raise RequirementError(f"tensor_parallel_size={tp} 超过实际 GPU 数 {self.caps.gpu_count}")
        if cfg.get("quantization") == "fp8" and not cc_at_least(self.caps.compute_capability, 8, 9):
            raise RequirementError(f"FP8 量化需要 CC ≥ 8.9，当前 {self.caps.compute_capability or '未知'}")

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        cmd = ["vllm", "serve", str(cfg["model"]),
               "--host", "0.0.0.0", "--port", str(self.profile.port),
               "--tensor-parallel-size", str(cfg.get("tensor_parallel_size", 1)),
               "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.9))]
        if cfg.get("max_model_len"):
            cmd += ["--max-model-len", str(cfg["max_model_len"])]
        if cfg.get("quantization"):
            cmd += ["--quantization", str(cfg["quantization"])]
        cmd += self.api_key_args()
        if cfg.get("extra_args"):
            cmd += shlex.split(str(cfg["extra_args"]))
        env = {"HF_HOME": os.environ["HF_HOME"]} if os.environ.get("HF_HOME") else {}
        return cmd, env

    def metrics_mapping(self) -> dict[str, list[str]]:
        return {
            "prompt_total": ["vllm:prompt_tokens_total"],
            "predicted_total": ["vllm:generation_tokens_total"],
            "prompt_rate": [],
            "predicted_rate": [],
        }

    def stop_patterns(self) -> list[str]:
        return ["vllm"]
```

- [ ] **Step 4: 实现 `script/engines/sglang.py`**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engines/sglang.py — SGLang 适配器。"""
from __future__ import annotations

import os
import shlex
import sys

from engines.base import EngineAdapter, RequirementError


class SglangAdapter(EngineAdapter):
    def check_requirements(self) -> None:
        if not self.caps.binaries.get("sglang"):
            raise RequirementError("未安装 sglang（PATH 中找不到 sglang 命令）")
        cfg = self.profile.engine_config
        if not cfg.get("model"):
            raise RequirementError(f"{self.profile.name}：sglang.model 必填")
        tp = int(cfg.get("tensor_parallel_size", 1))
        if self.caps.gpu_count and tp > self.caps.gpu_count:
            raise RequirementError(f"tensor_parallel_size={tp} 超过实际 GPU 数 {self.caps.gpu_count}")

    def build_command(self) -> tuple[list[str], dict[str, str]]:
        cfg = self.profile.engine_config
        cmd = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", str(cfg["model"]),
               "--host", "0.0.0.0", "--port", str(self.profile.port),
               "--tp", str(cfg.get("tensor_parallel_size", 1))]
        if cfg.get("context_length"):
            cmd += ["--context-length", str(cfg["context_length"])]
        if cfg.get("mem_fraction_static"):
            cmd += ["--mem-fraction-static", str(cfg["mem_fraction_static"])]
        if cfg.get("extra_args"):
            cmd += shlex.split(str(cfg["extra_args"]))
        env = {"HF_HOME": os.environ["HF_HOME"]} if os.environ.get("HF_HOME") else {}
        return cmd, env

    def metrics_mapping(self) -> dict[str, list[str]]:
        return {
            "prompt_total": ["sglang:prompt_tokens_total"],
            "predicted_total": ["sglang:generation_tokens_total"],
            "prompt_rate": [],
            "predicted_rate": [],
        }

    def stop_patterns(self) -> list[str]:
        return ["sglang"]
```

注册表 `_REGISTRY` 增加 `"vllm": VllmAdapter`、`"sglang": SglangAdapter`（含 import）。

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_engines_vllm.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add script/engines/vllm.py script/engines/sglang.py script/engines/__init__.py tests/test_engines_vllm.py
git commit -m "feat(engines): vllm 与 sglang 适配器"
```

---

### Task 8: core/stats.py — 用量统计多引擎适配

**Files:**
- Create: `script/core/stats.py`（迁移自 `script/usage_stats_server.py`）
- Test: `tests/test_stats.py`（迁移并扩展 `tests/test_usage_stats_server.py`）

**Interfaces:**
- Consumes: `EngineAdapter.metrics_mapping()` 的四组指标名
- Produces:
  - `parse_metrics(text: str, mapping: dict[str, list[str]]) -> dict[str, float]` — 从 Prometheus 文本按 mapping 各键的候选名取第一个命中（无命中为 0.0）；返回键：`prompt_total / predicted_total / prompt_rate / predicted_rate`
  - `build_usage_payload(tokens: dict[str, float], usage_cfg: dict, start_time: float, now: float) -> dict` — 按 usage_cfg 的 `price_in / price_out / budget` 折算，输出字段与现版 `/api/usage` 一致（cc-switch 无感）
  - `run_server(targets: list[StatsTarget]) -> None` — HTTP 服务（ThreadingHTTPServer），`/api/usage` 与 `/api/usage?model=<name>`
  - `@dataclass StatsTarget`: `name: str`、`metrics_url: str`、`mapping: dict[str, list[str]] | None`、`usage_cfg: dict`、`api_key: str | None`
  - 环境变量（沿用现版 .env 语义）：`USAGE_HOST`（默认 0.0.0.0）、`USAGE_PORT`（默认 5002）、`USAGE_MODE`（poll/on-demand）、`USAGE_POLL_INTERVAL`（默认 5）

行为：
- 保留现版两种模式（poll 后台线程 / on-demand 同步拉取）与累计口径（自服务启动以来）。
- `mapping` 为 None（ollama）→ `/api/usage?model=<name>` 返回 `{"error": "该引擎不支持精确统计"}`，HTTP 200。
- `?model=` 缺省：返回 targets 中第一个。
- 现版 HTTP handler、价格折算、预算逻辑原样迁移，仅把"单一 llama-server 数据源"改为 `targets` 列表 + 按 model 路由。
- `/api/usage` 输出 JSON 顶层字段与现版完全一致，额外加 `"model": <name>` 字段（新增字段不影响 cc-switch 解析）。

- [ ] **Step 1: 写失败测试 `tests/test_stats.py`**

```python
# -*- coding: utf-8 -*-
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

from core.stats import build_usage_payload, parse_metrics  # noqa: E402

LLAMACPP_MAPPING = {
    "prompt_total": ["llamacpp:prompt_tokens_total"],
    "predicted_total": ["llamacpp:tokens_predicted_total"],
    "prompt_rate": ["llamacpp:prompt_tokens_seconds"],
    "predicted_rate": ["llamacpp:predicted_tokens_seconds"],
}
VLLM_MAPPING = {
    "prompt_total": ["vllm:prompt_tokens_total"],
    "predicted_total": ["vllm:generation_tokens_total"],
    "prompt_rate": [],
    "predicted_rate": [],
}

METRICS_TEXT = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
llamacpp:prompt_tokens_total 1234
llamacpp:tokens_predicted_total 5678
llamacpp:prompt_tokens_seconds 100.5
llamacpp:predicted_tokens_seconds 55.0
"""


def test_parse_metrics_llamacpp():
    got = parse_metrics(METRICS_TEXT, LLAMACPP_MAPPING)
    assert got["prompt_total"] == 1234
    assert got["predicted_total"] == 5678
    assert got["prompt_rate"] == 100.5
    assert got["predicted_rate"] == 55.0


def test_parse_metrics_vllm_no_rate():
    got = parse_metrics("vllm:prompt_tokens_total 10\nvllm:generation_tokens_total 20\n", VLLM_MAPPING)
    assert got["prompt_total"] == 10
    assert got["predicted_total"] == 20
    assert got["prompt_rate"] == 0.0


def test_build_payload_with_budget():
    tokens = {"prompt_total": 1_000_000, "predicted_total": 500_000,
              "prompt_rate": 0.0, "predicted_rate": 0.0}
    payload = build_usage_payload(tokens, {"price_in": 1.0, "price_out": 2.0, "budget": 100},
                                  start_time=time.time() - 60, now=time.time())
    # 1M 输入 × 1元/M + 0.5M 输出 × 2元/M = 2 元
    assert payload["used_cost"] == 2.0
    assert payload["total"] == 100
    assert payload["remaining"] == 98.0


def test_build_payload_no_budget():
    tokens = {"prompt_total": 0, "predicted_rate": 0.0, "prompt_rate": 0.0,
              "predicted_total": 0}
    payload = build_usage_payload(tokens, {"price_in": 1.0, "price_out": 2.0},
                                  start_time=time.time(), now=time.time())
    assert "total" not in payload and "remaining" not in payload
```

注意：`build_usage_payload` 的金额字段名（`used_cost`/`total`/`remaining`）必须与现版 `usage_stats_server.py` 的 `/api/usage` 输出一致——实现前先读现版代码确认字段名，若现版不同则以现版为准并同步修改本测试。

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL（ModuleNotFoundError: core.stats）

- [ ] **Step 3: 实现 `script/core/stats.py`**

迁移步骤：
1. 读 `script/usage_stats_server.py` 全文，把指标解析逻辑抽为 `parse_metrics(text, mapping)`（现版是全局 METRIC_NAMES，改为参数传入）。
2. 把"折算 + 组装 /api/usage 响应"抽为 `build_usage_payload(tokens, usage_cfg, start_time, now)`。
3. HTTP handler 改为按 `?model=` 从 `targets` 选择数据源；`StatsTarget.mapping` 为 None 时返回不支持提示。
4. `run_server(targets)` 保留 poll/on-demand 两种模式与 USAGE_* 环境变量。
5. 保留原有 `test_usage_stats_server.py` 中仍适用的用例（迁移到本测试文件并改为调用新接口）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_stats.py -v`
Expected: 4 passed（含迁移用例则更多）

- [ ] **Step 5: Commit**

```bash
git add script/core/stats.py tests/test_stats.py
git commit -m "feat(core): 用量统计迁移至多引擎指标映射（/api/usage 兼容）"
```

---

### Task 9: modelctl.py CLI 与 modelctl.sh

**Files:**
- Create: `script/modelctl.py`、`script/modelctl.sh`
- Test: `tests/test_modelctl.py`

**Interfaces:**
- Consumes：前面所有任务的公开接口
- Produces：
  - `main(argv: list[str] | None = None) -> int`
  - 子命令：`start <name> [--timeout 300]` / `stop <name>` / `restart <name>` / `status [name]` / `list` / `probe`

CLI 行为：
- 所有子命令先 `load_env()`，再 `probe()` 探测能力（`status`/`list`/`probe` 也需要）。
- `start`：`load_profile` → `get_adapter` → `check_requirements`（打印所有 warnings）→ `pre_start` → `build_command` → `start_detached(name, cmd, env)` → `wait_health(adapter.health_url(), timeout, profile.api_key)`，超时打印 `tail_file(launch_log, 50)` 并返回 1 → 成功则 `post_start()` → 打印访问地址与日志路径 → 若 profile.usage 非空或 engine 有 mapping，提示 stats 由 `modelctl stats` 拉起（见下）。
- 额外子命令 `stats start|stop`：收集所有 is_running 的 profile → 构造 `StatsTarget`（metrics_url = `http://127.0.0.1:{port}/metrics`）→ `core.stats.run_server(targets)` 后台化（复用 `start_detached`，命令为 `[sys.executable, "-m", "core.stats"]` 形式独立进程入口，在 `core/stats.py` 加 `if __name__ == "__main__"` 或提供 `-m core.stats` 支持）；stop 复用 `stop_instance("usage-stats", USAGE_PORT, ["core.stats"])`。
- `stop <name>`：ollama 引擎 → 若 serve 由 modelctl 拉起（PID 文件存在）且无其他 ollama profile 在运行 → `stop_instance(name, port, [])`；否则仅 `adapter.unload_model()` + 删除该模型 PID 记录。其他引擎 → `stop_instance(name, port, adapter.stop_patterns())`。
- `status`：表格输出每个 profile 的 name/engine/port/状态（运行中/已停止/PID 异常）+ 健康检查结果。
- `probe`：打印 GPU 摘要、CC、各引擎二进制可用性。
- 错误处理：`ProfileError`/`RequirementError` 捕获后打印消息并返回 2。

`modelctl.sh`（bash 薄封装，POSIX）：

```bash
#!/usr/bin/env bash
# modelctl.sh — modelctl.py 的 bash 入口（后台 start/stop/restart 语义）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/modelctl.py" "$@"
```

- [ ] **Step 1: 写失败测试 `tests/test_modelctl.py`**

```python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))

import modelctl  # noqa: E402


def test_list_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    rc = modelctl.main(["list", "--models-dir", str(tmp_path)])
    assert rc == 0


def test_profile_error_exit_code(tmp_path, capsys):
    rc = modelctl.main(["start", "ghost", "--models-dir", str(tmp_path)])
    assert rc == 2
    assert "不存在" in capsys.readouterr().out


def test_status_output(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.yaml").write_text("name: a\nengine: vllm\nport: 8000\n", encoding="utf-8")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    rc = modelctl.main(["status", "--models-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "a" in out and "vllm" in out and "8000" in out
```

（CLI 需支持 `--models-dir` 全局参数便于测试；start 的完整流程测试在部署机上手工验证，本任务仅覆盖参数解析、错误路径与 status/list 输出。）

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_modelctl.py -v`
Expected: FAIL（ModuleNotFoundError: modelctl）

- [ ] **Step 3: 实现 `script/modelctl.py`**

按上述 CLI 行为实现。argparse 结构：

```python
parser = argparse.ArgumentParser(prog="modelctl", description="多模型部署启动器")
parser.add_argument("--models-dir", type=Path, default=None)
sub = parser.add_subparsers(dest="command", required=True)
for cmd in ("start", "stop", "restart", "status"):
    p = sub.add_parser(cmd); p.add_argument("name", nargs="?" if cmd == "status" else None)
sub.add_parser("list"); sub.add_parser("probe")
sp = sub.add_parser("stats"); sp.add_argument("action", choices=["start", "stop"])
start_parser 增加 --timeout（默认 300）
```

- [ ] **Step 4: 实现 `script/modelctl.sh`（内容见上）**

- [ ] **Step 5: 运行确认通过**

Run: `python -m pytest tests/test_modelctl.py -v`
Expected: 3 passed

- [ ] **Step 6: 全量回归**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

- [ ] **Step 7: Commit**

```bash
git add script/modelctl.py script/modelctl.sh tests/test_modelctl.py
git commit -m "feat(cli): modelctl 统一入口与 bash 薄封装"
```

---

### Task 10: profile 样例、.env.example、文档与旧文件清理

**Files:**
- Create: `models/deepseek-v4.yaml`、`models/qwen3-ollama.yaml`、`models/qwen3-vllm.yaml`
- Modify: `.env.example`、`README.md`
- Delete: `script/start_v4_flash_gguf.py`、`script/start_v4_flash_background.sh`、`script/usage_stats_server.py`、`tests/test_usage_stats_server.py`

- [ ] **Step 1: 创建 `models/deepseek-v4.yaml`**（值对齐现 .env.example）

```yaml
# DeepSeek-V4-Flash-0731（官方 llama.cpp + DSpark）
name: deepseek-v4
engine: llamacpp
port: 18888
api_key: ${API_KEY}

llamacpp:
  model: /raid5/sh/model/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
  draft: ""                # 留空自动发现 dspark*.gguf
  parallel: 2
  ctx_size: ""             # 留空 = parallel × 1M
  reasoning: on
  reasoning_format: deepseek
  dspark: on
  spec_type: draft-dspark
  spec_draft_n_max: 3
  n_gpu_layers_draft: 999
  cache_type_k: q8_0
  cache_type_v: q8_0
  gpu_count: 8
  fit: off
  download:
    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF
    quant: UD-Q8_K_XL

usage:
  price_in: 1.0
  price_out: 2.0
```

- [ ] **Step 2: 创建 `models/qwen3-ollama.yaml` 与 `models/qwen3-vllm.yaml`**

```yaml
# qwen3-ollama.yaml
name: qwen3-ollama
engine: ollama
port: 11434

ollama:
  model: qwen3:32b
  keep_alive: -1
  num_parallel: 2
  context_length: 32768
```

```yaml
# qwen3-vllm.yaml
name: qwen3-vllm
engine: vllm
port: 8000
api_key: ${API_KEY}

vllm:
  model: Qwen/Qwen3-32B
  tensor_parallel_size: 2
  max_model_len: 32768
  gpu_memory_utilization: 0.9
  extra_args: ""

usage:
  price_in: 0.5
  price_out: 1.0
```

- [ ] **Step 3: 重写 `.env.example`**

保留全局项并新增存储目录；删除已迁移进 profile 的模型级配置（MODEL/DRAFT/PARALLEL 等）：

```bash
# 全局配置（模型级配置见 models/*.yaml）
API_KEY=root123456

# 模型存储目录
MODEL_ROOT=/raid5/sh/model/model-gguf
MODELSCOPE_CACHE=/raid5/sh/model/modelscope
OLLAMA_MODELS=/raid5/sh/model/ollama-models
HF_HOME=/raid5/sh/model/huggingface

# llama.cpp 源码目录（llamacpp 引擎编译用）
LLAMACPP_SOURCE_DIR=/raid5/sh/code/llama.cpp

# 日志目录
LOG_DIR=/raid5/sh/logs

# 用量统计服务
USAGE_HOST=0.0.0.0
USAGE_PORT=5002
USAGE_MODE=poll
USAGE_POLL_INTERVAL=5
```

- [ ] **Step 4: 验证 profile 可加载**

Run: `python -c "import sys; sys.path.insert(0,'script'); from core.envfile import load_env; from core.profile import list_profiles; load_env(); [print(p.name, p.engine, p.port) for p in list_profiles()]"`
Expected: 打印三个 profile（需 .env 存在 API_KEY；无 .env 时 `API_KEY=xxx python -c ...` 注入）

- [ ] **Step 5: 删除旧文件并更新 README**

- `git rm script/start_v4_flash_gguf.py script/start_v4_flash_background.sh script/usage_stats_server.py tests/test_usage_stats_server.py`
- README 改写为 modelctl 用法：`python3 script/modelctl.py start deepseek-v4` / `bash script/modelctl.sh deepseek-v4 start` 不对——统一为 `bash script/modelctl.sh start deepseek-v4`（modelctl.sh 透传参数，子命令在前）；更新目录结构、快速开始、停止/重启说明。

- [ ] **Step 6: 全量回归 + Commit**

Run: `python -m pytest tests/ -v`
Expected: 全部通过

```bash
git add -A
git commit -m "feat: profile 样例、.env.example 重写、README 更新、移除旧启动脚本"
```

---

## 自审记录

- **Spec 覆盖**：引擎插件架构（T5-T7）、YAML profile（T2、T10）、能力探测与降级（T3、T5/T7 check_requirements）、进程管理（T4、T9）、用量统计适配（T8）、ollama 共享 serve 语义（T6、T9 stop 分支）、存储目录 .env（T1 全局约束、T10）、命名调整（T9、T10）、测试（每任务均含）。
- **类型一致性**：`EngineAdapter.__init__(profile, caps)`、`build_command() -> tuple[list[str], dict[str, str]]`、`metrics_mapping() -> dict[str, list[str]] | None`、`RequirementError`、`ProfileError`、`get_adapter(engine) -> type[EngineAdapter]`、`StatsTarget` 字段在各任务间一致。
- **已知留待实现期确认**：`build_usage_payload` 输出字段名以现版 usage_stats_server.py 为准（T8 Step 1 已注明）；sglang 指标名以部署版本实际 /metrics 为准（映射为单元素列表，易于修正）。
