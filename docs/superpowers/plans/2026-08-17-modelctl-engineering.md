# modelctl 工程化改造 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 modelctl 从 `script/` 目录脚本改造为标准、可安装、可维护、可自动化验证的 Python 工程（src 布局 + console script + loguru + ruff/mypy + CI）。

**Architecture:** 采用 `src/modelctl/` 标准布局，包内保持现有分层（cli → core → engines 适配器）。通过 `[project.scripts]` 提供 `modelctl` 命令，`__main__.py` 支持 `python -m modelctl`。日志统一用 loguru，静态检查用 ruff/mypy，CI 用 GitHub Actions。

**Tech Stack:** Python 3.12、uv、PyYAML、loguru、pytest、ruff、mypy、GitHub Actions。

## Global Constraints

- Python 版本下限：`>=3.12`（来自 spec 与现有 pyproject）。
- 目标平台：Linux/CUDA；**不**做 Windows 支持。
- 包名：`modelctl`；采用 `src/` 布局。
- 依赖：`PyYAML>=6.0`（必需）、`loguru`（必需）、`modelscope`（可选，llamacpp 下载用）、`pytest>=8.0`/`ruff`/`mypy`（dev）。
- 配置优先级保持不变：profile YAML > 环境变量 > .env > 代码默认值。
- 现有 CLI 子命令与行为保持不变：start/stop/restart/status/list/probe/stats。
- 所有 `print()` 替换为 loguru `logger`，但 status/list 的表格对齐输出保留 `print`。
- 移除 `core/process.py` 中所有 `sys.platform == "win32"` 分支。
- `engines/llamacpp.py` 的 `SystemExit` 改为抛 `RequirementError`/`ProfileError`。
- `USAGE_PORT` 默认值收敛到单一来源 `core/stats.py`。

---

### Task 1: 建立 src 布局并迁移 core 模块

**Files:**
- Create: `src/modelctl/__init__.py`
- Create: `src/modelctl/core/__init__.py`
- Create: `src/modelctl/engines/__init__.py`
- Create: `src/modelctl/py.typed`
- Move: `script/core/*.py` → `src/modelctl/core/*.py`
- Move: `script/engines/*.py` → `src/modelctl/engines/*.py`
- Delete: `script/core/`、`script/engines/`

**Interfaces:**
- Consumes: 现有 `script/core/`、`script/engines/` 全部模块。
- Produces: 包 `modelctl`，内部模块路径从 `core.xxx` 变为 `modelctl.core.xxx`、`engines.xxx` 变为 `modelctl.engines.xxx`。所有内部 import 需同步改为绝对导入 `from modelctl.core.xxx import ...`。

- [ ] **Step 1: 创建目录与包标记文件**

创建 `src/modelctl/__init__.py`：

```python
"""modelctl — 多模型部署启动器。"""

__version__ = "0.3.0"
```

创建 `src/modelctl/py.typed`（空文件，标记类型）。

创建 `src/modelctl/core/__init__.py` 与 `src/modelctl/engines/__init__.py`（空文件）。

- [ ] **Step 2: 迁移 core 模块并改写 import**

将 `script/core/envfile.py`、`profile.py`、`capabilities.py`、`process.py`、`stats.py` 移动到 `src/modelctl/core/`。

将每个文件内的相对导入改为绝对导入：
- `from core.envfile import PROJECT_ROOT` → `from modelctl.core.envfile import PROJECT_ROOT`
- `from core.capabilities import ...` → `from modelctl.core.capabilities import ...`
- `from core.profile import ...` → `from modelctl.core.profile import ...`
- `from core.envfile import load_env` → `from modelctl.core.envfile import load_env`

- [ ] **Step 3: 迁移 engines 模块并改写 import**

将 `script/engines/base.py`、`llamacpp.py`、`ollama.py`、`vllm.py`、`sglang.py` 移动到 `src/modelctl/engines/`。

改写 import：
- `from core.capabilities import ...` → `from modelctl.core.capabilities import ...`
- `from core.envfile import PROJECT_ROOT` → `from modelctl.core.envfile import PROJECT_ROOT`
- `from core.profile import Profile` → `from modelctl.core.profile import Profile`
- `from engines.base import ...` → `from modelctl.engines.base import ...`

- [ ] **Step 4: 删除旧目录**

删除 `script/core/` 与 `script/engines/` 目录。

- [ ] **Step 5: 验证导入**

运行：`python -c "import modelctl.core.profile, modelctl.engines.vllm; print('ok')"`
Expected: 输出 `ok`（需在项目根，且 `src` 在 sys.path 上；若未安装，用 `PYTHONPATH=src`）。

- [ ] **Step 6: Commit**

```bash
git add src/ script/
git rm -r script/core script/engines
git commit -m "refactor: 迁移 core/engines 到 src/modelctl 标准布局"
```

---

### Task 2: 迁移 CLI 入口为 cli.py 并新增 __main__

**Files:**
- Create: `src/modelctl/cli.py`
- Create: `src/modelctl/__main__.py`
- Delete: `script/modelctl.py`

**Interfaces:**
- Consumes: `modelctl.core.*`、`modelctl.engines.*`（Task 1 产物）。
- Produces: `modelctl.cli.main(argv: list[str] | None = None) -> int`；`modelctl.__main__` 调用 `main()`。

- [ ] **Step 1: 迁移 modelctl.py 为 cli.py**

将 `script/modelctl.py` 内容复制到 `src/modelctl/cli.py`，改写 import：
- `from core.capabilities import probe` → `from modelctl.core.capabilities import probe`
- `from core.envfile import load_env` → `from modelctl.core.envfile import load_env`
- `from core.process import ...` → `from modelctl.core.process import ...`
- `from core.profile import ...` → `from modelctl.core.profile import ...`
- `from engines import get_adapter` → `from modelctl.engines import get_adapter`
- `from engines.base import RequirementError` → `from modelctl.engines.base import RequirementError`

保留 `main(argv)` 签名与全部子命令逻辑。

- [ ] **Step 2: 新增 __main__.py**

创建 `src/modelctl/__main__.py`：

```python
"""python -m modelctl 入口。"""
from modelctl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: 删除旧入口**

删除 `script/modelctl.py`。

- [ ] **Step 4: 验证**

运行：`PYTHONPATH=src python -m modelctl list --models-dir models`
Expected: 输出 profile 表格（deepseek-v4 / qwen3-ollama / qwen3-vllm）。

- [ ] **Step 5: Commit**

```bash
git add src/modelctl/cli.py src/modelctl/__main__.py
git rm script/modelctl.py
git commit -m "feat: CLI 迁移为 modelctl.cli，新增 python -m modelctl 入口"
```

---

### Task 3: 更新 pyproject.toml（打包 + 可选依赖 + 工具配置）

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Task 1/2 的包结构。
- Produces: 可安装包 `modelctl`，提供 `modelctl` 命令；`uv.lock`。

- [ ] **Step 1: 重写 pyproject.toml**

```toml
[project]
name = "modelctl"
version = "0.3.0"
description = "多模型部署启动器（llamacpp / ollama / vllm / sglang）"
requires-python = ">=3.12"
dependencies = [
    "PyYAML>=6.0",
    "loguru>=0.7",
]

[project.optional-dependencies]
modelscope = ["modelscope>=1.0"]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
modelctl = "modelctl.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/modelctl"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.12"
packages = ["modelctl"]
```

- [ ] **Step 2: 生成 lock 并安装**

运行：`uv lock`
运行：`uv sync --extra dev`
Expected: 生成 `uv.lock`，安装成功。

- [ ] **Step 3: 验证命令可用**

运行：`uv run modelctl list --models-dir models`
Expected: 输出 profile 表格。

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: 打包为可安装包，新增 console script 与工具配置"
```

---

### Task 4: 新增 loguru 日志模块并替换 print

**Files:**
- Create: `src/modelctl/core/logging.py`
- Modify: `src/modelctl/cli.py`
- Modify: `src/modelctl/engines/llamacpp.py`

**Interfaces:**
- Consumes: `modelctl.core.envfile.PROJECT_ROOT`。
- Produces: `modelctl.core.logging.setup_logging() -> None`；`logger` 实例。

- [ ] **Step 1: 编写日志模块**

创建 `src/modelctl/core/logging.py`：

```python
"""loguru 统一日志初始化。"""
from __future__ import annotations

import os
import sys

from loguru import logger

from modelctl.core.envfile import PROJECT_ROOT


def setup_logging() -> None:
    """配置控制台与文件日志（LOG_DIR，默认项目根上级 logs/）。"""
    logger.remove()
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    log_dir = Path(os.environ.get("LOG_DIR") or PROJECT_ROOT.parent / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "modelctl.log", level=level, rotation="10 MB", retention="7 days", encoding="utf-8")
```

（注：需在文件顶部 `from pathlib import Path`。）

- [ ] **Step 2: 在 cli.py 接入日志**

在 `cli.py` 顶部 import 并初始化：

```python
from modelctl.core.logging import setup_logging
from loguru import logger
```

在 `main()` 开头调用 `setup_logging()`。

将 `cli.py` 中的 `print(f"警告: {warning}")` 改为 `logger.warning(warning)`；
`print(f"错误: {error}")` 改为 `logger.error(str(error))`；
`print(f"已启动 {name}（PID {pid}）...")` 改为 `logger.info(...)`；
`print(f"启动成功：{name} 运行于 ...")` 改为 `logger.info(...)`；
`print(f"日志：{log}")` 改为 `logger.info(f"日志：{log}")`；
`print(f"提示：用量统计可通过 ...")` 改为 `logger.info(...)`；
`print(f"健康检查超时，日志尾部 50 行（{log}）：")` 改为 `logger.warning(...)`；
`print(tail_file(log, 50))` 改为 `logger.warning(tail_file(log, 50))`；
`print("健康检查超时，且未找到启动日志")` 改为 `logger.warning(...)`；
`print(f"已停止：{profile.name}")` 改为 `logger.info(...)`；
`print(f"已停止：{profile.name}，正在重新启动...")` 改为 `logger.info(...)`；
`print(f"未找到 profile：{args.name}")` 改为 `logger.warning(...)`。

**保留** status/list/probe 的表格 `print` 输出（对齐格式）。

- [ ] **Step 3: 替换 llamacpp.py 的 print**

将 `llamacpp.py` 中 `run()` 的 `print("\n$ " + ...)` 改为 `logger.info(...)`；
`download_gguf` 中的 `print(...)` 改为 `logger.info(...)`；
`print(f"找到 DSpark 草稿：{draft_match}")` 改为 `logger.info(...)`；
警告类 `print(...)` 改为 `logger.warning(...)`。
在文件顶部 `from loguru import logger`。

- [ ] **Step 4: 验证**

运行：`uv run modelctl list --models-dir models`
Expected: 正常输出表格，无报错。

- [ ] **Step 5: Commit**

```bash
git add src/modelctl/core/logging.py src/modelctl/cli.py src/modelctl/engines/llamacpp.py
git commit -m "feat: 引入 loguru 统一日志，替换 print"
```

---

### Task 5: 统一异常策略（llamacpp SystemExit → 异常）

**Files:**
- Modify: `src/modelctl/engines/llamacpp.py`

**Interfaces:**
- Consumes: `modelctl.engines.base.RequirementError`。
- Produces: `run()`/`require()`/`download_gguf()` 抛 `RequirementError` 而非 `SystemExit`。

- [ ] **Step 1: 改写 run/require**

将：

```python
def require(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"缺少 {name}。请安装后再运行脚本。")
```

改为：

```python
def require(name: str) -> None:
    if shutil.which(name) is None:
        raise RequirementError(f"缺少 {name}。请安装后再运行脚本。")
```

`run()` 保持 `subprocess.run(..., check=True)`（失败抛 `CalledProcessError`，由 CLI 统一捕获）。

- [ ] **Step 2: 改写 download_gguf 的 SystemExit**

将 `download_gguf` 中两处 `raise SystemExit(...)` 改为 `raise RequirementError(...)`。

- [ ] **Step 3: 验证**

运行：`uv run pytest tests/ -q`
Expected: 现有测试通过（llamacpp 相关测试若存在则通过）。

- [ ] **Step 4: Commit**

```bash
git add src/modelctl/engines/llamacpp.py
git commit -m "fix: llamacpp 异常统一为 RequirementError"
```

---

### Task 6: 常量去重 + 移除 win32 死代码

**Files:**
- Modify: `src/modelctl/core/stats.py`
- Modify: `src/modelctl/cli.py`
- Modify: `src/modelctl/core/process.py`

**Interfaces:**
- Consumes: `modelctl.core.stats`。
- Produces: `modelctl.core.stats.USAGE_PORT`（默认 5002）；`cli.py` 引用它。

- [ ] **Step 1: 在 stats.py 定义 USAGE_PORT**

在 `src/modelctl/core/stats.py` 顶部新增：

```python
USAGE_PORT = 5002
```

将 `run_server` 中 `port = int(os.environ.get("USAGE_PORT", "5002"))` 改为 `port = int(os.environ.get("USAGE_PORT", str(USAGE_PORT)))`。

- [ ] **Step 2: cli.py 引用 USAGE_PORT**

删除 `cli.py` 中的 `DEFAULT_USAGE_PORT = 5002`，改为：

```python
from modelctl.core.stats import USAGE_PORT
```

将 `cli.py` 中 `os.environ.get("USAGE_PORT", str(DEFAULT_USAGE_PORT))` 改为 `os.environ.get("USAGE_PORT", str(USAGE_PORT))`。

- [ ] **Step 3: 移除 process.py 的 win32 分支**

在 `src/modelctl/core/process.py` 中：
- `start_detached`：删除 `if sys.platform != "win32": kwargs["start_new_session"] = True`，直接设置 `kwargs["start_new_session"] = True`。
- `is_running`：删除 win32 分支，仅保留 `os.kill(pid, 0)` 逻辑。
- `stop_instance`：删除 win32 分支与 `if sys.platform != "win32":` 包裹，直接执行 `os.killpg`/`fuser`/`pkill` 逻辑。
- 删除不再使用的 `import sys`（若仅用于平台判断）。

- [ ] **Step 4: 验证**

运行：`uv run pytest tests/ -q`
Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add src/modelctl/core/stats.py src/modelctl/cli.py src/modelctl/core/process.py
git commit -m "refactor: USAGE_PORT 收敛单一来源，移除 win32 死代码"
```

---

### Task 7: 更新测试（去掉 sys.path hack）

**Files:**
- Modify: `tests/test_modelctl.py`
- Modify: `tests/test_stats.py`
- Modify: `tests/test_capabilities.py`
- Modify: `tests/test_envfile.py`
- Modify: `tests/test_process.py`
- Modify: `tests/test_profile.py`
- Modify: `tests/test_engines_llamacpp.py`
- Modify: `tests/test_engines_ollama.py`
- Modify: `tests/test_engines_vllm.py`

**Interfaces:**
- Consumes: 已安装的 `modelctl` 包。
- Produces: 通过 `import modelctl` 的测试。

- [ ] **Step 1: 移除 sys.path hack**

在每个测试文件顶部删除：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
```

将 `import modelctl` 改为 `from modelctl import cli`（或按需 `from modelctl.core.stats import ...`）。

- [ ] **Step 2: 修正 test_modelctl.py 引用**

将 `import modelctl` 改为 `from modelctl import cli`，并把 `modelctl.main(...)` 改为 `cli.main(...)`。

- [ ] **Step 3: 修正 test_stats.py 引用**

将 `from core.stats import build_usage_payload, parse_metrics` 改为 `from modelctl.core.stats import build_usage_payload, parse_metrics`。

- [ ] **Step 4: 运行全部测试**

运行：`uv run pytest tests/ -q`
Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: 测试改为 import modelctl 包，移除 sys.path hack"
```

---

### Task 8: 配置 ruff/mypy 并修复问题

**Files:**
- Modify: `src/modelctl/**/*.py`（按需）
- Modify: `tests/**/*.py`（按需）

**Interfaces:**
- Consumes: Task 3 的 `[tool.ruff]`/`[tool.mypy]` 配置。
- Produces: 通过 `ruff check` 与 `mypy` 的代码。

- [ ] **Step 1: 运行 ruff 并修复**

运行：`uv run ruff check src tests`
Expected: 无错误。若有错误，逐项修复（未使用 import、未定义变量、行过长等）。

- [ ] **Step 2: 运行 ruff format**

运行：`uv run ruff format src tests`
Expected: 格式化完成。

- [ ] **Step 3: 运行 mypy 并修复**

运行：`uv run mypy src/modelctl`
Expected: 无错误。若有类型错误，补充类型标注或修正。

- [ ] **Step 4: 回归测试**

运行：`uv run pytest tests/ -q`
Expected: 全部通过。

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "chore: 通过 ruff/mypy 静态检查"
```

---

### Task 9: 新增 CI（GitHub Actions）

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Task 3 的 pyproject 配置。
- Produces: CI 流水线。

- [ ] **Step 1: 编写 CI 配置**

创建 `.github/workflows/ci.yml`：

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --extra dev
      - run: uv run ruff check src tests
      - run: uv run mypy src/modelctl
      - run: uv run pytest tests/ -q
```

- [ ] **Step 2: 验证 YAML 语法**

运行：`uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: 无异常。

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: 新增 GitHub Actions 流水线"
```

---

### Task 10: 更新 modelctl.sh 与文档

**Files:**
- Modify: `script/modelctl.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: 已安装的 `modelctl` 命令。
- Produces: 调用 `modelctl` 命令的 bash 封装。

- [ ] **Step 1: 改写 modelctl.sh**

将 `script/modelctl.sh` 改为：

```bash
#!/usr/bin/env bash
# modelctl.sh — 调用已安装的 modelctl 命令
set -euo pipefail
exec modelctl "$@"
```

- [ ] **Step 2: 更新 README 安装说明**

在 `README.md` 的「快速开始」前新增安装步骤：

```markdown
## 安装

```bash
uv sync --extra dev
uv run modelctl list
```
```

并更新「目录结构」章节，将 `script/` 相关描述改为 `src/modelctl/`。

- [ ] **Step 3: 验证**

运行：`uv run modelctl list --models-dir models`
Expected: 正常输出。

- [ ] **Step 4: Commit**

```bash
git add script/modelctl.sh README.md
git commit -m "docs: 更新安装说明与目录结构，modelctl.sh 调用已安装命令"
```

---

## Self-Review

- **Spec 覆盖**：包布局（Task 1/2/3）、loguru（Task 4）、质量工具（Task 8）、CI（Task 9）、异常统一（Task 5）、常量去重与 win32 移除（Task 6）、测试（Task 7）、bash 封装与文档（Task 10）——全部覆盖。
- **占位符扫描**：无 TBD/TODO，所有步骤含具体代码。
- **类型一致性**：`modelctl.cli.main`、`modelctl.core.stats.USAGE_PORT`、`modelctl.core.logging.setup_logging` 在各任务间签名一致。
