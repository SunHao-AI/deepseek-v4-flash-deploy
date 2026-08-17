# modelctl 工程化改造设计

日期：2026-08-17
状态：待评审

## 背景与目标

当前 `modelctl` 多模型部署启动器功能完整、分层清晰（CLI → core → engines 适配器），
但存在工程化缺口：不可安装、目录布局非标准、日志用 `print()`、无静态检查/CI、
依赖治理不完善、异常策略不统一、存在平台死代码。

本次改造目标：将项目提升为**标准、可安装、可维护、可自动化验证**的 Python 工程。

## 决策记录

| 决策点 | 结论 |
|--------|------|
| 改造范围 | 完整工程化（包化 + 日志 + 质量工具 + CI + 依赖治理） |
| 目录布局 | 迁移到 `src/modelctl/` 标准布局 |
| 日志方案 | loguru |
| 平台分支 | 移除 process.py 的 win32 死代码（目标环境 Linux/CUDA） |
| CI 平台 | GitHub Actions |
| bash 封装 | `script/modelctl.sh` 改为调用已安装的 `modelctl` 命令 |

## 目标目录结构

```
deepseek-v4-flash-deploy/
├── src/modelctl/
│   ├── __init__.py          # 版本号、导出
│   ├── __main__.py          # python -m modelctl 入口
│   ├── cli.py               # 原 script/modelctl.py 的 main/子命令
│   ├── core/
│   │   ├── __init__.py
│   │   ├── envfile.py
│   │   ├── profile.py
│   │   ├── capabilities.py
│   │   ├── process.py
│   │   ├── stats.py
│   │   └── logging.py       # 新增：loguru 统一初始化
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llamacpp.py
│   │   ├── ollama.py
│   │   ├── vllm.py
│   │   └── sglang.py
│   └── py.typed             # 类型标记
├── models/                  # 保持不变
├── tests/                   # 改为 import modelctl（去掉 sys.path hack）
├── pyproject.toml           # 增加 [project.scripts]、可选依赖、ruff/mypy 配置
├── uv.lock                  # 依赖锁定
├── .github/workflows/ci.yml # CI
└── script/modelctl.sh       # 改为 exec modelctl "$@"
```

## 详细设计

### 1. 包布局与打包

- 采用 `src/` 布局，包名 `modelctl`。
- `pyproject.toml` 增加：
  - `[project.scripts] modelctl = "modelctl.cli:main"`
  - `[project.optional-dependencies]`：`modelscope`（llamacpp 下载用）、`dev`（pytest、ruff、mypy、loguru）
  - `[tool.uv]` 保留，生成 `uv.lock`
- 新增 `src/modelctl/__main__.py`，支持 `python -m modelctl`。
- `script/modelctl.sh` 改为 `exec modelctl "$@"`，保持现有调用习惯。

### 2. 日志（loguru）

- 新增 `core/logging.py`：`setup_logging()` 初始化控制台 + 文件（`LOG_DIR`）handler，
  分级、带时间戳、支持 `LOG_LEVEL` 环境变量。
- 将 `print()` 替换为 `logger.info/warning/error`。
- CLI 表格类输出（status/list）保留 `print` 以维持对齐格式。

### 3. 代码质量工具

- `pyproject.toml` 配置 `[tool.ruff]`（lint + format）、`[tool.mypy]`。
- 修复现有类型与 lint 问题。

### 4. CI（GitHub Actions）

- `.github/workflows/ci.yml`：`uv sync` → `ruff check` → `mypy` → `pytest`。

### 5. 错误处理与清理

- `engines/llamacpp.py` 的 `run/require/download_gguf` 中 `SystemExit` 改为抛
  `RequirementError`/`ProfileError`，由 CLI 统一捕获返回退出码。
- 常量去重：`USAGE_PORT` 收敛到单一来源（`core/stats.py` 定义，CLI 引用）。
- 移除 `core/process.py` 的 win32 分支（`sys.platform == "win32"` 相关代码）。

### 6. 测试

- 去掉 `tests/*.py` 的 `sys.path.insert` hack，改为安装后 `import modelctl`。
- 保持现有测试逻辑，补充 CLI 入口与日志初始化测试。

## 非目标（YAGNI）

- 不引入 Web 框架、不新增引擎。
- 不做跨平台（Windows）支持。
- 不重构 stats.py 内部结构（本次仅做包化与清理）。

## 验证方式

- `uv sync` 成功，`uv lock` 生成。
- `ruff check .`、`mypy src`、`pytest` 全部通过。
- `pip install -e .` 后 `modelctl list` 可运行。
- `python -m modelctl list` 可运行。
