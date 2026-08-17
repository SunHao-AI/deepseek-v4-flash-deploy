# Task 1 报告：项目脚手架与 core/envfile.py

## 状态

DONE

## 修改文件列表

- `pyproject.toml`
  - name: `modelctl`
  - version: `0.2.0`
  - description: `多模型部署启动器（llamacpp / ollama / vllm / sglang）`
  - requires-python: `>=3.12`
  - dependencies: `["PyYAML>=6.0"]`
  - optional-dependencies.dev: `["pytest>=8.0"]`
  - 新增 `[tool.pytest.ini_options]`，`testpaths = ["tests"]`
- 新建 `script/core/__init__.py`（空，包标记）
- 新建 `script/engines/__init__.py`（空，包标记；本任务先空，后续任务注册引擎）
- 新建 `script/core/envfile.py`
  - 提供 `PROJECT_ROOT = Path(__file__).resolve().parents[2]`
  - 提供 `parse_env_file(path: Path) -> dict[str, str]`
  - 提供 `load_env(env_path: Path | None = None) -> Path`
- 新建 `tests/test_envfile.py`
  - `test_parse_env_file_basic`
  - `test_parse_env_file_missing`
  - `test_load_env_no_override`

## 测试命令与输出摘要

```powershell
python -m pytest tests/test_envfile.py -v
```

输出摘要（3 passed）：

```
tests/test_envfile.py::test_parse_env_file_basic PASSED
tests/test_envfile.py::test_parse_env_file_missing PASSED
tests/test_envfile.py::test_load_env_no_override PASSED

============================== 3 passed in 0.09s ===============================
```

注：首次运行因 `core.envfile` 模块不存在而失败（`ModuleNotFoundError: No module named 'core'`）；实现后再次运行全部通过。

## 提交哈希

`f76501a`

提交信息：`feat(core): 项目脚手架升级 Python 3.12 + PyYAML，新增 envfile 模块`

## 后续任务注意事项

1. `script/usage_stats_server.py` 与 `tests/test_usage_stats_server.py` 在工作区被 Git 标记为已修改，但 `git diff` 无实际内容差异，推测为行尾符（CRLF/LF）规范化预警；本次未纳入 Task 1 提交范围，Task 8 迁移/删除旧文件时请留意。
2. `script/engines/__init__.py` 当前为空，Task 5 开始需在此注册 `LlamaCppAdapter`、`OllamaAdapter`、`VllmAdapter`、`SglangAdapter`。
3. `.env` 文件目前不存在于仓库，`load_env()` 默认返回 `PROJECT_ROOT / ".env"` 路径；后续 Task 10 提供 `.env.example` 时可按需复制。
