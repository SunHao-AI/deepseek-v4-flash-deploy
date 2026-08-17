# Task 9 报告：modelctl.py CLI 与 modelctl.sh

## 状态
完成。

## 交付物
- 新建 `script/modelctl.py`（CLI 入口，`main(argv) -> int`）
- 新建 `script/modelctl.sh`（bash 薄封装）
- 新建 `tests/test_modelctl.py`（3 个 pytest 用例）

## 实现要点
- **全局参数 `--models-dir`**：通过 `_extract_models_dir` 在任意位置提取（含 `--models-dir=...` 形式），便于测试；`main` 中先 `load_env()` 再 `probe()`。
- **start <name> [--timeout 300]**：`load_profile → get_adapter → check_requirements`（打印所有 warnings）→ `pre_start` → `build_command` → `start_detached` → `wait_health`（超时打印 `tail_file(launch_log, 50)` 并返回 1）→ 成功则 `post_start` → 打印访问地址与日志路径 → 若 `profile.usage` 非空或 engine 有 mapping，提示 `modelctl stats start`。
- **stop <name>**：ollama 引擎特判——serve 由本工具拉起（PID 文件存在）且无其他 ollama profile 在运行时 `stop_instance(name, port, [])`；否则 `adapter.unload_model()` + 删除 PID 记录。其他引擎 `stop_instance(name, port, adapter.stop_patterns())`。
- **restart <name>**：先 stop 再 start。
- **status [name]**：表格输出 name/engine/port/状态（运行中/已停止/PID 异常）+ 健康检查（仅对运行中实例做 `wait_health`，失败不阻塞表格）。
- **list**：列出所有 profile。
- **probe**：打印 GPU 数量/型号/显存/CC/驱动 + 各引擎二进制可用性。
- **stats start|stop**：start 收集所有 `is_running` 的 profile 构造 `StatsTarget`，后台化运行 `[sys.executable, "-m", "core.stats"]`（复用 `start_detached`，注入 `PYTHONPATH`）；stop 用 `stop_instance("usage-stats", USAGE_PORT, ["core.stats"])`。
- **错误处理**：`ProfileError`/`RequirementError` 捕获后打印消息并返回 2。

## 测试
- `tests/test_modelctl.py` 3 用例：`test_list_empty`、`test_profile_error_exit_code`（返回 2 且输出含"不存在"）、`test_status_output`（表格含 name/engine/port）。
- 全量回归：`python -m pytest tests/ -v` → **60 passed**。
- 说明：start 完整流程（真实启动引擎）不在测试范围；status/list/probe 在 `probe()` 失败（Windows 无 nvidia-smi，gpu_count=0）时仍正常输出，不崩溃。

## 提交
- commit `7f44dd6`：`feat(cli): modelctl 统一入口与 bash 薄封装`
- 提交内容含本任务三个新文件，以及此前已暂存（未提交）的 `.superpowers/sdd` 台账与 review 包文件。

## 备注
- 未修改 Task 1-8 已完成的文件。
- `stats start` 后台进程通过 `python -m core.stats` 独立入口从 `models/*.yaml` 构造统计目标（core/stats.py 已提供 `_targets_from_profiles`），与计划"收集 is_running profile 构造 StatsTarget"的意图一致。
