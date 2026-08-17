# Task 4 完成报告：core/process.py — 进程生命周期管理

## 完成内容

- 创建 `script/core/process.py`，实现以下接口：
  - `log_dir()`：读取 `LOG_DIR` 环境变量，缺省为 `PROJECT_ROOT.parent / "logs"`，自动创建目录。
  - `pid_file(name)`：返回 `log_dir() / f"{name}.pid"`。
  - `launch_log(name)`：返回最新的 `launch-{name}-*.log` 文件路径（按文件名排序）。
  - `start_detached(name, command, extra_env)`：使用 `subprocess.Popen` 后台启动，POSIX 设置 `start_new_session=True`，输出重定向到 `launch-{name}-{时间戳}.log`，写入 PID 文件，返回 PID。
  - `is_running(name)`：Windows 使用 `tasklist /FI "PID eq {pid}"`；POSIX 使用 `os.kill(pid, 0)`。
  - `stop_instance(name, port, patterns)`：Windows 使用 `taskkill /PID {pid} /T /F`；POSIX 先 `SIGTERM` 进程组，10s 超时后 `SIGKILL`，再用 `fuser -k` 与 `pkill -f` 兜底。
  - `wait_health(url, timeout, api_key)`：每 2s 轮询 GET，5s 读超时，可选 `Authorization: Bearer {api_key}`，2xx 返回 True。
  - `tail_file(path, lines)`：读取文件末尾指定行数。
- 创建 `tests/test_process.py`，包含计划中 5 个测试用例。

## 关键实现点

- Windows 兼容：`is_running` 使用 `tasklist`，`stop_instance` 使用 `taskkill`；`fuser`/`pkill` 仅在 POSIX 执行。
- `start_detached` 在 POSIX 设置 `start_new_session=True`，实现 nohup 语义，避免 SSH 断开后子进程被 SIGHUP 终止。
- `stop_instance` 先读 PID 文件终止；POSIX 若 10s 后进程仍存活则强制 `SIGKILL`。
- `wait_health` 使用标准库 `urllib.request`，5s 超时避免单次请求挂死。

## 测试执行记录

1. 先运行测试确认失败：
   ```
   ImportError: cannot import name 'process' from 'core'
   ```
2. 实现后运行 `tests/test_process.py`：
   ```
   5 passed in 2.29s
   ```
3. 全量回归 `tests/`：
   ```
   39 passed in 5.28s
   ```

## Git 提交

```
feat(core): 进程生命周期管理（后台启动/PID/停止/健康检查）
```

包含文件：
- `script/core/process.py`
- `tests/test_process.py`
