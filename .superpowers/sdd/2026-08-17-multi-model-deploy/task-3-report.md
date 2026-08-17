# Task 3 Report: core/capabilities.py — 硬件能力探测

## 完成内容

- 创建 `tests/test_capabilities.py`，包含计划中的 4 个测试用例。
- 创建 `script/core/capabilities.py`，实现：
  - `Capabilities` 数据类
  - `which_binaries`
  - `probe`（支持 `nvidia_smi_output` 参数 mock）
  - `cc_at_least`
  - `free_vram_total_mb`
- 注释均使用中文。

## 执行过程

1. 编写测试文件后运行 `python -m pytest tests/test_capabilities.py -v`，确认失败：`ModuleNotFoundError: core.capabilities`。
2. 实现 `script/core/capabilities.py` 后再次运行，4 个测试全部通过。
3. 全量回归测试 `python -m pytest tests/ -v`：共 34 个测试通过（含 Task 1、Task 2 与既有 usage_stats_server 测试）。
4. 提交 git：`7992a18 feat(core): 硬件能力探测（GPU/CC/显存/引擎二进制）`。

## 关键实现说明

- `probe(nvidia_smi_output=None)` 仅在无参数时调用 `_safe_smi()`，该辅助函数捕获 `OSError` 与 `subprocess.SubprocessError`，调用失败/异常均返回空字符串，最终生成 `gpu_count=0` 的兜底 `Capabilities` 对象。
- nvidia-smi 查询命令严格使用计划指定格式：
  `--query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits`
- `which_binaries` 使用标准库 `shutil.which`，无需外部依赖。
- `cc_at_least` 解析 `major.minor` 为元组后做字典序比较；非法输入返回 `False`。
- 所有代码仅使用 Python 标准库，符合 Windows 可测试约束。
