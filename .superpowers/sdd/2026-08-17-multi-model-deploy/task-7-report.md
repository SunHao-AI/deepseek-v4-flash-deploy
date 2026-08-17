# Task 7 报告：vllm 与 sglang 适配器

## 完成内容

按计划文件 `docs/superpowers/plans/2026-08-17-multi-model-deploy.md` 的 Task 7 执行完毕。

### 新增文件
- `script/engines/vllm.py` — `VllmAdapter`
  - `check_requirements`：vllm 二进制缺失 → RequirementError；`model` 必填；`tensor_parallel_size` 超过实际 GPU 数 → RequirementError("GPU")；`quantization == "fp8"` 且 CC < 8.9 → RequirementError("8.9")
  - `build_command`：`vllm serve <model> --host 0.0.0.0 --port <port> --tensor-parallel-size <tp> --gpu-memory-utilization <gmu>` + 可选 `--max-model-len`/`--quantization` + `api_key_args()` + `extra_args`（shlex 拆分）；env 透传 `HF_HOME`
  - `metrics_mapping`：prompt_total=`["vllm:prompt_tokens_total"]`、predicted_total=`["vllm:generation_tokens_total"]`、两个 rate 为空
  - `stop_patterns`：`["vllm"]`
- `script/engines/sglang.py` — `SglangAdapter`
  - `check_requirements`：sglang 二进制缺失 → RequirementError；`model` 必填；TP 校验
  - `build_command`：`[sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--host", "0.0.0.0", "--port", port, "--tp", tp]` + 可选 `--context-length`/`--mem-fraction-static` + `extra_args`；env 透传 `HF_HOME`
  - `metrics_mapping`：prompt_total=`["sglang:prompt_tokens_total"]`、predicted_total=`["sglang:generation_tokens_total"]`、两个 rate 为空
  - `stop_patterns`：`["sglang"]`
- `tests/test_engines_vllm.py` — 计划中的 5 个测试用例（vllm 命令构建、FP8 CC 校验、TP 超限、sglang 命令构建、vllm 指标映射）

### 修改文件
- `script/engines/__init__.py` — 注册 `"vllm": VllmAdapter`、`"sglang": SglangAdapter`

## 测试结果
- 先运行失败测试确认失败（`ProfileError: 引擎未实现：vllm/sglang`）
- 实现后 `tests/test_engines_vllm.py`：5 passed
- 全量回归 `python -m pytest tests/ -v`：**53 passed**

## 提交
- commit `9d52985` — `feat(engines): vllm 与 sglang 适配器`
- 仅提交本任务文件（vllm.py、sglang.py、__init__.py、test_engines_vllm.py）；`script/usage_stats_server.py` 与 `tests/test_usage_stats_server.py` 的改动属 Task 8 范围，未纳入本次提交。

## 备注
- 测试使用 `Capabilities(gpu_count=8, compute_capability="8.9", binaries={"vllm": True, "sglang": True})` 构造场景，未调用真实 vllm/sglang。
- 代码注释为中文。
- 未修改 Task 1-6 已完成的文件（`engines/__init__.py` 除外，本任务按计划注册 vllm/sglang）。
