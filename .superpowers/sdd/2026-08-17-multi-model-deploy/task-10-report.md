# Task 10 报告 — profile 样例、.env.example、文档与旧文件清理

## 状态
- **完成**（commit `7472cf2`，review pending）

## 完成内容

### Step 1-2：创建 profile 样例
- `models/deepseek-v4.yaml` — DeepSeek-V4-Flash-0731（llamacpp + DSpark），含 `download` 段与 `usage` 单价
- `models/qwen3-ollama.yaml` — Qwen3-32B（ollama），`keep_alive: -1`、`num_parallel: 2`、`context_length: 32768`
- `models/qwen3-vllm.yaml` — Qwen3-32B（vllm），`tensor_parallel_size: 2`、`max_model_len: 32768`、`usage` 单价

三个 YAML 均严格按计划 Step 1-2 内容编写。

### Step 3：重写 .env.example
- 删除已迁移进 profile 的模型级配置（MODEL/DRAFT/PARALLEL/PORT/REASONING/DSPARK/CACHE_TYPE 等）
- 保留全局配置：`API_KEY`
- 新增存储目录：`MODEL_ROOT` / `MODELSCOPE_CACHE` / `OLLAMA_MODELS` / `HF_HOME`
- 保留 `LLAMACPP_SOURCE_DIR`、`LOG_DIR`
- 用量统计：`USAGE_HOST` / `USAGE_PORT` / `USAGE_MODE` / `USAGE_POLL_INTERVAL`
- 删除已废弃的 `USAGE_LLAMA_BASE` / `USAGE_PRICE_IN` / `USAGE_PRICE_OUT` / `USAGE_BUDGET` / `LLAMA_API_KEY`（单价/预算已移入 profile 的 `usage` 段）

### Step 4：验证 profile 可加载
- 无 `.env` 时注入 `API_KEY` 后运行计划命令，成功打印三个 profile：
  ```
  deepseek-v4 llamacpp 18888
  qwen3-ollama ollama 11434
  qwen3-vllm vllm 8000
  ```

### Step 5：删除旧文件 + 更新 README
- `git rm` 删除：`script/start_v4_flash_gguf.py`、`script/start_v4_flash_background.sh`、`script/usage_stats_server.py`、`tests/test_usage_stats_server.py`
- `README.md` 改写为 modelctl 用法：`bash script/modelctl.sh start <model>`，更新目录结构（含 `models/`、`script/core/`、`script/engines/`）、快速开始、停止/重启/状态/用量统计说明

### Step 6：全量回归
- `python -m pytest tests/ -v` → **40 passed**（test_envfile / test_profile / test_capabilities / test_process / test_engines_llamacpp / test_engines_ollama / test_engines_vllm / test_stats / test_modelctl）
- 已删除的 `test_usage_stats_server.py` 不再收集

### Step 7：提交
- `git add -A` + commit `7472cf2`（13 files changed, 559 insertions, 1318 deletions）
- 同时提交了 Task 9 fix round 遗留的 SDD 文件（progress.md、task-9-report.md、task-9-fix1.diff、task-9-review-package.diff）

## 关键说明
- 未修改 Task 1-9 已完成的 `script/core`、`script/engines`、`script/modelctl.py` 等文件
- 代码注释均为中文
- 无 .env 时 profile 加载需注入 `API_KEY`（deepseek-v4 / qwen3-vllm 的 `api_key: ${API_KEY}` 插值依赖），符合计划预期
