# 多模型部署启动器改造设计

日期：2026-08-17
状态：已确认（用户逐节评审通过）

## 1. 背景与目标

现有项目仅支持通过官方 llama.cpp 启动 DeepSeek-V4-Flash 单一模型（`start_v4_flash_gguf.py` + `start_v4_flash_background.sh`）。改造目标：**支持多种模型、多种推理引擎的统一部署启动器**，作为自定义模型调用时使用。

### 已确认的关键决策

| 决策点 | 结论 |
|---|---|
| 核心形态 | 多模型启动器（无统一网关，调用方直连各模型端口） |
| 支持引擎 | llama.cpp (GGUF)、Ollama、vLLM、SGLang；引擎可扩展 |
| profile 定义 | 每模型一个 YAML（`models/<name>.yaml`） |
| 用量统计 | 统一适配所有引擎，对 cc-switch 输出格式不变 |
| Python 版本 | 最低 3.12，允许引入 PyYAML |
| 模型存储 | 由 `.env` 配置，缺省用默认路径 |
| 硬件适配 | 启动前探测能力，不支持的功能自动降级 + warning |

## 2. 整体架构与目录结构

```
deepseek-v4-flash-deploy/
├── models/                        # 模型 profile（每模型一个 YAML）
│   ├── deepseek-v4.yaml
│   ├── qwen3-ollama.yaml
│   └── qwen3-vllm.yaml
├── script/
│   ├── modelctl.py                # 统一 CLI 入口
│   ├── modelctl.sh                # bash 薄封装（后台 start/stop/restart）
│   ├── core/
│   │   ├── profile.py             # YAML 加载、校验、${VAR} 插值、默认值合并
│   │   ├── process.py             # 后台启动、PID/端口管理、日志落盘、健康检查
│   │   ├── capabilities.py        # 硬件/环境能力探测
│   │   └── stats.py               # 用量统计（现 usage_stats_server.py 迁移）
│   └── engines/
│       ├── base.py                # EngineAdapter 抽象基类
│       ├── llamacpp.py            # 现 start_v4_flash_gguf.py 逻辑迁移
│       ├── ollama.py
│       ├── vllm.py
│       └── sglang.py
├── tests/
├── .env.example                   # 全局默认值（存储目录、统计端口等）
└── docs/
```

### 命名调整

| 现有 | 改为 |
|---|---|
| `start_v4_flash_background.sh` | `script/modelctl.sh`（`bash script/modelctl.sh <model> start\|stop\|restart`） |
| `start_v4_flash_gguf.py` | 拆解迁移到 `script/engines/llamacpp.py`，不再作为独立入口 |
| `usage_stats_server.py` | 迁移为 `script/core/stats.py`，由 modelctl 统一拉起 |

所有模型相关命名（v4_flash 等）从代码中消除，模型差异只体现在 `models/*.yaml`。

### 职责划分

- **modelctl.py**：解析命令、加载 profile、分发到引擎适配器；不含引擎细节
- **engines/base.py**：统一接口 `build_command()` / `start()` / `stop()` / `health_check()` / `metrics_mapping()`；新增引擎 = 新增一个实现文件
- **core/process.py**：引擎无关的进程生命周期管理，替代现 bash 脚本中的 fuser/pkill 逻辑
- **llamacpp 特有能力**（自动编译、DSpark、ModelScope 下载）保留在 llamacpp 适配器内，作为该引擎 profile 的特有配置项，不污染通用接口

## 3. Profile YAML 格式

### 通用字段

- `name`（必填）：模型实例名，用于 CLI、PID 文件、日志命名
- `engine`（必填）：`llamacpp | ollama | vllm | sglang`
- `port`（必填）：服务端口
- `api_key`（可选）：支持 `${VAR}` 从 `.env`/环境变量插值，缺失时报错
- `usage`（可选）：`price_in` / `price_out` / `budget`，用于用量统计
- 引擎同名段（`llamacpp:` / `ollama:` / ...）：引擎特有配置，由对应适配器校验并给默认值

### 示例：models/deepseek-v4.yaml（llamacpp）

```yaml
name: deepseek-v4
engine: llamacpp
port: 18888
api_key: ${API_KEY}

llamacpp:
  model: /raid5/sh/model/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
  draft: ""                  # 留空自动发现 dspark*.gguf
  parallel: 2
  ctx_size: ""               # 留空 = parallel × 1M
  reasoning: on
  dspark: on
  cache_type_k: q8_0
  cache_type_v: q8_0
  gpu_count: 8
  source_dir: ""             # 留空用 .env 的 LLAMACPP_SOURCE_DIR
  download:
    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF
    quant: UD-Q8_K_XL
```

### 示例：models/qwen3-ollama.yaml（ollama）

```yaml
name: qwen3-ollama
engine: ollama
port: 11434

ollama:
  model: qwen3:32b           # 不存在时自动 ollama pull
  keep_alive: -1             # 启动后预加载并常驻显存
  num_parallel: 2
  context_length: 32768
```

### 示例：models/qwen3-vllm.yaml（vllm）

```yaml
name: qwen3-vllm
engine: vllm
port: 8000
api_key: ${API_KEY}

vllm:
  model: Qwen/Qwen3-32B      # HF/ModelScope 模型 ID 或本地路径
  tensor_parallel_size: 2
  max_model_len: 32768
  gpu_memory_utilization: 0.9
  extra_args: ""             # 透传其他 vllm serve 参数
```

## 4. 模型存储目录（.env 全局配置）

部署服务器目录布局 `/raid5/sh/model/`，`.env` 默认值：

```bash
MODEL_ROOT=/raid5/sh/model/model-gguf        # llama.cpp GGUF 下载父目录
MODELSCOPE_CACHE=/raid5/sh/model/modelscope  # ModelScope 缓存
OLLAMA_MODELS=/raid5/sh/model/ollama-models  # ollama 模型目录（确认用此名）
HF_HOME=/raid5/sh/model/huggingface          # vllm / sglang HF 缓存
```

引擎适配器启动子进程时注入对应环境变量；profile 中的相对模型路径基于这些根目录解析。

## 5. 硬件能力探测（core/capabilities.py）

目标服务器：8× RTX 5880 Ada（CC 8.9，384GB 总显存）。部分功能未必支持，**先探测再决定开关**。

### 探测内容

启动时执行一次，结果缓存到 `logs/capabilities.json`：

- `nvidia-smi`：GPU 型号、数量、单卡显存、CUDA 驱动版本
- 计算能力（CC）→ 影响 FP8、FlashAttention 后端选择
- 引擎二进制可用性：`ollama` / `vllm` / `sglang` 是否已安装、llama.cpp 是否已编译

### 功能门槛与降级策略

| 特性 | 门槛 | 不满足时 |
|---|---|---|
| llamacpp DSpark | 草稿模型存在 + 剩余显存 ≥ ~11GB | 自动关闭 DSpark，warning |
| llamacpp `--parallel N>1` | 试启后健康检查 | 失败则降级 parallel=1 重试一次 |
| llamacpp KV cache q8_0 | 已知官方 b10298+ 正常 | 保留配置，启动失败回退 f16 |
| vllm FP8 量化 | CC ≥ 8.9 | profile 配了 fp8 但 CC 不足 → 拒绝启动并提示 |
| vllm/sglang tensor_parallel_size | ≤ 实际 GPU 数 | 拒绝启动并提示 |
| ollama num_parallel | 无硬门槛 | 仅记录 |

### 显存预检

启动前读 `nvidia-smi` 剩余显存，与 profile 预估需求比对（llamacpp 按模型文件大小估算，vllm 按 `gpu_memory_utilization`），不足则拒绝启动。

## 6. 进程管理（core/process.py）

- 每个模型实例：`nohup` 后台化 + PID 文件（`logs/<name>.pid`）+ 启动日志（`logs/launch-<name>-<时间戳>.log`）
- `stop`：PID 文件优雅终止 → 超时按端口 kill → 兜底按进程特征 pkill
- `status` / `list`：PID 文件 + 端口探测 + `/health` 检查，输出每个 profile 的 运行/停止/异常 状态
- **ollama 特例**：`ollama serve` 为常驻服务、多模型共享：
  - 服务未运行则启动并记录"由 modelctl 拉起"
  - `stop` 时若该服务还承载其他 ollama profile 的模型，只卸载模型（`ollama stop <model>`），不杀服务

## 7. 用量统计适配（core/stats.py）

- 按引擎映射 Prometheus 指标名：
  - llamacpp：`llamacpp:prompt_tokens_total` / `llamacpp:tokens_predicted_total`
  - vllm：`vllm:prompt_tokens_total` / `vllm:generation_tokens_total`
  - sglang：`/metrics`（指标名以实际版本为准）
  - ollama：无 Prometheus 指标 → 用 `/api/ps` 轮询估算，或标记"不支持精确统计"
- 对外 `/api/usage` 输出格式不变，cc-switch 无感
- 多模型并存时按 `?model=<name>` 区分，缺省返回第一个运行中的

## 8. 错误处理

- profile 校验失败（缺字段、端口冲突、YAML 语法错、插值变量缺失）→ 启动前报错，不产生任何进程
- 引擎启动后健康检查超时（默认 300s，可配）→ 输出最后 50 行日志并退出非零
- 所有自动降级行为写入启动日志，`modelctl status` 可见

## 9. 测试（tests/）

- profile 加载/校验/插值：纯单元测试
- 能力探测：mock `nvidia-smi` 输出，覆盖 8× 5880 Ada 场景
- 引擎命令构建：mock 子进程，断言生成的命令行参数
- 用量统计：沿用现有 `test_usage_stats_server.py` 思路，mock 各引擎 `/metrics` 响应

## 10. 兼容性说明

- 现有 `.env` 中的 llama.cpp/DeepSeek 配置迁移到 `models/deepseek-v4.yaml` + 全局 `.env`（存储目录、统计服务）
- 现有启动/停止习惯通过 `modelctl.sh` 保持：`start/stop/restart` 语义不变，仅多了 `<model>` 参数
