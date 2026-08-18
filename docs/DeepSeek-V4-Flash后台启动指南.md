# DeepSeek-V4-Flash 部署与运维指南

在远程服务器（8× RTX 5880 Ada）上部署 DeepSeek-V4-Flash-0731（官方 llama.cpp + DSpark）的完整说明。工程化改造后，统一使用 `modelctl` CLI 管理生命周期，配置分层为：

- **全局配置**：项目根 `.env`（API 密钥、模型存储目录、日志目录、llama.cpp 源码目录、用量统计服务）
- **模型级配置**：`models/<engine>/<name>.yaml` 或兼容旧式 `models/<name>.yaml`（模型路径、端口、并行度、量化、DSpark 参数、下载配置、用量单价）

配置优先级：**profile YAML > 环境变量 > .env 文件 > 代码默认值**。

## 前置条件

- Python 3.12+，已安装项目依赖：`uv sync --extra dev`
- `git`、`cmake`、CUDA 工具链、`nvidia-smi`
- 官方 llama.cpp 源码目录（用于编译 `llama-server`），默认由 `.env` 的 `LLAMACPP_SOURCE_DIR` 指定
- 如尚未下载模型，首次启动时会根据 `models/deepseek-v4.yaml` 的 `download` 段自动从 ModelScope 拉取

模型文件预期布局：

```
${MODEL_ROOT}/DeepSeek-V4-Flash-0731-GGUF/
├── UD-Q8_K_XL/
│   └── DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf  (+00002~00005)
└── dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
```

> 默认使用 **UD-Q8_K_XL 无损量化**（162GB，与官方权重 bit-identical）。若需近无损的 Q4（155GB），修改 `models/deepseek-v4.yaml` 中的 `model` 路径为 `UD-Q4_K_XL/` 对应分片即可。

## 配置管理

### 1. 复制并编辑 `.env`

```bash
cp .env.example .env
vi .env
```

与 DeepSeek-V4-Flash 启动直接相关的全局变量：

| 变量 | 默认值示例 | 说明 |
| --- | --- | --- |
| `API_KEY` | `root123456` | API 密钥（供 profile 的 `${API_KEY}` 插值；留空则不校验） |
| `MODEL_ROOT` | `/raid5/sh/model/model-gguf` | GGUF 模型根目录（llamacpp 下载段保存父目录） |
| `MODELSCOPE_CACHE` | `/raid5/sh/model/modelscope` | ModelScope 下载缓存目录 |
| `LLAMACPP_SOURCE_DIR` | `/raid5/sh/code/llama.cpp` | llama.cpp 源码目录（编译用） |
| `LOG_DIR` | `/raid5/sh/logs` | 启动日志与服务运行日志目录 |
| `USAGE_HOST` | `0.0.0.0` | 用量统计服务监听地址 |
| `USAGE_PORT` | `5002` | 用量统计服务监听端口 |

### 1.5 models 目录布局

profile 支持两种存放方式（按引擎分目录为推荐方式，旧的根目录方式仍兼容）：

```
models/
├── deepseek-v4.yaml            # llamacpp + DSpark（根目录，兼容旧式）
├── qwen3-llama.yaml            # llamacpp（根目录）
├── llamacpp/                   # llamacpp 引擎 profile 子目录
│   └── qwen3-llamacpp.yaml
├── ollama/                     # ollama 引擎 profile 子目录（预留）
└── vllm/                       # vllm 引擎 profile 子目录（预留）
```

同一 `name` 同时存在于根目录与子目录时，以根目录为准（`modelctl list` 会打印忽略警告）。

### 2. 按需修改 `models/deepseek-v4.yaml`

```yaml
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

> **model 字段可留空自动下载**：若 `model` 为空或指向的 GGUF 不存在，且配置了 `download` 段，
> 首次启动会自动从 ModelScope 下载指定量化分片，并把本地绝对路径**持久化写回** YAML 的 `model`
> 字段（原文件备份为 `.yaml.bak`）。下次启动直接复用本地文件，不再触发下载。
> 下载目录由 `.env` 的 `MODEL_ROOT` 控制。

## 启动 / 停止 / 重启 / 状态

启动前请确认 `.env` 中的 `MODEL_ROOT`、`MODELSCOPE_CACHE`、`LLAMACPP_SOURCE_DIR`、`LOG_DIR` 已按实际环境设置。

```bash
# 启动（首次会自动编译 llama.cpp 并下载模型到 MODEL_ROOT / MODELSCOPE_CACHE 指定位置）
bash script/modelctl.sh start deepseek-v4

# 停止
bash script/modelctl.sh stop deepseek-v4

# 重启
bash script/modelctl.sh restart deepseek-v4

# 查看状态
bash script/modelctl.sh status

# 列出所有 profile
bash script/modelctl.sh list

# 探测硬件与引擎二进制可用性
bash script/modelctl.sh probe
```

也可直接调用已安装的 `modelctl` 命令：

```bash
uv run modelctl start deepseek-v4
```

## 验证服务

模型加载需要 1-2 分钟，之后健康检查：

```bash
curl http://127.0.0.1:18888/health
```

预期返回 `{"status":"ok"}`。再做一次推理测试（注意带 API key 头）：

```bash
curl http://127.0.0.1:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer root123456" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好，用一句话自我介绍"}],"max_tokens":100}'
```

## 用量统计服务

```bash
# 启动用量统计服务（/api/usage，cc-switch 兼容）
bash script/modelctl.sh stats start

# 停止用量统计服务
bash script/modelctl.sh stats stop
```

## 查看日志

日志目录由 `.env` 的 `LOG_DIR` 决定。

```bash
# 启动过程日志
tail -f ${LOG_DIR}/launch-deepseek-v4-*.log

# 服务运行日志（llama-server 输出）
tail -f ${LOG_DIR}/llama-server-18888-*.log
```

## 重启 / 换量化

1. 停止服务：`bash script/modelctl.sh stop deepseek-v4`
2. 修改 `models/deepseek-v4.yaml` 中的 `model` 路径（例如换 `UD-Q4_K_XL/`）
3. 重新启动：`bash script/modelctl.sh start deepseek-v4`

> 若首次启动时 `model` 留空、由 `download` 段自动下载并写回了路径，再次换量化时需同时修改
> `model` 与 `download.quant`（或先删除已写回的 `model` 路径让其重新下载）。

如需调整 `extra_args` 等额外参数，目前 llamacpp 引擎尚未支持该字段，请直接修改 `build_command()` 输出或提交 issue。

## 参数速查（models/deepseek-v4.yaml）

| YAML 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `UD-Q8_K_XL/...-00001-of-00005.gguf` | GGUF 模型第一个分片；留空 + `download` 段时自动下载并写回本地路径 |
| `draft` | 空（自动发现） | DSpark 草稿路径 |
| `port` | `18888` | 服务端口 |
| `ctx_size` | 空（自动） | 总上下文长度。留空自动计算：`parallel × 1048576`（每并发 1M），可手动覆盖 |
| `parallel` | `2` | 并发序列数 |
| `gpu_count` | `8` | GPU 数量 |
| `dspark` | `on` | DSpark 投机解码开关 |
| `reasoning` | `on` | 思考模式 |
| `cache_type_k` / `cache_type_v` | `q8_0` | KV cache 量化 |
| `download.modelscope_id` | `unsloth/DeepSeek-V4-Flash-0731-GGUF` | ModelScope 模型 ID |
| `download.quant` | `UD-Q8_K_XL` | 下载的量化版本 |

## 已知注意事项

1. **`--spec-draft-n-max` 会被钳制到 5**（checkpoint 的 `dspark_block_size`），默认 3 是实测最优
2. **不要传 `--spec-draft-device`**：草稿模型借用主模型的 embedding/输出头，必须跨同一批 GPU
3. **`--no-mmap` 已从默认命令移除**：fork 时代遗留参数，官方版 mmap 加载更快
4. **OpenSSL 未找到**：HTTPS 禁用，本机 HTTP 使用无影响

## 可选优化

- **安装 NCCL**：多卡 layer split 跨卡通信依赖 NCCL，当前编译警告 `NCCL not found`。安装后需重新编译 llama.cpp 才能生效：
  ```bash
  apt-get install -y libnccl-dev
  # 重新配置 + 编译
  cmake -S ${LLAMACPP_SOURCE_DIR} -B ${LLAMACPP_SOURCE_DIR}/build \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build ${LLAMACPP_SOURCE_DIR}/build --config Release -j
  ```
- **换 Q4 近无损量化**：如果显存吃紧或想加快加载，把 `models/deepseek-v4.yaml` 的 `model` 改为 `UD-Q4_K_XL/` 路径，分片约小 7GB

## Unsloth 引擎（实验性）

基于 Unsloth 无头 API 服务（`unsloth studio --api-only`）部署 Unsloth 动态量化 GGUF 模型。

### 前置条件

- 在目标服务器安装 Unsloth：`curl -fsSL https://unsloth.ai/install.sh | sh`（或独立 venv 安装，避免重依赖污染项目环境）
- `.env` 配置 `UNSLOTH_API_KEY`（必填，健康检查依赖）、可选 `HF_ENDPOINT`（HF 兜底镜像）、复用 `MODEL_ROOT`/`MODELSCOPE_CACHE`
- 启动前用 `unsloth --help` 核实无头服务 flag（`--api-only`、`--model`、`-p` 等），与本工具内置常量不一致时需调整 `engines/unsloth.py`

### 使用

```bash
bash script/modelctl.sh start deepseek-v4-unsloth   # 首次自动从 ModelScope 下载并写回 profile
curl http://127.0.0.1:8001/v1/models -H "Authorization: Bearer $UNSLOTH_API_KEY"
bash script/modelctl.sh status
```

### 已知限制

- 用量统计暂不支持精确统计（`/metrics` 端点未验证，`modelctl stats` 对该模型返回"不支持精确统计"）
- 健康检查使用 `/v1/models`（需认证），非 `/health`
