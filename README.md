# modelctl — 多模型部署启动器

在 CUDA/Ada GPU 上通过**引擎插件式架构**启动多种模型服务（llamacpp / ollama / vllm / sglang），每模型一个 YAML profile，统一 CLI 管理启动、停止、重启、状态与用量统计。

## 特性

- **多引擎支持**：llamacpp（官方 llama.cpp + DSpark 投机解码）、ollama、vllm、sglang
- **YAML profile**：每模型一个 `models/<name>.yaml`，配置模型路径、端口、引擎参数、用量单价
- **能力探测与自动降级**：启动前探测 GPU/CC/显存/引擎二进制，硬性不满足拒绝启动并说明原因，可降级项自动降级并告警
- **统一生命周期**：后台启动、PID 管理、健康检查、优雅停止
- **用量统计**：`/api/usage` 输出与 cc-switch 兼容，支持多模型按 `?model=` 路由
- **配置外置**：全局配置通过 `.env` 管理，模型级配置通过 profile YAML 管理

## 目录结构

```
deepseek-v4-flash/
├── README.md                       # 本文档（入口）
├── docs/
│   └── DeepSeek-V4-Flash后台启动指南.md   # 部署与运维详细指南
├── src/modelctl/
│   ├── cli.py                      # 统一 CLI 入口（start/stop/restart/status/list/probe/stats）
│   ├── __main__.py                 # python -m modelctl 入口
│   ├── core/                       # 核心模块：envfile / profile / capabilities / process / stats
│   ├── engines/                    # 引擎适配器：base / llamacpp / ollama / vllm / sglang
│   └── py.typed                    # PEP 561 类型标记
├── script/
│   └── modelctl.sh                 # bash 薄封装（调用已安装的 modelctl 命令）
├── models/                         # 模型 profile（每模型一个 YAML）
│   ├── deepseek-v4.yaml            # DeepSeek-V4-Flash（llamacpp + DSpark）
│   ├── qwen3-ollama.yaml           # Qwen3-32B（ollama）
│   └── qwen3-vllm.yaml             # Qwen3-32B（vllm）
├── .env.example                    # 全局配置模板（复制为 .env 后修改）
├── .env                            # 本地配置（含密钥，不入库）
├── .gitignore
└── pyproject.toml
```

## 安装

```bash
uv sync --extra dev
uv run modelctl list
```

## 快速开始

### 1. 安装依赖

- Python 3.12+、PyYAML
- `git`、`cmake`、CUDA 工具链、`nvidia-smi`（llamacpp 引擎编译用）
- 各引擎二进制：`ollama` / `vllm` / `sglang`（按需安装）

### 2. 配置 .env 与 profile

```bash
cp .env.example .env
vi .env        # 修改 API 密钥、存储目录、日志目录等全局配置
```

模型级配置（模型路径、端口、并行度、量化等）在 `models/*.yaml` 中修改。

配置优先级：**profile YAML > 环境变量 > .env 文件 > 代码默认值**。

### 3. 启动服务

```bash
# 启动 DeepSeek-V4-Flash（llamacpp，首次运行会自动编译 llama.cpp 并下载模型）
bash script/modelctl.sh start deepseek-v4

# 启动 Qwen3（ollama）
bash script/modelctl.sh start qwen3-ollama

# 启动 Qwen3（vllm）
bash script/modelctl.sh start qwen3-vllm
```

也可直接调用已安装的 `modelctl` 命令：

```bash
uv run modelctl start deepseek-v4
```

### 4. 验证

```bash
curl http://127.0.0.1:18888/health   # deepseek-v4
curl http://127.0.0.1:11434/         # qwen3-ollama
curl http://127.0.0.1:8000/health    # qwen3-vllm
```

### 5. 停止 / 重启 / 状态

```bash
# 停止
bash script/modelctl.sh stop deepseek-v4

# 重启（先停后启）
bash script/modelctl.sh restart deepseek-v4

# 查看所有模型状态（含健康检查）
bash script/modelctl.sh status

# 列出所有 profile
bash script/modelctl.sh list

# 探测硬件与引擎二进制可用性
bash script/modelctl.sh probe
```

### 6. 用量统计服务

```bash
# 启动用量统计服务（/api/usage，cc-switch 兼容）
bash script/modelctl.sh stats start

# 停止用量统计服务
bash script/modelctl.sh stats stop
```

查看日志（LOG_DIR 默认 = 项目根目录上级的 `../logs/`）：

```bash
tail -f ../logs/launch-deepseek-v4-*.log   # 最近一次启动日志
```

## 文档

部署前置条件、目录布局、日志/停止/重启、参数速查等详见 [docs/DeepSeek-V4-Flash后台启动指南.md](docs/DeepSeek-V4-Flash后台启动指南.md)。

## 说明

- 模型级配置（模型路径、端口、并行度、量化、用量单价）在 `models/*.yaml` 中管理，全局配置（API 密钥、存储目录、日志目录、统计服务）在 `.env` 中管理
- `.env` 含 API 密钥等敏感信息，已加入 `.gitignore`，请勿提交
- 详细注意事项（KV cache 量化、DSpark 参数、NCCL 优化等）见上方文档
