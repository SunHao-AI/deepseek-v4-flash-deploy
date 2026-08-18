# modelctl 新增 Unsloth 推理引擎设计

日期：2026-08-18
状态：已确认（用户评审通过）

## 1. 背景与目标

在现有引擎插件式架构（llamacpp / ollama / vllm / sglang）基础上，为 modelctl 新增 **Unsloth** 推理引擎，支持以 Unsloth 无头服务作为模型部署后端，从而获得 Unsloth 的差异化能力：动态量化 GGUF（UD-*）、自愈工具调用、自动推理参数调优、OpenAI/Anthropic 双方言 API。

### 已确认的关键决策

| 决策点 | 结论 |
|---|---|
| 集成形态 | 新增 `unsloth` 引擎适配器（非复用 llamacpp、非仅代理启动） |
| 部署形态 | 无头 CLI 模式（`unsloth studio --api-only`），不依赖 Unsloth Desktop GUI |
| 模型来源 | ModelScope 优先，HuggingFace 兜底（`HF_ENDPOINT` 镜像） |
| 进程模型 | 主模式"一 profile 一进程"；检测到已有 unsloth 服务则连接复用 |

## 2. 技术可行性分析

**结论：可行**，与现有架构契合度较高。

### 2.1 有利条件

1. **插件架构天然支持**：引擎注册表机制（`src/modelctl/engines/__init__.py`）新增引擎只需"1 个适配器文件 + 1 条注册"。llamacpp 适配器已验证"外部二进制 + 后台进程 + GGUF 分片下载"完整链路，可直接复用。
2. **Unsloth 具备无头服务能力**：
   - `unsloth studio --api-only`：无头 API 服务器，支持 `-p <port>`、`-H 0.0.0.0`（绑定所有接口）、`--secure`（HTTPS）
   - `unsloth start <agent> --serve`：托管模型并启动临时服务器；`--no-launch` 打印生成的环境与命令（便于调试与核对）
   - `--model <hf_id>[:<gguf_variant>]`、`--context-length`、`--tensor-parallel`、`--load-in-4bit` 等加载选项
3. **开放 API 兼容现有消费方**：同一端口提供 OpenAI 兼容（`/v1/chat/completions`、`/v1/responses`）与 Anthropic 兼容（`/v1/messages`），认证为 `sk-unsloth-…`，与项目 api_key 机制对接顺畅。
4. **模型源已打通**：项目已在用 `unsloth/DeepSeek-V4-Flash-0731-GGUF`（ModelScope），Unsloth 动态量化 GGUF（`UD-Q8_K_XL` 等）可直接复用现有 `download_gguf` 分片下载逻辑；HF 兜底可走 `HF_ENDPOINT` 镜像。
5. **硬件满足**：8× RTX 5880 Ada（CC 8.9 / 384GB）符合 Unsloth 要求，`--tensor-parallel` 支持多卡 GGUF 加载。

### 2.2 需验证的不确定点（实施/测试阶段验证，已预留降级路径）

| 不确定点 | 影响 | 降级路径 |
|---|---|---|
| 无头 API 模式是否暴露 Prometheus `/metrics` | 用量统计依赖 | 标记"不支持精确统计"（同 ollama 策略） |
| 无 `/health` 端点 | 健康检查 | 改用带认证的 `/v1/models` 作为健康检查 |
| 无头模式具体 flag 与文档差异 | 命令构建 | 测试阶段先 `unsloth --help` / `--no-launch` 实测，flag 常量集中管理 |

## 3. 环境配置要求

### 3.1 基础环境

- Python 3.12+、CUDA 12.x、NVIDIA 驱动（项目已有）
- Unsloth 安装（Linux 服务器）：
  - `curl -fsSL https://unsloth.ai/install.sh | sh`；或
  - 独立 venv / uv tool 安装 Unsloth Studio（**推荐**：避免 torch 等重依赖污染项目环境）；或
  - 官方 Docker 镜像（备选）

### 3.2 新增环境变量（`.env.example` 补充）

| 变量 | 用途 |
|---|---|
| `UNSLOTH_API_KEY` | Unsloth API key（`sk-unsloth-…`），profile `api_key` 可插值引用 |
| `UNSLOTH_STUDIO_URL` | 可选，远程 Unsloth 服务器地址 |
| `HF_ENDPOINT` | 国内 HF 镜像（如 `https://hf-mirror.com`），HF 兜底下载时使用 |

### 3.3 复用环境变量

| 变量 | 用途 |
|---|---|
| `MODEL_ROOT` / `MODELSCOPE_CACHE` | ModelScope 下载目录与缓存 |
| `HF_HOME` | HF 缓存目录 |

### 3.4 能力探测扩展（`core/capabilities.py`）

- 新增 `unsloth` 命令可用性探测与版本记录（用于兼容性判断与升级提示）

## 4. 代码适配要点

### 4.1 新增/修改文件清单

| 文件 | 动作 |
|---|---|
| `src/modelctl/engines/unsloth.py` | 新增 `UnslothAdapter` |
| `src/modelctl/engines/__init__.py` | 注册 `"unsloth"` → `UnslothAdapter` |
| `src/modelctl/core/capabilities.py` | 探测 unsloth 二进制/版本（`ENGINE_BINARIES` 增加 `"unsloth"`） |
| `src/modelctl/core/profile.py` | `KNOWN_ENGINES` 增加 `"unsloth"` |
| `src/modelctl/cli.py` | `probe` 子命令引擎列表增加 `"unsloth"` |
| `.env.example` | 新增环境变量（见 3.2） |
| `models/unsloth/deepseek-v4-unsloth.yaml` | 示例 profile（Unsloth 引擎；name 避开根目录已有的 `deepseek-v4`） |
| `tests/test_engines_unsloth.py` | 适配器单元测试 |

### 4.2 UnslothAdapter 钩子实现

| 钩子 | 实现要点 |
|---|---|
| `check_requirements()` | `unsloth` 命令存在；model 必填或配 download；`tensor_parallel` ≤ 实际 GPU 数；显存预检（模型文件大小 ×1.1，复用 llamacpp 逻辑） |
| `pre_start()` | model 缺失或不存在时，从 ModelScope 下载指定 quant 分片并持久化写回（复用 `download_gguf`）；下载失败时提示 HF 兜底（`HF_ENDPOINT` 镜像） |
| `build_command()` | 构造 `unsloth studio --api-only -H 0.0.0.0 -p <port> --model <hf_id>[:<gguf_variant>] --context-length <n>` 类命令；具体 flag 以 `unsloth --help` / `--no-launch` 实测为准，集中为模块常量 |
| `health_url()` | **覆盖**为 `/v1/models`（base 默认 `/health` 不适用）。认证头由 `wait_health(url, timeout, profile.api_key)` 自动携带（见 `src/modelctl/cli.py`），无需新增机制；**因此 unsloth 引擎必须配置 `api_key`，否则健康检查因 401 持续重试直至超时** |
| `metrics_mapping()` | 验证存在 `/metrics` 则复用 llamacpp 风格指标名；否则返回 `None` + warning |
| `post_start()` | 启动后发一个最小请求预热，避免首个请求冷启动慢 |
| `stop_patterns()` | `["unsloth"]` |
| `api_key_args()` | 复用现有 `--api-key` 逻辑 |

### 4.3 进程模型适配策略（关键决策）

- **主模式（一 profile 一进程）**：`modelctl start` 时用 `--model` 拉起独立 unsloth 无头服务，进程管理 / PID / 日志 / 停止全部复用现有 `core/process.py`，与 llamacpp 模式一致。
- **兼容模式（连接复用）**：启动前先探测目标端口，若检测到已有 unsloth 服务在运行，则健康检查通过即视为启动成功，不重复拉起。

### 4.4 Profile YAML 设计

示例 `models/unsloth/deepseek-v4-unsloth.yaml`：

> 命名说明：根目录已有 `models/deepseek-v4.yaml`（llamacpp 引擎），按"根目录优先 + 重复警告"规则，unsloth 实例必须使用不同 name，故命名为 `deepseek-v4-unsloth`。

```yaml
name: deepseek-v4-unsloth
engine: unsloth
port: 8000
api_key: ${UNSLOTH_API_KEY}   # 必填：健康检查 /v1/models 依赖 Bearer 认证

unsloth:
  model: unsloth/DeepSeek-V4-Flash-0731-GGUF   # HF ID 或本地路径
  gguf_variant: UD-Q8_K_XL                      # Unsloth 动态量化后缀
  context_length: 131072                        # 请求的上下文长度
  tensor_parallel: false                        # 多卡 GGUF tensor-parallel 加载
  load_in_4bit: false                           # 非 GGUF HF 模型的 4bit 加载
  download:
    modelscope_id: unsloth/DeepSeek-V4-Flash-0731-GGUF
    quant: UD-Q8_K_XL
  extra_args: ""                                # 透传其他 unsloth 参数

usage:
  price_in: 0.5
  price_out: 1.0
```

## 5. 性能优化策略

1. **Unsloth 动态量化 GGUF（UD-*）**：同显存下比标准量化精度更高，可跑更大模型 / 更长上下文。
2. **`--tensor-parallel`**：8 卡环境启用多卡 GGUF 加载。
3. **显式 `--context-length`**：避免默认上下文过大挤占显存。
4. **`--jinja` + `--flash-attn on`**：后端聊天模板修复与 FlashAttention 加速（Unsloth 后端默认推荐组合）。
5. **分片下载只拉所需 quant**：复用现有 `download_gguf`，不下载整个仓库。
6. **启动后预热**：`post_start()` 发一个最小请求，降低首个请求冷启动延迟。
7. **自动推理调参**：GGUF 模型的推理参数（temp / top-k 等）由 Unsloth 自动推理，无需手动调。

## 6. 部署测试步骤

### 6.1 环境与命令验证

1. 安装 Unsloth 后执行 `unsloth --help`，确认 `--api-only`、`--model`、`-p`、`-H` 等 flag 存在。
2. 使用 `unsloth start <agent> --no-launch` 打印生成的环境与命令，核对端口 / key / 模型参数。

### 6.2 单元测试

- `uv run pytest tests/test_engines_unsloth.py`：覆盖命令构建、requirement 校验、metrics 映射（含降级）、profile 解析、显存预检。

### 6.3 集成测试（真实 GPU 环境）

1. `modelctl start deepseek-v4` → 通过 `/v1/models` 健康检查。
2. 用 OpenAI SDK（`base_url=http://127.0.0.1:<port>/v1`，`api_key=sk-unsloth-…`）发起 chat 请求：普通对话、流式、tool calling。
3. `modelctl stop` 验证优雅停止与进程清理。
4. 下载链路：ModelScope 下载 → 路径写回 YAML；HF 兜底（`HF_ENDPOINT` 镜像）验证。
5. 统计验证：确认 `/metrics` 可用性或正确降级（metrics_mapping 返回 None + warning）。
6. 回归：确认 llamacpp / vllm / ollama / sglang 现有引擎不受影响（注册表新增无破坏）。

## 7. 潜在风险与应对措施

| 风险 | 应对 |
|---|---|
| 无头模式 flag 与文档存在差异 | 测试阶段先 `--help` / `--no-launch` 实测；flag 集中为模块常量 |
| 无 `/metrics` 或指标名变化 | `metrics_mapping` 返回 `None` + warning（同 ollama 策略），stats 标记"不支持精确统计" |
| 无 `/health` 端点 | 健康检查覆盖为 `/v1/models` + 认证头 |
| 多模型共享进程冲突 | 主模式一进程一模型；检测到已有服务则连接复用 |
| 国内网络 HF 下载慢/失败 | ModelScope 优先；`HF_ENDPOINT=hf-mirror.com` 兜底 |
| Unsloth 迭代快、flag/API 变化 | 能力探测记录版本，适配器按版本兼容 + 升级提示 |
| 认证 key 管理 | profile `api_key` 或 `UNSLOTH_API_KEY` 环境变量，不硬编码；**unsloth 引擎 api_key 为必填**（健康检查 `/v1/models` 依赖 Bearer 认证，缺失会 401 超时） |
| torch 等重依赖体积大 | 独立 venv（uv tool）安装 Unsloth，避免污染项目环境；或 Docker |
| 8 卡 tensor-parallel 显存碎片 | 显存预检 + 剩余显存告警（复用 llamacpp 降级策略） |

## 8. 非目标（明确不做）

- 不实现 `unsloth start` 编码代理（Claude Code / Codex 等）的启动与编排 —— 属于后续独立扩展。
- 不将 Unsloth 管理 UI / 训练能力纳入 modelctl。
- 不对现有引擎做任何行为变更（仅注册表新增）。

## 9. 兼容性说明

- 现有 `.env` 配置与各引擎 profile 不受影响。
- 新增 `unsloth` 引擎后，`modelctl list/status/probe` 等命令自然识别新引擎（仅需 capabilities 增加探测项）。
- `models/unsloth/` 子目录遵循既有 `models/<engine>/<name>.yaml` 目录规则。
