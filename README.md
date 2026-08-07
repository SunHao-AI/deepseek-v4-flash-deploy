# DeepSeek-V4-Flash 服务启动工具

在 CUDA/Ada GPU 上通过**官方 llama.cpp** 启动 DeepSeek-V4-Flash-0731 GGUF，提供 OpenAI 兼容接口，并集成 **DSpark 投机解码**（约 1.5x-1.9x 解码加速）。

## 特性

- 自动编译官方 llama.cpp（CUDA）并启动 `llama-server`
- 支持从 ModelScope 按量化版本按需下载 GGUF 分片（`--download`）
- DSpark 投机解码（自动发现/指定草稿模型）
- 上下文自动分配：默认**每个并发槽位 1M 上下文**（总 ctx = `PARALLEL × 1048576`），可手动覆盖
- 配置外置：全部参数通过 `.env` 管理，命令行参数可覆盖
- 支持前后台两种启动方式，日志自动落盘

## 目录结构

```
deepseek-v4-flash/
├── README.md                       # 本文档（入口）
├── docs/
│   └── DeepSeek-V4-Flash后台启动指南.md   # 部署与运维详细指南
├── script/
│   ├── start_v4_flash_gguf.py      # 主启动脚本（构建 + 启动 llama-server）
│   └── start_v4_flash_background.sh # 后台启动脚本（推荐）
├── .env.example                    # 配置模板（复制为 .env 后修改）
├── .env                            # 本地配置（含密钥，不入库）
├── .gitignore
└── pyproject.toml
```

## 快速开始

### 1. 安装依赖

- Python 3.10+（运行期零第三方依赖，`.env` 解析内置实现）
- `git`、`cmake`、CUDA 工具链、`nvidia-smi`、`tee`

### 2. 配置 .env

```bash
cp .env.example .env
vi .env        # 修改模型路径、端口、API 密钥等
```

配置优先级：**命令行参数 > 环境变量 > .env 文件 > 脚本内置默认值**。

### 3. 启动服务

前台（首次运行会自动编译 llama.cpp 并启动）：

```bash
python3 script/start_v4_flash_gguf.py
```

后台（推荐，SSH 断开不影响；需先编译好，见下方说明）：

```bash
bash script/start_v4_flash_background.sh
```

### 4. 验证

```bash
curl http://127.0.0.1:18888/health
```

返回 `{"status":"ok"}` 即成功。接口地址：`http://127.0.0.1:18888/v1/chat/completions`。

## 文档

部署前置条件、目录布局、日志/停止/重启、参数速查等详见 [docs/DeepSeek-V4-Flash后台启动指南.md](docs/DeepSeek-V4-Flash后台启动指南.md)。

## 说明

- 默认使用 **UD-Q8_K_XL 无损量化**；显存吃紧可换 `UD-Q4_K_XL`（近无损），修改 `.env` 的 `MODEL` 即可
- `.env` 含 API 密钥等敏感信息，已加入 `.gitignore`，请勿提交
- 详细注意事项（KV cache 量化、DSpark 参数、NCCL 优化等）见上方文档
