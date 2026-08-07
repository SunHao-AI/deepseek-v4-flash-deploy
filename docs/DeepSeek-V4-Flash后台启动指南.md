# DeepSeek-V4-Flash 后台启动指南

在远程服务器（8× RTX 5880 Ada）上以后台方式运行 DeepSeek-V4-Flash-0731（官方 llama.cpp + DSpark）的完整说明。

## 前置条件

- 已完成官方 llama.cpp 编译（`/raid5/sh/code/llama.cpp/build/bin/llama-server`，版本 **b10269+**）
- 已下载模型 GGUF 分片与 DSpark 草稿：

```
/raid5/sh/model-gguf/DeepSeek-V4-Flash-0731-GGUF/
├── UD-Q8_K_XL/
│   └── DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf  (+00002~00005)
└── dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
```

> 默认使用 **UD-Q8_K_XL 无损量化**（162GB，与官方权重 bit-identical）。若需近无损的 Q4（155GB），下载对应分片并把 `.env` 中的 `MODEL` 路径改为 `UD-Q4_K_XL/`。

## 配置管理（.env）

所有可调配置集中放在项目根目录的 `.env` 文件中（该文件不入库，含 API 密钥等敏感信息）：

```bash
cp .env.example .env
vi .env          # 按部署环境修改
```

配置优先级：**命令行参数 > 环境变量 > .env 文件 > 脚本内置默认值**。`.env` 中留空的值会自动回退到内置默认。常用配置项：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL` | `/raid5/sh/model-gguf/.../UD-Q8_K_XL/...-00001-of-00005.gguf` | GGUF 模型第一个分片（必填） |
| `DRAFT` | 空（自动发现） | DSpark 草稿路径 |
| `PORT` | `18888` | 服务端口 |
| `CTX_SIZE` | 空（自动） | 总上下文长度。留空自动计算：`PARALLEL × 1048576`（每并发 1M），可手动覆盖 |
| `PARALLEL` | `2` | 并发序列数（并行崩溃可改 1） |
| `GPU_COUNT` | `8` | GPU 数量 |
| `API_KEY` | `root123456` | API 密钥（留空则不校验） |
| `DSPARK` | `on` | DSpark 投机解码开关 |
| `REASONING` | `on` | 思考模式 |
| `CACHE_TYPE_K` / `CACHE_TYPE_V` | `q8_0` | KV cache 量化 |
| `LOG_DIR` | `/raid5/sh/logs` | 日志目录（绝对路径） |
| `SOURCE_DIR` | `/raid5/sh/code/llama.cpp` | llama.cpp 源码目录（绝对路径） |

> 注意：`set -a` 后 `source .env` 时，值含空格需用双引号包裹（如 `API_KEY="a b c"`）。

## 方式一：用后台启动脚本（推荐）

两个脚本位于 `script/` 目录，启动时会自动读取项目根目录的 `.env`。确认 `.env` 配置无误后：

```bash
bash script/start_v4_flash_background.sh
```

脚本会：

1. 自动加载项目根的 `.env`（未配置的环境变量已存在时不被覆盖）
2. 用 `nohup` 把 `start_v4_flash_gguf.py` 放到后台运行，即使 SSH 断开也不受影响
3. 传入 `--skip-build`（跳过重复编译）和 `--no-console`（只写日志、不占用终端）
4. 启动日志写入 `logs/launch-<时间戳>.log`（默认 `<项目根>/../../logs`）
5. 默认带 API key 启动（`.env` 的 `API_KEY`，留空则不校验）
6. 打印 PID、日志路径、健康检查命令
7. 未显式设置 `CTX_SIZE` 时，由 Python 按并发数自动计算总上下文（每并发 1M）

脚本内容：

```bash
#!/usr/bin/env bash
# ============================================================
# 后台启动 DeepSeek-V4-Flash 服务（官方 llama.cpp + DSpark）
# 用法：bash script/start_v4_flash_background.sh
#
# 配置优先级：环境变量 > .env 文件 > 脚本内置默认值
# 首次使用请先复制配置模板并按需修改：cp .env.example .env
#
# 所有路径均以项目根（脚本所在目录的上级）为基准，默认目录布局：
#   <项目根>/script/        e.g. /raid5/sh/code/deepseek-v4-flash/script/
#   <项目根>/../llama.cpp   e.g. /raid5/sh/code/llama.cpp
#   <项目根>/../../model-gguf  e.g. /raid5/sh/model-gguf
#   <项目根>/../../logs     e.g. /raid5/sh/logs
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 加载 .env（已存在的环境变量优先，不覆盖）
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# ---- 模型 ----
MODEL="${MODEL:-$(cd "$PROJECT_ROOT/../.." && pwd)/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}"

# ---- 服务 ----
PORT="${PORT:-18888}"
# 上下文长度：留空则由 Python 按并发数自动计算（每个并发槽位 1M 上下文）
CTX_SIZE="${CTX_SIZE:-}"
PARALLEL="${PARALLEL:-2}"
API_KEY="${API_KEY:-root123456}"

LOG_DIR="${LOG_DIR:-$(cd "$PROJECT_ROOT/../.." && pwd)/logs}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="$LOG_DIR/launch-$STAMP.log"

ARGS=(--model "$MODEL" --skip-build --no-console --port "$PORT" --parallel "$PARALLEL")
if [[ -n "$CTX_SIZE" ]]; then
  ARGS+=(--ctx-size "$CTX_SIZE")
fi
if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi

nohup python3 "$SCRIPT_DIR/start_v4_flash_gguf.py" "${ARGS[@]}" \
  > "$LAUNCH_LOG" 2>&1 &

echo "======================================"
echo " 已后台启动，PID: $!"
echo " 启动日志:  $LAUNCH_LOG"
echo " 服务日志:  $LOG_DIR/llama-server-${PORT}-*.log"
echo " 健康检查:  curl http://127.0.0.1:${PORT}/health"
echo " 停止服务:  kill $!"
echo "======================================"
```

## 方式二：直接 nohup 原始命令

```bash
nohup python3 /raid5/sh/code/deepseek-v4-flash/script/start_v4_flash_gguf.py \
  --model /raid5/sh/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
  --skip-build \
  --no-console \
  --api-key "root123456" \
  > /raid5/sh/logs/launch.log 2>&1 &
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

## 查看日志

```bash
# 启动过程日志
tail -f /raid5/sh/logs/launch-*.log

# 服务运行日志（llama-server 输出）
tail -f /raid5/sh/logs/llama-server-18888-*.log
```

## 停止服务

```bash
# 方式一：用脚本打印的 PID
kill <PID>

# 方式二：按端口找到进程
fuser -k 18888/tcp
# 或
pkill -f llama-server

# 确认已停止
curl http://127.0.0.1:18888/health   # 应连接失败
```

## 重启 / 换量化

1. 停止服务（见上）
2. 修改 `.env` 中的 `MODEL` 路径（默认 `UD-Q8_K_XL`；例如换 `UD-Q4_K_XL`），或用命令行直接传 `--model`
3. 重新 `bash script/start_v4_flash_background.sh`

## 参数速查（start_v4_flash_gguf.py）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `.env: MODEL` | 模型分片路径（第一个分片） |
| `--quant` | `.env: QUANT`（`UD-Q8_K_XL`） | `--download` 时下载的量化版本（无损 Q8 默认，近无损 Q4 可改） |
| `--draft` | `.env: DRAFT`（自动发现） | DSpark 草稿路径（不传会自动向上级目录找 `dspark*.gguf`） |
| `--skip-build` | 关 | 跳过 CMake 编译（已编译过必须加） |
| `--no-dspark` | `.env: DSPARK=on` | 禁用 DSpark 投机解码 |
| `--fit` | `.env: FIT=off` | DSpark README 推荐；`on` 时草稿显存测量被跳过、~11GB 落在预算外 |
| `--cache-type-k` / `--cache-type-v` | `.env: CACHE_TYPE_K/V=q8_0` | KV cache 量化（官方 b10298 已验证正常；f16 更保守） |
| `--ctx-size` | 自动（`PARALLEL × 1048576`） | 总上下文长度；命令行或 `.env: CTX_SIZE` 手动覆盖 |
| `--port` | `.env: PORT=18888` | 服务端口 |
| `--api-key` | `.env: API_KEY`（留空不校验） | API 密钥；设置后请求需带 `Authorization: Bearer <key>` |
| `--parallel` | `.env: PARALLEL=2` | 并发序列数（fork 上 --parallel 2 曾崩溃；官方版需实测确认正常） |
| `--no-console` | 关 | 只写日志文件，终端不输出 |
| `--reasoning` | `.env: REASONING=on` | 思考模式开关 |
| `--reasoning-format` | `.env: REASONING_FORMAT=deepseek` | 推理内容格式 |

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
  cmake -S /raid5/sh/code/llama.cpp -B /raid5/sh/code/llama.cpp/build \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build /raid5/sh/code/llama.cpp/build --config Release -j
  ```
- **prefill 吞吐调优**：可尝试 `--ubatch-size 1024`（默认 512）观察长提示 prefill 提升，需实测对比
- **换 Q4 近无损量化**：如果显存吃紧或想加快加载，把 `.env` 的 `MODEL` 改为 `UD-Q4_K_XL/` 路径，分片约小 7GB
