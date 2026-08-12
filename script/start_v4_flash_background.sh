#!/usr/bin/env bash
# ============================================================
# 后台启动 DeepSeek-V4-Flash 服务（官方 llama.cpp + DSpark）
# 用法：bash script/start_v4_flash_background.sh
#
# 配置优先级：环境变量 > .env 文件 > 脚本内置默认值
# 首次使用请先复制配置模板并按需修改：
#   cp .env.example .env
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
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# ---- 模型 ----
MODEL="${MODEL:-$(cd "$PROJECT_ROOT/../.." && pwd)/model-gguf/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}"

# ---- 服务 ----
PORT="${PORT:-18888}"
# 上下文长度：留空则由 Python 按并发数自动计算（每个并发槽位 1M 上下文）；
# 显式设置（.env 或环境变量）时传给 --ctx-size 覆盖。
CTX_SIZE="${CTX_SIZE:-}"
# 并发序列数（llama-server --parallel）。若官方版并行崩溃可改 1。
PARALLEL="${PARALLEL:-2}"
# API 密钥（.env 留空则不校验）
API_KEY="${API_KEY:-root123456}"
# 重复惩罚系数（.env 留空则不启用，llama-server 默认 1.0；如 1.15 可抑制复读）
REPEAT_PENALTY="${REPEAT_PENALTY:-}"

LOG_DIR="${LOG_DIR:-$(cd "$PROJECT_ROOT/../.." && pwd)/logs}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
LAUNCH_LOG="$LOG_DIR/launch-$STAMP.log"

ARGS=(--model "$MODEL" --skip-build --no-console --port "$PORT" --parallel "$PARALLEL" --metrics)
if [[ -n "$CTX_SIZE" ]]; then
  ARGS+=(--ctx-size "$CTX_SIZE")
fi
if [[ -n "$API_KEY" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi
if [[ -n "$REPEAT_PENALTY" ]]; then
  ARGS+=(--repeat-penalty "$REPEAT_PENALTY")
fi

nohup python3 "$SCRIPT_DIR/start_v4_flash_gguf.py" "${ARGS[@]}" \
  > "$LAUNCH_LOG" 2>&1 &
LLAMA_PID=$!

# ---- cc-switch 用量统计服务（已运行则跳过，避免端口冲突） ----
USAGE_LOG="$LOG_DIR/usage-stats.log"
if ! curl -s -o /dev/null http://127.0.0.1:"${USAGE_PORT:-5002}"/api/usage; then
  nohup python3 "$SCRIPT_DIR/usage_stats_server.py" >> "$USAGE_LOG" 2>&1 &
  USAGE_PID=$!
else
  USAGE_PID=""
fi

echo "======================================"
echo " 已后台启动，llama-server PID: $LLAMA_PID"
echo " 启动日志:  $LAUNCH_LOG"
echo " 服务日志:  $LOG_DIR/llama-server-${PORT}-*.log"
if [[ -n "$USAGE_PID" ]]; then
  echo " 用量统计服务已启动，PID: $USAGE_PID"
else
  echo " 用量统计服务已在运行"
fi
echo " 用量统计服务日志: $USAGE_LOG"
echo " 健康检查:  curl http://127.0.0.1:${PORT}/health"
echo " 用量查询:  curl http://127.0.0.1:${USAGE_PORT:-5002}/api/usage"
if [[ -n "$USAGE_PID" ]]; then
  echo " 停止服务:  kill $LLAMA_PID $USAGE_PID"
else
  echo " 停止服务:  kill $LLAMA_PID ; pkill -f usage_stats_server.py"
fi
echo " 强制停止:  fuser -k ${PORT}/tcp ; pkill -f usage_stats_server.py"
echo "======================================"
