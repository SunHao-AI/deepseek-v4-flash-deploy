#!/usr/bin/env bash
# modelctl.sh — modelctl.py 的 bash 入口（后台 start/stop/restart 语义）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/modelctl.py" "$@"
