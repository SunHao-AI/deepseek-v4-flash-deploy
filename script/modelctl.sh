#!/usr/bin/env bash
# modelctl.sh — 调用已安装的 modelctl 命令
set -euo pipefail
exec modelctl "$@"
