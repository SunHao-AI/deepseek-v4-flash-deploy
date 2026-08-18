#!/usr/bin/env bash
# modelctl.sh — 通过 uv 调用 modelctl 命令（自动适配 Windows / Linux）
set -euo pipefail

# Windows（含 WSL）下 uv 可执行文件带 .exe 后缀，Linux 下不带
if command -v uv.exe >/dev/null 2>&1; then
    UV=uv.exe
else
    UV=uv
fi

"$UV" run modelctl "$@"
