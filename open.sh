#!/bin/bash
# open.sh — 启动 PicoClaw 服务
# 用法：bash open.sh
#       bash open.sh --fg    前台运行（看日志用）

cd "$(dirname "$0")"

# 先调用 kill.sh 干净地停止旧服务并释放 BPU/CSI 资源
bash "$(dirname "$0")/kill.sh"

PYTHON=/root/pico_view/venv/bin/python

if [ "${1:-}" = "--fg" ]; then
    echo "[open.sh] 前台启动（Ctrl+C 停止）..."
    exec $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000
else
    echo "[open.sh] 后台启动，日志 → /tmp/pico_server.log"
    nohup $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 \
        > /tmp/pico_server.log 2>&1 &
    echo "[open.sh] PID=$!  tail -f /tmp/pico_server.log 查看日志"
fi
