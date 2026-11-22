#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# MolBase 局域网知识库服务器一键启动脚本
# 部署位置：虚拟机 /home/zoky/molbase/deploy-lan/start-server.sh
# 用法：bash start-server.sh   （前台运行；常驻建议配合 molbase-lan.service）
#
# 硬编码约定（项目约定：不加密、不用 ${} 占位符）：
#   监听 0.0.0.0:8787 —— 开放局域网，供所有设备共用；不鉴权（受信内网）。

set -e

# 回到 molbase 项目根目录（本脚本位于 deploy-lan/ 子目录下）
cd "$(dirname "$0")/.."

# 探测 Python 解释器（Ubuntu 用 python3，兼容纯 python）
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

echo "[molbase] 启动局域网知识库服务器：0.0.0.0:8787（项目目录：$(pwd)）"
echo "[molbase] 监控大屏：http://10.0.0.200:8787/"

# 前台启动；Ctrl+C 优雅退出（systemd 由 Restart=always 兜底）
exec "$PY" -m molbase.server --host 0.0.0.0 --port 8787
