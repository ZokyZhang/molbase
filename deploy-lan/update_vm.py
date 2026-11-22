#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolBase 虚拟机热更新脚本：上传本次改动的代码文件 + 用 systemd 常驻重启服务。

改动文件（本次监控改版）：
    molbase/store/__init__.py、molbase/orchestrator.py、molbase/server.py、
    molbase/dashboard.html、molbase_mcp.py

不覆盖 data/molbase.db（虚拟机库是中心，本机测试数据不回流）。

硬编码参数：VM_HOST=10.0.0.200, VM_USER=zoky, VM_PASS=FamilyEdu@2026,
VM_TARGET=/home/zoky/molbase。sudo 密码经 stdin 传给 sudo -S。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

VM_HOST = "10.0.0.200"
VM_USER = "zoky"
VM_PASS = "FamilyEdu@2026"
VM_TARGET = "/home/zoky/molbase"
LOCAL_ROOT = Path(__file__).resolve().parent.parent

# 本次要同步的文件（相对项目根）
FILES = [
    "molbase/store/__init__.py",
    "molbase/orchestrator.py",
    "molbase/server.py",
    "molbase/config.py",
    "molbase/llm.py",
    "molbase/summarize.py",
    "molbase/retrieval/__init__.py",
    "molbase/dashboard.html",
    "molbase_mcp.py",
    "deploy-lan/molbase-lan.service",
    "deploy-lan/start-server.sh",
]


def connect() -> paramiko.SSHClient:
    """建立 SSH 连接。"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_HOST, username=VM_USER, password=VM_PASS, timeout=15)
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[str, str]:
    """执行远程命令，返回 (stdout, stderr)。"""
    _, o, e = ssh.exec_command(cmd, timeout=timeout)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")


def sudo(ssh: paramiko.SSHClient, cmd: str) -> tuple[str, str]:
    """带密码的 sudo 执行。"""
    return run(ssh, f"echo '{VM_PASS}' | sudo -S {cmd}")


def main() -> None:
    ssh = connect()
    sftp = ssh.open_sftp()
    try:
        # 1) 上传改动文件
        print("=== 上传改动文件 ===")
        for rel in FILES:
            local = LOCAL_ROOT / rel
            remote = f"{VM_TARGET}/{rel}"
            sftp.put(str(local), remote)
            print(f"  [上传] {rel}")

        # 2) 停旧进程（setsid 启动的旧服务）
        print("=== 停旧服务 ===")
        run(ssh, "pkill -f 'molbase.server' || true")
        time.sleep(1)

        # 3) 安装 systemd 常驻（覆盖旧 unit 后重载）
        print("=== 安装 systemd ===")
        sudo(ssh, f"cp {VM_TARGET}/deploy-lan/molbase-lan.service /etc/systemd/system/molbase-lan.service")
        sudo(ssh, "systemctl daemon-reload")
        # 先停再启，确保用新代码
        sudo(ssh, "systemctl stop molbase-lan 2>/dev/null || true")
        sudo(ssh, "systemctl enable molbase-lan 2>&1 | tail -1")
        sudo(ssh, "systemctl start molbase-lan")
        time.sleep(3)

        # 4) 验证
        print("=== 验证 ===")
        out, err = run(ssh, "systemctl is-active molbase-lan; curl -s -m 5 http://127.0.0.1:8787/health")
        print(out.strip())
        out, _ = run(ssh, "curl -s -m 10 http://127.0.0.1:8787/overview | head -c 500")
        print("overview:", out.strip()[:400])
        if err.strip():
            print("stderr:", err.strip()[:300])
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
