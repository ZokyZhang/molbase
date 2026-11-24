#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolBase 虚拟机热更新脚本：上传本次改动的代码文件 + 用 systemd 常驻重启服务。

改动文件（本次监控改版）：
    molbase/store/__init__.py、molbase/orchestrator.py、molbase/server.py、
    molbase/dashboard.html、molbase_mcp.py

不覆盖 data/molbase.db（虚拟机库是中心，本机测试数据不回流）。

硬编码参数：VM_HOST=molbase.lan（hosts 解析到 192.168.101.200，IP 漂移不影响）,
VM_USER=zoky, VM_PASS=FamilyEdu@2026,
VM_TARGET=/home/zoky/molbase。sudo 密码经 stdin 传给 sudo -S。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

VM_HOST = "molbase.lan"
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
    "molbase/embedding.py",
    "molbase/llm.py",
    "molbase/summarize.py",
    "molbase/retrieval/__init__.py",
    # v9.0/v9.1 新增：向量预计算、Markdown 切块、知识库四层（树/网/标签/原文）
    "molbase/vectorize.py",
    "molbase/chunking.py",
    "molbase/kb_layers.py",
    "molbase/dashboard.html",
    "molbase_mcp.py",
    "deploy-lan/molbase-lan.service",
    "deploy-lan/start-server.sh",
    "deploy-lan/backfill_vectors.py",
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


def _unit_backend() -> str:
    """
    从本地 systemd unit 里读出服务实际使用的向量后端。

    为什么不能硬编码：unit 与 config.py 的后端可能不同（VM 为了性能锁 hash，
    开发机默认 bge-m3）。回填必须跟服务一致，否则算出的向量服务查不到。
    """
    unit = LOCAL_ROOT / "deploy-lan" / "molbase-lan.service"
    for line in unit.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Environment=MOLBASE_EMBEDDING_BACKEND="):
            return line.split("=", 2)[2].strip()
    return "hash"


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

        # 2.5) 离线回填向量（v9.1）：必须在服务起来之前跑完。
        # 理由：730 条目要切 12000+ 块并算向量，实测 8~110 秒；放启动路径会堵住端口。
        # 后端必须与服务一致：systemd 里写的 hash 在 ssh 会话里不存在，
        # 不显式传就会用 config 默认的 bge-m3，12000+ 块逐条推理要几小时。
        print("=== 离线回填向量与原文层 ===")
        backend = _unit_backend()
        print(f"  使用后端: {backend}")
        out, err = run(
            ssh,
            f"cd {VM_TARGET} && MOLBASE_EMBEDDING_BACKEND={backend} "
            f"timeout 900 python3 deploy-lan/backfill_vectors.py --chunks --backend {backend} 2>&1 | tail -20",
            timeout=960,
        )
        print(out.strip() or "(无输出)")
        if err.strip():
            print("backfill stderr:", err.strip()[:300])

        # 3) 安装 systemd 常驻（覆盖旧 unit 后重载）
        print("=== 安装 systemd ===")
        sudo(ssh, f"cp {VM_TARGET}/deploy-lan/molbase-lan.service /etc/systemd/system/molbase-lan.service")
        sudo(ssh, "systemctl daemon-reload")
        # 先停再启，确保用新代码
        sudo(ssh, "systemctl stop molbase-lan 2>/dev/null || true")
        sudo(ssh, "systemctl enable molbase-lan 2>&1 | tail -1")
        sudo(ssh, "systemctl start molbase-lan")
        time.sleep(3)

        # 4) 确保局域网域名反代配置存在
        # 历史踩坑：lan-services.conf 被改名成 .bak 后，molbase.lan 会落到默认站点
        # （family-edu default_server）返回 502/404。此处幂等恢复并热重载，防止复发。
        print("=== 确保 nginx 反代配置 ===")
        sudo(ssh, "bash -c \"test -f /etc/nginx/conf.d/lan-services.conf || "
                  "cp /etc/nginx/conf.d/lan-services.conf.bak-030800 /etc/nginx/conf.d/lan-services.conf\"")
        sudo(ssh, "bash -c 'nginx -t >/dev/null 2>&1 && nginx -s reload 2>/dev/null || true'")

        # 5) 验证
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
