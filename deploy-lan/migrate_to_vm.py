#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolBase 知识库迁移 + 部署脚本（开发机 → 局域网虚拟机）。

功能：
  --check   只读探测：虚拟机 Python 版本、是否已有 molbase 目录、8787 端口占用。
  --deploy  执行部署：SFTP 递归上传项目 → 启动局域网服务（0.0.0.0:8787）。

依赖：paramiko（开发机已装 5.1.0）。
用法：
  python migrate_to_vm.py --check
  python migrate_to_vm.py --deploy

硬编码参数（项目约定：不加密、不用 ${} 占位符）：
  VM_HOST      10.0.0.200
  VM_USER      zoky
  VM_PASS      FamilyEdu@2026
  VM_TARGET    /home/zoky/molbase          虚拟机项目落点
  LOCAL_ROOT   ../                          开发机 molbase 项目根（本脚本位于 deploy-lan/）
  SERVER_ARGS  --host 0.0.0.0 --port 8787   服务监听参数
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import paramiko

# ---- 硬编码参数 ----
VM_HOST = "10.0.0.200"
VM_PORT = 22
VM_USER = "zoky"
VM_PASS = "FamilyEdu@2026"
VM_TARGET = "/home/zoky/molbase"
SERVER_ARGS = "--host 0.0.0.0 --port 8787"
# 本地项目根（本脚本位于 deploy-lan/ 下，上一级即 molbase 项目根）
LOCAL_ROOT = Path(__file__).resolve().parent.parent
# 递归上传时跳过的目录/文件
SKIP_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
SKIP_FILES = {"molbase.db-wal", "molbase.db-shm"}


def connect() -> paramiko.SSHClient:
    """建立 SSH 连接并返回客户端。"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_HOST, port=VM_PORT, username=VM_USER, password=VM_PASS, timeout=15)
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str) -> tuple[str, str]:
    """执行远程命令，返回 (stdout, stderr)。"""
    _, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def do_check() -> None:
    """只读探测虚拟机环境。"""
    print(f"=== 探测 {VM_HOST} ===")
    ssh = connect()
    try:
        out, _ = run(ssh, "hostname && python3 --version 2>&1; command -v python3 || echo 'no python3'")
        print("[主机名/Python]", out.strip())

        out, _ = run(ssh, f"ls -ld {VM_TARGET} 2>&1 || echo '无 molbase 目录'")
        print("[已有 molbase]", out.strip())

        out, _ = run(ssh, "ss -lntp 2>/dev/null | grep 8787 || echo '8787 未监听'")
        print("[8787 端口]", out.strip())

        out, _ = run(ssh, "sudo -n true 2>&1 && echo 'sudo 免密可用' || echo 'sudo 需密码'")
        print("[sudo]", out.strip())

        out, _ = run(ssh, "curl -s http://127.0.0.1:8787/health 2>&1 || echo '本机无知识库服务'")
        print("[知识库探活]", out.strip()[:200])
    finally:
        ssh.close()


def _sftp_upload_dir(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    """递归上传目录（跳过缓存/临时文件）。"""
    sftp.mkdir(remote) if not _remote_exists(sftp, remote) else None
    for item in sorted(local.iterdir()):
        if item.name in SKIP_DIRS:
            continue
        if item.name in SKIP_FILES:
            continue
        if item.name.startswith("."):
            continue
        rpath = f"{remote}/{item.name}"
        if item.is_dir():
            _sftp_upload_dir(sftp, item, rpath)
        else:
            sftp.put(str(item), rpath)
            print(f"  [上传] {item.relative_to(LOCAL_ROOT)}")


def _remote_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    """判断远程路径是否存在。"""
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def do_deploy() -> None:
    """执行迁移 + 启动。"""
    print(f"=== 迁移 {LOCAL_ROOT} → {VM_HOST}:{VM_TARGET} ===")
    ssh = connect()
    sftp = ssh.open_sftp()
    try:
        # 1) 递归上传项目
        sftp.mkdir(VM_TARGET) if not _remote_exists(sftp, VM_TARGET) else print("  [提示] 目标目录已存在，将增量覆盖")
        _sftp_upload_dir(sftp, LOCAL_ROOT, VM_TARGET)

        # 2) 给启动脚本加执行权限
        run(ssh, f"chmod +x {VM_TARGET}/deploy-lan/start-server.sh")

        # 3) 检测 Python3
        out, _ = run(ssh, "command -v python3 || echo python")
        py = out.strip().splitlines()[0] if out.strip() else "python3"

        # 4) 探活：若已在运行则跳过启动
        out, _ = run(ssh, "curl -s -m 3 http://127.0.0.1:8787/health || echo NO")
        if '"status": "ok"' in out:
            print("[已运行] 虚拟机已有知识库服务在 8787，跳过启动")
        else:
            # 5) 用 setsid 后台启动（远程，脱离 SSH 会话；systemd 常驻见部署说明）
            cmd = (f"cd {VM_TARGET} && setsid {py} -m molbase.server {SERVER_ARGS} "
                   f">> {VM_TARGET}/server.log 2>&1 < /dev/null &")
            run(ssh, cmd)
            print("[启动] 已触发后台启动，等待探活…")
            time.sleep(3)

        # 6) 最终探活
        out, _ = run(ssh, "curl -s -m 5 http://127.0.0.1:8787/health || echo NO")
        print("=== 探活结果 ===")
        print(out[:300])
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        do_check()
    elif "--deploy" in sys.argv:
        do_deploy()
    else:
        print(__doc__)
