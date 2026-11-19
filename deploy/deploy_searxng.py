# -*- coding: utf-8 -*-
"""从开发机一键部署 SearXNG 到 Linux 虚拟机（paramiko SSH）。

用法（密码不落盘，走环境变量）：
    VM_PASSWORD='xxx' python deploy/deploy_searxng.py

说明：paramiko 仅是本部署工具的运行依赖（开发机一次性使用），
      MolBase 运行时仍是零第三方依赖，不受影响。
"""
import os
import sys

import paramiko

HOST = os.environ.get("VM_HOST", "")  # 虚拟机 IP，从环境变量读（不落盘）
USER = os.environ.get("VM_USER", "zoky")
PASSWORD = os.environ.get("VM_PASSWORD", "")
DEPLOY_DIR = "molbase/deploy/searxng"

if not HOST:
    print("请设置 VM_HOST 环境变量（虚拟机 IP）")
    sys.exit(1)
if not PASSWORD:
    print("请设置 VM_PASSWORD 环境变量")
    sys.exit(1)


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 180, sudo: bool = False):
    """执行远程命令并回显输出，返回 (exit_code, stdout, stderr)。"""
    print(f"\n$ {('sudo ' if sudo else '')}{cmd}")
    if sudo:
        stdin, stdout, stderr = client.exec_command(f"sudo -S {cmd}", timeout=timeout)
        stdin.write(PASSWORD + "\n")
        stdin.flush()
    else:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out.rstrip())
    if err and code != 0:
        print("[stderr]", err.rstrip())
    print(f"[exit={code}]")
    return code, out, err


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print(f"[SSH 连接成功] {USER}@{HOST}\n")

    # 1. 检测 Docker 与 sudo 需求
    code, _, _ = run(client, "docker ps", timeout=30)
    need_sudo = code != 0
    if need_sudo:
        print(">>> docker 需 sudo，后续命令自动加 sudo")
        run(client, "docker ps", sudo=True)

    run(client, "docker compose version 2>&1 || docker-compose --version 2>&1")

    # 2. 上传部署文件（SFTP）
    run(client, f"mkdir -p ~/{DEPLOY_DIR} && echo MKDIR_OK", timeout=30)
    sftp = client.open_sftp()
    sftp.put("deploy/searxng/docker-compose.yml", f"{DEPLOY_DIR}/docker-compose.yml")
    sftp.put("deploy/searxng/settings.yml", f"{DEPLOY_DIR}/settings.yml")
    sftp.close()
    print("\n[文件上传完成] docker-compose.yml + settings.yml -> ~/" + DEPLOY_DIR)

    # 3. 部署 SearXNG（首次会拉镜像，超时放宽；用 -f 绝对路径避免 sudo cd 内建命令问题）
    run(client, f"docker compose -f /home/{USER}/{DEPLOY_DIR}/docker-compose.yml up -d",
        timeout=600, sudo=need_sudo)

    # 4. 验证
    run(client, "docker ps --format '{{.Names}} {{.Status}} {{.Ports}}' | grep -i searxng",
        timeout=30, sudo=need_sudo)
    run(client, "sleep 5 && curl -s 'http://127.0.0.1:8888/search?q=python&format=json' | head -c 400",
        timeout=30)

    client.close()
    print("\n[完成] SearXNG 部署结束")


if __name__ == "__main__":
    main()
