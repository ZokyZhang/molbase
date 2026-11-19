# MolBase 部署指南（deploy/）

本目录存放 MolBase 及配套中间件的部署编排文件，目标机为 Linux 虚拟机（IP 用 `<虚拟机IP>` 占位，部署前自行替换为真实地址）。

## 目录结构

```
deploy/
├── searxng/               # SearXNG（MolBase 网络搜索兜底后端）
│   ├── docker-compose.yml # 8888 -> 8080
│   └── settings.yml       # 开启 JSON + 关限流 + 启用引擎
├── molbase/                  # MolBase 服务化容器
│   ├── Dockerfile         # python:3.13-slim，纯标准库，镜像极简
│   └── docker-compose.yml # 8787 + 数据卷持久化 + 接 SearXNG
└── README.md              # 本文件
```

## 部署顺序

1. **SearXNG**（网络搜索兜底）：
   ```bash
   cd deploy/searxng && docker compose up -d
   curl "http://<虚拟机IP>:8888/search?q=test&format=json"   # 验证
   ```
2. **MolBase**（服务化 API）：
   ```bash
   docker compose -f deploy/molbase/docker-compose.yml up -d --build
   curl "http://<虚拟机IP>:8787/health"                      # 验证
   ```

> 虚拟机 / 网络搜索 / 部署知识的详细说明，已沉淀进 MolBase 知识库（`knowledge_repo/`），
> 供 MolBase 检索复用，而非只停留在会话记录中。
