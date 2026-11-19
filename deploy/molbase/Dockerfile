# Loom 容器镜像（零第三方依赖，纯标准库，镜像极简）
# 用法：在项目根目录 loom/ 下执行
#   docker build -f deploy/loom/Dockerfile -t loom:latest .
FROM python:3.13-slim

WORKDIR /app

# 只拷贝运行所需：包 + 语料目录 + 配置文件模板
COPY loom ./loom
COPY knowledge_repo ./knowledge_repo
COPY skills_repo ./skills_repo
COPY code_repo ./code_repo
COPY config.example.json ./config.example.json

EXPOSE 8787

ENV LOOM_SERVER_HOST=0.0.0.0
ENV LOOM_SERVER_PORT=8787

# 以服务化方式启动，供局域网内其他应用调用
CMD ["python", "-m", "loom.server", "--host", "0.0.0.0", "--port", "8787"]
