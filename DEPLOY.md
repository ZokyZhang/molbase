# MolBase 部署指南（Linux / Windows 双端）

> 对应 REPOSITION.md 的 P0（配置外部化）+ P1（服务化 HTTP API）。零第三方依赖，装个 Python 3.10+ 就能跑。

---

## 一、快速启动（双端通用，一步到位）

```bash
# 仅本机可用（默认）
python -m molbase.server

# 开放局域网，供其他应用调用
python -m molbase.server --host 0.0.0.0 --port 8787
```

**一个命令同时拉起：知识库（表）+ 代工中间层（存储过程）+ 监控大屏（配套服务）+ token 埋点。**
启动后：

- 监控大屏：`http://127.0.0.1:8787/`（实时可视化 token 优化，数据源自知识库 /metrics + /stats）
- 存活探活：`curl http://127.0.0.1:8787/health`
- 检索调用：`curl -X POST http://127.0.0.1:8787/query -H "Content-Type: application/json" -d '{"query":"作业管理怎么做","top_k":5}'`

这正是「架构分层、部署一体」：像装一个 Oracle 就同时有了表、存储过程、监控视图，无需分开部署。

---

## 二、配置（全核心参数可配置）

优先级：**环境变量 > config.json > config.py 默认值**。

1. 复制模板并修改：
   ```bash
   cp config.example.json config.json
   ```
2. 编辑 `config.json`，例如改模型、改端口、改知识库路径：
   ```json
   {
     "LLM_ENABLED": true,
     "LLM_ENDPOINT": "http://<宿主机IP>:11434/v1",
     "LLM_MODEL": "qwen3.5-9b",
     "SERVER_HOST": "0.0.0.0",
     "SERVER_PORT": 8787,
     "SERVER_AUTH_TOKEN": "your-secret-token",
     "DB_PATH": "/data/kb/molbase.db",
     "KNOWLEDGE_DIR": "/data/kb/knowledge_repo",
     "SKILLS_DIR": "/data/kb/skills_repo",
     "CODE_REPO_DIR": "/data/kb/code_repo"
   }
   ```
3. 或用环境变量（优先级更高，适合容器/进程管理）：
   ```bash
   MOLBASE_LLM_ENDPOINT=http://<宿主机IP>:11434/v1 \
   MOLBASE_SERVER_AUTH_TOKEN=your-secret-token \
   python -m molbase.server
   ```

**可配置项**：模型地址/版本/超时、向量后端、知识/技能/代码路径、检索与重排参数、
自我进化参数、网络搜索、成本单价、服务 host/port/auth_token（完整清单见 config.example.json）。

---

## 三、Linux 服务化（systemd）

新建 `/etc/systemd/system/molbase.service`：

```ini
[Unit]
Description=MolBase 知识代工层服务
After=network.target

[Service]
Type=simple
User=kb
WorkingDirectory=/opt/molbase
# 用 Environment 注入配置（或放 config.json）
Environment=MOLBASE_SERVER_HOST=0.0.0.0
Environment=MOLBASE_SERVER_PORT=8787
Environment=MOLBASE_SERVER_AUTH_TOKEN=your-secret-token
ExecStart=/usr/bin/python3 -m molbase.server --host 0.0.0.0 --port 8787
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now molbase
sudo systemctl status molbase
```

---

## 四、Windows 服务化

**方式 A：命令行直接跑（开发/调试）**

```powershell
python -m molbase.server --host 0.0.0.0 --port 8787
```

**方式 B：注册为系统服务（推荐 NSSM）**

```powershell
nssm install MolBaseService "C:\path\to\python.exe" -m molbase.server --host 0.0.0.0 --port 8787
nssm set MolBaseService AppDirectory "C:\path\to\molbase"
nssm start MolBaseService
```

**方式 C：计划任务（无 NSSM 时）**

```powershell
schtasks /Create /TN "MolBaseService" /TR "C:\path\to\python.exe -m molbase.server --host 0.0.0.0 --port 8787" /SC ONSTART /RL HIGHEST
```

> 注意：Windows 下知识库路径避免中文目录（历史踩坑：中文路径引发 GBK 乱码）。

---

## 五、HTTP API 一览

| 方法 | 路径 | 说明 | 请求体示例 |
|---|---|---|---|
| GET | `/health` | 存活 + 统计 | - |
| GET | `/stats` | 条目数量统计 | - |
| GET | `/token` | 会话级 token 节省统计 | - |
| GET | `/metrics` | 持久化埋点统计（累计节省 + 最近明细 + 工具轨迹聚合） | - |
| GET | `/` / `/dashboard` | 监控大屏（配套服务，数据源自知识库） | - |
| POST | `/query` | 检索最小上下文 | `{"query":"...","top_k":5}` |
| POST | `/feedback` | 反馈回写 | `{"entry_id":"...","action":"adopt"}` |
| POST | `/summarize` | 上下文压缩 | `{"text":"...","max_chars":200}` |
| POST | `/skill` | 匹配技能包（工具执行见 P2） | `{"intent":"解析 PDF"}` |
| POST | `/ingest/knowledge` | 摄入知识 | `{"entry_id","title","content"}` |
| POST | `/harvest` | 会话回写：会话→核心资产沉淀（LLM 优先、TextRank 兜底），自动关联同会话操作指纹（结论↔过程） | `{"session_id":"...","transcript":"...","source":"...","meta":{}}` |
| POST | `/tooltrace` | 工具调用轨迹沉淀：观察→聚类→达标自动生成 skill（N+1→N） | `{"session_id":"...","intent":"...","tool_calls":[{"tool":"...","args":{...}}]}` |

**鉴权**：配置 `SERVER_AUTH_TOKEN` 后，请求须带 `Authorization: Bearer <token>` 头。

**响应**：统一 JSON，`ensure_ascii=False`（中文原样返回）；`/` 返回 HTML 监控大屏。

---

## 六、供其他应用调用（Java 示例）

任何语言都能调。以 Java 为例（Spring RestTemplate / OkHttp 同理）：

```java
// 检索最小上下文
String url = "http://<宿主机IP>:8787/query";
String body = "{\"query\":\"作业管理怎么做\",\"top_k\":5}";
HttpResponse<String> resp = HttpClient.newHttpClient().send(
    HttpRequest.newBuilder(URI.create(url))
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer your-secret-token")
        .POST(HttpRequest.BodyPublishers.ofString(body))
        .build(),
    HttpResponse.BodyHandlers.ofString());
```

```java
// 会话回写：会话收尾时把整段会话提炼成核心资产沉淀（供复用）
String harvestUrl = "http://<宿主机IP>:8787/harvest";
String harvestBody = "{\"session_id\":\"sess-20261118\",\"transcript\":\"整段会话内容...\",\"source\":\"agent:autoclaw\"}";
HttpResponse<String> hr = HttpClient.newHttpClient().send(
    HttpRequest.newBuilder(URI.create(harvestUrl))
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer your-secret-token")
        .POST(HttpRequest.BodyPublishers.ofString(harvestBody))
        .build(),
    HttpResponse.BodyHandlers.ofString());
// 返回 {"entry_id":"session::sess-20261118","state":"new","engine":"textrank","title":"...","reason":""}
```

```java
// 工具调用轨迹沉淀：把大模型本次实际调用的工具喂给代工层，
// 相同操作模式累计达阈值（默认 3 次）自动生成可复用 skill，实现 N+1 -> N 的 token 收敛
String ttUrl = "http://<宿主机IP>:8787/tooltrace";
String ttBody = "{\"session_id\":\"sess-20261118\",\"intent\":\"检索并更新文档\","
    + "\"tool_calls\":[{\"tool\":\"web_search\",\"args\":{\"query\":\"x\"}},"
    + "{\"tool\":\"file_read\",\"args\":{\"path\":\"/x.md\"}},"
    + "{\"tool\":\"file_edit\",\"args\":{\"path\":\"/x.md\",\"content\":\"y\"}}]}";
HttpResponse<String> tr = HttpClient.newHttpClient().send(
    HttpRequest.newBuilder(URI.create(ttUrl))
        .header("Content-Type", "application/json")
        .header("Authorization", "Bearer your-secret-token")
        .POST(HttpRequest.BodyPublishers.ofString(ttBody))
        .build(),
    HttpResponse.BodyHandlers.ofString());
// 返回 {"fingerprint":"web_search(query) > file_read(path) > file_edit(content,path)",
//        "count":3,"generated":true,"skill_id":"skill::auto-web-search-file-read-file-edit","reason":"已达阈值（3），自动生成 skill"}
```

主系统只发意图、消费结果，不关心检索/重排/工具调用细节——这就是「轻应用、重知识」。
