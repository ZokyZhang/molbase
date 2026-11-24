# MolBase（猫贝）— 知识库 × 模型联动新范式

> **MolBase** = **Mol**（Model，模型）× **Base**（底座 / 知识库）。
> 中文名「猫贝」：以知识库为本体、与 Agent 和大模型联动的新范式。
> 把散乱的知识**织**成精准的上下文，让模型只做计算与推理。

把「知识库」从被动内容载体升级为**上下文代工中间层**：大模型退化为纯推理内核，
只负责计算与推算；检索、技能加载、源文件读取、代码编译/反编译、关联方法定位等
「高 token、低推理」动作全部下沉到本层，由**确定性算法 + 本地大模型协同**完成。

## 渐进增强（对齐「对标 MySQL，自我运行」）

**只有一个内核：零模型、零第三方依赖，装个 Python 就能跑。** 本地大模型与向量模型
都是「可插拔的增强插件」——有则更准（意图路由语义化、生成式摘要），没有也完整能跑。
拔掉或失效都不影响自足运行，这正是杜绝「模型版本匹配不上就失效」的根本解法。

| 能力 | 零模型内核（默认，永不失效） | 可插拔增强（可选，自动降级） |
| --- | --- | --- |
| 意图路由 | 关键词规则 | 本地大模型语义分类 |
| 查询改写 | 原查询直接检索 | 本地大模型改写为检索友好关键词 |
| 摘要压缩 | TextRank 抽取式 / 截断 | 本地大模型生成式摘要 |
| 语义向量 | 哈希 / TF-IDF 加权 | 本地 embedding 模型 |
| 网络搜索 | 不联网（内部知识库优先） | 内部命中弱时联网兜底（SearXNG 等） |

- **本地大模型**：`config.LLM_ENABLED = True` 接 Qwen3.5 9B 等（OpenAI 兼容，Ollama/vLLM 通用），
  模型名/地址参数化，换模型或升版本只改一个参数，不可用自动降级。
- **向量模型**：`config.EMBEDDING_BACKEND` 可选 `hash` / `tfidf` / `sentence-transformers`，
  模型名即版本指纹，失效自动降级 hash。

## 自我进化（零模型，全自动，无需人工）

知识库会**从使用痕迹中自动学习**，使用本身就在优化检索：

| 机制 | 说明 |
| --- | --- |
| 自动命中热度 | query 命中的条目自动累加热度（用得多自然热） |
| 共现自动建边 | 同一查询经常一起命中的条目，自动建立关联（系统自我发现关联） |
| 类型感知建边 | 共现建边仅限同类型（知识↔知识、代码↔代码、技能↔技能），减少跨类型噪音 |
| 冷知识自动衰减 | 长期未访问的条目热度自然衰减（自然淘汰） |
| 负反馈隔离 | `feedback(id, "reject")` 明确拒绝降权；累计达阈值自动隔离（quarantine，不参与召回） |
| 死知识归档 | 长期未访问且低热度/已隔离的知识自动归档（保留在库，可人工恢复） |

闭环：**使用 → 记录 → 学习 → 反哺检索**，query 会自动节流触发进化（`EVOLVE_INTERVAL`）。

## 核心思想：数据库 + 中间件

| 层 | 类比 | 本系统 |
| --- | --- | --- |
| 底座 | 存储引擎 | 知识 / 代码 / 技能 + 热度/新鲜度 + 共现信号 |
| 代工层 | 查询优化器 + 执行计划 | 意图路由 → 召回 → 重排 → 最小上下文 |
| 大模型 | 计算算子 | 只吃结果集做推算 |

传统 `agent.md` 的「检索 → 读源文件 → 加载 skill」本质是**全表扫描**；
本系统给它加上「索引 + 执行计划 + 反馈闭环」，正是数据库从全表扫描进化到
「索引 + 优化器 + 统计信息」的同一条路。

## 能力地图（集百家之长，开源直接用）

| 能力块 | 借鉴的开源 | 落地 |
| --- | --- | --- |
| 技能代管 | anthropics/skills（agentskills.io） | `skills/` SKILL.md 解析 + 注册 |
| 代码代工 | Aider repo-map（AST + PageRank） | `code/` 标准库 ast 符号索引 + 调用图 |
| 检索召回 | RAG Recall（BM25）→ BM25F | `index/` 倒排索引 + 字段加权 + 词典分词 |
| 重排 | RAG Rerank | `retrieval/` 热度 + 新鲜度 + 语义 |
| 关联检索 | LightRAG（图检索） | `graph/` 关联图 + PageRank + Personalized PR |
| 反馈闭环 | mem0 信号 | `store/` heat + last_access 回写 + 共现 |
| 语义向量 | 自研零模型基线 | `embedding.py` 哈希 / TF-IDF + 可插拔模型后端 |
| 摘要压缩 | TextRank | `summarize.py` 抽取式摘要 + 智能切块 |
| 去重 | SimHash | `embedding.py` 近重复检测 |
| 质量底线 | 自研「四道防线 + 两道闸门」 | `orchestrator.py` abstain/confidence + 冲突四态；`sanitize.py` PII 脱敏 |
| 本地大模型 | OpenAI 兼容 /v1/chat/completions | `llm.py` 意图路由 + 查询改写 + 摘要（零库依赖） |
| 网络搜索 | SearXNG 兼容 JSON 接口 | `websearch.py` 外部召回源适配器（兜底 + 采纳沉淀） |
| token 埋点 | 字符-token 粗估 | `tokens.py` 量化「降 token」效果 |

**差异化空白（自研）**：自我进化反馈闭环（共现建边 + 冷知识衰减）、四块能力统一成
「查询优化器」式中间层、以及零模型可自足运行的哈希/TF-IDF 向量基线。

## 架构分层（架构分层、部署一体，对标 TiDB/Oracle「表 + 存储过程」）

```
┌──────────────────────────────────────────────────────────────┐
│              MolBase · 一个进程一步到位（部署不分层）              │
│                                                              │
│   知识库本体（= 表）                                           │
│     ├─ 四层结构 树 topic_tree · 网 cooccur(PMI) ·              │
│     │           标签 tag_vocab(受控词表) · 原文 chunks         │
│     ├─ 信号层 entries(热度/新鲜度) · conflicts · feedback_log   │
│     └─ 观测层 metrics · tooltraces · abstain_log · draft_log   │
│        ▲                                    ▲                │
│        │ 读写                                │ 读取             │
│   ┌────┴───────────────┐            ┌────────┴──────────┐     │
│   │  代工中间层（= 存储过程）│           │  监控大屏（= 配套服务） │     │
│   │  路由→召回→重排→闸门    │           │  web 页面 + 埋点监控   │     │
│   │  →最小上下文 + 四层维护  │           │                    │     │
│   └────┬───────────────┘            └───────────────────┘     │
│        │ 服务化 HTTP API（/query /metrics /stats …）            │
│        ▼                                                     │
│   大模型（推理内核，只做计算/推算）                                 │
└──────────────────────────────────────────────────────────────┘
```

- **表（数据本体）**：知识/代码/技能条目 + 热度/新鲜度/共现信号 + token 埋点 + 工具调用轨迹。
- **存储过程（配套逻辑）**：代工中间层——意图路由 → 召回 → 重排 → 质量闸门 → 最小上下文，
  检索、技能加载、代码定位等「高 token、低推理」动作下沉到这里。
- **监控大屏（配套服务）**：`/` 一步托管的 web 页面，数据从 `/metrics` + `/stats` 实时读取。
- **部署一体**：`python -m molbase.server` 一步拉起三者。

传统 `agent.md` 的「检索 → 读源文件 → 加载 skill」本质是**全表扫描**；
本系统给它加上「索引 + 执行计划 + 反馈闭环」，正是数据库从全表扫描进化到
「索引 + 优化器 + 统计信息」的同一条路。

### 知识库四层模型（v9.1）

单纯堆条目只能做「词汇匹配」。要让检索长出结构信号，知识必须同时是四张网：

| 层 | 表 | 回答什么问题 |
|---|---|---|
| **树** | `topic_tree`（受控标签 → 条目标题，两层） | 我从哪个主题找到你 |
| **网** | `cooccur` + PMI 重定权 | 你和谁一起出现，权重有多可信 |
| **标签** | `tag_vocab`（别名归一 + 文档频次 df） | 这个概念在库里被多少条覆盖 |
| **原文** | `chunks`（标题栈路径 + 字符区间） | 这句答案出自哪一节第几段 |

`/query` 响应的 `refs` 字段直接给出 `path / heading / char_start / char_end`，
可以把结论精确还原到原文位置，不必把整段正文重新塞回上下文。

### 质量闸门（v9.2）

```
召回 → 重排 → 语义余弦 ┐
                    ├→ strength = max(余弦, 词法覆盖率×0.55)
       词法覆盖率 ────┘
                    ↓
     conf = 1/(1+exp(-SLOPE×(strength−MID)))   ← MID/SLOPE 按运行时真实后端标定
                    ↓
   阶段1 判库域（词表覆盖率<0.15 且 strength<0.30 → out_of_domain）
   阶段2 判质量（conf ≥ 0.43 放行）
```

弃答、拒答、低置信三种情况可归因到 `abstain_reason / abstain_subreason`，
配合 `draft_log` 形成「弃答 → 补数草稿 → 复测」的闭环。

> 实测（100 条历史查询回放）：弃答率 52.0% → **13.0%**，置信度上限从公式封顶 0.786 抬到 0.921。
> 每项参数为什么这么定、哪些改法被实测否决，见 `reports/MolBase-v9.2降弃答与知识库四层升级.md`。

## 目录结构

```
molbase/
├── AGENT.md                 # agent 唯一改动：一条「先查知识库」约束
├── demo.py                  # 端到端演示
├── requirements.txt         # 核心零依赖，可选依赖注释说明
├── molbase/
│   ├── config.py            # 全局配置（参数硬编码）
│   ├── orchestrator.py      # 统一编排入口（query / feedback / summarize / evolve / metrics）
│   ├── evolution.py         # 自我进化引擎（共现建边 / 冷知识衰减）
│   ├── summarize.py         # TextRank 摘要 + 智能切块
│   ├── embedding.py         # 可插拔向量层（哈希 / TF-IDF / 模型）+ SimHash
│   ├── sanitize.py          # PII 脱敏（手机号/邮箱/身份证/银行卡/密码，输出前打码）
│   ├── llm.py               # 本地大模型接入层（意图路由/查询改写/摘要，零库依赖）
│   ├── websearch.py         # 网络搜索外部源适配器（兜底 + 采纳沉淀）
│   ├── tokens.py            # token 粗估埋点（量化降 token 效果）
│   ├── server.py            # 服务化 HTTP API + 监控大屏一步托管（/ /dashboard /metrics）
│   ├── dashboard.html       # 监控大屏页面（配套服务，数据从知识库实时读取）
│   ├── store/               # 存储底座（SQLite + 共现表 + metrics 埋点表 + conflicts 待裁决表 + tooltraces 轨迹表）
│   ├── index/               # BM25F 倒排索引 + 词典分词
│   ├── graph/               # 关联图 + PageRank + Personalized PR（纯 Python）
│   ├── code/                # AST 符号索引（代码代工）
│   ├── skills/              # SKILL.md 解析（技能代管）
│   └── retrieval/           # 混合召回 + 重排
├── code_repo/               # 示例代码仓库
├── skills_repo/             # 示例技能目录
└── tests/                   # 单元测试
```

## 快速开始

```bash
# 零模型内核：零依赖，直接运行
python demo.py

# 运行测试（76 个，全部零依赖）
python -m unittest discover -s tests -v

# 一步到位部署（知识库 + 中间层 + 监控大屏 + 埋点）
python -m molbase.server            # 监控大屏: http://127.0.0.1:8787/

# 接本地大模型（可选）：先启动本地推理服务，再改 config
#   Ollama 启动 Qwen3.5 9B 后，修改 molbase/config.py：
#     LLM_ENABLED  = True
#     LLM_MODEL    = "qwen3.5-9b"         # 模型名即版本指纹
#     LLM_ENDPOINT = "http://localhost:11434/v1"
# 向量模型同理参数化（可选）：EMBEDDING_BACKEND = "sentence-transformers"
```

## 核心接口

```python
from molbase.orchestrator import Orchestrator

orc = Orchestrator()                              # 纯零模型内核

# 摄入知识 / 代码 / 技能（ingest 带冲突四态，质量底线输入闸门）
res = orc.ingest_knowledge("id", "标题", "内容", source="knowledge_repo/x.md")
print(res.state)  # new | merge | duplicate | conflict | pinned（宁可少存不错存）
orc.ingest_code_repo("code_repo/")
orc.ingest_skill_dir("skills_repo/")

# 查询（唯一信息获取入口，自动命中热度 + 共现记录；LLM 可用时先改写查询提升召回）
resp = orc.query("人脸识别的代码方法")   # -> 路由 + 最小上下文块（resp.rewritten 为改写后查询）
if resp.abstain:                       # 质量底线输出闸门：置信不足则宁可不答也不错答
    print("暂无可靠答案，confidence =", resp.confidence)

# 反馈（驱动热度/新鲜度，越用越准）
orc.feedback("entry_id", "adopt")
orc.feedback("entry_id", "reject")       # 负反馈

# 上下文压缩（LLM > TextRank > 截断 三级降级）
summary = orc.summarize(resp.results[0].content, max_chars=200)

# 手动触发一次自我进化（正常 query 会自动节流触发）
orc.evolve()

# 网络搜索兜底：内部命中弱时，resp.web_results 有外部结果；采纳即沉淀为内部知识
resp = orc.query("一个内部知识库没有的主题")
if resp.web_results:
    orc.adopt_web_result(resp.web_results[0])   # 搜索 -> 沉淀 -> 复用

# token 埋点：本次返回 token vs 全表扫描基线 token
print(resp.token_estimate, resp.baseline_tokens, resp.content_chars)

# 会话回写：把整段会话提炼成一份核心资产沉淀回知识库（LLM 优先、TextRank 兜底）
harvest = orc.harvest_session("sess-20261118", "整段会话记录文本……")
print(harvest.entry_id, harvest.state, harvest.engine)  # session::sess-20261118 new textrank
print(harvest.related_fingerprints)  # 关联的同会话操作指纹（结论↔过程，不靠反推）

# 工具调用轨迹沉淀：观察大模型实际调用了哪些工具，相同操作模式累计达阈值
# 自动生成可复用 skill（实现 N+1 -> N 的 token 收敛，越用越聪明）
trace = orc.record_tooltrace("sess-20261118", "检索并更新文档", [
    {"tool": "web_search", "args": {"query": "数据库索引"}},
    {"tool": "file_read", "args": {"path": "/docs/index.md"}},
    {"tool": "file_edit", "args": {"path": "/docs/index.md", "content": "更新"}},
])
print(trace.fingerprint, trace.count, trace.generated)
# web_search(query) > file_read(path) > file_edit(content,path) 3 True
# 第 3 次自动生成 skill::auto-web-search-file-read-file-edit 并写回 skills_repo
```

## 越用越顺手：自我进化闭环

- 命中 +1、采纳 +3、引用 +2、拒绝 -3（`config.FEEDBACK_WEIGHT_*`）
- query 自动命中热度 +1（`config.AUTO_HIT_HEAT`）
- 共现 ≥ 3 次自动建关联边（`config.COOCCUR_THRESHOLD`）
- 共现建边仅限同类型条目，减少跨类型噪音（`config.COOCCUR_SAME_TYPE_ONLY`）
- 查询改写：LLM 可用时口语化查询改写为检索友好关键词（`config.LLM_REWRITE`）
- 冷知识超 90 天热度减半（`config.COLD_DECAY_DAYS` / `COLD_DECAY_FACTOR`）
- 负反馈 reject 累计 ≥ 3 次自动隔离（`config.QUARANTINE_REJECT_THRESHOLD`）
- 死知识超 180 天未访问且低热度自动归档（`config.ARCHIVE_DAYS` / `ARCHIVE_HEAT_MAX`）
- 工具调用模式自进化：同一操作指纹累计 ≥ 3 次自动生成可复用 skill（`config.TOOL_TRACE_GENERATE_THRESHOLD`）
- 冷启动精度问题是「数据库统计信息冷启动」的必经阶段，跑久了越用越准。

## 网络搜索：内部优先、外部兜底

网络搜索接在**代工层的「混合召回」环节**，作为内部 BM25F/图检索之外的第三个召回源：

| 定位 | 说明 |
| --- | --- |
| 内部知识库 | 权威沉淀，**优先**（有热度/共现/采纳反馈，索引定位降 token） |
| 网络搜索 | 实时兜底，**补充**（内部 top-1 得分低于阈值才触发，默认关闭） |
| 采纳沉淀 | `adopt_web_result` 把采纳的网络结果写入内部知识，下次直接命中、不再联网 |

「搜索 → 沉淀 → 复用」让网络搜索从「每次必查」退化为「冷知识/新知识的临时补充」——
**越用越少依赖网络**。启用方式：`config.WEB_SEARCH_ENABLED = True` + 指向 SearXNG 兼容后端。

## token 埋点（粗估，非实测）

每次 `query` 响应附带 `token_estimate`（本次返回内容粗估 token）、`baseline_tokens`
（传统「全表扫描」读全部条目的粗估 token）、`content_chars`（返回字符数）。口径：
中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token（见 `tokens.py`）。用于量化「降 token」效果，
真实数字需接入真实场景埋点后校准。

**持久化埋点（部署后实时监控）**：每次 query 额外写入知识库 `metrics` 表（重启不丢），
由 `/metrics` 接口聚合透出，监控大屏 `/` 每 3 秒刷新可视化——这是「基于知识库做的应用 demo」，
既反映 token 优化结果，也验证「传统数据结构项目（表 + 存储过程 + 监控台）」可用
「知识库 + 中间层 + 大屏」替代。
