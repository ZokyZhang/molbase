# MolBase 与 GitHub 高星近似项目对比

> 日期：2026-11-18
> 前提：MolBase 已落地质量底线（abstain/confidence + 冲突四态 + PII 脱敏 + reject 隔离 + 死知识归档），
>       本对比基于「更新后的 MolBase 框架」与 GitHub 上 star 最高、定位最接近的记忆/知识中间层项目。

## 结论先行

高星近似项目全都在做「给 agent 喂最小上下文、降 token」，但走两条相反的路：

- **高星项目**：靠「LLM 抽取事实 + 向量/图库」堆生产级记忆能力；
- **MolBase**：靠「零模型确定性算法」做自足内核 + 代工边界。

高星项目的价值在「记忆的工程化与生态」，MolBase 的价值在「零依赖的克制与代工边界」。

## 一、五个高星近似项目（按 star，2026-11 第三方口径）

| 项目 | star | 定位 | 技术路线 |
|---|---|---|---|
| mem0（mem0ai/mem0） | ~60K | AI 记忆层 | LLM 抽取事实 → 向量 + 图（Mem0g） |
| Cognee（topoteretes/cognee） | ~31K | 知识图谱记忆引擎 | remember/recall/improve/forget |
| Graphiti（getzep/graphiti） | ~30K | 时序知识图谱 | bi-temporal（真实时间线 + 摄入时间线） |
| Letta/MemGPT（letta-ai/letta） | ~25K | 记忆 OS | Core/Recall/Archival 三层 + 自编辑 |
| Basic Memory（basicmachines-co） | ~4K | 本地记忆 | 本地 Markdown KG + MCP |

## 二、深度对比（7 维度）

| 维度 | MolBase | 高星记忆项目 |
|---|---|---|
| 模型依赖 | 零模型自足（哈希/TF-IDF + 规则 + TextRank） | 强依赖 LLM + embedding/向量库 |
| 第三方依赖 | 零（标准库 + SQLite） | 重（Qdrant/Neo4j + 16~20 后端） |
| 定位层级 | 知识代工中间层（检索/代码/技能代工） | 纯记忆层/记忆 OS |
| 存储 | SQLite 单库（entries/cooccur/conflicts/metrics） | 向量库 + 图数据库 |
| 检索 | BM25F + 图 + 语义余弦 | 向量相似 + 图遍历 + 实体匹配 |
| 进化/学习 | 热度 + 共现建边 + 冷衰减（统计信息级） | LLM 抽取 + 冲突检测 + 时序失效 |
| 质量底线 | abstain + 冲突四态 + PII 脱敏 + reject 隔离 + 归档 | 冲突检测/去重有，无 abstain |

## 三、相似点

1. 都在解决「AI 写/记快、上下文贵」，做「抽取最小上下文喂 agent」。
2. 都有「写入冲突 + 衰减遗忘」（mem0 冲突+decay、Graphiti 时序失效、Cognee forget ↔ MolBase 冲突四态+冷衰减+归档）。
3. 都通过 MCP 接入 agent 生态。
4. 都强调溯源/可审计（Graphiti provenance、mem0 audit ↔ MolBase entry_id+source）。

## 四、不同点（护城河）

1. 模型依赖是分水岭：五个高星项目没一个能离了 LLM 跑，MolBase 是唯一零模型零依赖。
2. 定位边界不同：它们只做记忆，MolBase 做代工，多出代码代工（AST）+ 技能代管（SKILL.md）。
3. 进化机制不同：它们靠 LLM 抽取事实，MolBase 靠数据库统计信息（热度/共现），零模型可解释可回滚。
4. 降 token：MolBase 是一等公民（量化埋点），它们只是副产品。

## 五、可借鉴（按优先级）

| 优先级 | 借鉴点 | 来源 |
|---|---|---|
| P0 | MCP server 入口 | Basic Memory / Graphiti / mem0 |
| P0 | 时序失效（valid_from/valid_to） | Graphiti |
| P1 | 显式 forget/improve 语义 | Cognee |
| P1 | metadata filtering（检索时按元数据过滤） | mem0 |
| P2 | 闲时重组（sleep-time compute） | Letta |

## 六、MolBase 独有（他们没有的）

1. 零模型自足内核
2. 自我进化闭环（统计信息级）
3. 降 token 量化埋点
4. 代码代工 + 技能代管
5. abstain「宁可不答不错答」（零依赖实现）

## 结论

MolBase 在「零依赖 + 代工边界 + 降 token 量化」上独一份，短板在「生态接入（MCP）+ 质量底线」；
后者已补齐，前者（MCP）建议从 P3 提前到下一步。
