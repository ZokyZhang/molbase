# 迭代记录（CHANGELOG）

> 每版写清「为什么改、改了什么、实测结果、否决了什么」。
> 深度报告在 `reports/`（.gitignore 内，本地保留），此处保留可在 git 里追溯的结论。

---

## v9.2（2026-11-24）降弃答 + 知识库四层升级

### 目标
弃答率 52% → <20%；置信度从「普遍 0.2x」抬到可用；树/网/标签/原文四层从空表变成真结构。

### 改动
| 模块 | 内容 |
|---|---|
| `config.py` | logistic 校准（`CONF_MID` 按后端标定、`CONF_SLOPE` 分后端）；`ABSTAIN_CONFIDENCE` 0.55→0.50→**0.43**；新增 `LEX_COV_*`、`CHUNK_*`、`LAYER_BOOTSTRAP_ON_START` |
| `orchestrator.py` | `_compute_confidence` 重写（logistic 主项 + 乘性加成 + 词法覆盖率下限）；`_judge_abstain` 两阶段闸门，子因按中点派生；`_lexical_coverage`；`_rerank_ranked` 走预计算向量；`_apply_chunk_boost`（默认关）；`_collect_refs`；ingest 增量补向量/切块 |
| `store/__init__.py` | 新增 `vector_index` / `chunks` / `tag_vocab` 表；`upsert_vectors` 批量提交；`delete_vectors` 禁止无条件全表删；`_unpack_vec` 长度校验；`rebuild_topic_tree` / `tree_children` / `tree_path`；`cooccur_pmi_pairs` |
| `retrieval/__init__.py` | `rerank` 支持 `content_limit` 与 `semantic_text` 统一口径；修非批量后端把字符串当向量算余弦的 bug；新增 `rerank_with_vectors`（显式归一化 + 与 `rerank` 同窗口） |
| `vectorize.py`（新） | 条目向量预计算（`semantic_text` / `model_fingerprint` / `ensure_entry_vectors` 增量） |
| `chunking.py`（新） | Markdown 切块：标题栈路径快照 + 代码围栏识别 + 段落贪心装箱 + 超长硬切 overlap |
| `kb_layers.py`（新） | 四层加工：切块入库、块向量、受控标签词表、两层主题树、`rebuild_layers` |
| `server.py` | `/query` 响应补 `refs`（原文定位） |
| `deploy-lan/` | `backfill_vectors.py`（离线回填，后端显式对齐 systemd）；`update_vm.py` 停服→回填→启服，文件清单补 3 个新模块 |
| `bench_abstain.py`（新） | 弃答率回放台，同一批查询做前后对照 |

### 实测
| 指标 | 旧 | 新 |
|---|---|---|
| 弃答率（100 条历史查询回放） | **52.0%** | **13.0%** |
| conf 均值 / 上限 | 0.526 / **0.786（公式封顶）** | 0.652 / 0.921 |
| 回放耗时 | 147s | 51s |
| 单测 + 黑盒 | — | 135/135 OK |
| 反向测试 | — | 34/34 PASS |
| bandit | — | High 0 / Medium 12（历史 SQL 拼接 + 固定 urlopen） |

四层规模（VM）：`chunks` 15269 / 905 条目（均值 16.9 块）、`topic_tree` 1462 节点 / 93 根、`tag_vocab` 92 标签。

### 被实测否决的改法
1. 语义窗口 300 → 1500 字：策略类问题 top1 从第 4 掉到第 8，已回退。
2. 长文档块级相似度补强：5 条金标 conf 全线下滑、1 条由放行变弃答，`CHUNK_BOOST_ENABLED=False`。
3. BM25 原始分作词法下限：「给我讲个冷笑话」5.63 > 「MolBase 架构」4.89，跨查询不可比 → 改 idf 归一化覆盖率。
4. 启动时同步建四层：堵住端口致 `RemoteDisconnected`；改后台线程又长期占用 SQLite 连接致测试 `WinError 32` → 最终改离线回填 + 显式开关。


### 第二轮评审修复（同版本内追加）
| 修复 | 证据 |
|---|---|
| `semantic_text` 摘要不参与截断 | 1 条条目 summary 3135 字 → 实际返回 3181 字（预算 300 的 10.6 倍），成了吸附点；修后全库最长 = 300 |
| 后端故障静默全放行 | `rerank` 异常分支用 BM25 原始分（>1）clamp 成 1.0 → conf≈0.98 放行噪音；改为 0.0 显式弃答 |
| v9.x 参数 env 回滚失效 | 常量写在 `_apply_overrides` 之后，`MOLBASE_VECTOR_PRECOMPUTE_ENABLED=false` 无效；模块末尾幂等再应用 |
| 共现网 PMI 恒返回空 | 用 `tag_df`（标签名）当边际，但端点是 entry_id → 823 条边全丢；改度数归一，`learn()` 真正消费，保留 543/823 |
| 引用定位偏移 | `_collect_refs` 固定取 chunks[0]，长文档答案在后段；改按查询词命中数选块 |
| 树静默丢 52% 节点 | `max_children=200` 把 1462 砍成 705；默认不截断 |
| 微块污染块向量 | 标题行也算了向量；加 `CHUNK_VECTOR_MIN_CHARS=120` |
| 标签归一化两套口径 | store 侧不 lower → df 被拆散；统一 |
| 围栏混用提前闭合 | 单一布尔；改为按标记闭合 |
| `CONF_MID_CURRENT` 死变量 | 全仓无引用，且是"按 env 缓存中点"事故源；删除 |
| chunk boost 死代码 | 改的是置信度输入 `sim_norm`；改成 `raise`，默认关的布尔值防护为零 |

修复后复测：135/135 单测黑盒、34/34 反向测试、bandit High 0、弃答率回放 **16.0%**。

未做（已写明原因）：`CONF_MID` 按 `model_fp` 标定（需 ≥200 条标注集）；块向量加 `path`（需与 bge 回填一起做）。

### 已知局限
- 噪音泄漏 4/12（关键词撞词面），hash 词袋无法区分「词面像」与「真相关」，需 bge-m3。
- 块向量已算齐但未参与打分，仅用于 `refs` 溯源。
- conf 口径变更，历史 metrics 需按 `embedding_backend` / `confidence_threshold` 分段对比。

---

## v8.0（历史）飞轮传动修复
接上传动轴（feedback 触发进化）、刹车生效（阈值拦截）、反馈自锁解除。详见 `reports/MolBase-v8.0飞轮传动修复-验收报告.md`。

## v7.2（历史）三方向资产落地
场景锚定三表灌数、技能隐式复用、弃答补数草稿闭环。

## v7.0 / v7.1（历史）自进化闭环
自动正反馈、反馈触发进化、轨迹归因；测试契约对齐 V2 → 当前设计。

## v6（历史）全量明细监控 + 批量 embed 降延迟
22 维明细监控、9 维分组对比、批量推理。
