# -*- coding: utf-8 -*-
"""
MolBase（Mol = Model 模型 × Base 底座/知识库）全局配置。

设计原则（对齐用户约定）：
- 所有参数硬编码，不使用 `${}` 占位符，不加密。
- 底座 = 存储引擎；代工层 = 查询优化器 + 执行计划；大模型 = 计算算子。
- 所有方法及代码块需注释。
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置（硬编码）
# ---------------------------------------------------------------------------
# 包目录：本文件所在目录（molbase/）
PACKAGE_ROOT = Path(__file__).resolve().parent
# 项目根目录：molbase/ 的上一级（含 code_repo、skills_repo、data 等）
PROJECT_ROOT = PACKAGE_ROOT.parent

# 数据目录：存储 SQLite 数据库、索引缓存等
DATA_DIR = PROJECT_ROOT / "data"

# SQLite 数据库文件路径
DB_PATH = DATA_DIR / "molbase.db"

# 技能目录：存放 SKILL.md 技能包（按 agentskills.io 标准组织）
SKILLS_DIR = PROJECT_ROOT / "skills_repo"

# 知识库内容根目录：存放知识文档（markdown 等）
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge_repo"


# ---------------------------------------------------------------------------
# 检索与重排参数（硬编码）
# ---------------------------------------------------------------------------
# BM25 超参数：b 控制文档长度归一化强度（0=完全不归一，1=完全归一）
BM25_B = 0.75
# BM25 超参数：k1 控制词频饱和速度（越大越接近线性）
BM25_K1 = 1.5

# BM25F 字段权重：标题命中权重高于正文（标题更能表达条目主题）
BM25_FIELD_WEIGHT_TITLE = 3.0
BM25_FIELD_WEIGHT_CONTENT = 1.0

# 混合召回候选池大小：BM25 与图检索各自取 top-N 再合并
RECALL_TOP_N = 20

# 最终返回给大模型的最小上下文条数
FINAL_TOP_K = 5

# 重排权重（热度 / 新鲜度 / 语义相关度，三者之和为 1.0）
RERANK_WEIGHT_HEAT = 0.05  # 历史热度权重（进一步降低，避免热门旧内容压过精准语义）
RERANK_WEIGHT_FRESH = 0.10  # 新鲜度权重（降低，避免新发布但低质内容干扰）
RERANK_WEIGHT_SIM = 0.85  # 语义相似度权重（进一步提高，精准知识点优先于热度）

# 设备加权：检索时本机（meta.device_mac 匹配）条目的加分系数，
# 叠加在归一化综合分之上（软加权：本机优先，不排除其他设备）。
DEVICE_BOOST = 0.15

# 热度信号权重（命中 +1、采纳 +3、引用 +2）
FEEDBACK_WEIGHT_HIT = 1
FEEDBACK_WEIGHT_ADOPT = 3
FEEDBACK_WEIGHT_CITE = 2
# 负反馈：明确拒绝某条目时热度下降（自然沉底）
FEEDBACK_WEIGHT_REJECT = -3

# 新鲜度半衰期（天）：每过半衰期，新鲜度信号衰减一半
FRESHNESS_HALF_LIFE_DAYS = 30.0


# ---------------------------------------------------------------------------
# 自我进化参数（零模型，全自动，硬编码）
# ---------------------------------------------------------------------------
# query 命中的条目自动累加的热度增量（弱信号，用得多自然热）
AUTO_HIT_HEAT = 1.0

# 共现建边阈值：两个条目在同一查询中共同命中达到该次数后，自动建立关联
COOCCUR_THRESHOLD = 3

# 类型感知共现建边：True = 仅同类型条目（知识↔知识、代码↔代码、技能↔技能）自动建边，
# 减少跨类型噪音关联（知识误连技能等）；False = 允许跨类型关联。
COOCCUR_SAME_TYPE_ONLY = True

# 自动学习节流：每 N 次查询触发一次进化步进（共现建边 + 冷知识衰减）
EVOLVE_INTERVAL = 10

# 冷知识判定：超过 N 天未被访问的条目视为冷知识
COLD_DECAY_DAYS = 90.0

# 冷知识衰减因子：每触发一次进化，冷知识热度乘以该因子（自然下沉）
COLD_DECAY_FACTOR = 0.5

# 工具调用模式自进化：同一「操作指纹」（工具名+参数键签名）累计达到该次数后，
# 自动生成可复用 skill，实现 N+1 -> N 的 token 收敛（大模型不再反复自创同一套操作）。
TOOL_TRACE_GENERATE_THRESHOLD = 3

# 指纹最少工具调用数：少于该数的调用序列不视为「可复用流程」，仅记录轨迹、不自动生成 skill。
# （单工具偶发调用无需固化成流程；2 个及以上工具组成的序列才是值得复用的操作模式。）
TOOL_TRACE_FINGERPRINT_MIN_LEN = 2


# ---------------------------------------------------------------------------
# 质量底线参数（硬编码）
#   原则：宁可不答，不可错答；宁可少存，不可错存。
#   落成「四道防线 + 两道闸门」：输出 abstain 闸门 + 输入冲突闸门。
# ---------------------------------------------------------------------------
# 输出闸门：综合置信度（confidence）低于该阈值时，/query 返回 abstain=true，
# 即「宁可不答也不错答」，调用方应据此明确说「不知道」而非硬答。
# 注意：confidence 基于「未归一化的原始语义余弦」（hash 下真命中 >0.5、假命中 ~0），
#       阈值按当前 embedding 后端校准；换 sentence-transformers 等后端需重调。
ABSTAIN_CONFIDENCE = 0.30

# 弃答归因分档阈值（可解释反馈：为什么拦截、怎么整改）。
# - strength < NO_MATCH_STRENGTH         ：语义几乎不相关，等价「没数据」→ 补数据；
# - NO_MATCH_STRENGTH <= strength < WEAK_STRENGTH_THRESHOLD ：弱相关但不准 → 补更精准条目；
# - strength >= WEAK_STRENGTH_THRESHOLD 且 conf < 阈值       ：多条太像、区分度不足 → 去重/细化。
NO_MATCH_STRENGTH = 0.15

# 弃答归因二级阈值：当 top1 的原始语义余弦（strength）低于该值时，
# 弃答归因为 weak_strength（召回内容与查询相关性弱）；否则归因为
# low_margin（top1/top2 区分度不足）。用于「为什么拦截、怎么整改」的可解释反馈。
WEAK_STRENGTH_THRESHOLD = 0.35

# 输入闸门：写入冲突判定参数（硬编码）。
# - 内容相似度 >= CONFLICT_DUPLICATE_SIM：视为近重复（duplicate），跳过写入；
# - 同 ID 且来源不同（source 字段变了）：视为可能矛盾（conflict），
#   不覆盖、进入待裁决（「宁可少存，不可错存」）；
# - 同 ID 且来源相同：视为同源更新（merge），正常覆盖。
CONFLICT_DUPLICATE_SIM = 0.85

# 输出前 PII 脱敏开关：命中内容里的手机号/邮箱/身份证/银行卡/密码 token 自动打码。
SANITIZE_ENABLED = True

# 负反馈隔离：某条目被 reject 累计达到该次数后自动 quarantine（隔离，不参与召回）。
QUARANTINE_REJECT_THRESHOLD = 3

# 死知识归档：超过该天数未被访问、且热度 <= ARCHIVE_HEAT_MAX（或已 quarantine）的
# 条目自动归档（不参与召回，但保留在库，可人工恢复）。
ARCHIVE_DAYS = 180.0
ARCHIVE_HEAT_MAX = 0.0


# ---------------------------------------------------------------------------
# 向量化（embedding）后端配置（硬编码）
# ---------------------------------------------------------------------------
# 后端名：
#   "hash"                  零模型确定性哈希向量（默认，离线无大模型版，永不失效）
#   "sentence-transformers" 本地 embedding 模型（增强，失败自动降级回 hash）
EMBEDDING_BACKEND = "hash"

# 哈希向量的维度（仅 backend="hash" 时生效）
EMBEDDING_HASH_DIM = 256

# 本地向量模型名（仅 backend="sentence-transformers" 时生效）
# 说明：模型名即「版本指纹」——换模型 / 升版本只需改这一个参数；
#       模型加载失败会自动降级回 hash，绝不因版本匹配不上而失效。
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# 本地大模型（LLM）配置（V2 接本地大模型，硬编码）
# ---------------------------------------------------------------------------
# 总开关：False = 纯 V1 离线无大模型（零网络调用）；True = 启用本地 LLM 增强。
LLM_ENABLED = False

# 本地推理服务地址（OpenAI 兼容，Ollama/vLLM/llama.cpp 通用）：
#   Ollama  默认 http://localhost:11434/v1
#   vLLM    默认 http://localhost:8000/v1
# 注意：不含末尾的 /chat/completions，代码会自动拼接。
LLM_ENDPOINT = "http://localhost:11434/v1"

# 本地大模型名（即「版本指纹」，换模型/版本只改这一处）。
# 例：Ollama 中 Qwen3.5 9B 通常为 "qwen3.5-9b"，请按实际模型名填写。
LLM_MODEL = "qwen3.5-9b"

# 单次请求超时（秒）
LLM_TIMEOUT = 30.0

# 采样温度：0.0 表示确定性输出，降低幻觉（路由/结构化任务推荐 0.0）
LLM_TEMPERATURE = 0.0

# 单次生成 token 上限（路由/摘要够用）
LLM_MAX_TOKENS = 512

# 查询改写开关：LLM 启用时，是否用本地大模型把口语化查询改写成检索友好关键词。
# True = 启用改写（提升召回，多一次 LLM 调用）；False = 关闭（直接用原查询检索）。
LLM_REWRITE = True

# 本地 LLM 协议：ollama-native（Ollama 原生 /api/chat，自动关思考 think=false，
# qwen 等思考模型必须用此协议，否则 reasoning 吞 token 导致正文为空/不压缩）|
# openai（OpenAI 兼容 /v1/chat/completions，vLLM/llama.cpp/自建网关通用）。
LLM_PROTOCOL = "ollama-native"

# 摘要用 LLM 的输入长度上限（字符）：超过此长度的文本跳过 LLM 摘要，
# 直接走去冗余兜底（保留原文 + 指针），避免超上下文窗口被截断丢语义。
# 中文约 1 字 ≈ 1 token，12000 字远低于 qwen 32k 窗口，留足安全余量。
LLM_MAX_CONTEXT_CHARS = 12000


# ---------------------------------------------------------------------------
# 网络搜索外部源适配器（硬编码，可插拔增强）
# ---------------------------------------------------------------------------
# 总开关：False = 不联网（纯内部知识库，默认）；True = 内部召回不足时触发网络兜底。
WEB_SEARCH_ENABLED = False

# 搜索后端地址（SearXNG 兼容 JSON 接口，不含 /search 路径）。
# 本地/自建 SearXNG 默认 http://localhost:8888；可替换为任意兼容接口。
WEB_SEARCH_ENDPOINT = "http://localhost:8888"

# 单次网络搜索超时（秒）
WEB_SEARCH_TIMEOUT = 10.0

# 网络搜索返回条数（兜底补充，不宜过多，避免 token 膨胀）
WEB_SEARCH_TOP_N = 3

# 网络搜索触发阈值：内部召回 top-1 的 BM25 原始得分（rerank 前）低于此值才触发网络兜底。
# 用原始得分而非归一化分——归一化后「只要有命中」得分就接近 1，无法反映匹配真实强度。
# 兜底语义下阈值偏低（宁可多搜不漏搜）：小语料真命中约 2.3~10，假命中约 1.0~1.3；
# 大语料 IDF 变大、得分整体抬升，可按实际语料规模调参。
WEB_SEARCH_TRIGGER_SCORE = 2.0


# ---------------------------------------------------------------------------
# Token 埋点参数（硬编码，估算用）
# ---------------------------------------------------------------------------
# 估算口径说明见 molbase/tokens.py：中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token。
# 是否在 query 响应中附带 token 埋点（token_estimate / baseline_tokens / content_chars）。
TOKEN_TRACE = True


# ---------------------------------------------------------------------------
# 语义保持压缩参数（硬编码）
# ---------------------------------------------------------------------------
# 总开关：False 即回退现状（query 返回全文、不生成摘要），一键回滚
COMPACT_ENABLED = True
# 压缩模式：semantic（语义保持，默认）| lossy（现有有损：LLM生成式/TextRank/截断）
COMPACT_MODE = "semantic"
# 摘要上限（字符），超出继续压缩
SUMMARY_MAX_CHARS = 500
# 去冗余相似度阈值：两段 SimHash 相似度 >= 该值视为重复
DEDUP_SIM_THRESHOLD = 0.90
# /expand 按需展开的 LRU 缓存大小
EXPAND_CACHE_SIZE = 128
# 批量回填存量的单批大小
BACKFILL_BATCH = 50


# ---------------------------------------------------------------------------
# 成本折算参数（硬编码，性能报告用）
# ---------------------------------------------------------------------------
# 本地大模型「免费」（仅电费 / 算力），为量化「降本」，按「若上云等效单价」折算。
# 单位：人民币元 / 每百万 token；区间参考主流商用大模型定价，可自行调整。
# 说明见 molbase/cost.py：输入与输出单价分开计价（生成通常更贵）。
COST_INPUT_PER_1M = 2.0     # 输入 token 单价（元 / 百万 token）
COST_OUTPUT_PER_1M = 8.0    # 输出 token 单价（元 / 百万 token）
COST_CURRENCY = "¥"         # 成本展示货币符号


# ---------------------------------------------------------------------------
# 服务化 HTTP API 参数（硬编码默认值，可被外部配置覆盖）
# ---------------------------------------------------------------------------
# 服务监听地址：127.0.0.1 仅本机；0.0.0.0 开放局域网供其他应用调用。
SERVER_HOST = "127.0.0.1"

# 服务监听端口
SERVER_PORT = 8787

# 可选访问令牌：非空时，客户端须带 Authorization: Bearer <token> 才能调用。
# 空字符串 = 不鉴权（默认，适合受信局域网；公网部署务必设置）。
SERVER_AUTH_TOKEN = ""


# ---------------------------------------------------------------------------
# 配置外部化（可配置，覆盖上面的默认值）
# ---------------------------------------------------------------------------
# 优先级（高 → 低）：环境变量 > 外部配置文件（JSON）> 本文件默认值。
# - 环境变量：MOLBASE_ 前缀 + 常量名，如 MOLBASE_LLM_ENDPOINT、MOLBASE_SERVER_PORT、MOLBASE_DB_PATH。
# - 配置文件：JSON 扁平 key 直接对应常量名，如 {"LLM_ENDPOINT": "...", "SERVER_PORT": 8787}。
#   默认读取 PROJECT_ROOT/config.json，也可用环境变量 MOLBASE_CONFIG 指定路径。
# 设计原则：零第三方依赖（json + os.environ），无配置文件时行为与硬编码完全一致。

import json as _json
import os as _os


def _config_file_path() -> str | None:
    """返回外部配置文件路径：优先 MOLBASE_CONFIG，其次 PROJECT_ROOT/config.json。"""
    env_path = _os.environ.get("MOLBASE_CONFIG")
    if env_path:
        return env_path
    default = PROJECT_ROOT / "config.json"
    return str(default) if default.exists() else None


def _coerce(value: str, target):
    """把字符串/原始值转换为目标类型（按现有默认值的类型推断）。"""
    if isinstance(target, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float):
        return float(value)
    if isinstance(target, Path):
        return Path(str(value))
    return value  # 字符串等原样返回


def _load_external_overrides() -> dict:
    """
    读取外部配置（配置文件 + 环境变量），返回覆盖字典（key 对应常量名）。

    - 配置文件：JSON，扁平 key 映射到常量名。
    - 环境变量：MOLBASE_ 前缀，如 MOLBASE_SERVER_PORT -> SERVER_PORT。
    两者都不存在时返回空 dict（行为 = 纯硬编码默认值）。
    """
    overrides: dict = {}
    cfg_path = _config_file_path()
    if cfg_path:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            if isinstance(data, dict):
                overrides.update(data)
        except Exception:
            # 配置文件缺失 / 格式错误 -> 忽略，回退默认值，绝不崩溃
            pass
    # 环境变量优先级更高：MOLBASE_ + 常量名
    for key, val in _os.environ.items():
        if key.startswith("MOLBASE_") and len(key) > len("MOLBASE_"):
            overrides[key[len("MOLBASE_"):]] = val
    return overrides


def _apply_overrides(overrides: dict) -> None:
    """把覆盖值写回模块级常量（类型按默认值自动转换，未知 key 忽略）。"""
    g = globals()
    for key, val in overrides.items():
        if key in g and not key.startswith("_"):
            try:
                g[key] = _coerce(val, g[key])
            except (TypeError, ValueError):
                # 类型转换失败 -> 保留默认值，绝不崩溃
                pass



# ============================================================
# v3 整改配置（P0：语义后端切换 + 阈值重标 + 场景锚定层）
# 本块放在 _apply_overrides 之前，确保外部覆盖仍作用于全部常量。
# ============================================================
# --- 向量后端配置 (Ollama + bge-m3) ---
EMBEDDING_BACKEND = "ollama"
OLLAMA_EMBED_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_EMBED_MODEL = "bge-m3"
OLLAMA_EMBED_DIM = 1024
OLLAMA_EMBED_TIMEOUT = 20.0
# 语义文本窗口（唯一口径）：标题+摘要完整保留，正文只取前 N 字。
# 现场精排（rerank）、预计算入库（vectorize）、预计算重排（rerank_with_vectors）三处必须同口径，
# 否则同一对「查询-条目」在两条路径下会算出不同相似度，导致排序抖动。
# 实测 300 优于 1500：放宽到 1500 会让泛化型大文档压过精准小文档（策略类问题 top1 从第4掉到第8）。
SEMANTIC_SIM_CONTENT_CHARS = 300
# v6 性能：语义精排只对 BM25 召回池的前 N 条做向量推理（CPU 上每条约 16ms）。
# 40 条全量精排需 600ms+；16 条虽快但会漏掉排名靠后的真实命中（实测 golden 6/8），
# 折中 32 条：golden 8/8 且延迟约 400-500ms。未进窗口者语义分记 0，不用 BM25 冒充。
SEMANTIC_RERANK_LIMIT = 32

# --- 阈值重标 (适配 bge-m3 值域) ---
ABSTAIN_CONFIDENCE = 0.55
NO_MATCH_STRENGTH = 0.30
WEAK_STRENGTH_THRESHOLD = 0.50

# --- 置信度融合权重 (动态归一化，只纳入有信号的项) ---
CONF_W_STRENGTH = 0.55
CONF_W_MARGIN = 0.15
CONF_W_TAG = 0.15
CONF_W_SCENE = 0.10
CONF_W_ENTITY = 0.05

# --- 场景锚定层 ---
SCENE_DEFAULT = "global"
SCENE_MATCH_WEAK = 0.30  # 条目场景与查询场景不符时的弱匹配分
SCENE_RELAX_MIN = 2       # 场景硬过滤后候选少于该值时 relax 到 global
ENTITY_ANCHOR_BOOST = 0.10
SEMANTIC_CACHE_SIM = 0.92 # 语义缓存命中阈值
TURNSTATE_MAX_TOKENS = 50 # 多轮指代状态预算

# --- 标签策略 ---
TAG_FILTER_MODE = "or_boost"  # 由后置AND减法改为前置OR召回+boost
TAG_BOOST = 0.20

# --- 数据生命周期 (淘汰加速，适配6天龄系统) ---
COLD_DECAY_DAYS = 14.0
ARCHIVE_DAYS = 30.0

_apply_overrides(_load_external_overrides())


# ============================================================
# v8.0 飞轮修复配置（放在 _apply_overrides 之后，按最终 EMBEDDING_BACKEND 标定阈值）
#
# 根因：不同 embedding 后端相似度值域差异极大（hash 余弦普遍 0.2~0.6，
# bge-m3 0.4~0.85），用同一套阈值服务两个引擎必然「一套过松、一套过严」，
# 导致弃答率随后端漂移（实测 hash 期弃答 48.4% vs ollama 期 27.2%）。
# ============================================================
_ENGINE_THRESHOLDS = {
    "hash": {"abstain": 0.35, "no_match": 0.18, "weak": 0.30},
    "ollama": {"abstain": 0.55, "no_match": 0.30, "weak": 0.50},
    "sentence-transformers": {"abstain": 0.45, "no_match": 0.25, "weak": 0.40},
}
_engine_name = str(EMBEDDING_BACKEND).strip().lower()
_engine_thr = _ENGINE_THRESHOLDS.get(_engine_name, _ENGINE_THRESHOLDS["ollama"])
# 环境变量显式指定阈值时尊重用户值，否则按后端自动标定
if "MOLBASE_ABSTAIN_CONFIDENCE" not in _os.environ:
    ABSTAIN_CONFIDENCE = _engine_thr["abstain"]
if "MOLBASE_NO_MATCH_STRENGTH" not in _os.environ:
    NO_MATCH_STRENGTH = _engine_thr["no_match"]
if "MOLBASE_WEAK_STRENGTH_THRESHOLD" not in _os.environ:
    WEAK_STRENGTH_THRESHOLD = _engine_thr["weak"]

# 反馈触发线：永远低于弃答线，保证「答了就会反馈」，
# 杜绝旧版 0.6 > 0.55 造成的 0.55~0.6「答而不馈」空档（飞轮自锁根因之一）。
FEEDBACK_MIN_CONFIDENCE = round(ABSTAIN_CONFIDENCE * 0.85, 4)

# 热度权重提升：旧值 0.05 让反馈几乎不影响排序（飞轮转了也不传动）。
# 提到 0.15，语义仍主导（0.75），但正反馈能被排序感知，飞轮传动轴接上。
RERANK_WEIGHT_HEAT = 0.15
RERANK_WEIGHT_SIM = 0.75

# 数据生命周期适配系统真实寿命：旧值 14/30 天 > 系统龄（6 天），衰减归档永不触发，
# 导致「只增不减的伪进化」。冷知识 3 天未访问即衰减、7 天归档；
# 归档热度上限 2.0 允许「低热且久未用」下沉，但不误杀刚被正反馈抬高的条目。
COLD_DECAY_DAYS = 3.0
ARCHIVE_DAYS = 7.0
ARCHIVE_HEAT_MAX = 2.0


# ============================================================
# v9.0 语义置信度重构（logistic 校准 + 两阶段闸门 + 向量预计算）
#
# 根因：旧 conf 是「加权平均」，只有 strength+margin 有信号时天花板 0.786，
# 且 strength 用原始余弦（hash 后端值域 0.2~0.5），导致 conf 普遍 0.2x 被误拦。
# 改法：① 语义文本窗口统一走 SEMANTIC_SIM_CONTENT_CHARS（实测 300，勿放宽到 1500）；
#       ② conf 改为按后端标定中点的 logistic 单调映射，输出可当概率解释；
#       ③ 闸门拆两阶段：先判「问题是否在库域内」，再判「条目质量」；
#       ④ 条目向量离线预计算，查询时只编码 1 条 query，消除 32 条现场推理的超时风险。
# ============================================================
# 向量预计算总开关：入库时离线算好条目向量，查询时只编码 1 条 query
VECTOR_PRECOMPUTE_ENABLED = True
# 预计算批大小（每批 embed 条数，过大在弱 CPU 上易超时）
VECTOR_PRECOMPUTE_BATCH = 16
# 预计算向量最小覆盖率：候选集中有向量的比例低于该值时回退现场重排，避免部分向量污染排序
VECTOR_MIN_COVERAGE = 0.8

# 置信度校准中点：该后端下「刚好算命中」的余弦值，按实测分布取值
#   hash  ：正常命中余弦中位约 0.311，负样本 0.12~0.18 → 中点 0.26
#   ollama：bge-m3 真命中 0.47~0.83，负样本 0.22~0.46 → 中点 0.50
CONF_MID = {"hash": 0.26, "ollama": 0.50, "sentence-transformers": 0.42}
# logistic 斜率：越大越"陡"，区分度越高但同时容错空间越小。
# 按后端区分：hash 词袋余弦标准差仅约 0.05，用 8.0 会把 0.26~0.36 压成 0~1 的二值跳变，
# 因此降到 4.0 保持连续可分；bge-m3 分布更宽（σ≈0.12），8.0 才能拉开差距。
CONF_SLOPE = 8.0
CONF_SLOPE_MAP = {"hash": 4.0, "ollama": 8.0, "sentence-transformers": 6.0}
# 当前后端的校准中点（由 v8.0 块解析出的最终后端决定，作为默认值兜底）
CONF_MID_CURRENT = CONF_MID.get(_engine_name, CONF_MID["ollama"])


def conf_params(backend_name: str = "") -> tuple:
    """
    按运行时真实后端名返回 (中点, 斜率)。

    为什么必须按运行时判定：create_embedder 可能在 ollama 不可达时静默降级到 hash，
    此时 config 顶部按 env 解析出的 _engine_name 仍是 ollama，沿用 ollama 的中点 0.50
    会让所有 hash 余弦（0.2~0.4）恒低于中点，conf 永远 <0.5 → 全部弃答。
    """
    name = backend_name or _engine_name
    mid = CONF_MID.get(name, CONF_MID["ollama"])
    slope = CONF_SLOPE_MAP.get(name, CONF_SLOPE)
    return mid, slope

# 两阶段闸门：阶段1 域内判定——查询词（长度>=2）在库词表中的覆盖率低于该值视为库外问题
DOMAIN_COV_MIN = 0.15
# 参与覆盖率统计的最短词长（单字噪音大，不计入）
DOMAIN_COV_TOKEN_MIN_LEN = 2

# logistic 校准后的弃答线（策略旋钮，不是标定旋钮）
#
# confidence 本身仍是校准值（strength==中点时为 0.5），这里调的是「多大概率才敢答」。
# 实测依据（100 条历史真实查询全量回放，hash 后端）：
#   弃答线 0.50 → 弃答 31%   0.45 → 20%   0.44 → 20%   0.43 → 13%   0.40 → 9%
# 人工标注的 12 条噪音 / 12 条真问题分布：
#   噪音 0.31~0.51（多数 ≤0.35），真问题 0.43~0.91。
# 0.50→0.43 这一档减少的弃答全部落在「真问题」区间，噪音泄漏条数不变（4/12），
# 因此 0.43 是当前后端下的占优选择（同样泄漏、更少弃答）。
# 剩余 4 条噪音泄漏（关键词撞词面）需靠 bge-m3 的语义区分度解决，不是阈值能解决的。
if "MOLBASE_ABSTAIN_CONFIDENCE" not in _os.environ:
    ABSTAIN_CONFIDENCE = 0.43
    FEEDBACK_MIN_CONFIDENCE = round(ABSTAIN_CONFIDENCE * 0.85, 4)


# ============================================================
# v9.1 知识库四层模型（树 / 网 / 标签 / 原文）
#
# 目的：把「一堆平铺条目」升级成「可导航、可关联、可统计、可溯源」的结构，
#       让检索除了「词汇匹配」之外还有结构信号可用。
# ============================================================
# 原文层切块参数：单块上限字符数与硬切重叠字符数。
# 1200 字对应一个完整小节，能装下「背景+做法+结论」而不至于把多个主题混进一块。
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 100
# 入库时是否同步生成原文切块（切块本身毫秒级，块向量另由 CHUNK_VECTOR_ENABLED 控制）
CHUNK_INDEX_ENABLED = True
# 启动自愈时是否重建四层（树/标签/原文）。
# 默认关闭：首次构建 730 条目要切 12000+ 块（8~110 秒），
#   同步跑会堵住服务端口让客户端 RemoteDisconnected；
#   放后台线程又会长期持有 SQLite 连接，导致测试临时库 tearDown 时报 WinError 32。
# 因此改为「部署时用 deploy-lan/backfill_vectors.py 离线建一次」，
# 之后新条目靠入库钩子增量维护（毫秒级），运维需要时可把此项打开。
LAYER_BOOTSTRAP_ON_START = False
# 入库时是否同步计算块向量（hash 后端毫秒级可开；ollama 后端建议离线回填后置 False）
CHUNK_VECTOR_ENABLED = True
# 启动自愈时是否连块向量一起算（仅在 LAYER_BOOTSTRAP_ON_START 打开时有意义）
CHUNK_VECTOR_BOOTSTRAP = True
# 块级补强：只对正文长度超过该值的条目启用（短文档条目级向量已覆盖全文，无需块级）
CHUNK_BOOST_MIN_CHARS = 1200
# 块级相似度在最终语义分中的权重：0.35 表示「以条目级为主，块级做修正」
CHUNK_BOOST_WEIGHT = 0.35
# 词法覆盖下限（v9.2）
#
# 现象：一批「关键词堆叠型」真问题被误杀——「MySQL Redis EMQX 局域网 启动配置」
# 词面高度匹配，但 hash 词袋把 20 个关键词平均成一个向量，与 300 字条目的余弦只有 0.2x，
# conf 卡在 0.42~0.49 被判弃答。这不是「库里没有」，是语义平均化的度量缺陷。
#
# 为什么不用 BM25 原始分做下限：实测 BM25 分值跨查询不可比——
# 「给我讲个冷笑话」5.63 高于「MolBase 架构」4.89，用绝对阈值会把噪音放进来。
# 改用「查询自身信息量归一化的词法覆盖率」：
#     cov = top1 命中的查询词 idf 之和 / 查询全部有效词 idf 之和
# 它对查询长度免疫，衡量的是「这条命中吃掉了提问的多少信息」。
#
# 取值依据：cov=1.0（全命中）等价余弦 0.55，高于 hash 中点 0.26、也高于 bge-m3 中点 0.50，
# 只作为下限抬升，不改变任何排序。
LEX_COV_ENABLED = True
LEX_COV_W = 0.55
# 参与统计的词的最低 idf：低于该值的词（如「如何」「怎么」）几乎不携带信息，不计入分母
LEX_COV_MIN_IDF = 1.0
# 有效词少于该数量时不做词法推断（一个词的查询无从判断覆盖率）
LEX_COV_MIN_TOKENS = 2

# 块级补强总开关。
# 实测结论（730 条目 / 12426 块 / hash 后端）：开启后长文档被块级分抬进 top1，
# 挤掉了原本精准的短文档，5 条金标问题的 conf 从 0.85/0.60/0.75/0.91/0.85
# 掉到 0.68/0.41/0.57/0.78/0.71，其中 1 条由放行变弃答。
# 因此默认关闭：块级向量继续用于「原文溯源」与后续块级召回实验，但不参与打分。
CHUNK_BOOST_ENABLED = False


# ---------------------------------------------------------------------------
# 代码代工参数（硬编码）
# ---------------------------------------------------------------------------
# 代码仓库目录：用于 AST 符号索引 + 反编译定位
CODE_REPO_DIR = PROJECT_ROOT / "code_repo"

# PageRank 阻尼系数（关联方法定位）
PAGERANK_DAMPING = 0.85
# PageRank 最大迭代次数
PAGERANK_MAX_ITER = 100
# PageRank 收敛阈值
PAGERANK_TOLERANCE = 1e-6


def ensure_dirs() -> None:
    """确保所有数据目录存在。"""
    for d in (DATA_DIR, SKILLS_DIR, KNOWLEDGE_DIR, CODE_REPO_DIR):
        d.mkdir(parents=True, exist_ok=True)
