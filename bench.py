# -*- coding: utf-8 -*-
"""
性能测试工具（bench）：像性能测试一样，量化展示MolBase的「降 token / 降本」效果。

运行：
    python bench.py              # 默认：MockLLM 模拟本地大模型（无需真实 Ollama）
    python bench.py --llm        # 接真实本地大模型（需先启动 Ollama/vLLM）
    python bench.py --web        # 附加网络搜索兜底演示

产出：
    reports/bench_report.html    # 自包含 HTML 性能报告（零外部依赖，离线可看）
    命令行摘要                   # 关键指标直接打印

设计思路（对齐用户诉求「完整看到优化效果」）：
- 把一次端到端调用拆成「环节（stage）」，逐环节记录调用次数 / LLM 调用次数 /
  输入输出 token / 耗时（见 molbase/profiler.py）。
- 两类环节：
    模型环节（消耗 token）  ：意图路由 / 查询改写 / 摘要压缩 / 最终生成；
    代工环节（不消耗模型 token，替代了模型劳动）：混合召回 / 重排 / 网络搜索。
- 对比两条链路：
    有代工层   ：模型只吃 top-K 压缩上下文 + 少量路由/改写/摘要调用；
    无代工层（基线）：模型自读全表（全量文档）再作答。
- 成本用「若上云等效单价」折算（本地模型仅电费），见 molbase/cost.py。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from molbase.config import CODE_REPO_DIR, SKILLS_DIR, LLM_ENDPOINT, LLM_MODEL
from molbase.cost import cost_of, format_cost
from molbase.orchestrator import Orchestrator
from molbase.profiler import Profiler
from molbase.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# 基准测试参数（硬编码）
# ---------------------------------------------------------------------------
# 模拟「最终生成」环节的系统提示词（真实场景由 agent / 上层大模型注入）
FINAL_SYSTEM_PROMPT = "你是家庭教育助手，基于下方给定上下文回答问题，不要编造。"

# 报告输出目录与文件名
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_PATH = REPORT_DIR / "bench_report.html"

# 默认查询集（覆盖知识 / 代码 / 技能三类，验证路由与召回）
BENCH_QUERIES = [
    "孩子作业管理怎么做",
    "儿童屏幕时间怎么控制",
    "亲子阅读习惯怎么培养",
    "数据库索引是怎么加速查询的",
    "人脸识别的代码方法",
    "解析 PDF 的技能",
    "睡眠与学习效率的关系",
    "如何培养孩子的专注力",
]


class MockLLM:
    """
    模拟本地大模型（无需真实 Ollama/vLLM 即可跑通完整链路）。

    只实现 Orchestrator 用到的两个接口 complete_json / chat，返回确定性结果，
    让「意图路由 / 查询改写 / 摘要压缩」三个模型环节都能被触发并埋点。
    """

    name = "mock-qwen3.5-9b"

    def complete_json(self, system: str, user: str) -> dict:
        """按关键词规则返回意图路由结果（模拟语义分类）。"""
        if any(k in user for k in ("代码", "方法", "函数", "源码")):
            return {"route": "code", "entities": []}
        if any(k in user for k in ("技能", "流程", "PDF", "CSV", "解析")):
            return {"route": "skill", "entities": []}
        return {"route": "knowledge", "entities": []}

    def chat(self, system: str, user: str) -> str:
        """按提示词类型返回改写 / 摘要结果（模拟）。"""
        if "改写" in system:
            # 模拟改写：去寒暄语气词，保留核心词
            for filler in ("我想了解", "我想知道", "怎么做", "怎么", "呀", "呢", "吗", "啊"):
                user = user.replace(filler, " ")
            rewritten = " ".join(user.split())
            return rewritten or user
        if "压缩" in system:
            # 模拟摘要：取前约一半长度
            half = max(1, len(user) // 2)
            return user[:half]
        return "模拟回答"


def _mock_answer(query: str, results: list) -> str:
    """模拟「最终生成」的答案文本（基于命中的 top-1 结果）。"""
    if not results:
        return "根据当前知识库，暂未检索到与该问题直接相关的内容。"
    top = results[0]
    return (
        f"根据知识库，{top.title}：{top.content[:120]}。"
        f"以上为针对「{query}」的参考回答。"
    )


def _ingest_corpus(orc: Orchestrator) -> None:
    """摄入基准语料：真实技能 / 代码 + 一批模拟知识（让「全表基线」有分量）。"""
    # 真实技能目录（SKILL.md 解析）
    orc.ingest_skill_dir(SKILLS_DIR)
    # 真实代码仓库（AST 符号索引）
    orc.ingest_code_repo(CODE_REPO_DIR)

    # 模拟知识条目：家庭教育 + 学习/技术主题（贴近真实知识库规模，每条约 100~160 字）。
    # 条目越多，全表基线越大，代工层「只喂 top-K」的节省效果越显著。
    knowledge = [
        # --- 家庭教育 ---
        ("kb::homework", "孩子作业管理方法",
         "作业管理要分阶段：先做难的科目再做简单科目，每25分钟休息一次（番茄工作法）。"
         "家长只负责监督节奏不代写，完成一项勾一项，用可视化进度清单增强成就感。"),
        ("kb::screen-time", "儿童屏幕时间控制策略",
         "儿童屏幕时间建议每天不超过2小时，采用定时器与番茄工作法结合。"
         "优先安排学习类内容，娱乐类控制在30分钟以内，睡前1小时不接触电子屏幕。"),
        ("kb::reading", "亲子阅读习惯培养",
         "每天睡前进行30分钟亲子阅读，采用指读与提问结合的方式。"
         "让孩子自己选书自己翻页，家长用开放性问题引导思考，重在兴趣而非识字量。"),
        ("kb::focus", "儿童专注力培养方法",
         "专注力培养从短时段开始：5分钟专注配2分钟休息，逐步拉长。"
         "一次只做一件事，减少桌面干扰物，完成后及时正向反馈。"),
        ("kb::sleep", "睡眠与学习效率的关系",
         "充足睡眠是记忆巩固的关键，中小学生建议每天9小时以上睡眠。"
         "睡眠不足会显著降低次日注意力和记忆提取能力，熬夜学习往往得不偿失。"),
        ("kb::emotion", "儿童情绪管理引导",
         "先接纳情绪再讲道理：帮孩子命名情绪（生气/难过/害怕），共情后再引导。"
         "家长示范冷静表达，避免在孩子情绪爆发时说教，建立安全的情绪出口。"),
        ("kb::habit", "好习惯养成四步法",
         "习惯养成遵循提示-渴求-反应-奖励闭环。把新习惯绑定到已有习惯之后，"
         "让目标小到无法失败，即时奖励强化，重复21天以上逐渐自动化。"),
        ("kb::communication", "与孩子的有效沟通技巧",
         "多用开放式提问代替命令，先倾听再回应。用我语句表达感受而非指责，"
         "避免贴标签和翻旧账，把冲突转化为共同解决问题的机会。"),
        ("kb::reward", "奖励与惩罚的正确使用",
         "奖励重过程不重结果，表扬具体行为而非天赋。惩罚要有边界、与行为相关、"
         "可预期且不伤害自尊，避免体罚和过度物质奖励。"),
        ("kb::time-mgmt", "儿童时间管理训练",
         "用可视化时间表和沙漏培养时间感，把任务拆成小块并设定截止。"
         "教会优先级排序（重要紧急四象限），逐步从家长安排过渡到自主规划。"),
        # --- 数据库 ---
        ("kb::db-index", "数据库索引原理",
         "数据库索引本质是B+树结构，能快速定位数据避免全表扫描。"
         "查询优化器根据统计信息选择索引。索引带来写入开销，需要权衡列的选择性。"),
        ("kb::query-opt", "SQL查询优化技巧",
         "查询优化要避免全表扫描、减少回表、合理使用覆盖索引。"
         "用EXPLAIN分析执行计划，对高频字段建立联合索引，注意最左前缀原则。"),
        ("kb::b-tree", "B树与B+树区别",
         "B树所有节点都存数据，B+树只有叶子节点存数据且叶子串成链表。"
         "B+树更适合范围查询，是数据库索引的主流实现。"),
        ("kb::txn", "数据库事务与隔离级别",
         "事务保证ACID：原子性、一致性、隔离性、持久性。"
         "四种隔离级别为读未提交、读已提交、可重复读、串行化，级别越高并发越差。"),
        ("kb::lock", "数据库锁机制",
         "锁分为行锁、表锁、间隙锁、意向锁。悲观锁阻塞等待，乐观锁用版本号或CAS。"
         "死锁靠超时和等待图检测，避免交叉加锁顺序。"),
        ("kb::norm", "数据库范式设计",
         "第一范式保证原子性，第二范式消除部分依赖，第三范式消除传递依赖。"
         "实际设计常在范式与性能间取舍，适度反范式减少连接。"),
        ("kb::shard", "分库分表方案",
         "分库分表解决单库容量与性能瓶颈，按哈希、范围、一致性哈希路由。"
         "带来跨库事务和分页难题，需引入分布式ID和中间件。"),
        ("kb::cache", "缓存穿透雪崩击穿",
         "缓存穿透查不存在的数据，用布隆过滤器或空值缓存。"
         "雪崩是缓存同时失效，击穿是热点key失效，分别用随机过期和互斥锁应对。"),
        ("kb::explain", "EXPLAIN执行计划解读",
         "EXPLAIN输出包含type、key、rows、Extra等列。type从优到劣：system、const、"
         "eq_ref、ref、range、index、ALL，出现ALL说明全表扫描需优化。"),
        ("kb::master-slave", "数据库主从复制",
         "主从复制通过binlog同步，主库写从库读实现读写分离。"
         "复制有异步、半同步、同步三种，异步可能丢数据，同步影响性能。"),
        # --- 算法 ---
        ("kb::sort", "常见排序算法复杂度",
         "快排平均O(nlogn)最坏O(n²)，归并稳定O(nlogn)，堆排序O(nlogn)原地。"
         "插入排序小规模最优，计数排序适合整数范围小的情况。"),
        ("kb::binary-search", "二分查找适用条件",
         "二分查找要求有序且随机访问，时间复杂度O(logn)。"
         "变体包括查找第一个等于、最后一个小于等于、旋转数组查找等。"),
        ("kb::dp", "动态规划解题框架",
         "动态规划三要素：状态定义、状态转移方程、边界与初始化。"
         "先暴力递归找重叠子问题，再记忆化，最后转自底向上迭代。"),
        ("kb::greedy", "贪心算法的适用场景",
         "贪心每步取局部最优，需要贪心选择性质和最优子结构才保证全局最优。"
         "典型应用：区间调度、哈夫曼编码、最小生成树。"),
        ("kb::graph", "图的两种遍历",
         "深度优先DFS用栈递归，适合连通性、拓扑排序。"
         "广度优先BFS用队列，适合最短路径（无权）、层序遍历。"),
        ("kb::hash", "哈希表与冲突解决",
         "哈希表平均O(1)查找，冲突用链地址法或开放寻址法解决。"
         "负载因子过高需扩容，Java HashMap在0.75时扩容为2倍。"),
        ("kb::heap", "堆与优先队列",
         "堆是完全二叉树，大顶堆根最大。插入删除O(logn)，建堆O(n)。"
         "优先队列用于TopK、任务调度、Dijkstra。"),
        ("kb::complexity", "时间复杂度分析要点",
         "关注最坏情况和增长阶，忽略常数和低阶项。"
         "常见量级O(1)<O(logn)<O(n)<O(nlogn)<O(n²)<O(2ⁿ)。"),
        # --- 学习方法 ---
        ("kb::feynman", "费曼学习法",
         "把概念用自己的话讲给外行听，讲不清楚说明没学透。"
         "讲-发现缺口-回去补-再讲，循环往复直到能简洁清晰表达。"),
        ("kb::spaced", "间隔重复记忆法",
         "间隔重复根据遗忘曲线安排复习：学完1天、3天、7天、30天各复习一次。"
         "比集中刷题更省时间且记得更牢，适合背单词和概念记忆。"),
        ("kb::cornell", "康奈尔笔记法",
         "页面分三区：右侧笔记栏、左侧线索栏、底部总结栏。"
         "课后用线索栏自测，总结栏用自己的话概括，强化主动回忆。"),
        ("kb::mindmap", "思维导图学习法",
         "以中心主题向外发散分支，用关键词和图像组织层级关系。"
         "适合梳理知识结构和复习回顾，促进发散与联想记忆。"),
        ("kb::pomodoro", "番茄工作法",
         "25分钟专注加5分钟休息为一个番茄，四个番茄后长休息。"
         "减少干扰、保持心流，适合需要长时间专注的任务。"),
        ("kb::active-recall", "主动回忆法",
         "合上书本回忆所学内容，比反复阅读更有效。"
         "用自测、默写、给别人讲解等方式提取记忆，暴露掌握盲区。"),
        ("kb::wrong-book", "错题本的正确用法",
         "错题本不是抄题本，要记录错误原因、正确思路和同类变式。"
         "定期重做并淘汰已掌握的题，避免低效积累。"),
        ("kb::output", "输出倒逼输入",
         "以教代学、以写代读，输出能检验输入质量。"
         "写博客、做分享、回答问题是高效的学习闭环。"),
        # --- 心理健康 ---
        ("kb::anxiety", "考试焦虑缓解",
         "焦虑源于对结果的高估和对能力的低估。用深呼吸、正念和积极自我对话。"
         "把大目标拆小、模拟考试场景脱敏，降低不确定感。"),
        ("kb::mindfulness", "正念练习入门",
         "正念是有意识地觉察当下而不评判。从观呼吸开始，每天5分钟。"
         "觉察到走神就温和拉回，训练注意力和情绪调节能力。"),
        ("kb::stress", "压力管理策略",
         "压力分良性和恶性，适度压力提升表现。恶性压力需运动、社交支持、"
         "睡眠和认知重构来调节，避免长期透支。"),
        ("kb::motivation", "内在动机与外在动机",
         "内在动机来自兴趣、自主、胜任感，比外在奖励更持久。"
         "过度外在奖励会削弱内在动机，即德西效应。"),
        ("kb::procrastinate", "拖延症的破解",
         "拖延源于任务厌恶和即时满足，用两分钟法则启动。"
         "拆解任务、设定截止、消除诱惑、先做五分钟，行动能缓解焦虑。"),
        ("kb::resilience", "心理韧性的培养",
         "韧性是从挫折中恢复的能力，可通过成长型思维训练。"
         "把失败归因为可改变的努力而非天赋，在支持关系中重建信心。"),
        # --- 工具技能 ---
        ("kb::pdf-skill", "PDF文档处理",
         "PDF提取文本用pdfplumber或PyPDF2，表格提取需识别结构。"
         "扫描件需OCR，可编辑PDF需检查字体嵌入和许可。"),
        ("kb::csv-skill", "CSV数据处理",
         "CSV用逗号分隔，处理时注意编码、引号转义和空值。"
         "大文件用pandas分块读取，避免一次性载入内存。"),
        ("kb::json", "JSON数据格式",
         "JSON是键值对结构，支持对象、数组、字符串、数字、布尔和null。"
         "解析时注意转义和嵌套层级，大JSON用流式解析。"),
        ("kb::regex", "正则表达式要点",
         "正则用模式匹配字符串，元字符有. * + ? [] () {}。"
         "贪婪与懒惰匹配、捕获组与回溯引用是进阶重点。"),
        ("kb::git", "Git版本控制",
         "Git用快照记录版本，工作区-暂存区-仓库三层结构。"
         "常用命令add、commit、branch、merge、rebase，冲突需手动解决。"),
        ("kb::testing", "单元测试最佳实践",
         "单元测试用AAA模式（准备-执行-断言），覆盖边界和异常。"
         "测试隔离、可重复、快，用Mock隔离外部依赖。"),
        ("kb::debug", "调试技巧",
         "二分定位问题范围，打印关键变量或用断点单步跟踪。"
         "先复现再假设再验证，缩小差异找到根因。"),
        ("kb::logging", "日志规范",
         "日志分级DEBUG、INFO、WARN、ERROR，记录时间、级别、上下文。"
         "敏感信息脱敏，异常日志带堆栈，避免高频刷屏。"),
    ]
    for eid, title, content in knowledge:
        orc.ingest_knowledge(eid, title, content)


def _run_queries(orc: Orchestrator, profiler: Profiler) -> dict:
    """
    跑一组查询，逐条记录「有代工层」的环节统计与「无代工层」的基线 token。

    返回汇总字典，含：
    - per_query：每条查询的明细（route / 命中数 / 本次 token / 基线 token / 节省）
    - baseline_tokens / baseline_cost：无代工层累计
    - full_text：全表内容（供基线计算）
    """
    full_text = "\n".join(e.title + " " + e.content for e in orc.store.all_entries())
    per_query: list[dict] = []
    total_baseline_tokens = 0
    total_baseline_cost = 0.0
    prev_total = 0  # 上一次 query 结束时的累计 token（用于拆分单次增量）

    for q in BENCH_QUERIES:
        resp = orc.query(q)

        # 模拟「最终生成」：有代工层下，模型输入 = 系统提示 + 压缩上下文 + 查询
        ctx = "".join(r.title + " " + r.content for r in resp.results)
        ctx += "".join(r.title + " " + r.content for r in resp.web_results)
        final_input = FINAL_SYSTEM_PROMPT + q + ctx
        final_output = _mock_answer(q, resp.results)
        profiler.record(
            "最终生成",
            llm_call=True,
            input_tokens=estimate_tokens(final_input),
            output_tokens=estimate_tokens(final_output),
        )

        # 无代工层基线：模型自读全表（全量文档）再作答，输出长度与有代工层一致
        baseline_input = estimate_tokens(FINAL_SYSTEM_PROMPT + q + full_text)
        baseline_output = estimate_tokens(final_output)
        baseline_tokens = baseline_input + baseline_output

        # 本次 query 的「有代工层」token 增量 = 累计 token 减去上一次累计
        with_kb_tokens = profiler.total_tokens() - prev_total
        prev_total = profiler.total_tokens()

        total_baseline_tokens += baseline_tokens
        total_baseline_cost += cost_of(baseline_input, baseline_output)

        per_query.append({
            "query": q,
            "route": resp.route,
            "hits": len(resp.results),
            "web": len(resp.web_results),
            "with_kb_tokens": with_kb_tokens,
            "baseline_tokens": baseline_tokens,
        })

    return {
        "per_query": per_query,
        "baseline_tokens": total_baseline_tokens,
        "baseline_cost": total_baseline_cost,
        "full_text": full_text,
    }


def _with_kb_cost(profiler: Profiler) -> float:
    """有代工层总成本：逐环节按输入 / 输出 token 分别折算。"""
    total = 0.0
    for s in profiler.stages():
        total += cost_of(s.input_tokens, s.output_tokens)
    return total


def _render_html(report: dict, profiler: Profiler, orc: Orchestrator) -> str:
    """
    渲染自包含 HTML 性能报告（零外部依赖）。

    report  ：_run_queries 返回的汇总字典。
    profiler：累计剖析器（含全部环节统计）。
    orc     ：编排器（取 embedding / llm / search 后端名）。
    """
    stages = profiler.stages()
    with_kb_tokens = profiler.total_tokens()
    baseline_tokens = report["baseline_tokens"]
    with_kb_cost = _with_kb_cost(profiler)
    baseline_cost = report["baseline_cost"]

    saved_tokens = baseline_tokens - with_kb_tokens
    saved_pct = (saved_tokens / baseline_tokens * 100) if baseline_tokens else 0.0
    saved_cost = baseline_cost - with_kb_cost

    embed_name = getattr(orc.embedder, "name", "none")
    llm_name = orc.llm.name if orc.llm else "未启用（规则兜底）"
    search_name = orc.search_provider.name

    # 环节统计表行
    stage_rows = []
    for s in stages:
        kind = "模型环节" if s.llm_calls > 0 else "代工环节"
        stage_rows.append(
            f"<tr><td>{s.name}</td><td>{kind}</td><td>{s.calls}</td>"
            f"<td>{s.llm_calls}</td><td>{s.input_tokens:,}</td>"
            f"<td>{s.output_tokens:,}</td><td>{s.total_tokens:,}</td>"
            f"<td>{s.duration_ms:,.1f}</td><td>{s.note or '-'}</td></tr>"
        )

    # 每次查询明细表行
    query_rows = []
    for p in report["per_query"]:
        saved = p["baseline_tokens"] - p["with_kb_tokens"]
        query_rows.append(
            f"<tr><td>{p['query']}</td><td>{p['route']}</td><td>{p['hits']}</td>"
            f"<td>{p['web']}</td><td>{p['with_kb_tokens']:,}</td>"
            f"<td>{p['baseline_tokens']:,}</td><td>{saved:,}</td></tr>"
        )

    # 条形图比例（token 对比）
    max_tok = max(baseline_tokens, with_kb_tokens, 1)
    bar_with_kb = with_kb_tokens / max_tok * 100
    bar_base = baseline_tokens / max_tok * 100

    # 成本对比条形图
    max_cost = max(baseline_cost, with_kb_cost, 1e-9)
    bar_cost_kb = with_kb_cost / max_cost * 100
    bar_cost_base = baseline_cost / max_cost * 100

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MolBase · 优化效果性能报告</title>
<style>
  :root {{
    --bg: #f5f6f8; --card: #ffffff; --ink: #1a1d24; --muted: #6b7280;
    --brand: #2563eb; --save: #16a34a; --base: #94a3b8; --line: #e5e7eb;
    --model: #7c3aed; --proxy: #0891b2;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
         font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           padding: 16px; }}
  .card .k {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
  .card .v {{ font-size: 24px; font-weight: 700; }}
  .card .v.green {{ color: var(--save); }}
  .card .v.blue {{ color: var(--brand); }}
  .card .v.purple {{ color: var(--model); }}
  .card .u {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  section {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
             padding: 20px; margin-bottom: 20px; }}
  h2 {{ font-size: 16px; margin: 0 0 14px; padding-left: 10px; border-left: 3px solid var(--brand); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; background: #fafbfc; }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; }}
  .tag.model {{ background: #f3e8ff; color: var(--model); }}
  .tag.proxy {{ background: #ecfeff; color: var(--proxy); }}
  .bars {{ margin-top: 8px; }}
  .bar-row {{ margin-bottom: 16px; }}
  .bar-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
  .bar-track {{ height: 22px; background: var(--bg); border-radius: 6px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 6px; color: #fff; font-size: 11px;
               display: flex; align-items: center; padding: 0 8px; white-space: nowrap; }}
  .bar-fill.blue {{ background: var(--brand); }}
  .bar-fill.base {{ background: var(--base); }}
  .bar-fill.save {{ background: var(--save); }}
  .concl {{ background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px;
            padding: 16px 20px; font-size: 14px; line-height: 1.8; }}
  .concl b {{ color: var(--save); }}
  .meta {{ font-size: 12px; color: var(--muted); margin-top: 16px; }}
  @media (max-width: 720px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>MolBase · 优化效果性能报告</h1>
  <div class="sub">一次调用模型过程中，各环节调用次数、token 消耗与成本节省的完整观测（token 为粗估值，成本为「若上云」等效折算）</div>

  <div class="cards">
    <div class="card"><div class="k">查询总数</div><div class="v">{len(report['per_query'])}</div><div class="u">覆盖知识/代码/技能</div></div>
    <div class="card"><div class="k">模型调用次数（有代工层）</div><div class="v purple">{profiler.total_llm_calls()}</div><div class="u">路由+改写+摘要+生成</div></div>
    <div class="card"><div class="k">节省 token</div><div class="v green">{saved_tokens:,}</div><div class="u">节省 {saved_pct:.1f}%</div></div>
    <div class="card"><div class="k">节省成本</div><div class="v green">{format_cost(max(saved_cost, 0.0))}</div><div class="u">等效 API 折算</div></div>
  </div>

  <section>
    <h2>环节调用统计（哪些环节执行了多少次调用）</h2>
    <table>
      <thead><tr>
        <th>环节</th><th>类型</th><th class="num">调用次数</th><th class="num">模型调用</th>
        <th class="num">输入 token</th><th class="num">输出 token</th><th class="num">合计 token</th>
        <th class="num">耗时(ms)</th><th>备注</th>
      </tr></thead>
      <tbody>{''.join(stage_rows)}</tbody>
    </table>
    <div class="meta">「模型环节」消耗 token（意图路由/查询改写/摘要压缩/最终生成）；「代工环节」不消耗模型 token，是代工层替代模型完成的劳动（混合召回/重排/网络搜索）。</div>
  </section>

  <section>
    <h2>Token 消耗对比（有代工层 vs 无代工层基线）</h2>
    <div class="bars">
      <div class="bar-row">
        <div class="bar-label"><span>有代工层</span><span>{with_kb_tokens:,} token · {format_cost(with_kb_cost)}</span></div>
        <div class="bar-track"><div class="bar-fill blue" style="width:{bar_with_kb:.1f}%">{with_kb_tokens:,}</div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>无代工层（模型自读全表）</span><span>{baseline_tokens:,} token · {format_cost(baseline_cost)}</span></div>
        <div class="bar-track"><div class="bar-fill base" style="width:{bar_base:.1f}%">{baseline_tokens:,}</div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>节省</span><span>{saved_tokens:,} token（{saved_pct:.1f}%）</span></div>
        <div class="bar-track"><div class="bar-fill save" style="width:{saved_pct:.1f}%">-{saved_pct:.1f}%</div></div>
      </div>
    </div>
  </section>

  <section>
    <h2>成本对比（等效 API 折算）</h2>
    <div class="bars">
      <div class="bar-row">
        <div class="bar-label"><span>有代工层</span><span>{format_cost(with_kb_cost)}</span></div>
        <div class="bar-track"><div class="bar-fill blue" style="width:{bar_cost_kb:.1f}%">{format_cost(with_kb_cost)}</div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>无代工层（基线）</span><span>{format_cost(baseline_cost)}</span></div>
        <div class="bar-track"><div class="bar-fill base" style="width:{bar_cost_base:.1f}%">{format_cost(baseline_cost)}</div></div>
      </div>
      <div class="bar-row">
        <div class="bar-label"><span>节省</span><span>{format_cost(max(saved_cost, 0.0))}</span></div>
        <div class="bar-track"><div class="bar-fill save" style="width:{(saved_cost / baseline_cost * 100) if baseline_cost else 0:.1f}%">-{format_cost(max(saved_cost, 0.0))}</div></div>
      </div>
    </div>
  </section>

  <section>
    <h2>逐次查询明细</h2>
    <table>
      <thead><tr>
        <th>查询</th><th>路由</th><th class="num">命中</th><th class="num">网络兜底</th>
        <th class="num">有代工层 token</th><th class="num">基线 token</th><th class="num">节省 token</th>
      </tr></thead>
      <tbody>{''.join(query_rows)}</tbody>
    </table>
  </section>

  <div class="concl">
    <b>结论：</b>本次共 {len(report['per_query'])} 次调用，有代工层总消耗 <b>{with_kb_tokens:,} token</b>，
    相比「模型自读全表」基线 <b>{baseline_tokens:,} token</b>，节省 <b>{saved_tokens:,} token（{saved_pct:.1f}%）</b>，
    等效降低成本 <b>{format_cost(max(saved_cost, 0.0))}</b>。
    代工层以「混合召回 / 重排 / 意图路由 / 查询改写 / 摘要压缩」等少量低 token 环节，
    替代了模型逐文档阅读的高 token 劳动——这就是「越用越省」的量化来源。
  </div>

  <div class="meta">
    后端：embedding={embed_name} · LLM={llm_name} · 网络搜索={search_name}。
    token 为粗估值（中文 1 字≈1 token，英文 4 字符≈1 token）；成本按输入 ¥2.0/1M、输出 ¥8.0/1M 等效折算（见 config.py）。
  </div>
</div>
</body>
</html>"""


def main() -> None:
    """性能测试主流程：摄入语料 → 跑查询 → 汇总 → 生成 HTML 报告 + 命令行摘要。"""
    # 独立临时库，避免污染 config.DB_PATH（测试 / 多次运行互不干扰）
    tmp_dir = Path(tempfile.mkdtemp(prefix="kb_bench_"))
    db_path = tmp_dir / "bench.db"

    kwargs = {"db_path": db_path, "llm": MockLLM()}
    if "--web" in sys.argv:
        from molbase.websearch import SearchHit, SearchProvider

        class _BenchSearch(SearchProvider):
            name = "demo"

            def search(self, query, top_n=3):
                return [SearchHit(
                    title="外部补充知识",
                    content="网络搜索到的该主题最新资料（模拟结果，真实场景由 SearXNG 返回）。",
                    url="http://example.com/bench", score=1.0)]
        kwargs["search_provider"] = _BenchSearch()

    # --llm：接真实本地大模型（覆盖 MockLLM）
    if "--llm" in sys.argv:
        from molbase.llm import create_llm
        real = create_llm(LLM_ENDPOINT, LLM_MODEL)
        if real is not None:
            kwargs["llm"] = real
            print(f"[本地大模型] 已接入真实模型 {LLM_MODEL} @ {LLM_ENDPOINT}")
        else:
            print("[本地大模型] 未配置，回退 MockLLM 模拟")

    orc = Orchestrator(**kwargs)
    profiler = Profiler()
    orc.profiler = profiler  # 挂剖析器，让 query 内部各环节自动埋点

    print(f"[向量后端] {orc.embedder.name}")
    print(f"[本地大模型] {orc.llm.name}")
    print(f"[网络搜索] {orc.search_provider.name}")

    # 摄入语料
    _ingest_corpus(orc)
    stats = orc.stats()
    print(f"[语料] 知识 {stats['knowledge']} 条 / 代码 {stats['code']} 条 / 技能 {stats['skill']} 条")

    # 跑查询
    report = _run_queries(orc, profiler)

    # 汇总关键指标
    with_kb_tokens = profiler.total_tokens()
    baseline_tokens = report["baseline_tokens"]
    with_kb_cost = _with_kb_cost(profiler)
    baseline_cost = report["baseline_cost"]
    saved_tokens = baseline_tokens - with_kb_tokens
    saved_pct = (saved_tokens / baseline_tokens * 100) if baseline_tokens else 0.0
    saved_cost = baseline_cost - with_kb_cost

    # 命令行摘要
    print("\n" + "=" * 56)
    print("  性能测试摘要（有代工层 vs 无代工层基线）")
    print("=" * 56)
    print(f"  查询总数        : {len(report['per_query'])}")
    print(f"  模型调用次数    : {profiler.total_llm_calls()}")
    print(f"  有代工层总 token: {with_kb_tokens:,}")
    print(f"  无代工层基线    : {baseline_tokens:,}")
    print(f"  节省 token      : {saved_tokens:,}（{saved_pct:.1f}%）")
    print(f"  有代工层成本    : {format_cost(with_kb_cost)}")
    print(f"  无代工层成本    : {format_cost(baseline_cost)}")
    print(f"  节省成本        : {format_cost(max(saved_cost, 0.0))}")
    print("-" * 56)
    print("  环节调用明细（环节 | 调用次数 | 模型调用 | 输入token | 输出token）")
    for s in profiler.stages():
        kind = "模型" if s.llm_calls else "代工"
        print(f"    {s.name:<6} | {s.calls:>3} 次 | {s.llm_calls:>2} 次 | "
              f"{s.input_tokens:>6} | {s.output_tokens:>6}  ({kind})")

    # 写 HTML 报告
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_html(report, profiler, orc), encoding="utf-8")
    print("\n[报告] 已生成 " + str(REPORT_PATH))

    orc.close()


if __name__ == "__main__":
    main()
