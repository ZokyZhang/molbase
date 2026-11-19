# -*- coding: utf-8 -*-
"""
端到端演示脚本：展示MolBase的完整能力。

运行：
    python demo.py              # 零模型内核（零依赖，默认）
    python demo.py --tfidf      # TF-IDF 加权向量（零模型增强）
    python demo.py --llm        # 接本地大模型（需先启动 Ollama/vLLM）
    python demo.py --web        # 演示网络搜索兜底 + 采纳沉淀

演示流程：
1. 摄入知识条目（家庭教育领域）
2. 摄入代码仓库（人脸识别）并展示代码代工
3. 摄入技能目录（PDF/CSV）并展示技能代管
4. 查询召回 + 重排（BM25F + 词典分词）
5. 反馈闭环（正/负反馈）
6. 自我进化（自动命中热度 + 共现建边 + 冷衰减）
7. 摘要压缩 + 智能切块（TextRank，降 token）
8. token 埋点 + 网络搜索兜底（外部召回源，--web 时演示）
"""

import sys

from molbase.orchestrator import Orchestrator
from molbase.config import CODE_REPO_DIR, SKILLS_DIR, LLM_ENDPOINT, LLM_MODEL
from molbase.websearch import SearchHit, SearchProvider


class _DemoSearchProvider(SearchProvider):
    """演示用假搜索提供者：返回固定外部结果，不真联网（真实场景换成 SearXNG）。"""

    name = "demo"

    def search(self, query: str, top_n: int = 3) -> list[SearchHit]:
        """返回一条模拟的外部搜索结果。"""
        return [
            SearchHit(
                title="外部补充知识",
                content="网络搜索到的关于该主题的最新资料（模拟结果，真实场景由 SearXNG 等后端返回）。",
                url="http://example.com/demo",
                score=1.0,
            )
        ]


def section(title: str) -> None:
    """打印分节标题。"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """演示主流程。"""
    # 渐进增强：默认零模型；--tfidf 用 TF-IDF 向量；--llm 接本地大模型；--web 演示网络搜索
    kwargs = {}
    if "--tfidf" in sys.argv:
        kwargs["embedding_backend"] = "tfidf"
    if "--llm" in sys.argv:
        kwargs["llm_endpoint"] = LLM_ENDPOINT
        kwargs["llm_model"] = LLM_MODEL
    if "--web" in sys.argv:
        kwargs["search_provider"] = _DemoSearchProvider()
    orc = Orchestrator(**kwargs)
    print(f"[向量后端] {orc.embedder.name}")
    print(f"[本地大模型] {'已接入 ' + orc.llm.model if orc.llm else '未启用（零模型内核）'}")
    print(f"[网络搜索] {'已接入 ' + orc.search_provider.name if orc.search_provider.name != 'null' else '未启用（内部知识库优先）'}")

    # ------------------------------------------------------------------ 摄入
    section("1. 摄入知识条目")
    orc.ingest_knowledge(
        "kb::family-edu::homework",
        "孩子作业管理方法",
        "家庭教育中，作业管理要分阶段：先做难的科目，再做简单科目，每25分钟休息一次。",
    )
    orc.ingest_knowledge(
        "kb::family-edu::screen-time",
        "屏幕时间控制策略",
        "儿童屏幕时间控制建议每天不超过2小时，采用定时器与番茄工作法结合。",
    )
    orc.ingest_knowledge(
        "kb::family-edu::reading",
        "亲子阅读习惯培养",
        "每天睡前进行30分钟亲子阅读，采用指读与提问结合的方式培养阅读兴趣。",
    )
    print("知识条目：", orc.stats()["knowledge"])

    section("2. 摄入代码仓库（代码代工）")
    n_code = orc.ingest_code_repo(CODE_REPO_DIR)
    print(f"代码符号索引数：{n_code}")

    section("3. 摄入技能目录（技能代管）")
    n_skill = orc.ingest_skill_dir(SKILLS_DIR)
    print(f"技能代管数：{n_skill}")
    print("全量统计：", orc.stats())

    # ------------------------------------------------------------------ 查询
    section("4. 查询召回 + 重排")
    print("\n[查询] 作业管理怎么做 -> 路由 knowledge")
    resp = orc.query("作业管理怎么做")
    print(f"路由：{resp.route}")
    for r in resp.results:
        print(f"  - [{r.entry_type}] {r.title} (score={r.score})")

    print("\n[查询] 人脸识别的代码方法 -> 路由 code")
    resp = orc.query("人脸识别的代码方法")
    print(f"路由：{resp.route}")
    for r in resp.results:
        print(f"  - [{r.entry_type}] {r.title} (score={r.score})  source={r.source}")

    print("\n[查询] 解析 PDF 的技能 -> 路由 skill")
    resp = orc.query("解析 PDF 的技能")
    print(f"路由：{resp.route}")
    for r in resp.results:
        print(f"  - [{r.entry_type}] {r.title} (score={r.score})")

    # ------------------------------------------------------------------ 反馈
    section("5. 反馈闭环（正/负反馈）")
    print("\n[反馈] 对 '屏幕时间控制策略' 采纳 +3，并引用 +2")
    orc.feedback("kb::family-edu::screen-time", "adopt")
    orc.feedback("kb::family-edu::screen-time", "cite")
    print("\n再次查询 '儿童 屏幕 时间'：")
    resp = orc.query("儿童 屏幕 时间")
    for r in resp.results:
        print(f"  - [{r.entry_type}] {r.title} (score={r.score})")
    print("说明：屏幕时间条目热度上升，在相关查询中排名提升。")
    print("\n[负反馈] 对 '亲子阅读' 拒绝一次（热度下降）")
    orc.feedback("kb::family-edu::reading", "reject")

    # ------------------------------------------------------------------ 自我进化
    section("6. 自我进化（零模型，全自动）")
    before = orc.store.get("kb::family-edu::homework").heat
    orc.query("作业管理")
    after = orc.store.get("kb::family-edu::homework").heat
    print(f"作业管理条目热度：{before} -> {after}（query 自动命中 +1）")

    print("\n模拟 3 次查询同时命中 '作业管理' 与 '屏幕时间'（共现）：")
    for _ in range(3):
        orc.evolution.record_query(
            ["kb::family-edu::homework", "kb::family-edu::screen-time"]
        )
    stats = orc.evolve()
    print(f"共现学习：自动建立关联边 {stats['edges']} 条，冷知识衰减 {stats['cold_decayed']} 条")
    neighbors = orc.graph.neighbors("kb::family-edu::homework", resolver=orc._resolver)
    print("作业管理的新关联：", [n.entry.title for n in neighbors])

    # ------------------------------------------------------------------ 摘要
    section("7. 摘要压缩 + 智能切块（降 token）")
    long_text = (
        "数据库索引是加速查询的关键数据结构。索引本质上是 B 树结构。"
        "B 树能够快速定位数据，避免全表扫描。查询优化器会根据统计信息选择合适的索引。"
        "没有索引的情况下，数据库只能逐行扫描所有数据。合理的索引设计能显著提升查询性能。"
        "但是索引也会带来写入开销，因此需要权衡。"
    )
    summary = orc.summarize(long_text, max_chars=100)
    print("原始长度：", len(long_text), "字符")
    print("摘要结果：", summary)
    blocks = orc.chunk(long_text, max_chars=80)
    print(f"\n智能切块（每块 ≤80 字符，共 {len(blocks)} 块）：")
    for i, b in enumerate(blocks, 1):
        print(f"  [{i}] {b}")

    # ------------------------------------------------------------------ token 埋点 + 网络搜索
    section("8. token 埋点 + 网络搜索兜底")
    resp = orc.query("作业管理怎么做")
    saved = round((1 - resp.token_estimate / resp.baseline_tokens) * 100, 1)
    print(f"[token 埋点] 本次返回 {resp.token_estimate} token（{resp.content_chars} 字符）")
    print(f"             vs 传统「全表扫描」基线 {resp.baseline_tokens} token")
    print(f"             预测节省约 {saved}%（粗估，待真实场景埋点校准）")

    if orc.search_provider.name != "null":
        print("\n[网络搜索兜底] 查询内部知识库没有的主题：")
        resp = orc.query("区块链共识机制")
        print(f"  内部命中：{len(resp.results)} 条，网络兜底：{len(resp.web_results)} 条")
        for r in resp.web_results:
            print(f"    - [web] {r.title} (score={r.score}) source={r.source}")
        if resp.web_results:
            new_id = orc.adopt_web_result(resp.web_results[0])
            entry = orc.store.get(new_id)
            print(f"\n[采纳沉淀] 采纳网络结果 -> 沉淀为内部知识：{new_id}")
            print(f"  沉淀后类型={entry.entry_type}，来源标记={entry.meta.get('origin')}")
            print("  效果：下次同类查询直接命中内部知识，不再联网（越用越少依赖网络）。")
    else:
        print("\n[网络搜索] 未启用。加 --web 参数可演示兜底 + 采纳沉淀闭环。")

    orc.close()


if __name__ == "__main__":
    main()
