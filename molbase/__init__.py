# -*- coding: utf-8 -*-
"""MolBase（Mol = Model 模型 × Base 底座/知识库）核心包。

架构分层：
- 底座（store）：存储引擎，管理知识/代码/技能三类条目及其热度、新鲜度信号。
- 索引（index）：倒排索引 + BM25 召回。
- 图（graph）：关联图检索，定位关联方法/关联知识。
- 代码（code）：AST 符号索引 + PageRank 关联方法定位（编译/反编译代工）。
- 技能（skills）：SKILL.md 解析与注册（代管）。
- 检索（retrieval）：混合召回 + 热度/新鲜度/语义重排。
- 编排（orchestrator）：统一入口，意图路由 → 召回 → 重排 → 最小上下文输出。
"""

__version__ = "0.1.0"
