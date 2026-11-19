---
name: auto-web-search-file-read-file-edit
description: 执行「检索并更新文档」的可复用操作流程：web_search → file_read → file_edit
---

# auto-web-search-file-read-file-edit

## 步骤
1. 调用 web_search（关键参数：query）
2. 调用 file_read（关键参数：path）
3. 调用 file_edit（关键参数：content, path）

> 本技能由 MolBase 自动生成（工具调用模式自进化）。