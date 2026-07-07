"""兩份大型文件比較 Agent（超大 PDF 不塞整份）。

核心策略：串流解析 → 切塊帶頁碼 metadata → 向量嵌入建索引 →
章節分桶 + 向量檢索對齊（不物化 O(n×m) 稠密矩陣）→ map（逐對結構化差異）→
reduce（階層彙總）→ 渲染結構化報告，並把「未成功比對段落」明確列入報告。

LLM / Embedding 呼叫統一走 `part2/common/` 的共用 provider 抽象層（OpenAI 相容，
預設 LLM=Groq llama-3.3-70b-versatile、Embedding=Mistral mistral-embed 1024 維）。

各子模組：
    config       集中管理的可調常數（皆可由環境變數覆寫）；並把 part2/ 加入匯入路徑
    models       Chunk 資料類別與 Pydantic 輸出 schema（PairDiff / SectionSummary）
    parsing      串流解析 + 切塊（PDF / Markdown / 純文字，帶頁碼區間）
    embeddings   嵌入向量（common.embed_texts＝Mistral；離線降級用確定性 hash），含 matrix()
    alignment    章節分桶 + 向量檢索對齊（不物化 O(n×m) 稠密矩陣；top-k 候選）
    ratelimit    TPM/RPM token-bucket 限流
    llm          common 層 LLM 呼叫、比對準則、complete_json（結構化輸出驗證）
    map_stage    逐對結構化差異（含短路、失敗記錄）
    reduce_stage 章節層階層彙總（Structured Output）
    report       結構化欄位渲染表格 + LLM 敘事結論
    checkpoint   map/reduce 中間結果落地，支援續跑
    pipeline     串接整條 pipeline 的 compare() 進入點
"""

__all__ = ["__version__"]
__version__ = "2.0.0"
