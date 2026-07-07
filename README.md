# LLM Agent 能力評測 —— 作答與實作專案

> 一份 LLM / AI Agent 專業考題的完整交付：**第一部分（觀念簡答）** 的作答，加上 **第二部分（程式實作）** 四個「真正可執行」的 AI Agent 專案。
>
> 全部內容都經過 **對抗性審查 → 驗證** 兩輪把關：先由多個代理挑毛病（共 314 項），再逐條驗證剔除假陽性（確認 210 項），並把確認的 bug／缺陷修進答案與程式碼。

---

## 📂 專案結構

```
llm-agent-assessment/
├── README.md                          ← 你在這裡
├── LICENSE                            (MIT)
├── docs/
│   └── part1-conceptual-answers.md    ← 第一部分：觀念簡答題作答
└── part2/                             ← 第二部分：四個可執行的 AI Agent 實作
    ├── common/                        🔌 Provider 抽象層（OpenAI 相容，多供應商可切換）
    ├── 01_rag_qa_agent/               ① 問答 Agent（Agent + 進階 RAG）
    ├── 02_document_compare/           ② 兩份大型文件比較 Agent
    ├── 03_survey_to_google_sheet/     ③ 問卷 Agent → 寫入 Google Sheet
    └── 04_deep_research/              ④ Deep Research Agent
```

每個 `part2/*` 子專案都是**獨立、可各自安裝執行**的，內含自己的 `README.md`、`requirements.txt`、`.env.example` 與範例資料。

---

## 📘 第一部分：觀念簡答題

完整作答見 **[`docs/part1-conceptual-answers.md`](docs/part1-conceptual-answers.md)**，涵蓋四題：

| # | 題目 | 重點 |
|---|------|------|
| 1 | RAG 的原理與應用 | 核心流程與雙面優勢、企業知識庫場景與挑戰、11 種 RAG 技術比較表、RAG 無法解決的問題 |
| 2 | AI Agent 的概念 | ReAct 迴圈、Structured Output、Function Calling、大量工具處理、記憶分層、設計考量；加分題 MCP↔Function Calling、Prompt Caching |
| 3 | Prompt Engineering | 上下文／角色／輸出格式／指令明確度四要素 + 進階技巧 + 與 RAG／微調的界線 |
| 4 | 電商客服情境 | Agent 架構設計、對電商的價值、幻覺防治、三層成效評估指標 |

---

## 💻 第二部分：四個 AI Agent 實作

| # | 專案 | 一句話 | 主要能力 / 亮點 |
|---|------|--------|----------------|
| ① | **[問答 Agent（RAG）](part2/01_rag_qa_agent/)** | 企業 HR 制度問答助理 | 進階 RAG（多查詢改寫 + Cross-Encoder 重排 + 相關度門檻 + 帶出處引用）、對話記憶、**多工具資料互用**（檢索到的累進表帶入試算工具）、Streaming、Structured Output（strict tools） |
| ② | **[大型文件比較 Agent](part2/02_document_compare/)** | 比較兩份超大合約／文件 | **超大 PDF 不塞整份**：切塊帶頁碼、向量對齊配對、map-reduce 逐段差異、reduce 出結構化報告；失敗配對不靜默丟棄（重試＋記錄＋標示）、對齊避免 O(n×m) 稠密矩陣 |
| ③ | **[問卷 Agent → Google Sheet](part2/03_survey_to_google_sheet/)** | 對話式問卷並寫入試算表 | 對話式逐題收集＋LLM 解析驗證、gspread 寫入（service account）、**防公式/CSV 注入**、寫入失敗有本地備援與重試 |
| ④ | **[Deep Research Agent](part2/04_deep_research/)** | 多來源多步驟研究並產出引用報告 | planner→searcher→verifier→synthesizer 流程、tool-calling agent loop、去重與 loop-until-dry 收斂、**多輪 thinking block 正確回傳**、強制引用可溯源 |

> 每個專案的詳細架構、安裝與執行方式，請見各自資料夾內的 `README.md`。

---

## 🔌 Provider 抽象層（可切換 LLM / Embedding 供應商）

四個實作的 LLM 與 Embedding 呼叫**都走同一套共用抽象層** [`part2/common/`](part2/common/)，底層採 **OpenAI 相容 API**，只要換環境變數即可切換供應商，程式碼零改動：

| 用途 | 環境變數 | 預設 | 可選 |
|------|----------|------|------|
| LLM | `LLM_PROVIDER` / `LLM_MODEL` | `groq` / `llama-3.3-70b-versatile` | groq、openrouter、cerebras、together、nvidia、mistral |
| Embedding | `EMBED_PROVIDER` / `EMBED_MODEL` | `mistral` / `mistral-embed` | mistral、nvidia、together、openai |

金鑰各自來自對應的 `*_API_KEY`（如 `GROQ_API_KEY`、`MISTRAL_API_KEY`）。抽象層自帶 429/5xx 指數退避重試。

## 🚀 快速開始

以「① 問答 Agent（RAG）」為例，其餘專案步驟相同（進到對應資料夾、看該 README）：

```bash
cd part2/01_rag_qa_agent

# 1) 建虛擬環境
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

# 2) 安裝相依
pip install -r requirements.txt

# 3) 設定金鑰（複製範本後填入）——預設用 Groq(LLM)+Mistral(Embedding)
cp .env.example .env      # 編輯 .env：至少填 GROQ_API_KEY 與 MISTRAL_API_KEY

# 4) 執行（每個專案都附非互動 demo）
python run_demo.py        # 一鍵跑完整流程並印出真實結果
# 或互動模式（若該專案有 main.py）：python main.py
```

> **需求**：Python 3.10+、一把 OpenAI 相容供應商金鑰（預設 Groq，免費）＋一把 Embedding 金鑰（預設 Mistral，免費層即可）。
> ③ 若要真的寫入 Google Sheet 另需 service account 憑證（沒有則自動降級寫本地 CSV）；④ 用 DuckDuckGo 免金鑰搜尋，無需 Search API。細節見各專案 README。

---

## 🧪 實測結果（真實 API 執行）

四個實作皆已用**真實 API** 端到端跑通（因測試當下 Groq 免費層暫時性 IP 限流，改用 **Mistral** 作為 LLM＋Embedding；任一 OpenAI 相容供應商皆可）：

| # | 專案 | 實測結果 |
|---|------|----------|
| ① | RAG 問答 | ✅ 3 文件嵌入建索引、out-of-scope 誠實拒答＝0、**多工具資料互用**（檢索累進表→試算特休 18 天 vs 勞基法 15 天）、帶引用、對話記憶、串流 |
| ② | 文件比較 | ✅ 正確標出賠償上限🔴/保密期🟡/個資新增/72h 資安通報等差異；限流下失敗段落**不靜默丟棄**、如實列入「未成功比對段落」 |
| ③ | 問卷→Sheet | ✅ 5 題自由文字經 LLM 正確歸類、**公式注入防護生效**（`=…`→`'=…`）、無憑證自動降級寫本地 CSV |
| ④ | Deep Research | ✅ 規劃 2 子問題→DuckDuckGo 搜尋→抓 4 真實來源→產出帶引用 [1][2] 報告；證據不足時**誠實標註低信心**（反幻覺設計運作） |

> 註：Mistral 免費層約 1 req/s，高並行時會觸發 429；抽象層的退避重試與各專案的降級/容錯設計均已在此條件下驗證有效。

---

## 🔐 安全與機密

- 所有金鑰／憑證一律走環境變數 / `.env`，**不寫死於程式碼**；各專案提供 `.env.example` 範本（僅佔位符）。
- 根目錄 `.gitignore` 已排除 `.env`、`credentials.json`、`service_account*.json`、向量索引快取、問卷本地備援等，避免機密與個資進版控。

---

## ✅ 品質保證流程

本專案的答案與程式碼並非一次生成即交付，而是經過三個階段：

1. **逐題作答** — 由專家代理草擬並經技術審稿潤飾。
2. **對抗性審查** — 每題兩個不同視角的代理挑毛病，共找出 **314 項**問題（事實錯誤、程式 bug、安全漏洞、需求未達成等）。
3. **驗證與修訂** — 逐條複核，剔除假陽性 32 項、去重 71 項，確認 **210 項**有效問題並全部修進答案與實作。

代表性修正：修掉 RAG 答案的內部矛盾、補齊間接提示注入與資料治理；修正 Structured Output／Function Calling 的「絕對保證」誤述；修掉文件比較的靜默丟棄與 O(n×m) 矩陣；修掉問卷的公式注入與無回滾；修掉 Deep Research 多輪 thinking block 未回傳導致的 API 400。

---

## 📄 授權

本專案以 [MIT License](LICENSE) 釋出。
