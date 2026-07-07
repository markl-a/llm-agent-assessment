# 企業 HR 制度問答 AI Agent（Agent + RAG）

以**共用 provider 抽象層（`part2/common`）＋進階 RAG** 打造的企業內部「HR 制度問答助理」。員工可用口語、跨文件、帶前後脈絡的方式提問，助理會**帶出處**地回答，需要時**呼叫試算工具**算給你聽，並以**工具事件 / 串流**即時呈現處理過程。

> 本服務的 **LLM 與 Embedding 皆走共用 provider 層**：
> **LLM = Groq `llama-3.3-70b-versatile`**、**Embedding = Mistral `mistral-embed`（1024 維、多語含繁中）**，皆走 OpenAI 相容 API，靠環境變數切換供應商。

---

## 一、商業情境與價值

大型企業的人資制度散落在請假辦法、差旅報支、薪酬福利、勞健保、績效考核等數十份文件中，員工的提問往往口語化、跨文件、且帶前後脈絡（例如「那我這種年資可以休幾天？報帳上限多少？」）。傳統關鍵字搜尋命中率低，也無法「算給你聽」。

本服務讓助理能夠：

1. 從制度文件庫檢索**帶出處**的答案；
2. 需要時**呼叫計算工具**（依到職日試算特休天數，並帶入公司制度累進表對照勞基法）；
3. 記住對話脈絡，支援連續追問；
4. 以工具事件 / 串流即時呈現。

**價值：** 降低 HR 重複性諮詢工作量、縮短員工找答案的時間，並以引用來源降低制度誤答的合規風險。

---

## 二、架構

```
   使用者輸入      對話記憶                ┌─────────────────────────────────────────────┐
   (連續追問) ─►  HRAgent.messages ──────►│  Agent 主迴圈 common.run_tool_loop (Groq)     │
                       ▲                   │  system prompt + tools（OpenAI function call） │
                       │                   │  ── on_event 逐段輸出工具軌跡 ──► 使用者      │
                       │                   └───────────────┬─────────────┬───────────────┘
                       │            需要工具時              │             │ 收斂
                       │                                    ▼            （最終回答＋引用）
                       │                   ┌──────────────────────────────┐
                       │                   │ 工具層 (function calling)      │
                       │                   │ ① search_hr_knowledge_base    │
                       │                   │ ② calculate_annual_leave      │
                       │                   └───────┬───────────────┬──────┘
                       │        tool result        │  (資料互用)    │ tool result
                       └───────────────────────────┘               │
                                                   ▼               ▼
                    ┌─────────────────────────────────┐   模型把檢索到的公司制度累進表整理成
                    │ 進階 RAG Pipeline (rag.py)        │   accrual_table，帶入 calculate_annual_leave
                    │  1. Query Rewriting (multi-query) │   計算公司實際特休天數並對照勞基法基準
                    │  2. 向量粗召回 (Mistral + cosine) │
                    │  3. LLM 相關度重排 (top-N)        │
                    │  4. 相關度分數門檻 + 穩定來源編號 │
                    └───────────┬─────────────────────┘
              載入→切塊→嵌入→建索引 │
                    ┌────────────▼─────────────┐
                    │ 文件庫  data/*.md         │
                    │ Mistral embeddings (list) │
                    └───────────────────────────┘
```

### 檔案結構

| 檔案 | 職責 |
|---|---|
| `config.py` | 全域設定：RAG 調參、`.env` 載入（模型 / 供應商由共用層 env 決定） |
| `rag.py` | 進階 RAG：載入 `data/*.md` → 切塊 → Mistral 嵌入 → 多查詢召回 → **LLM 相關度重排** → 分數門檻；以及 Query Rewriting |
| `agent.py` | 工具定義（OpenAI function schema）、工具執行、對話記憶；以 `common.run_tool_loop` 跑 agent 迴圈 |
| `main.py` | CLI 入口（互動）：建索引、啟動自檢、互動問答 |
| `run_demo.py` | **非互動示範**：內建問題，印出檢索來源、工具軌跡、帶引用回答、串流 |
| `data/*.md` | 範例 HR 制度文件（請假、差旅、福利），帶 `title` / `source` front-matter |

### 技術選型與理由

| 元件 | 選型 | 理由 |
|---|---|---|
| LLM / Agent | **Groq `llama-3.3-70b-versatile`**（經 `common.run_tool_loop`） | 免費、快速、原生支援 OpenAI 相容 tool calling；共用層可靠 env 一鍵切換其他供應商 |
| Embedding | **Mistral `mistral-embed`**（經 `common.embed_texts`） | 1024 維、多語（含繁中）、走 API 免本地權重；通用對稱嵌入，查詢與段落共用同一種編碼、無需加特殊前綴 |
| 向量索引 | **純 Python list + `cosine_sim`** | 示範資料量小（數個切塊），免 FAISS 外部索引；正式環境資料量大時可換向量資料庫 |
| Rerank | **LLM 相關度評分**（`common.complete` 一次評所有候選） | 以一次呼叫對所有候選打 0~1 相關度分數、免載入本地重排權重；分數讓 `RERANK_MIN_SCORE` 仍是有意義的門檻、可觸發「查無規定」誠實拒答 |
| Query Rewriting | Groq（`common.complete`） | 追問常帶指代，改寫成獨立且多角度查詢提高召回；有對話脈絡才啟用以省成本 |
| 記憶 | `HRAgent.messages`（`run_tool_loop` 就地維護） | OpenAI 相容 messages 歷史，跨輪續談；同一 `HRAgent` 連續 `ask()` 即具記憶 |

---

## 三、安裝

需要 Python 3.10+（使用了 `list[str]`、`X | None` 型別註記）。

```bash
# 1) 建議使用虛擬環境
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

# 2) 安裝相依（僅 openai + python-dotenv，安裝輕量、無需本地 ML 權重）
pip install -r requirements.txt
```

> 本服務**不**下載任何本地 embedding / reranker 權重；嵌入與重排都走 API，安裝輕量。

---

## 四、環境變數

金鑰一律走環境變數 / `.env`，**不寫死於程式碼**。

```bash
cp .env.example .env
# 編輯 .env，填入 GROQ_API_KEY 與 MISTRAL_API_KEY
```

| 變數 | 必填 | 說明 |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM（Groq）金鑰，至 <https://console.groq.com> 取得 |
| `MISTRAL_API_KEY` | ✅ | Embedding（Mistral）金鑰，至 <https://console.mistral.ai> 取得 |
| `LLM_PROVIDER` / `LLM_MODEL` | ✗ | 覆寫 LLM 供應商 / 模型（預設 `groq` / `llama-3.3-70b-versatile`） |
| `EMBED_PROVIDER` / `EMBED_MODEL` | ✗ | 覆寫 Embedding 供應商 / 模型（預設 `mistral` / `mistral-embed`） |

也可直接以 shell 設定（不透過 `.env`）：

```bash
# macOS / Linux
export GROQ_API_KEY="gsk_..."; export MISTRAL_API_KEY="..."
# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."; $env:MISTRAL_API_KEY="..."
```

> `.env` 內含機密，請加入 `.gitignore`，切勿提交進版控。

---

## 五、執行方式與預期輸出

### （A）非互動示範：`run_demo.py`（建議先跑這個）

```bash
python run_demo.py
```

不需任何輸入，會依序：**建索引 → out-of-scope 自檢 → 連續兩題問答（含特休試算）→ 串流小結**，並印出**檢索到的來源、工具呼叫軌跡、最終帶引用的回答**。若缺金鑰會印出清楚指引並以非 0 退出（優雅降級）。預期輸出摘要：

```
步驟 1／3：建立知識庫（Mistral mistral-embed 向量化）
已載入文件並切塊：共 3 個切塊。
  - [HR-POL-009#0] 員工福利與獎金辦法（HR-POL-009 v1）
  - [HR-POL-002#0] 請假與特別休假辦法（HR-POL-002 v3）
  - [HR-POL-005#0] 國內外差旅報支辦法（HR-POL-005 v2）

步驟 2／3：out-of-scope 自檢（確認『查無規定』拒答路徑可觸發）
通過相關度門檻的段落數 = 0（應為 0）

步驟 3／3：連續問答（同一 HRAgent＝具對話記憶）
──────── 問題 1 ────────
使用者：我 2021-03-15 到職，以 2026-07-06 為基準，今年可以休幾天特休？…
  [工具呼叫] search_hr_knowledge_base 參數={'question': '公司特休制度'}
  [檢索到的來源]
     - [HR-POL-002#0] 請假與特別休假辦法（HR-POL-002 v3）：特別休假採「週年制」…
  [工具呼叫] calculate_annual_leave 參數={'accrual_table': [...], 'max_days': 30, ...}
  [工具回傳] {... "公司特休天數": 18, "勞基法特休天數": 15, "優於勞基法天數": 3}
助理（帶引用）：… 公司實際特休天數為 18 天，優於勞基法 3 天。
參考來源：[HR-POL-002#0] 請假與特別休假辦法（HR-POL-002 v3）
```

- 問題 1 會示範**多工具資料互用**：模型先檢索公司累進表，整理成 `accrual_table` 後**帶入 `calculate_annual_leave`** 算出公司實際天數（18）並對照勞基法（15）。
- 問題 2 為帶指代的追問，沿用**對話記憶**並自動觸發**多查詢改寫**解析指代，命中另一份文件來源。

### （B）互動 CLI：`main.py`

```bash
python main.py
```

啟動後建索引、跑自檢，然後可連續輸入問題（`exit` 離開）。同樣會逐段印出工具軌跡與帶引用回答。

---

## 六、已實作能力（與程式一致）

- ✅ **持續對話記憶（memory）**：`HRAgent.messages` 由 `common.run_tool_loop` 就地維護跨輪 messages（user / assistant / tool），同一 `HRAgent` 連續 `ask()` 即具記憶。
- ✅ **多工具回傳資料互用**：`search_hr_knowledge_base` 檢索到的制度條文，由模型整理成 `accrual_table`（累進表）與 `max_days`，**實際帶入 `calculate_annual_leave` 的工具參數**計算公司實際天數，並與工具回傳的勞基法基準對照——資料互用發生在**工具輸入層**，而非僅停在文字彙整。
- ✅ **進階 RAG**：Query Rewriting（multi-query 改寫）＋ **LLM 相關度重排**（一次呼叫對候選段落打 0~1 分數）＋ **相關度分數門檻**（低於門檻即誠實回覆查無規定，避免以出處包裝幻覺）＋ **穩定引用來源**（以「文件代號#切塊序號」如 `[HR-POL-002#0]` 標註，跨輪不重編）。
- ✅ **Structured Output（工具參數）**：兩個工具皆以 JSON Schema + `additionalProperties:false` 約束模型產出的工具參數。
- ✅ **工具軌跡 / 串流呈現**：`run_tool_loop` 的 `on_event` 逐段印出 `tool_call` / `tool_result` / `final`；`run_demo.py` 另以 `common.stream_complete` 展示逐字串流。

> **公司制度真正「優於」勞基法**：`data/leave_policy.md` 的累進天數（滿一年 10 日、滿三年 16 日、滿五年 18 日…）在每個年資級距皆高於 `agent.py._legal_leave_days()` 的勞基法基準（第 38 條），因此工具互用會產生**可見差異**（18 vs 15），而非數字相同的空示範。

---

## 七、限制與正式化建議

本專案為可執行的示範，落地正式環境前請注意：

- **對話狀態持久化**：記憶僅存於單一行程的 `HRAgent.messages`，無外部持久化。多 worker 部署時應把 session 狀態外置到 **Redis / DB**（以 `session_id` 存取），並加 TTL 與並發鎖。
- **記憶收斂與成本**：每輪把完整歷史（含整段檢索條文的 tool 回傳）累加，長對話 token 只增不減。可導入 **compaction / context editing**，或保留最近 N 輪＋摘要，並讓檢索條文以「引用摘要」回存而非全文。
- **LLM 重排的成本與延遲**：重排以一次 `complete` 呼叫評所有候選（省去本地模型），但仍是關鍵路徑上的一次 LLM 呼叫。熱門查詢可快取重排結果，或先以向量分數粗篩後只送少量候選重排。若重排呼叫失敗，程式會退化為「向量分數排序」（此退化路徑下拒答門檻不保證觸發，需監控）。
- **示範資料量**：`data/` 僅 3 份短文件（約數個切塊），`RETRIEVE_TOP_K` / `RERANK_TOP_N` 在此規模形同虛設；正式化時應擴充知識庫或依規模調參（見 `config.py`）。
- **資料治理與存取控管**：知識庫含薪酬、健檢、生育婚喪等敏感內容。正式環境應納入**依部門 / 角色的資料分區與檢索 ACL、輸入輸出 PII 遮罩與稽核**。
- **門檻校準**：`RERANK_MIN_SCORE=0.50`（LLM 相關度分數）為示意值，須以標註驗證集校準；程式啟動 / demo 的 out-of-scope 自檢即用來確認拒答路徑真的可觸發。
- **持久化索引**：目前每次啟動重新嵌入。正式環境可持久化 chunk 向量與 metadata，啟動時載入避免重嵌入。

### 可能的延伸

- **Hybrid Search**（BM25 + dense，RRF 融合）改善數字 / 代碼型查詢；Contextual Retrieval；以 `ragas` 做 faithfulness / answer-relevancy 離線評測。
- **Multi-Agent**：拆成「檢索 Agent ＋回答 Agent ＋合規稽核 Agent（驗證僅引用檢索內容、無幻覺）」。
- **供應商彈性**：共用層支援以 `LLM_PROVIDER` / `EMBED_PROVIDER` 切換 OpenRouter / Cerebras / Together / NVIDIA 等 OpenAI 相容端點。
