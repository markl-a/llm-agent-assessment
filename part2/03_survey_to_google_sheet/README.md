# 問卷 AI Agent（結果寫入 Google Sheet／本地 CSV）

一個「對話式問卷 → 寫入資料 sink」的命令列 Agent。透過終端機逐題問答收集「產品使用回饋」，
以 LLM 把選擇題的口語輸入解析成合法選項，最後把整份作答附加一列寫進資料 sink，並在每列前置版本欄與 UTC 時間戳。

LLM 呼叫走 **`part2/common` 共用 provider 抽象層**（OpenAI 相容），預設 **LLM = Groq `llama-3.3-70b-versatile`**、
Embedding = Mistral `mistral-embed`（本專案只用 LLM）。**核心相依僅 `openai` SDK，不使用任何本地 ML 模型**，
供應商 / 模型 / 金鑰全部由環境變數決定。

資料 sink **可插拔**：偵測到 Google service account 憑證時走 **gspread → Google Sheet**；
**偵測不到憑證時自動降級為寫本地 CSV（`responses_local.csv`）**，並在輸出明確標示已略過 Google Sheet。

重點特性：

- **對話式作答**：支援 `/back` 回上一題、`/skip` 略過選填題、`/cancel` 中止；送出前可複查、依題號重答。
- **LLM 只當「答案解析器」**：把「差不多每天用吧」對應到 `每天`、「儀表板跟 API」對應到 `["儀表板","API"]`；
  要求模型**只輸出符合 schema 的 JSON**，程式端再用 pydantic 驗一次、並以題目白名單比對，攔截模型幻覺。
- **開放題不進 LLM**：稱呼、改善建議原樣保留，省成本也避免硬套選項。
- **安全防護（重點）**：
  1. **公式/CSV 注入防護** —— 寫入前對以 `=` `+` `-` `@`（含 `\t \r \n` 前綴）開頭的自由文字加前導單引號，
     且資料列**一律以 `RAW` 寫入**（Google Sheet 路徑），本地 CSV 亦沿用同一淨化，杜絕自由文字被當公式執行。
  2. **本地持久化備援** —— 送出前先把整份作答落地成本地 JSON，寫入採指數退避重試，仍失敗則保留待補檔，
     下次啟動自動補送，**任何網路／配額／權限問題都不會讓已收集的答案憑空消失**。

---

## 一、問卷設計

主題為「產品使用回饋」，刻意混合三種型態（單選 `single`／多選 `multi`／開放 `open`）與必填／選填。
每題有四個屬性：`id`（對應欄位）、`type`、`options`、`required`。

| # | 題目 (id) | 型態 | 選項 | 必填 |
|---|---|---|---|---|
| 1 | 稱呼 (`name`) | 開放題 | — | 選填 |
| 2 | 主要使用角色 (`role`) | 單選 | 開發者 / 產品經理 / 設計師 / 其他 | 必填 |
| 3 | 使用頻率 (`frequency`) | 單選 | 每天 / 每週 / 每月 / 很少 | 必填 |
| 4 | 最常使用的功能 (`features`) | 多選 | 儀表板 / 報表匯出 / API / 通知 / 協作 / 都不使用 | 必填 |
| 5 | 整體滿意度 (`satisfaction`) | 單選 | 1 / 2 / 3 / 4 / 5 | 必填 |
| 6 | 最希望改善之處 (`improvement`) | 開放題 | — | 選填 |
| 7 | 是否願意後續訪談 (`interview`) | 單選 | 願意 / 不願意 | 必填 |

題庫定義於 `survey_agent.py` 的 `QUESTIONS`，並以 `SURVEY_VERSION` 標記版本；每列都會寫入該版本。

---

## 二、對話流程與驗證

- **狀態物件**：以 `answers: dict` 保存已收集答案，`idx` 為題號游標；整份流程是「以 `idx` 為游標的 while 迴圈」，
  因此 `/back`（`idx -= 1`）與「回上一題重答」得以實作。
- **導覽指令（規則式、不進 LLM）**：`/back` 回上一題、`/skip` 略過（僅選填題）、`/cancel` 中止並落地已收集作答。
- **驗證分層**：先做規則驗證（空輸入、必填、略過），再交由 LLM 把選擇題自由文字對應到合法選項；
  LLM 回傳後**於程式端再驗一次**（單選須恰一個合法選項、多選為合法子集合），並先**保序去重、正規化**（strip／大小寫）
  後才比對，只接受題庫定義內的字串。
- **收斂設計**：每題設最大嘗試次數（`MAX_ATTEMPTS`）；逾次降級為「保留原始輸入待人工檢視」，
  確保流程一定收斂、LLM 呼叫成本有上限。
- **多輪脈絡**：`needs_clarification` 後的下一輪，會把前一輪輸入與追問一併帶回解析上下文。
- **末段複查**：全部收集完後印出摘要，可輸入題號重答、按 Enter 送出、或 `/cancel` 放棄。

**LLM 解析（`parse_answer`）**：呼叫共用層的 `common.complete(system, user, temperature=0.0, max_tokens=512)`，
於 system prompt 要求模型**只輸出 JSON**（`status` / `selected_options` / `clarification`），程式端以 `_extract_json`
容錯抽出 JSON（可去除 ```json``` 包裹）後，用 `pydantic` 的 `AnswerParse` 驗證。互動情境重延遲，`temperature=0`
求穩定、`max_tokens` 小以控成本；品質退化由末段複查與程式端選項白名單兜底。

---

## 三、資料 sink：Google Sheet（有憑證）／本地 CSV（無憑證）

`save_record()` 依環境自動選擇 sink：

- **有憑證**：設定了 `GOOGLE_SERVICE_ACCOUNT_FILE`（且檔案存在）與 `SPREADSHEET_ID`，並安裝了 `gspread`/`google-auth`
  → 走 `gspread` 以 `open_by_key` 開啟試算表，寫入 `Responses` 分頁（分頁不存在會自動建立並寫一次表頭）。
- **無憑證**：偵測不到 `GOOGLE_SERVICE_ACCOUNT_FILE`（或未安裝 gspread）→ **自動降級寫本地 CSV `responses_local.csv`**
  （首寫自動補表頭，`utf-8-sig` 讓 Excel 直接開啟不亂碼），並在輸出標示
  「Google Sheet 寫入略過（無憑證），已寫入本地 CSV」。

兩條路徑共用同一組欄位與同一套注入防護（`_build_row` → `_sanitize_cell`）。

### Google Cloud 設定步驟（僅在你要真的寫進 Google Sheet 時才需要）

1. 到 [Google Cloud Console](https://console.cloud.google.com/) 建立（或選擇）一個專案。
2. 於「API 和服務 → 程式庫」中**啟用 Google Sheets API**。
3. 於「憑證 → 建立憑證 → 服務帳戶」建立一個 **Service Account**，並為它建立 **JSON 金鑰**，
   下載另存為例如 `service_account.json`（**切勿提交進版本庫**）。
4. 打開該 JSON，複製 `client_email`，開啟目標試算表按「共用」，**把該 email 加為「編輯者」**
   —— 這是**最常見的失敗點**（未分享會得到 403）。
5. 從試算表網址取得 **SPREADSHEET_ID**：`https://docs.google.com/spreadsheets/d/<這一段>/edit`。

---

## 四、安裝

需要 Python 3.9+（建議 3.10+）。

```bash
# （建議）建立虛擬環境
python -m venv .venv
# Windows: .venv\Scripts\activate ；macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

> `gspread`／`google-auth` 為**選用**：只有在你要寫進 Google Sheet 時才需要。
> 只跑 `run_demo.py`（本地 CSV）時，核心相依其實只有 `openai`、`pydantic`、`python-dotenv`。

---

## 五、環境變數設定

複製範本後填入你自己的值：

```bash
cp .env.example .env   # Windows PowerShell: Copy-Item .env.example .env
```

`.env` 內容（`survey_agent.py` / `run_demo.py` 會透過 python-dotenv 自動載入；也可改用系統環境變數）：

| 變數 | 必要性 | 說明 |
|---|---|---|
| `GROQ_API_KEY` | **必要** | Groq API 金鑰（預設 LLM 供應商，供選擇題答案解析）。到 https://console.groq.com/keys 取得。 |
| `LLM_PROVIDER` / `LLM_MODEL` | 選用 | 切換供應商 / 模型（留空用預設 Groq `llama-3.3-70b-versatile`）。 |
| `MISTRAL_API_KEY` | 選用 | 共用層 Embedding 預設走 Mistral；本專案未使用 embedding，可不填。 |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | 選用 | service account JSON 金鑰的**檔案路徑**。未設定時 sink 自動降級為本地 CSV。 |
| `SPREADSHEET_ID` | 選用 | 目標試算表 ID（與上一項一起提供時才走 Google Sheet）。 |

> 金鑰**一律走環境變數**，程式內不寫死。切勿把真實金鑰或 `.env`／`service_account.json` 提交進版本庫。

---

## 六、執行方式

### 1) 非互動 Demo（推薦先跑，不需 Google 憑證）

```bash
python run_demo.py
```

- 只需設定 `GROQ_API_KEY`。
- 以一位受訪者的**腳本化回答**逐題餵入既有對話流程（走與真人作答相同的程式路徑），其中：
  - 多題以**自由文字**作答，交給 LLM 歸類（如「差不多每天都會打開來用」→ `每天`、
    「最常看儀表板，偶爾也會串接 API」→ `儀表板；API`）；
  - 改善建議欄故意輸入 `=HYPERLINK("https://evil.example/steal","點我領獎")` 這種**惡意公式**，驗證注入防護。
- 無 Google 憑證 → 自動寫入本地 `responses_local.csv`。
- 跑完會印出：**實際寫入的每一欄**、sink 結果、以及**注入防護「前 / 後」的值對照**。

預期輸出（節錄）：

```
sink 結果：local_csv —— Google Sheet 寫入略過（無憑證），已寫入本地 CSV：.../responses_local.csv

公式/CSV 注入防護示範（improvement 欄）：
  防護前（受訪者原始輸入）：'=HYPERLINK("https://evil.example/steal","點我領獎")'
  防護後（實際寫入值）    ："'=HYPERLINK(\"https://evil.example/steal\",\"點我領獎\")"
  → 已加前導單引號、且以純文字寫入…：是
```

（LLM 呼叫次數＝選擇題數量 5 次，`temperature=0`、`max_tokens` 小，速度與成本可控。）

### 2) 互動作答（真人逐題輸入）

```bash
python survey_agent.py
```

- 啟動時先嘗試補送 `pending_submissions/` 內先前積壓的作答，接著逐題作答，全部完成後印出摘要供複查，按 Enter 送出。
- 有 Google 憑證 → 寫入 `Responses` 分頁；無憑證 → 降級寫 `responses_local.csv`。

欄位順序（Google Sheet 與本地 CSV 一致）：

```
survey_version | timestamp | respondent_id | name | role | frequency | features | satisfaction | improvement | interview
```

若因網路／配額暫時失敗（Google Sheet 路徑），作答會保存於 `pending_submissions/<uuid>.json`，下次啟動自動補送。

---

## 七、安全防護細節

### 1. 公式 / CSV 注入（嚴重）
不可信自由文字（稱呼、改善建議）若以 `=` `+` `-` `@` 開頭，可能被 Google Sheets 或匯出 CSV 後的
試算表軟體當公式解析執行。本專案採「雙重防線」：

- `_build_row` 逐格呼叫 `_sanitize_cell`：對以 `=/+/-/@`（含 `\t \r \n` 前綴）開頭者加**前導單引號**，強制當純文字。
- Google Sheet 路徑資料列**一律以 `value_input_option="RAW"`** 寫入（不使用 `USER_ENTERED`）；本地 CSV 亦寫入已淨化的值。

合法選項字串（如「開發者」「每天」「1」）不會以危險字元開頭，故上述淨化對它們是無害的 no-op。

### 2. 寫入失敗的本地持久化備援（嚴重）
- **先落地、後寫入**：`_new_record` 產生含 `respondent_id`（即冪等鍵）的紀錄，`_persist_pending` 先存成本地 JSON，
  之後才 `save_record` 寫入 sink；**寫入成功才移除**待補檔。
- **指數退避重試**（Google Sheet 路徑）：可重試錯誤（429/5xx/連線）退避重試；不可重試錯誤（403/404/憑證/設定錯）直接放棄並保留待補檔。
- **自動補送**：下次啟動 `resend_pending` 會補送 `pending_submissions/` 內所有待補檔（同樣走可插拔 sink）；單一壞檔不影響其他檔。

---

## 八、錯誤處理

| 類別 | 徵狀 | 處置 |
|---|---|---|
| LLM 設定/金鑰（不可重試） | 缺 `GROQ_API_KEY`、金鑰無效、`LLM_PROVIDER` 錯 | `run_agent` 開頭以 `get_llm()` 預檢即報錯；解析階段偵測到金鑰/權限特徵字串則以 `SurveyAborted` 中止並落地作答。 |
| LLM 暫時性（可重試） | 429 / 5xx / 連線 / 逾時 | 共用層已內建指數退避重試；仍失敗則提示「請換個說法再試一次」就地重試，不中斷整份問卷。 |
| Sheets 權限（不可重試） | gspread `APIError` **403/404** | ID 錯或未把試算表分享給 service account；給友善分享提示、保留待補檔。 |
| Sheets 配額/服務（可重試） | gspread `APIError` **429 / 5xx** / 連線 | `save_with_retry` 指數退避重試。 |
| 無 Google 憑證 | 未設定 `GOOGLE_SERVICE_ACCOUNT_FILE` | **不是錯誤**：自動降級寫本地 CSV，並在輸出標示已略過 Google Sheet。 |
| 中止 | EOF / Ctrl+C / `/cancel` | `prompt()` 攔截 EOF/Ctrl+C，優雅結束並落地部分作答。 |

---

## 九、與共用層（`part2/common`）的整合

- `survey_agent.py` 於檔首把 `part2/` 加入 `sys.path`，`from common import complete` 取得 LLM 呼叫入口。
- 只用到 `complete()`（選擇題解析）；本專案不需 embedding。若日後要做語意去重／相似回覆聚類，
  可改用 `common.embed_texts` / `cosine_sim`（走 Mistral API，非本地模型）。
- 供應商切換完全靠環境變數：`LLM_PROVIDER`（預設 groq）、`LLM_MODEL`、`GROQ_API_KEY` 等。

---

## 十、檔案結構

```
part2/03_survey_to_google_sheet/
├─ survey_agent.py       # 主程式：問卷結構、對話迴圈、LLM 解析(common 層)、可插拔 sink、落地/重試/補送
├─ run_demo.py           # 非互動 Demo：腳本化回答跑完整流程 → 本地 CSV，並展示注入防護前後值
├─ requirements.txt      # 依賴：openai、pydantic、python-dotenv（gspread/google-auth 選用）
├─ .env.example          # 環境變數範本（GROQ_API_KEY 必要；Google 為選用）
└─ README.md             # 本說明
# 執行後可能產生：
# ├─ responses_local.csv          # 無 Google 憑證時的降級輸出
# ├─ .env                         # 你的環境變數（請勿提交）
# └─ pending_submissions/         # 寫入失敗時暫存的待補作答（成功後自動移除）
```
