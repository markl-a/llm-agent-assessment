"""對話式問卷 Agent：逐題收集答案，用 LLM 解析選擇題，最後把整份作答寫入資料 sink。

設計重點（詳見 README.md）：
- 問卷結構以 dataclass 定義；對話迴圈以 idx 為游標的 while 迴圈，支援 /back、/skip、/cancel。
- LLM 只當「選擇題答案解析器」：把自由文字對應到合法選項；程式端再以題目白名單驗一次。
- 開放題（稱呼、改善建議）原樣保留、不進 LLM。
- 寫入前對自由文字做「公式/CSV 注入」防護（前導單引號 + RAW 寫入）。
- 寫入前先落地本地 JSON、失敗指數退避重試、仍失敗保留待補檔，任何情況都不遺失作答。
- 資料 sink 可插拔：偵測到 Google service account 憑證時走 gspread；偵測不到時自動降級寫本地 CSV。

LLM / Embedding 呼叫一律走 part2/common 共用抽象層（OpenAI 相容，預設 LLM=Groq、Embedding=Mistral），
本檔不依賴任何本地 ML 模型；供應商/模型/金鑰皆由環境變數決定。

依賴：openai（透過 common 層）、pydantic、python-dotenv（選用）、gspread + google-auth（選用，僅在有憑證時）。
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, ValidationError

# 把 part2/ 加入模組搜尋路徑，讓各專案能 `from common import ...`。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import complete  # noqa: E402 —— 共用 provider 抽象層（OpenAI 相容）

# python-dotenv 為選用相依：若未安裝，仍可直接以系統環境變數執行。
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 選用相依
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

# gspread / google-auth 為選用相依：只有在提供 service account 憑證時才會用到。
# 未安裝時整支程式仍可執行（資料 sink 自動降級為本地 CSV）。
try:  # pragma: no cover - 視環境而定
    import gspread
    from google.oauth2.service_account import Credentials as _GCredentials
except ImportError:  # pragma: no cover
    gspread = None
    _GCredentials = None


logger = logging.getLogger("survey_agent")


# =============================================================================
# (i) 問卷結構定義
# =============================================================================

SURVEY_VERSION = "1"  # 題庫版本；寫入每一列，題目增改時據此做欄位對映/遷移。


@dataclass
class Question:
    id: str                                    # 欄位鍵，對應輸出欄位
    text: str                                  # 題目文字
    qtype: Literal["single", "multi", "open"]  # 型態
    options: List[str] = field(default_factory=list)  # 選項候選（開放題為空）
    required: bool = True                       # 必填 / 選填


QUESTIONS: List[Question] = [
    Question("name", "方便怎麼稱呼你？", "open", required=False),
    Question("role", "你主要的使用角色是？", "single",
             ["開發者", "產品經理", "設計師", "其他"]),
    Question("frequency", "你多常使用本產品？", "single",
             ["每天", "每週", "每月", "很少"]),
    Question("features", "你最常使用哪些功能？（可多選）", "multi",
             ["儀表板", "報表匯出", "API", "通知", "協作", "都不使用"]),
    Question("satisfaction", "整體滿意度（1 很差 ~ 5 很好）？", "single",
             ["1", "2", "3", "4", "5"]),
    Question("improvement", "最希望我們改善的地方？", "open", required=False),
    Question("interview", "是否願意接受後續使用者訪談？", "single",
             ["願意", "不願意"]),
]


# =============================================================================
# (ii) LLM 解析與對話迴圈
# =============================================================================

# 導覽/略過/取消：不易與自由文字碰撞的前綴指令。
NAV_BACK, NAV_SKIP, NAV_CANCEL = "/back", "/skip", "/cancel"
MAX_ATTEMPTS = 3      # 每題最大嘗試次數，逾次降級以收斂、控成本。
OPEN_TEXT_MAX = 500   # 開放題長度上限（視為不可信輸入做基本清理）。


class SurveyAborted(Exception):
    """使用者中止（EOF/Ctrl+C 或 /cancel），或遇不可重試致命錯誤——需結束並落地已收集作答。"""


class AnswerParse(BaseModel):
    """LLM 對『選擇題』的結構化輸出。

    共用 common 層採 OpenAI 相容 chat：要求 LLM「只輸出符合此 schema 的 JSON」，
    程式端再用本模型驗證一次（缺欄位/型別錯 → 視為解析失敗）。
    """

    status: Literal["ok", "needs_clarification", "out_of_scope"]
    selected_options: List[str] = []  # 選擇題正規化後的合法選項；needs/out_of_scope 時留空。
    clarification: str = ""           # status != ok 時，給使用者的追問話術。


PARSE_SYSTEM = (
    "你是問卷『選擇題』答案解析器。會收到一題的定義（型態、合法選項、是否必填）與受訪者的自由輸入，"
    "請把輸入對應到合法選項並回傳結構化結果：\n"
    "- 單選(single)：status=ok 時 selected_options 必須是【恰一個】合法選項字串。\n"
    "- 多選(multi)：selected_options 為合法選項的子集合，可多個。\n"
    "- 若語意模糊無法確定選項，status=needs_clarification，並在 clarification 給一句友善追問。\n"
    "- 若答非所問或拒答，status=out_of_scope，clarification 引導回本題。\n"
    "- 若附上『先前對話』脈絡，請延續理解、不要重複追問已釐清處，並避開先前已排除的選項。\n"
    "selected_options 只能使用題目定義中的原字串，不可自創。\n\n"
    "【輸出格式（務必嚴格遵守）】只輸出一個 JSON 物件，不要有任何多餘文字、說明或 markdown code fence，"
    "鍵為：{\"status\": \"ok|needs_clarification|out_of_scope\", "
    "\"selected_options\": [字串, ...], \"clarification\": \"字串\"}。"
)


def _extract_json(raw: str) -> Optional[dict]:
    """從 LLM 回覆抽出第一個 JSON 物件。容忍 ```json ...``` 包裹與前後雜訊；失敗回 None。"""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1 and s[:nl].strip().lower() in ("json", ""):
            s = s[nl + 1:]
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_answer(q: Question, user_reply: str, prior_turns=()) -> Optional[AnswerParse]:
    """呼叫 common 層 LLM 把選擇題的自由輸入解析成結構化結果。解析失敗時回傳 None。"""
    ctx = {"type": q.qtype, "text": q.text, "options": q.options, "required": q.required}
    parts = [f"題目定義(JSON)：\n{json.dumps(ctx, ensure_ascii=False)}"]
    if prior_turns:
        # 澄清情境：把前幾輪輸入與追問一併帶入，補足單則解析看不到的脈絡。
        history = "\n".join(f"- 受訪者說：「{r}」→ 追問：{c}" for r, c in prior_turns)
        parts.append(f"先前對話（尚未成功對應）：\n{history}")
    parts.append(f"受訪者最新輸入：\n{user_reply}")

    # 走共用抽象層：temperature=0 求穩定、max_tokens 收斂控成本（只需一小段 JSON）。
    raw = complete(PARSE_SYSTEM, "\n\n".join(parts), temperature=0.0, max_tokens=512)
    data = _extract_json(raw)
    if data is None:
        logger.info("parse_answer id=%s 無法解析 JSON 輸出", q.id)
        return None
    try:
        parsed = AnswerParse(**data)
    except (ValidationError, TypeError) as e:
        logger.info("parse_answer id=%s schema 驗證失敗：%s", q.id, e)
        return None
    logger.info("parse_answer id=%s status=%s", q.id, parsed.status)
    return parsed


def _norm(s: str) -> str:
    return s.strip().casefold()


def _match_options(selected: List[str], options: List[str]) -> List[str]:
    """保序去重 + strip/大小寫正規化後比對，回傳題目定義中的『原字串』。"""
    canon = {_norm(o): o for o in options}
    seen, result = set(), []
    for s in selected:
        o = canon.get(_norm(s))
        if o is not None and o not in seen:
            seen.add(o)
            result.append(o)
    return result


# 設定/金鑰類錯誤的特徵字串：這類錯誤重試無用，應中止並提示修正設定。
_FATAL_LLM_HINTS = (
    "api_key", "api key", "unauthorized", "401", "403",
    "invalid api key", "permission", "缺少", "未知的 llm_provider",
)


def resolve_answer(q: Question, reply: str, prior_turns=()):
    """回傳 (value, error)。error 非 None 代表需停在原題重試；致命錯誤改以 SurveyAborted 上拋。"""
    # 1) 空輸入
    if not reply:
        if q.required:
            return None, "這一題為必填，請作答（/skip 略過僅適用選填題）。"
        return ([] if q.qtype == "multi" else ""), None

    # 2) 開放題：原樣保留、不進 LLM（自由文字不需正規化）；做長度清理，寫入時再做公式防護。
    if q.qtype == "open":
        return reply[:OPEN_TEXT_MAX], None

    # 3) 選擇題交給 LLM 正規化；區分「可重試」與「不可重試（設定/金鑰）」錯誤。
    #    common 層對 429/5xx/連線已內建退避重試，故此處拋出多屬設定或已逾次的暫時錯誤。
    try:
        result = parse_answer(q, reply, prior_turns)
    except SurveyAborted:
        raise
    except Exception as e:  # noqa: BLE001 —— 統一分類 LLM 呼叫錯誤
        msg = str(e).lower()
        if any(h in msg for h in _FATAL_LLM_HINTS):
            raise SurveyAborted(
                f"解析服務設定有誤（{type(e).__name__}）：請檢查 LLM 供應商金鑰（如 GROQ_API_KEY）"
                "與 LLM_PROVIDER/LLM_MODEL。這類錯誤重試無用，已中止作答。"
            ) from e
        # 其餘（暫時性/超時/已逾內建重試）：就地重試，不中斷整份問卷。
        return None, "抱歉，這次沒能解析你的回答，請換個說法再試一次。"

    if result is None or result.status != "ok":
        return None, (result.clarification if result else "") or "抱歉，我不太理解，請換個說法。"

    # 4) 防幻覺 + 邊界：保序去重（避免 ["每天","每天"] 被誤判為多於一個）+ 正規化比對。
    legal = _match_options(result.selected_options, q.options)
    if q.qtype == "single":
        if len(legal) != 1:
            return None, "請明確選擇其中一個選項。"
        return legal[0], None
    if not legal:
        return None, "請至少選擇一個有效選項（若都不用可選「都不使用」）。"
    return legal, None  # multi -> list


# 輸入來源可插拔：預設互動 input()，run_demo 等可注入腳本化回答（非互動）。
_INPUT_FN: Optional[Callable[[str], str]] = None


def set_input_fn(fn: Optional[Callable[[str], str]]) -> None:
    """注入輸入來源（回傳已 strip 的字串）；傳 None 還原為互動 input()。"""
    global _INPUT_FN
    _INPUT_FN = fn


def prompt(label: str) -> str:
    fn = _INPUT_FN
    if fn is not None:
        return fn(label)
    try:
        return input(label).strip()
    except (EOFError, KeyboardInterrupt):
        # 管線導入(EOF) 或 Ctrl+C：優雅中止，交由上層落地已收集作答。
        raise SurveyAborted()


def render_question(q: Question) -> str:
    tag = "必填" if q.required else "選填，可輸入 /skip 略過"
    body = f"{q.text}（{tag}）"
    if q.options:
        # 不加「1. 2.」序號：序號→選項映射未餵給模型，且會與 satisfaction 的數字選項重疊。
        body += "\n  可選：" + "　".join(q.options)
    return body


def ask_question(q: Question, allow_back: bool):
    """單題的『解析單則輸入（含導覽/略過/取消 + 重試上限）』，供作答與複查兩處共用。"""
    print("\n" + render_question(q))
    prior_turns = []  # 累積本題的 (輸入, 追問)，供澄清情境帶入脈絡。
    last_reply = ""
    for _ in range(MAX_ATTEMPTS):
        reply = prompt("你：")
        if reply == NAV_BACK:
            if allow_back:
                return "back", None
            print("　已經是第一題，無法返回上一題。")
            continue
        if reply == NAV_CANCEL:
            return "cancel", None
        if reply == NAV_SKIP:
            if q.required:
                print("　這一題為必填，不能略過。")
                continue
            return "value", ([] if q.qtype == "multi" else "")
        last_reply = reply
        value, error = resolve_answer(q, reply, prior_turns)
        if error is None:
            return "value", value
        print("　" + error)
        prior_turns.append((reply, error))
    # 逾次降級：保留原始輸入待人工檢視，確保流程收斂、LLM 呼叫成本有上限。
    print("　已多次無法辨識，先保留你的原始輸入待人工檢視，繼續下一題。")
    return "value", f"[待人工檢視] {last_reply}"


def run_survey(answers: dict) -> None:
    """以 idx 為游標逐題收集；idx 可前進或回退，支援 /back 與 /cancel。"""
    idx = 0
    while idx < len(QUESTIONS):
        q = QUESTIONS[idx]
        outcome, value = ask_question(q, allow_back=(idx > 0))
        if outcome == "back":
            idx -= 1
            continue
        if outcome == "cancel":
            raise SurveyAborted()
        answers[q.id] = value
        idx += 1


def review_and_confirm(answers: dict) -> None:
    """送出前複查：可輸入題號重答、按 Enter 送出、或 /cancel 放棄。"""
    while True:
        print("\n=== 你的回答 ===")
        for i, q in enumerate(QUESTIONS, start=1):
            v = answers.get(q.id, "")
            v = "；".join(v) if isinstance(v, list) else v
            print(f"[{i}] {q.text} → {v or '(未填)'}")
        cmd = prompt("輸入題號可修改，直接按 Enter 送出，或 /cancel 放棄：")
        if cmd == "":
            return
        if cmd == NAV_CANCEL:
            raise SurveyAborted()
        if cmd.isdigit() and 1 <= int(cmd) <= len(QUESTIONS):
            q = QUESTIONS[int(cmd) - 1]
            # 複查重答共用同一支 ask_question：一致支援 /skip、/cancel，並以 /back 取消回摘要。
            print("（重答時：/back 取消修改回摘要、/skip 略過選填、/cancel 放棄）")
            outcome, value = ask_question(q, allow_back=True)
            if outcome == "cancel":
                raise SurveyAborted()
            if outcome == "back":
                continue  # 取消修改，回到摘要
            answers[q.id] = value
        else:
            print("　請輸入有效題號，或按 Enter 送出。")


# =============================================================================
# (iii) 資料 sink 與注入防護（可插拔：Google Sheet / 本地 CSV）
# =============================================================================

WORKSHEET_NAME = "Responses"
LOCAL_CSV = "responses_local.csv"  # 無 Google 憑證時的降級輸出檔。
# 只用 open_by_key 開啟時，spreadsheets scope 即足夠（add_worksheet 亦屬同一 scope）。
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
HEADER = ["survey_version", "timestamp", "respondent_id"] + [q.id for q in QUESTIONS]

# 以 =/+/-/@ 開頭的儲存格可能被 Google Sheets 或匯出後的 CSV 工具當公式解析（公式/CSV 注入）。
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _sanitize_cell(value):
    """對『可能被當公式解析』的自由文字做防護：以 =/+/-/@（含 \\t \\r \\n 前綴）開頭者加前導單引號。

    即使資料列一律以 RAW 寫入（Sheets 端不會執行公式），日後把試算表匯出成 CSV 再於
    Excel/試算表軟體開啟時，這類字串仍可能被當公式執行——加前導單引號可強制當純文字。
    本地 CSV 輸出亦沿用同一防護。非字串或空字串原樣回傳；合法選項字串不會以危險字元開頭，
    因此對它們是無害的 no-op。
    """
    if not isinstance(value, str) or value == "":
        return value
    stripped = value.lstrip("\t\r\n")
    if stripped[:1] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def _build_row(record: dict) -> List:
    """把一筆紀錄攤平成一列（多選以「；」串接），並逐格套用公式/CSV 注入防護。"""
    row = [record["survey_version"], record["timestamp"], record["respondent_id"]]
    for q in QUESTIONS:
        v = record["answers"].get(q.id, "")
        row.append("；".join(v) if isinstance(v, list) else v)  # 多選攤平為字串
    return [_sanitize_cell(c) for c in row]


# ---- 本地 CSV sink（降級預設） -------------------------------------------------

def write_local_csv(record: dict, path: str = LOCAL_CSV) -> str:
    """把一筆作答附加到本地 CSV（首寫自動補表頭）。回傳實際寫入的絕對路徑。

    使用 utf-8-sig 讓 Excel 直接開啟不亂碼；欄值已於 _build_row 做注入防護。
    """
    row = _build_row(record)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(HEADER)
        w.writerow(row)
    return os.path.abspath(path)


# ---- Google Sheet sink（有憑證時） --------------------------------------------

class _FatalWriteError(Exception):
    """不可重試（403/404/憑證/設定錯）——重試無用，直接放棄並保留待補檔。"""


class _RetryableWriteError(Exception):
    """可重試（429/5xx/連線）——指數退避後再試。"""


def _credentials_available() -> bool:
    """是否偵測到可用的 Google service account 憑證（且已安裝 gspread）。"""
    path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    return bool(path) and os.path.isfile(path) and gspread is not None and _GCredentials is not None


def _classify_gspread_error(e) -> Exception:
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in (403, 404):
        # open_by_key 對錯誤或未分享的 ID 會拋 APIError(403/404)；403（未分享給 service account）
        # 是最常見的失敗點——這裡給友善的分享提示。
        return _FatalWriteError(
            "無法開啟試算表：請確認 SPREADSHEET_ID 正確，且已把試算表分享給 "
            "service account email（未分享是最常見原因）。"
        )
    if code == 429 or (code or 0) >= 500:
        return _RetryableWriteError(f"Sheets API 暫時性錯誤（{code}）：{e}")
    return _FatalWriteError(f"Sheets API 錯誤：{e}")


def _create_worksheet(sh):
    try:
        ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=1000, cols=len(HEADER) + 5)
    except gspread.exceptions.APIError:
        # 併發首寫：分頁標題須唯一，另一行程可能已建立同名分頁 → 改用既有分頁，避免該筆遺失。
        return sh.worksheet(WORKSHEET_NAME)
    # 表頭「於建表當下」寫一次：避免每次送出前 row_values(1) 檢查的 TOCTOU 與重複表頭列。
    ws.append_row(HEADER, value_input_option="RAW")
    return ws


def get_worksheet():
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")
    if gspread is None or _GCredentials is None:
        raise _FatalWriteError("未安裝 gspread / google-auth，無法寫入 Google Sheet。")
    if not service_account_file or not spreadsheet_id:
        raise _FatalWriteError("缺少 GOOGLE_SERVICE_ACCOUNT_FILE 或 SPREADSHEET_ID 環境變數。")
    try:
        creds = _GCredentials.from_service_account_file(service_account_file, scopes=SCOPES)
    except FileNotFoundError:
        raise _FatalWriteError(f"找不到憑證檔：{service_account_file}")
    except Exception as e:  # MalformedError / ValueError 等：JSON 無法解析
        raise _FatalWriteError(f"憑證 JSON 無法解析：{e}")
    gc = gspread.authorize(creds)
    try:
        sh = gc.open_by_key(spreadsheet_id)
        try:
            return sh.worksheet(WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            return _create_worksheet(sh)
    except gspread.exceptions.APIError as e:
        raise _classify_gspread_error(e)
    except (ConnectionError, TimeoutError) as e:
        raise _RetryableWriteError(f"連線 Google 服務失敗：{e}")


def append_response(ws, record: dict):
    # 資料列一律 RAW：既杜絕自由文字被當公式解析，也讓 ISO-8601 時間戳與「4」等值以純文字精確保存。
    ws.append_row(_build_row(record), value_input_option="RAW")


def _write_once(record: dict):
    try:
        ws = get_worksheet()
        append_response(ws, record)
    except (_FatalWriteError, _RetryableWriteError):
        raise
    except gspread.exceptions.APIError as e:  # append 階段的 API 錯誤也分類處理
        raise _classify_gspread_error(e)
    except (ConnectionError, TimeoutError) as e:
        raise _RetryableWriteError(f"連線 Google 服務失敗：{e}")


def save_with_retry(record: dict, max_attempts: int = 4) -> bool:
    """寫入 Google Sheet；可重試錯誤指數退避，逾次或遇不可重試錯誤則回 False（保留待補檔）。"""
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            _write_once(record)
            return True
        except _FatalWriteError as e:
            logger.error("寫入失敗（不可重試）：%s", e)
            print(f"　{e}")
            return False
        except _RetryableWriteError as e:
            logger.warning("寫入暫時失敗（第 %d/%d 次）：%s", attempt, max_attempts, e)
            if attempt == max_attempts:
                return False
            time.sleep(delay)
            delay *= 2
    return False


# ---- sink 分派：有憑證走 gspread，否則降級本地 CSV -----------------------------

def save_record(record: dict) -> dict:
    """把一筆作答寫入資料 sink（可插拔）。回傳 {"ok": bool, "sink": str, "detail": str}。"""
    if _credentials_available():
        if save_with_retry(record):
            return {"ok": True, "sink": "google_sheet", "detail": "已寫入 Google Sheet。"}
        return {"ok": False, "sink": "google_sheet",
                "detail": "Google Sheet 寫入失敗（設定有誤或暫時性錯誤已逾次），作答已保留待補送。"}
    # 降級：偵測不到 Google 憑證 → 寫本地 CSV。
    try:
        p = write_local_csv(record)
    except OSError as e:
        return {"ok": False, "sink": "local_csv", "detail": f"本地 CSV 寫入失敗：{e}，作答已保留待補送。"}
    return {"ok": True, "sink": "local_csv",
            "detail": f"Google Sheet 寫入略過（無憑證），已寫入本地 CSV：{p}"}


# =============================================================================
# 主流程：先落地、後寫入、失敗保留待補
# =============================================================================

PENDING_DIR = "pending_submissions"


def _new_record(answers: dict) -> dict:
    # respondent_id 隨作答一起落地，即為冪等鍵——補送同一筆不會改內容。
    return {
        "survey_version": SURVEY_VERSION,
        "respondent_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answers": answers,
    }


def _persist_pending(record: dict) -> str:
    os.makedirs(PENDING_DIR, exist_ok=True)
    path = os.path.join(PENDING_DIR, f"{record['respondent_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def resend_pending() -> None:
    """補送先前寫入失敗、暫存本地的作答；成功才移除待補檔。單一壞檔不影響其他檔與後續流程。"""
    if not os.path.isdir(PENDING_DIR):
        return
    for name in sorted(os.listdir(PENDING_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(PENDING_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("待補檔無法讀取，略過：%s（%s）", path, e)
            continue
        try:
            if save_record(record)["ok"]:
                os.remove(path)
                print(f"已補送 {record.get('respondent_id', '')[:8]}")
        except Exception as e:  # 防止單筆非預期例外中斷整批補送
            logger.warning("補送 %s 時發生非預期錯誤：%s", path, e)


def run_agent(*, input_fn: Optional[Callable[[str], str]] = None,
              do_resend: bool = True) -> Optional[dict]:
    """跑完整問卷流程：預檢金鑰 → 收集作答 → 複查 → 寫入 sink（先落地、失敗保留待補）。

    - input_fn：注入非互動輸入來源（如腳本化回答）；None 則用互動 input()。
    - 回傳 {"record": ..., "result": save_record 結果}；中止時回 None。
    """
    # 提早驗證 LLM 供應商金鑰（get_llm 不發 API 請求，只檢查設定/金鑰）。
    try:
        from common import get_llm  # type: ignore
        get_llm()
    except ImportError:
        pass  # 若 common 未匯出 get_llm，略過預檢
    except Exception as e:  # noqa: BLE001
        print(f"無法初始化 LLM 供應商：請設定對應金鑰（如 GROQ_API_KEY），並確認 LLM_PROVIDER。（{e}）")
        return None

    if input_fn is not None:
        set_input_fn(input_fn)
    try:
        print("歡迎填寫「產品使用回饋」問卷。")
        print("說明：作答中可輸入 /back 回上一題、對選填題輸入 /skip 略過、/cancel 中止。")
        sink_desc = ("Google Sheet（已偵測到 service account 憑證）"
                     if _credentials_available()
                     else "本地 CSV（未偵測到 Google 憑證，將降級寫入 responses_local.csv）")
        print("隱私：本問卷僅供內部改善產品使用；「稱呼」為選填、可留白；"
              f"作答將寫入 {sink_desc}。\n")

        if do_resend:
            resend_pending()  # 先嘗試補送先前積壓的作答。

        answers: dict = {}  # 由 run_survey 就地填入，中止時仍看得到已收集的部分。
        try:
            run_survey(answers)
            review_and_confirm(answers)
        except SurveyAborted as e:
            print(f"\n{e}" if str(e) else "\n已中止作答。")
            if answers:
                path = _persist_pending(_new_record(answers))
                print(f"目前已作答的部分已保存於 {path}，可稍後補送。")
            return None

        record = _new_record(answers)
        path = _persist_pending(record)  # 寫入前先落地，任何後續失敗都不會遺失。
        result = save_record(record)
        if result["ok"]:
            os.remove(path)  # 寫入成功才移除待補檔。
            print(f"\n{result['detail']}")
            print(f"已送出，感謝填寫！（編號 {record['respondent_id'][:8]}）")
        else:
            print(f"\n{result['detail']}")
            print(f"你的作答已安全保存於 {path}，稍後會自動補送。")
        return {"record": record, "result": result}
    finally:
        if input_fn is not None:
            set_input_fn(None)


def main() -> None:
    load_dotenv()  # 若有 .env 則載入（含 GROQ_API_KEY / MISTRAL_API_KEY / GOOGLE_* / SPREADSHEET_ID）。
    logging.basicConfig(level=logging.INFO)
    run_agent(input_fn=None, do_resend=True)


if __name__ == "__main__":
    main()
