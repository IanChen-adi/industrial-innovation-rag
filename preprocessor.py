"""
前處理判斷模組:一次 LLM 呼叫,輸出「服務識別 + 信心度 + 問題類型」。
接在規則式翻譯之後、retrieval 之前。支援對話歷史(處理反問後的回應)。

v2 修正:
- 改用 OpenAI JSON mode(response_format)強制 JSON 輸出
- 對話歷史改為「文字摘要」注入 user message,不用原生 role
  (原生 role 會讓 LLM 模仿歷史中反問的自然語言格式,忘記輸出 JSON)
- 新增領域術語規則(一般認定/專案認定、中小企業認定)
"""
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ══ 第2刀開關:前處理判斷的供應商(預設 openai)══════════════
#   切換:PREPROC_PROVIDER=gemini python xxx.py
PREPROC_PROVIDER = os.getenv("PREPROC_PROVIDER", "openai")

openai_client = OpenAI()
gemini_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)
_CLIENTS = {"openai": openai_client, "gemini": gemini_client}
_MODELS = {"openai": "gpt-4o-mini",
           "gemini": "gemini-3.1-flash-lite"}   # 獨立額度池:15 RPM / 500 RPD

client = _CLIENTS[PREPROC_PROVIDER]
PREPROCESS_MODEL = _MODELS[PREPROC_PROVIDER]

PREPROCESS_PROMPT = """你是「產業創新條例租稅優惠諮詢系統」的前處理判斷模組。
你的任務:分析使用者的問題,輸出結構化 JSON 判斷結果。你不回答問題本身。
無論輸入是什麼,你的輸出永遠只能是一個 JSON 物件。

【本系統涵蓋的 5 項服務】
1. 研發投資抵減(產創§10)— 公司研發支出抵減營所稅,law_code: 01_產創研發
2. 中小企業研發投資抵減(中小企§35)— 同上但適用中小企業,law_code: 05_中小企業研發
3. 設備投資抵減(產創§10-1)— 智慧機械/5G/資安/AI/節能減碳設備支出抵減營所稅,law_code: 02_設備
4. 個人投資新創減除(產創§23-2)— 個人投資高風險新創,投資額減除綜所稅,law_code: 03_個人投資新創減除
5. 所得基本稅額高風險新創認定(所基稅§12)— 高風險新創股票交易所得免計基本所得額,law_code: 04_所得基本稅額高風險新創認定

【服務關聯規則】
- 「高風險新創」身分同時涉及服務 4 和 5,兩者受益人都是「個人投資者」,
  但優惠時點與方式不同:
  · 服務 4(§23-2):投資「時」的優惠 — 現金投資高風險新創達 50 萬元且
    持股滿 3 年,投資金額之半數可自「綜合所得總額」減除(當年度上限 500 萬)。
  · 服務 5(§12):出售「時」的優惠 — 交易設立未滿 5 年之高風險新創
    未上市櫃股票,其交易所得免計入「基本所得額」。
  使用者只說「高風險新創」而無法判斷是「投資減除」還是「股票交易免稅」時,
  屬於模糊,confidence 必須為 low。
- 服務 1 和 2 核心邏輯相同:使用者明確提到「中小企業」時包含服務 2,
  否則預設服務 1。同一筆研發支出,服務 1 和 2 只能擇一申請(互斥)。
- 高風險新創「認定函」是服務 4 和 5 的前置資格。使用者問「認定」本身
  (如何取得、條件、補件)時,對應 law_code 為 03 和 04 兩者。

【領域術語規則】
- 「一般認定」「專案認定」:研發投資抵減的兩種認定申請方式。
  使用者提到這兩個詞,即屬研發投資抵減脈絡:未提中小企業 → services 為
  ["01_產創研發"],confidence 為 high;有提中小企業 → ["05_中小企業研發"]。
- 使用者問「如何判斷/怎樣算是中小企業」「中小企業的認定標準」:
  屬服務 2 的條件問題,services 為 ["05_中小企業研發"],confidence 為 high,
  question_type 為條件。
  - 「國家重點發展產業」:個人投資新創減除(03)的專屬概念,影響投資人
  可減除上限(屬國家重點發展產業者上限 500 萬元,非屬者 50 萬元)。
  使用者問及此詞 → services=["03_個人投資新創減除"],confidence=high。

【輸出格式】只輸出 JSON 物件,不要其他文字:
{
  "services": ["law_code", ...],
  "confidence": "high" | "low",
  "question_type": "文書" | "條件" | "程序" | "疑義",
  "reason": "一句話說明判斷依據"
}

【判斷規則】
- confidence 為 high 的條件:使用者明確講出服務名稱或明確可推斷的情境,
  且只對應到一項服務。
- 以下情況 confidence 必須為 low:
  (a) 對應到 2 項以上服務
  (b) 使用者用詞模糊、你是用猜的
  (c) 問題完全不屬於這 5 項服務(此時 services 為空陣列)
- 若【對話歷史】顯示系統先前正在反問釐清,而使用者本輪是在回應該反問
  (包括選了某個選項或編號、描述自己情境、或「我不知道」「還是不確定」等):
  · 能從回應確定服務 → 輸出該服務,confidence 為 high。
  · 使用者仍不確定 → 維持先前反問的候選 services,confidence 維持 low,
    不要輸出空陣列(這不是範圍外問題,是釐清尚未完成)。
  · 回應反問的輪次,question_type 沿用原始問題的類型。
  - 【歷史優先原則】對話歷史顯示已鎖定單一服務(先前輪次 confidence 為 high
  或使用者已在反問中選定)時,後續追問一律沿用該服務、confidence 維持 high
  ——即使追問中出現「高風險新創」「認定」等跨服務術語,也不重新觸發模糊判斷。
  僅在使用者明確表示要「換一個服務」「問別的優惠」或要求「比較兩個服務」時,
  才重新判斷。
  -文書 = 僅限行政作業事項:申請書怎麼填、用印核章、寄送方式、文號、
         送件形式。「怎麼認定」「認定標準」「資格怎麼算」「條件是什麼」
         一律屬「條件」,不屬文書。"""


def _history_to_text(history: list) -> str:
    """把對話歷史壓成文字段落(避免原生 role 讓模型模仿自然語言格式)"""
    if not history:
        return ""
    lines = []
    for m in history[-6:]:
        who = "使用者" if m["role"] == "user" else "系統"
        content = m["content"]
        if len(content) > 300:
            content = content[:300] + "…"
        lines.append(f"{who}:{content}")
    return "【對話歷史】\n" + "\n".join(lines) + "\n\n"


def preprocess(query: str, history: list = None) -> dict:
    """前處理判斷。可帶對話歷史(處理反問後的回應)。
    解析失敗時回傳保守的 fallback(low + 空)。"""
    user_content = _history_to_text(history) + f"【本輪問題】{query}"

    resp = client.chat.completions.create(
        model=PREPROCESS_MODEL,
        messages=[
            {"role": "system", "content": PREPROCESS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},   # 強制 JSON 輸出
    )
    raw = resp.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        result.setdefault("services", [])
        result.setdefault("confidence", "low")
        result.setdefault("question_type", "疑義")
        result.setdefault("reason", "")
        return result
    except json.JSONDecodeError:
        return {"services": [], "confidence": "low",
                "question_type": "疑義", "reason": f"JSON解析失敗:{raw[:50]}"}


# --- 自我測試 ---
if __name__ == "__main__":
    print(f"[前處理供應商] {PREPROC_PROVIDER} / {PREPROCESS_MODEL}\n")
    tests = [
        "我研發投資抵減哪一天以前要交件",
        "一般認定跟專案認定差在哪裡",          # 新術語規則:預期 high, [01]
        "怎麼判斷我是不是中小企",              # 新術語規則:預期 high, [05], 條件
        "高風險新創認定怎麼申請",
        "你們有提供上市櫃意見書嗎",
    ]
    for t in tests:
        r = preprocess(t)
        print(f"Q: {t}")
        print(f"   → services={r['services']} conf={r['confidence']} type={r['question_type']}")
        print(f"   → reason: {r['reason']}\n")

    print("=" * 60)
    print("多輪測試 1:反問後回應「我不知道」")
    fake_history = [
        {"role": "user", "content": "高風險新創認定怎麼申請"},
        {"role": "assistant", "content": "您想了解高風險新創認定,這涉及兩種服務:"
         "① 個人投資新創減除(投資時優惠)② 所得基本稅額高風險新創認定(出售時優惠)。"
         "請問您的情況接近哪一種?"},
    ]
    r = preprocess("我不知道我是哪一個", fake_history)
    print(f"→ services={r['services']} conf={r['confidence']}")
    print("(預期:維持 [03,04]、conf=low,且 reason 是正常 JSON 非解析失敗)\n")

    print("多輪測試 2:反問後回應編號「1.」")
    fake_history2 = [
        {"role": "user", "content": "研發投抵的一般認定跟專案認定差在哪"},
        {"role": "assistant", "content": "您想了解研發投資抵減的認定方式差異:"
         "1. 研發投資抵減(一般公司) 2. 中小企業研發投資抵減(中小企業)。"
         "請問您是一般公司還是中小企業?"},
    ]
    r = preprocess("1.", fake_history2)
    print(f"→ services={r['services']} conf={r['confidence']}")
    print("(預期:[01_產創研發]、conf=high)")