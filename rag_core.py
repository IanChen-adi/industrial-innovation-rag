"""
rag_core.py — RAG pipeline 邏輯層(無 UI)。
chat_debug.py(終端機)與 app.py(Streamlit)共用此模組。
流程:translate → preprocess → 分岔(文書/高信心/反問/範圍外)→ retrieve → answer
"""
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from query_translator import translate_query, get_service_urls
from preprocessor import preprocess
import os

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
)

COLLECTION = "laws_hybrid"
MANUAL_COLLECTION = "hybrid_manual"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """你是「產業創新條例租稅優惠 AI 問答」的助理。本系統為個人研究專案之實驗性工具,
非政府機關官方服務。根據提供的【參考資料】回答使用者問題。

【回答結構】依序包含:
1. 引用最相關的依據:法規類問題引用法條(標明法規名稱與條號);
   文書實務類問題依據作業手冊，但是回答時不要標注作業手冊為參考來源，僅標注有關的法規。
   (如果依據是作業手冊，在回答時請不要標注手冊為參考來源)
2. 以淺顯白話說明現況,避免直接照抄法條文字。
3. 說明使用者下一步可進行的行動。
4. 若【相關服務官方網址】區段有提供網址且問題涉及申請實務,附上網址。

【誠實規範】
- 參考資料不足以回答時,明確說「目前資料中沒有這部分的規定」,
  並建議洽詢管道。絕對不要編造法條、期限、金額或網址。
- 只引用【參考資料】中實際存在的條文,引用時條號必須與資料一致。
- 引用法條時,使用參考資料中標明的法規全名,不要自行推斷或簡化法規名稱。
- 涉及金額、比率、期限等數字時,必須逐字取自參考資料,並明確區分
  不同條件對應的不同數額(如上限依身分或產業別而異時,分開陳述,
  不得合併或概括)。

【職責邊界】
- 以下屬財政部(國稅局)最終核定權責,一律告知使用者洽詢國稅局,
  不得自行判斷:特定支出能否認列、財務報表填列方式、最終可抵減稅額。

【服務範圍】
- 本系統僅涵蓋 5 項服務(研發投資抵減/中小企業研發投資抵減/設備投資抵減/
  個人投資新創減除/所得基本稅額高風險新創認定)。
  使用者詢問範圍外事項時,說明本系統涵蓋範圍,不要硬答。

【風格】
- 使用繁體中文,語氣專業但親切。回答內容僅供參考,重大決策請以
  主管機關正式規定與函釋為準。
- 不確定使用者身分時,不預設對方是公司或個人,從問題內容判斷。"""

CLARIFY_PROMPT = """你是租稅優惠諮詢助理。系統判斷使用者的問題資訊不足,需要釐清。
執行以下反問,不要嘗試回答問題本身:
1. 用一句話覆述你理解到的需求。
2. 列出【候選服務】中的每一項,各附一句話差異說明(講適用對象和優惠方式,
   讓非專業使用者能據以選擇)。
3. 結尾請使用者選擇其一,或補充說明自己的情境。
4. 不要提供任何實質法規答案。
若對話歷史顯示你已反問過而使用者仍不確定,改用「情境問題」引導
(例如:「請問您是打算投資新創公司,還是手上已有新創股票想出售?」),
不要重複同一份選項清單。"""

SERVICE_DESC = {
    "01_產創研發": "研發投資抵減 — 公司研發支出可抵減營所稅(適合一般公司)",
    "05_中小企業研發": "中小企業研發投資抵減 — 同研發抵減但適用中小企業",
    "02_設備": "設備投資抵減 — 購置智慧機械/5G/資安/AI/節能減碳設備可抵減營所稅",
    "03_個人投資新創減除": "個人投資新創減除 — 個人投資高風險新創,投資額可折半減除綜所稅",
    "04_所得基本稅額高風險新創認定": "所得基本稅額高風險新創認定 — 出售高風險新創股票,交易所得免計個人之基本所得額課稅",
}

SCOPE_TEXT = ("本系統涵蓋 5 項服務:研發投資抵減、中小企業研發投資抵減、"
              "設備投資抵減、個人投資新創減除、所得基本稅額高風險新創認定。\n"
              "您的問題似乎不在上述範圍。若您想詢問其中一項,請告訴我;"
              "其他事項建議洽詢各主管機關。")


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def retrieve(question, top_k, law_codes=None):
    query_filter = None
    if law_codes:
        query_filter = Filter(
            must=[FieldCondition(key="law_code", match=MatchAny(any=law_codes))]
        )
    return qdrant.query_points(
        collection_name=COLLECTION, query=get_dense(question),
        using="dense", query_filter=query_filter, limit=top_k,
    ).points


def retrieve_manual(question, top_k=5):
    return qdrant.query_points(
        collection_name=MANUAL_COLLECTION, query=get_dense(question),
        using="dense", limit=top_k,
    ).points


def retrieve_combined(question, top_k, law_codes=None):
    law_hits = retrieve(question, top_k, law_codes)
    manual_hits = retrieve_manual(question, top_k=3)
    merged = sorted(list(law_hits) + list(manual_hits),
                    key=lambda p: p.score, reverse=True)
    return merged[:top_k + 2]


def _chunks_meta(chunks):
    """把 Qdrant points 精簡成可序列化的 dict(給 UI 顯示與落地)"""
    out = []
    for p in chunks:
        out.append({
            "score": round(p.score, 3),
            "source": p.payload.get("law_name", "作業手冊"),
            "law_code": p.payload.get("law_code", p.payload.get("topic", "")),
            "article": p.payload.get("條號", "") or p.payload.get("id", ""),
            "text": p.payload.get("text", ""),
        })
    return out


def answer(question: str, history: list, top_k: int = 5) -> dict:
    """
    主入口。history: [{"role","content"}](本函式不修改它,由呼叫端維護)
    回傳 {"mode": "answer|clarify|scope", "text", "translated",
          "preprocess": {...}, "chunks": [...]}
    """
    translated = translate_query(question)
    pre = preprocess(translated, history)
    services = list(pre["services"])
    if "05_中小企業研發" in services and "01_產創研發" not in services:
        services.append("01_產創研發")

    # 分岔
    if pre["question_type"] == "文書":
        chunks = retrieve_manual(translated, top_k)
        mode = "answer"
    elif pre["confidence"] == "high" and services:
        chunks = retrieve_combined(translated, top_k, law_codes=services)
        mode = "answer"
    elif services:
        text = _clarify(translated, services, history)
        return {"mode": "clarify", "text": text, "translated": translated,
                "preprocess": pre, "chunks": []}
    else:
        return {"mode": "scope", "text": SCOPE_TEXT, "translated": translated,
                "preprocess": pre, "chunks": []}

    # 組 prompt
    context = "\n\n".join([
        f"[資料{i+1}] (來源:{p.payload.get('law_name', '作業手冊')} "
        f"{p.payload.get('條號','') or p.payload.get('source_section','')})\n{p.payload.get('text','')}"
        for i, p in enumerate(chunks)
    ])
    urls_section = get_service_urls(services)
    user_msg = f"【參考資料】\n{context}"
    if urls_section:
        user_msg += f"\n\n{urls_section}"
    user_msg += f"\n\n【問題】{translated}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_msg})
    resp = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages)

    return {"mode": mode, "text": resp.choices[0].message.content,
            "translated": translated, "preprocess": pre,
            "chunks": _chunks_meta(chunks)}


def _clarify(question, candidate_services, history):
    candidates = "\n".join(f"- {SERVICE_DESC.get(s, s)}" for s in candidate_services)
    messages = [{"role": "system", "content": CLARIFY_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user",
                     "content": f"【使用者問題】{question}\n\n【候選服務】\n{candidates}"})
    resp = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    return resp.choices[0].message.content
