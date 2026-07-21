from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from query_translator import translate_query, get_service_urls
from preprocessor import preprocess

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION = "laws_hybrid"          # 用 dense 欄位(跟舊的一樣)
MANUAL_COLLECTION = "hybrid_manual"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """你是「產業創新條例租稅優惠諮詢助理」,由數位產業署維運。
根據提供的【參考資料】回答使用者問題。

【回答結構】依序包含:
1. 引用最相關的依據:法規類問題引用法條(標明法規名稱與條號);
   文書實務類問題依據署內作業手冊(標明「依本署作業規範」)。
2. 以淺顯白話說明現況,避免直接照抄法條文字。
3. 說明使用者下一步可進行的行動。
4. 若【相關服務官方網址】區段有提供網址且問題涉及申請實務,附上網址。

【誠實規範】
- 參考資料不足以回答時,明確說「目前資料中沒有這部分的規定」,
  並建議洽詢管道。絕對不要編造法條、期限、金額或網址。
- 只引用【參考資料】中實際存在的條文,引用時條號必須與資料一致。
- 引用法條時,使用參考資料中標明的法規全名,不要自行推斷或簡化法規名稱。

【職責邊界】
- 本署僅就條文明定事項提供說明。以下屬財政部(國稅局)最終核定權責,
  一律告知使用者洽詢國稅局,不得自行判斷:
  特定支出能否認列、財務報表填列方式、最終可抵減稅額。

【服務範圍】
- 本系統僅涵蓋 5 項服務(研發投資抵減/中小企業研發投資抵減/設備投資抵減/
  個人投資新創減除/所得基本稅額高風險新創認定)。
  使用者詢問範圍外事項時,說明本系統涵蓋範圍,不要硬答。

【風格】
- 使用繁體中文,語氣專業但親切,像有耐心的承辦人員。
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
    "04_所得基本稅額高風險新創認定": "所得基本稅額高風險新創認定 — 出售高風險新創股票,交易所得免計最低稅負",
}


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def retrieve(question, top_k, law_codes=None):
    """查法規 collection,可加 law_code filter"""
    query_filter = None
    if law_codes:
        query_filter = Filter(
            must=[FieldCondition(key="law_code", match=MatchAny(any=law_codes))]
        )
    return qdrant.query_points(
        collection_name=COLLECTION,
        query=get_dense(question),
        using="dense",
        query_filter=query_filter,
        limit=top_k,
    ).points


def retrieve_manual(question, top_k=5):
    """查手冊 collection(文書類問題用)"""
    return qdrant.query_points(
        collection_name=MANUAL_COLLECTION,
        query=get_dense(question),
        using="dense",
        limit=top_k,
    ).points

def retrieve_combined(question, top_k, law_codes=None):
    """法規(filter)+ 手冊 合併檢索,按分數排序"""
    law_hits = retrieve(question, top_k, law_codes)
    manual_hits = retrieve_manual(question, top_k=3)
    merged = sorted(list(law_hits) + list(manual_hits),
                    key=lambda p: p.score, reverse=True)
    return merged[:top_k + 2]

def do_clarify(question, candidate_services, history):
    """反問釐清。回傳反問文字(讓呼叫端存進 history)。"""
    candidates = "\n".join(f"- {SERVICE_DESC.get(s, s)}" for s in candidate_services)
    messages = [{"role": "system", "content": CLARIFY_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": f"【使用者問題】{question}\n\n【候選服務】\n{candidates}",
    })
    resp = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    text = resp.choices[0].message.content
    print("\n" + "─" * 70)
    print("💬 反問")
    print("─" * 70)
    print(text)
    return text


def ask(question, top_k, history):
    translated = translate_query(question)
    if translated != question:
        print(f"\n✏️  翻譯: {question} → {translated}")

    # 前處理判斷(帶對話歷史,支援反問後的回應)
    pre = preprocess(translated, history)
    print(f"🧠 前處理: services={pre['services']} conf={pre['confidence']} type={pre['question_type']}")
    print(f"   reason: {pre['reason']}")

    services = pre["services"]
    # 05 自動補 01(資料庫補償:05 只有 1 chunk,細節在 01)
    if "05_中小企業研發" in services and "01_產創研發" not in services:
        services.append("01_產創研發")
        print(f"   (自動補 01_產創研發:中小企細節借用 01)")

    # === 分岔 ===
    if pre["question_type"] == "文書":
        # ① 文書 → 查手冊
        print(f"📋 文書類 → 查手冊")
        chunks = retrieve_manual(translated, top_k)
    elif pre["confidence"] == "high" and services:
        # ② 高信心 → filter 查法規
        print(f"🎯 限定 law_code: {services}")
        chunks = retrieve_combined(translated, top_k, law_codes=services)
    elif services:
        # ③ 低信心但有候選 → 反問
        print(f"❓ 信心不足 → 反問(候選: {services})")
        clarify_text = do_clarify(translated, services, history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": clarify_text})
        return
    else:
        # ④ 範圍外
        print(f"🚫 範圍外")
        scope_text = ("本系統涵蓋 5 項服務:研發投資抵減、中小企業研發投資抵減、"
                      "設備投資抵減、個人投資新創減除、所得基本稅額高風險新創認定。\n"
                      "您的問題似乎不在上述範圍。若您想詢問其中一項,請告訴我;"
                      "其他事項建議洽詢數位產業署。")
        print("\n💬 回答")
        print(scope_text)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": scope_text})
        return

    # Retrieval 區
    print("\n" + "─" * 70)
    print(f"🔍 Retrieval(top {top_k})")
    print("─" * 70)
    for i, p in enumerate(chunks, 1):
        law = p.payload.get("law_code", p.payload.get("topic", "?"))
        art = p.payload.get("條號", "") or p.payload.get("id", "")
        print(f"  [{i}] score={p.score:.3f} | {law} {art}")
        print(f"      {p.payload.get('text','')[:40]}")

    # 組 prompt(來源用法規全名;手冊 chunk 無 law_name → 標「署內作業手冊」)
    context = "\n\n".join([
        f"[資料{i+1}] (來源:{p.payload.get('law_name', '署內作業手冊')} "
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
        messages.extend(history[-6:])   # 帶最近 3 輪
    messages.append({"role": "user", "content": user_msg})

    resp = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    answer = resp.choices[0].message.content
    print("\n" + "─" * 70)
    print("💬 回答")
    print("─" * 70)
    print(answer)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})


def main():
    print("=" * 70)
    print("  RAG 診斷工具(可看 retrieval + 回答,含多輪記憶)")
    print("=" * 70)
    print("指令:")
    print("  直接輸入問題 → 查詢")
    print("  topk=10      → 改 top_k 為 10")
    print("  reset        → 清空對話歷史(換話題用)")
    print("  quit / exit  → 離開")

    top_k = 5
    history = []

    while True:
        user = input(f"\n[top_k={top_k}] 你: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("再見!")
            break
        if user.lower().startswith("topk="):
            try:
                top_k = int(user.split("=")[1])
                print(f"✅ top_k 已改為 {top_k}")
            except ValueError:
                print("❌ 格式錯誤,範例:topk=10")
            continue
        if user.lower() == "reset":
            history = []
            print("✅ 對話歷史已清空")
            continue

        ask(user, top_k, history)


if __name__ == "__main__":
    main()