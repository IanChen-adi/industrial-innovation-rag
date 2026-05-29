from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION = "laws_hybrid"          # 用 dense 欄位（跟舊的一樣）
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """你是一個產業創新條例稅務助理。請根據提供的參考資料回答使用者問題。
如果參考資料中沒有足夠資訊，請明確說明，不要編造。"""


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def retrieve(question, top_k):
    return qdrant.query_points(
        collection_name=COLLECTION,
        query=get_dense(question),
        using="dense",
        limit=top_k,
    ).points


def ask(question, top_k):
    chunks = retrieve(question, top_k)

    # --- 【Retrieval 區】把撈到的攤開給你看 ---
    print("\n" + "─" * 70)
    print(f"🔍 Retrieval（top {top_k}）")
    print("─" * 70)
    for i, p in enumerate(chunks, 1):
        law = p.payload.get("law_code", "?")
        art = p.payload.get("條號", "")
        text = p.payload.get("text", "")[:40]
        print(f"  [{i}] score={p.score:.3f} | {law} {art}")
        print(f"      {text}")

    # --- 組 prompt ---
    context = "\n\n".join([
        f"[資料{i+1}] (來源:{p.payload.get('law_code','?')} {p.payload.get('條號','')})\n{p.payload.get('text','')}"
        for i, p in enumerate(chunks)
    ])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【參考資料】\n{context}\n\n【問題】{question}"},
    ]
    resp = openai_client.chat.completions.create(model=LLM_MODEL, messages=messages)
    answer = resp.choices[0].message.content

    # --- 【回答區】 ---
    print("\n" + "─" * 70)
    print("💬 回答")
    print("─" * 70)
    print(answer)


def main():
    print("=" * 70)
    print("  RAG 診斷工具（可看 retrieval + 回答）")
    print("=" * 70)
    print("指令：")
    print("  直接輸入問題 → 查詢")
    print("  topk=10      → 改 top_k 為 10")
    print("  quit / exit  → 離開")

    top_k = 5   # 預設

    while True:
        user = input(f"\n[top_k={top_k}] 你: ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("再見!")
            break
        # 調整 top_k 的指令
        if user.lower().startswith("topk="):
            try:
                top_k = int(user.split("=")[1])
                print(f"✅ top_k 已改為 {top_k}")
            except ValueError:
                print("❌ 格式錯誤，範例：topk=10")
            continue

        ask(user, top_k)


if __name__ == "__main__":
    main()