from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def retrieve(question: str, top_k: int = TOP_K):
    """把問題向量化,從 Qdrant 撈 top_k 個 chunks"""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    query_vector = response.data[0].embedding

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return results


def build_prompt(question: str, chunks: list) -> list:
    """組裝 baseline prompt(沒有特殊規範,測試 LLM 原始行為)"""
    # 把 chunks 格式化成參考資料區塊
    context = "\n\n".join([
        f"[資料 {i+1}] (來源:{point.payload.get('law_code', '?')} {point.payload.get('條號', '')})\n{point.payload.get('text', '')}"
        for i, point in enumerate(chunks)
    ])

    system_prompt = """你是一個產業創新條例稅務助理。請根據提供的參考資料回答使用者問題。

    【行為規範】
    1. 當【參考資料】中包含 2 個以上不同 law_code 的 chunks 時,請先向使用者確認他想要申請的是哪一個,並附上簡單的介紹與差異比較。

    2. 當使用者問題未提及具體服務名稱(例如「研發抵減」、「個人投資新創」、「高風險新創認定」、「設備抵減」)時,請先向使用者確認他想要申請的是哪一個。

    3. 當回答涉及多個 law_code 時,請明確分辨並標註資料來源(例如「依據產業創新條例§10 規定...」、「依據所得基本稅額條例§12 規定...」)。"""

    user_prompt = f"""【參考資料】
{context}

【使用者問題】
{question}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def ask(question: str):
    """完整 RAG 流程"""
    print(f"\n{'='*70}")
    print(f"❓ 問題: {question}")
    print(f"{'='*70}")

    # 1. Retrieval
    chunks = retrieve(question)
    print(f"\n📚 撈到的 chunks(top {TOP_K}):")
    for i, point in enumerate(chunks, 1):
        law_code = point.payload.get("law_code", "?")
        條號 = point.payload.get("條號", "")
        print(f"  [{i}] score={point.score:.4f} | {law_code} {條號}")

    # 2. 組 prompt
    messages = build_prompt(question, chunks)

    # 3. 呼叫 LLM
    print(f"\n💬 LLM 回答:")
    print("-" * 70)
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )
    answer = response.choices[0].message.content
    print(answer)
    print("-" * 70)


if __name__ == "__main__":
    # 跑測試 3 的問題(籠統問題,我們知道 retrieval 會失敗的那題)
    ask("我們公司要符合哪些條件才能申請?")