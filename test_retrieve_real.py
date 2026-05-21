from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 3. 三組測試問題(預期值改成 law_code)
test_queries = [
    {
        "question": "我們公司做研發,有哪些費用可以列入抵減?",
        "expected_law": "01_產創研發",
    },
    {
        "question": "我去年投資了一家新創公司,聽說可以從我的綜合所得減一些金額,這是真的嗎?有什麼依據?",
        "expected_law": "03_個人投資新創減除",
    },
    {
        "question": "我們公司要符合哪些條件才能申請?",
        "expected_law": "01_產創研發",   # 改一下:之前 id=3 是「高風險新創認定」,但這個問題太籠統,改成研發看
    },
]

# 4. 逐一查詢
for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"測試 {i}: {query['question']}")
    print(f"預期 law_code: {query['expected_law']}")
    print(f"{'='*70}")

    # 4a. 問題向量化
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query["question"]
    )
    query_vector = response.data[0].embedding

    # 4b. 查 Qdrant,拿 top 5
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
    ).points

    # 4c. 印出結果
    for rank, point in enumerate(results, 1):
        actual_law = point.payload.get("law_code", "?")
        match = "✅" if actual_law == query["expected_law"] else "  "
        條號 = point.payload.get("條號", "")
        text_preview = point.payload.get("text", "")[:50]

        print(f"  {match} Rank {rank} | score={point.score:.4f} | {actual_law} {條號}")
        print(f"      {text_preview}...")