from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 3. 三組測試問題
test_queries = [
    {
        "question": "我們公司做研發,有哪些費用可以列入抵減?",
        "expected_id": 1,
    },
    {
        "question": "我去年投資了一家新創公司,聽說可以從我的綜合所得減一些金額,這是真的嗎?有什麼依據?",
        "expected_id": 2,
    },
    {
        "question": "我們公司要符合哪些條件才能申請?",
        "expected_id": 3,
    },
]

# 4. 逐一查詢
for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"測試 {i}: {query['question']}")
    print(f"預期命中 id={query['expected_id']}")
    print(f"{'='*60}")

    # 4a. 把問題變向量
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query["question"]
    )
    query_vector = response.data[0].embedding

    # 4b. 在 Qdrant 搜尋
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,  # 拿 top 3
    ).points

    # 4c. 印出結果
    for rank, point in enumerate(results, 1):
        match = "✅" if point.id == query["expected_id"] else "  "
        print(f"  {match} Rank {rank} | id={point.id} | score={point.score:.4f}")
        print(f"      {point.payload['text']}")