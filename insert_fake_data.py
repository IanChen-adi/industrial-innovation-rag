from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 3. 假資料(3 筆,主題明顯不同)
fake_chunks = [
    {
        "id": 1,
        "text": "第 5 條 研究發展支出包含薪資、消耗性器材、專利權等費用。",
        "metadata": {
            "law_code": "01_產創研發",
            "topic": "研發抵減",
        }
    },
    {
        "id": 2,
        "text": "個人投資新創事業,投資金額 50% 可減除綜合所得總額。",
        "metadata": {
            "law_code": "03_個人投資新創減除",
            "topic": "高風險新創",
        }
    },
    {
        "id": 3,
        "text": "高風險新創事業公司認定,須符合技術創新性、市場化潛力等條件。",
        "metadata": {
            "law_code": "04_所得基本稅額高風險新創認定",
            "topic": "高風險新創",
        }
    },
]

# 4. 把 3 筆 chunk 變成 points
points = []
for chunk in fake_chunks:
    # 呼叫 OpenAI embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk["text"]
    )
    vector = response.data[0].embedding

    points.append(
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                **chunk["metadata"]
            }
        )
    )
    print(f"  ✅ id={chunk['id']} 向量化完成")

# 5. 一次寫入 3 筆
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(f"\n✅ 全部寫入完成,共 {len(points)} 筆")