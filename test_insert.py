from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# 1. 載入 .env(OpenAI key)
load_dotenv()

# 2. 建立 client(OpenAI + Qdrant 各一個)
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 3. 假資料(1 個 chunk)
fake_chunk = {
    "id": 1,
    "text": "第 5 條 研究發展支出包含薪資、消耗性器材、專利權等費用。",
    "metadata": {
        "law_code": "01_產創研發",
        "條號": "第 5 條",
        "topic": "研發抵減",
    }
}

# 4. 把 text 變成 1536 維向量
response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=fake_chunk["text"]
)
vector = response.data[0].embedding

# 5. 組成 PointStruct 並寫入 Qdrant
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=fake_chunk["id"],
            vector=vector,
            payload={
                "text": fake_chunk["text"],
                **fake_chunk["metadata"]
            }
        )
    ]
)

print(f"✅ 寫入成功")
print(f"   id: {fake_chunk['id']}")
print(f"   text: {fake_chunk['text']}")