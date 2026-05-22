import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_manual"   # 手冊專用 collection
EMBEDDING_MODEL = "text-embedding-3-small"

# 1. 建立/重建 collection
print(f"⚠️  重建 collection '{COLLECTION_NAME}'")
if qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.delete_collection(COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
print(f"✅ Collection 建立完成")

# 2. 讀手冊 chunks
with open("manual_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\n📂 載入 {len(chunks)} 筆手冊 chunk")

# 3. 逐一 embedding + 寫入
points = []
for i, chunk in enumerate(chunks, start=1):
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunk["text"],
    )
    vector = response.data[0].embedding

    points.append(
        PointStruct(
            id=i,                # 手冊 collection 從 1 開始(獨立命名空間)
            vector=vector,
            payload=chunk,       # 整個 chunk(含原始 id、related_law 等)放 payload
        )
    )
    print(f"  ✅ [{i}/{len(chunks)}] {chunk['id']}")

# 4. 寫入
qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"\n✅ 全部寫入完成,共 {len(points)} 筆")
