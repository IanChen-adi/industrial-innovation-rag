import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 3. 清空 collection(重建)
print(f"⚠️  清空 collection '{COLLECTION_NAME}'")
if qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.delete_collection(COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)
print(f"✅ Collection 重建完成")

# 4. 讀真實 chunks
with open("all_chunks_flat.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"\n📂 載入 {len(chunks)} 筆 chunks")

# 5. 一筆一筆 embedding + 寫入
points = []
for i, chunk in enumerate(chunks, start=1):
    # 5a. 呼叫 OpenAI embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk["text"]
    )
    vector = response.data[0].embedding

    # 5b. 組 PointStruct
    points.append(
        PointStruct(
            id=i,                    # Qdrant id 用遞增整數
            vector=vector,
            payload=chunk            # 整個 chunk 物件都放進 payload(包含原始 id)
        )
    )

    # 5c. 進度提示(每 25 筆印一次)
    if i % 25 == 0:
        print(f"  ✅ 已處理 {i}/{len(chunks)}")

# 6. 一次寫入 Qdrant
print(f"\n💾 寫入 Qdrant...")
qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
print(f"✅ 全部寫入完成,共 {len(points)} 筆")