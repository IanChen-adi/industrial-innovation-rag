from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 連 Qdrant
client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"

# 先檢查 collection 是否已存在
if client.collection_exists(COLLECTION_NAME):
    print(f"⚠️  Collection '{COLLECTION_NAME}' 已存在,跳過建立")
else:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{COLLECTION_NAME}' 建立成功")
    print(f"   維度: 1536")
    print(f"   距離演算法: Cosine")