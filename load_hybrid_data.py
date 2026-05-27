import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
)
from sparse_utils import text_to_sparse   # 用我們修好的（deterministic）

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "laws_hybrid"          # 新的雙向量 collection
SOURCE_FILE = "all_chunks_flat.json"     # 從原始資料重算
EMBEDDING_MODEL = "text-embedding-3-small"


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


# --- 1. 建立雙向量 collection ---
print(f"⚠️  建立 collection '{COLLECTION_NAME}'（dense + sparse）")
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=1536, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    },
)
print("✅ collection 建立完成")

# --- 2. 讀原始資料 ---
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"📂 載入 {len(chunks)} 筆 chunk")

# --- 3. 算 dense + sparse，寫入 ---
points = []
for i, chunk in enumerate(chunks, start=1):
    text = chunk["text"]
    points.append(
        PointStruct(
            id=i,
            vector={
                "dense": get_dense(text),
                "sparse": text_to_sparse(text),
            },
            payload=chunk,
        )
    )
    if i % 25 == 0:
        print(f"  ...已處理 {i}/{len(chunks)}")

# --- 4. 寫入 ---
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"✅ 寫入完成,共 {len(points)} 筆")

# --- 5. 確認 ---
info = qdrant.get_collection(COLLECTION_NAME)
print(f"📊 {COLLECTION_NAME} points = {info.points_count}")