from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    SparseVectorParams, SparseVector,
    PointStruct, NamedSparseVector, NamedVector,
)
from openai import OpenAI
from dotenv import load_dotenv
from sparse_utils import text_to_sparse

load_dotenv()
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

TEST_COLLECTION = "hybrid_smoke_test"   # 測試專用，壞了就刪
EMBEDDING_MODEL = "text-embedding-3-small"


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


# --- 1. 建立雙向量 collection ---
print("⚠️  重建測試 collection")
if qdrant.collection_exists(TEST_COLLECTION):
    qdrant.delete_collection(TEST_COLLECTION)

qdrant.create_collection(
    collection_name=TEST_COLLECTION,
    # dense 向量：具名欄位 "dense"
    vectors_config={
        "dense": VectorParams(size=1536, distance=Distance.COSINE),
    },
    # sparse 向量：具名欄位 "sparse"
    sparse_vectors_config={
        "sparse": SparseVectorParams(),
    },
)
print("✅ 雙向量 collection 建立完成（dense + sparse 兩個具名欄位）")


# --- 2. 準備 3 個假 chunk（smoke test）---
fake_chunks = [
    "高風險新創事業公司認定，指設立未滿五年且具創新性之公司。",
    "研發投資抵減適用於公司投入研究發展之支出。",
    "設備投資抵減適用於購置智慧機械、5G、資安等設備。",
]

# --- 3. 寫入：每個 chunk 同時存 dense + sparse ---
points = []
for i, text in enumerate(fake_chunks, start=1):
    points.append(
        PointStruct(
            id=i,
            vector={
                "dense": get_dense(text),        # dense 欄位
                "sparse": text_to_sparse(text),  # sparse 欄位
            },
            payload={"text": text},
        )
    )

qdrant.upsert(collection_name=TEST_COLLECTION, points=points)
print(f"✅ 寫入 {len(points)} 個假 chunk（每個都含 dense + sparse）")

# --- 4. 確認寫入 ---
info = qdrant.get_collection(TEST_COLLECTION)
print(f"\n📊 collection 狀態：points = {info.points_count}")


from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

def hybrid_search(query_text, top_k=3):
    """Hybrid 查詢：dense + sparse 兩路，用 RRF 融合"""
    dense_vec = get_dense(query_text)
    sparse_vec = text_to_sparse(query_text)

    results = qdrant.query_points(
        collection_name=TEST_COLLECTION,
        # prefetch：先各自跑兩路檢索
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k),
            Prefetch(
                query=SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                using="sparse",
                limit=top_k,
            ),
        ],
        # 用 RRF 融合兩路
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points
    return results


# --- 測試 Hybrid 查詢 ---
print("\n" + "="*50)
print("Hybrid 查詢測試")
print("="*50)

test_query = "高風險新創認定"
print(f"\n查詢: {test_query}\n")
hits = hybrid_search(test_query)
for i, h in enumerate(hits, 1):
    print(f"  [{i}] score={h.score:.4f}  {h.payload['text'][:30]}")