"""
load_gemini_data.py — 用 Gemini embedding 灌「平行 collection」(第3刀對照實驗)
  laws_hybrid_g   ← all_chunks_flat.json(256)
  hybrid_manual_g ← manual_chunks.json(17)
舊的 laws_hybrid / hybrid_manual 完全不動,兩套並存、隨時可切回。

用法(本機):
  python load_gemini_data.py
用法(雲端):
  QDRANT_URL="https://xxx:443" QDRANT_API_KEY="xxx" python load_gemini_data.py
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
)
from sparse_utils import text_to_sparse

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
DIM = 3072                    # 實測維度;若改用截斷版(1536)請同步改這裡與 API 參數
BATCH = 25                    # 分批上傳,避免大包被網路切斷
EMBED_BATCH = 20              # 分批取向量,避免撞 TPM

gemini = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    check_compatibility=False,
    timeout=120,
)

JOBS = [
    ("laws_hybrid_g", "all_chunks_flat.json", True),    # True = 建 law_code index
    ("hybrid_manual_g", "manual_chunks.json", False),
]


def get_dense_batch(texts):
    """一次取多筆向量(省呼叫次數,免費層 100 RPM / 1000 RPD)"""
    r = gemini.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]


def load_one(collection, source_file, need_index):
    print(f"\n{'=' * 56}\n  {collection} ← {source_file}\n{'=' * 56}")

    with open(source_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"📂 載入 {len(chunks)} 筆 chunk")

    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
        print(f"⚠️  舊的 {collection} 已刪除(重建)")
    qdrant.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"✅ collection 建立完成(dense {DIM} 維 + sparse)")

    # --- 取向量(分批)---
    points = []
    for s in range(0, len(chunks), EMBED_BATCH):
        part = chunks[s:s + EMBED_BATCH]
        vecs = get_dense_batch([c["text"] for c in part])
        for c, v in zip(part, vecs):
            points.append(PointStruct(
                id=len(points) + 1,
                vector={"dense": v, "sparse": text_to_sparse(c["text"])},
                payload=c,
            ))
        print(f"  🧮 已向量化 {min(s + EMBED_BATCH, len(chunks))}/{len(chunks)}")

    # --- 分批上傳 ---
    for s in range(0, len(points), BATCH):
        qdrant.upsert(collection_name=collection,
                      points=points[s:s + BATCH], wait=True)
        print(f"  ⬆️ 已上傳 {min(s + BATCH, len(points))}/{len(points)}")

    if need_index:
        qdrant.create_payload_index(collection_name=collection,
                                    field_name="law_code", field_schema="keyword")
        print("  🔑 law_code payload index 建立完成")

    n = qdrant.get_collection(collection).points_count
    print(f"📊 {collection} points = {n}")
    return n


if __name__ == "__main__":
    print(f"目標 Qdrant:{os.getenv('QDRANT_URL', 'http://localhost:6333')}")
    print(f"Embedding:{EMBED_MODEL}({DIM} 維)")
    results = {}
    for collection, src, need_index in JOBS:
        results[collection] = load_one(collection, src, need_index)

    print(f"\n{'=' * 56}")
    print("  完成:", results)
    print("  下一步:EMBED_PROVIDER=gemini python evaluate_retrieval.py")
    print("        與舊數據對照 Hit@5 / Precision@5 / MRR")