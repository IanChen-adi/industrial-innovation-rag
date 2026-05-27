import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from sparse_utils import text_to_sparse

load_dotenv()
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "laws_hybrid"   # ← 改成雙向量 collection
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def get_dense(text):
    r = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return r.data[0].embedding


def retrieve_dense(question, top_k=TOP_K):
    """純 dense（基準線）"""
    return qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=get_dense(question),
        using="dense",
        limit=top_k,
    ).points


def retrieve_hybrid(question, top_k=TOP_K):
    """hybrid（dense + sparse + RRF）"""
    sv = text_to_sparse(question)
    return qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=get_dense(question), using="dense", limit=top_k),
            Prefetch(query=SparseVector(indices=sv.indices, values=sv.values),
                     using="sparse", limit=top_k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    ).points



def calculate_metrics(retrieved_laws: list, expected_laws: list):
    """計算 Hit@K、Precision@K、MRR"""

    # Hit@K:top K 內有沒有任何一個 expected_law
    hit = any(law in expected_laws for law in retrieved_laws)

    # Precision@K:top K 內有多少 % 是 expected 的
    if len(retrieved_laws) == 0:
        precision = 0.0
    else:
        correct_count = sum(1 for law in retrieved_laws if law in expected_laws)
        precision = correct_count / len(retrieved_laws)

    # MRR:第一個正確答案的倒數排名
    mrr = 0.0
    for rank, law in enumerate(retrieved_laws, start=1):
        if law in expected_laws:
            mrr = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": precision,
        "mrr": mrr,
    }


def evaluate(retrieve_fn, label):
    with open("evaluation_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n{'='*80}")
    print(f"📋 [{label}] 評估 {len(questions)} 題")
    print('='*80)

    all_results = []
    for q in questions:
        points = retrieve_fn(q["question"])   # ← 用傳進來的 retrieve 函式
        retrieved_laws = [p.payload.get("law_code", "?") for p in points]
        metrics = calculate_metrics(retrieved_laws, q["expected_laws"])
        all_results.append({"id": q["id"], "category": q["category"], **metrics})

    # 印整體統計
    def stats(results, name):
        if not results:
            return
        h = sum(r["hit"] for r in results) / len(results)
        p = sum(r["precision"] for r in results) / len(results)
        m = sum(r["mrr"] for r in results) / len(results)
        print(f"  {name}({len(results)}題): Hit@5={h:.1%}  Prec@5={p:.1%}  MRR={m:.3f}")

    print(f"\n[{label}] 結果:")
    stats(all_results, "全體")
    for cat in ["precise", "colloquial", "difficult"]:
        stats([r for r in all_results if r["category"] == cat], cat)

    return all_results

    # 總結
    print("\n" + "=" * 80)
    print("📊 整體統計")
    print("=" * 80)

    # 全體平均
    avg_hit = sum(r["hit"] for r in all_results) / len(all_results)
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_mrr = sum(r["mrr"] for r in all_results) / len(all_results)
    print(f"\n全體({len(all_results)} 題):")
    print(f"  Hit@5      = {avg_hit:.2%}")
    print(f"  Precision@5 = {avg_precision:.2%}")
    print(f"  MRR        = {avg_mrr:.3f}")

    # 按類別統計
    for category in ["precise", "colloquial", "difficult"]:
        cat_results = [r for r in all_results if r["category"] == category]
        if not cat_results:
            continue
        avg_hit = sum(r["hit"] for r in cat_results) / len(cat_results)
        avg_precision = sum(r["precision"] for r in cat_results) / len(cat_results)
        avg_mrr = sum(r["mrr"] for r in cat_results) / len(cat_results)
        print(f"\n{category}({len(cat_results)} 題):")
        print(f"  Hit@5      = {avg_hit:.2%}")
        print(f"  Precision@5 = {avg_precision:.2%}")
        print(f"  MRR        = {avg_mrr:.3f}")


if __name__ == "__main__":
    evaluate(retrieve_dense, "純 Dense")
    evaluate(retrieve_hybrid, "Hybrid")