import json
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# 1. 載入 .env
load_dotenv()

# 2. 建立 clients
openai_client = OpenAI()
qdrant_client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "industrial_innovation_laws"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def retrieve(question: str, top_k: int = TOP_K):
    """把問題向量化,從 Qdrant 撈 top_k 個 chunks"""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    query_vector = response.data[0].embedding

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return results


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


def evaluate():
    # 載入評估問題
    with open("evaluation_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"📋 評估 {len(questions)} 題")
    print("=" * 80)

    # 收集所有結果
    all_results = []

    for q in questions:
        # 跑 retrieval
        points = retrieve(q["question"])
        retrieved_laws = [p.payload.get("law_code", "?") for p in points]

        # 算指標
        metrics = calculate_metrics(retrieved_laws, q["expected_laws"])

        # 印出單題結果
        hit_mark = "✅" if metrics["hit"] else "❌"
        print(f"\n[{q['id']}] {hit_mark} {q['question'][:50]}...")
        print(f"  Expected: {q['expected_laws']}")
        print(f"  Retrieved: {retrieved_laws}")
        print(f"  Hit={metrics['hit']} | Precision={metrics['precision']:.2f} | MRR={metrics['mrr']:.3f}")

        all_results.append({
            "id": q["id"],
            "category": q["category"],
            **metrics,
        })

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
    evaluate()