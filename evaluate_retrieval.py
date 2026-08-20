"""
evaluate_retrieval.py — 檢索品質評估(Hit@K / Precision@K / MRR)
改版重點:
  · 走 rag_core 的供應商開關,不再自帶一份 client/collection
    → EMBED_PROVIDER=openai|gemini 自動切換模型與對應 collection
  · 支援 named vector(using="dense"),對得上現行 laws_hybrid
  · 結果可存 JSON,方便兩版對照

用法:
  python evaluate_retrieval.py                        # 基準(OpenAI 向量)
  EMBED_PROVIDER=gemini python evaluate_retrieval.py  # 對照(Gemini 向量)
  python evaluate_retrieval.py --save                 # 順便存成 eval_<provider>.json
"""
import json
import sys

import rag_core   # 供應商開關、collection、get_dense 全部來自這裡

TOP_K = 5


def retrieve(question: str, top_k: int = TOP_K):
    """與線上 pipeline 同一條路徑(rag_core.retrieve,不帶 filter)"""
    return rag_core.retrieve(question, top_k)


def calculate_metrics(retrieved_laws: list, expected_laws: list):
    hit = any(law in expected_laws for law in retrieved_laws)
    if not retrieved_laws:
        precision = 0.0
    else:
        correct = sum(1 for law in retrieved_laws if law in expected_laws)
        precision = correct / len(retrieved_laws)
    mrr = 0.0
    for rank, law in enumerate(retrieved_laws, start=1):
        if law in expected_laws:
            mrr = 1.0 / rank
            break
    return {"hit": hit, "precision": precision, "mrr": mrr}


def evaluate(save=False):
    with open("evaluation_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    provider = rag_core.EMBED_PROVIDER
    print("=" * 80)
    print(f"  檢索評估 | 供應商={provider} | 模型={rag_core.EMBEDDING_MODEL}")
    print(f"  collection={rag_core.COLLECTION} | {len(questions)} 題")
    print("=" * 80)

    all_results = []
    for q in questions:
        points = retrieve(q["question"])
        retrieved_laws = [p.payload.get("law_code", "?") for p in points]
        metrics = calculate_metrics(retrieved_laws, q["expected_laws"])

        mark = "✅" if metrics["hit"] else "❌"
        print(f"\n[{q['id']}] {mark} {q['question'][:50]}…")
        print(f"  Expected : {q['expected_laws']}")
        print(f"  Retrieved: {retrieved_laws}")
        print(f"  Hit={metrics['hit']} | P@{TOP_K}={metrics['precision']:.2f} "
              f"| MRR={metrics['mrr']:.3f}")

        all_results.append({"id": q["id"], "category": q["category"],
                            "question": q["question"],
                            "retrieved": retrieved_laws, **metrics})

    def summary(rows):
        n = len(rows)
        return (sum(r["hit"] for r in rows) / n,
                sum(r["precision"] for r in rows) / n,
                sum(r["mrr"] for r in rows) / n)

    print("\n" + "=" * 80)
    print(f"📊 整體統計({provider})")
    print("=" * 80)
    h, p, m = summary(all_results)
    print(f"\n全體({len(all_results)} 題):")
    print(f"  Hit@{TOP_K}       = {h:.2%}")
    print(f"  Precision@{TOP_K} = {p:.2%}")
    print(f"  MRR          = {m:.3f}")

    per_cat = {}
    for category in ["precise", "colloquial", "difficult"]:
        rows = [r for r in all_results if r["category"] == category]
        if not rows:
            continue
        h, p, m = summary(rows)
        per_cat[category] = {"hit": h, "precision": p, "mrr": m, "n": len(rows)}
        print(f"\n{category}({len(rows)} 題):")
        print(f"  Hit@{TOP_K}       = {h:.2%}")
        print(f"  Precision@{TOP_K} = {p:.2%}")
        print(f"  MRR          = {m:.3f}")

    # 失敗題清單(對照時最有價值的部分)
    misses = [r for r in all_results if not r["hit"]]
    if misses:
        print(f"\n❌ 未命中 {len(misses)} 題:")
        for r in misses:
            print(f"  [{r['id']}] {r['question'][:40]}… → {r['retrieved']}")

    if save:
        out = f"eval_{provider}.json"
        h, p, m = summary(all_results)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"provider": provider,
                       "model": rag_core.EMBEDDING_MODEL,
                       "collection": rag_core.COLLECTION,
                       "overall": {"hit": h, "precision": p, "mrr": m},
                       "per_category": per_cat,
                       "details": all_results}, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已存 {out}(可與另一版對照)")


if __name__ == "__main__":
    evaluate(save="--save" in sys.argv)