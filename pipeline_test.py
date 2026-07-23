"""
pipeline_test.py — RAG pipeline 分段診斷
用法:
  本機打本機:  python pipeline_test.py
  本機打雲端:  QDRANT_URL="https://...:6333" QDRANT_API_KEY="..." python pipeline_test.py
  跳過 LLM 段: SKIP_LLM=1 python pipeline_test.py
每段獨立測試不互相中斷,結尾給總表 + 第一個斷點。
"""
import os
import json
import traceback
from dotenv import load_dotenv
load_dotenv()

RESULTS = []


def step(name):
    def deco(fn):
        def wrapper(ctx):
            try:
                detail = fn(ctx)
                RESULTS.append((name, True, detail or ""))
                print(f"✅ [{name}] {detail or ''}")
            except Exception as e:
                tb = traceback.format_exc().strip().splitlines()[-1]
                RESULTS.append((name, False, tb))
                print(f"❌ [{name}] {tb}")
        return wrapper
    return deco


# ─────────────────────────────────────────────
@step("1. 環境變數")
def t_env(ctx):
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    key = os.getenv("QDRANT_API_KEY") or ""
    oai = os.getenv("OPENAI_API_KEY") or ""
    ctx["url"], ctx["key"] = url, key
    assert oai, "OPENAI_API_KEY 未設定"
    return f"QDRANT_URL={url} | qdrant key 尾8碼={key[-8:] if key else '(無,本機模式)'} | openai key 有"


@step("2. raw HTTP 直打 /collections(繞過 qdrant-client)")
def t_raw(ctx):
    import httpx
    headers = {"api-key": ctx["key"]} if ctx["key"] else {}
    r = httpx.get(f"{ctx['url']}/collections", headers=headers, timeout=15)
    body = r.text[:300]
    ctx["raw_body"] = body
    assert r.status_code == 200, f"HTTP {r.status_code}: {body}"
    names = [c["name"] for c in r.json()["result"]["collections"]]
    ctx["raw_names"] = names
    return f"HTTP 200, {len(r.text)} bytes, server 回報 collections={names}"


@step("3. qdrant-client 連線 get_collections")
def t_client(ctx):
    from qdrant_client import QdrantClient
    import qdrant_client
    q = QdrantClient(url=ctx["url"], api_key=ctx["key"] or None,
                     check_compatibility=False)
    ctx["q"] = q
    names = [c.name for c in q.get_collections().collections]
    ctx["client_names"] = names
    ver = getattr(qdrant_client, "__version__", "?")
    # 關鍵比對:raw vs client
    raw = ctx.get("raw_names", [])
    match = "一致" if set(names) == set(raw) else f"⚠️ 不一致!raw={raw}"
    return f"client v{ver} 看到 {names}({match})"


@step("4. laws_hybrid 存在 + points 數")
def t_laws(ctx):
    info = ctx["q"].get_collection("laws_hybrid")
    n = info.points_count
    assert n and n > 0, f"points={n}(空的)"
    return f"points={n}(預期 256)"


@step("5. hybrid_manual 存在 + points 數")
def t_manual(ctx):
    info = ctx["q"].get_collection("hybrid_manual")
    n = info.points_count
    assert n and n > 0, f"points={n}(空的)"
    return f"points={n}(預期 17)"


@step("6. 撈一個 point 看 payload 完整性")
def t_point(ctx):
    pts = ctx["q"].retrieve("laws_hybrid", ids=[1], with_vectors=True)
    assert pts, "retrieve 回空"
    p = pts[0]
    fields = list(p.vector.keys()) if isinstance(p.vector, dict) else ["(單一向量)"]
    law = p.payload.get("law_name", "?")
    return f"id=1 向量欄位={fields} law_name={law[:20]}"


@step("7. OpenAI embedding")
def t_embed(ctx):
    from openai import OpenAI
    oc = OpenAI()
    v = oc.embeddings.create(model="text-embedding-3-small",
                             input="研發投資抵減").data[0].embedding
    ctx["vec"] = v
    assert len(v) == 1536
    return f"dense 1536 維 OK"


@step("8. dense 檢索 laws_hybrid")
def t_search(ctx):
    hits = ctx["q"].query_points(collection_name="laws_hybrid",
                                 query=ctx["vec"], using="dense", limit=3).points
    assert hits, "查詢回 0 筆"
    top = hits[0].payload.get("law_code", "?")
    return f"top1={top} score={hits[0].score:.3f}(共 {len(hits)} 筆)"


@step("9. manual 檢索 + filter 檢索")
def t_search2(ctx):
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    m = ctx["q"].query_points(collection_name="hybrid_manual",
                              query=ctx["vec"], using="dense", limit=2).points
    f = ctx["q"].query_points(
        collection_name="laws_hybrid", query=ctx["vec"], using="dense",
        query_filter=Filter(must=[FieldCondition(
            key="law_code", match=MatchAny(any=["01_產創研發"]))]),
        limit=3).points
    assert m and f
    codes = {h.payload.get("law_code") for h in f}
    assert codes == {"01_產創研發"}, f"filter 失效:{codes}"
    return f"manual {len(m)} 筆 / filter 後全為 01 ✓"


@step("10. 前處理 LLM(preprocess)")
def t_pre(ctx):
    if os.getenv("SKIP_LLM"):
        return "SKIP_LLM=1 跳過"
    from preprocessor import preprocess
    r = preprocess("我研發投資抵減哪一天以前要交件")
    assert r["services"] == ["01_產創研發"], f"判斷異常:{r}"
    return f"services={r['services']} conf={r['confidence']} type={r['question_type']}"


@step("11. 完整 answer() 三分岔")
def t_answer(ctx):
    if os.getenv("SKIP_LLM"):
        return "SKIP_LLM=1 跳過"
    import rag_core
    cases = [("申請書要蓋什麼章", "answer"),
             ("研發投資抵減哪天前要交件", "answer"),
             ("你們有辦護照嗎", "scope")]
    outs = []
    for q_text, expect in cases:
        res = rag_core.answer(q_text, [])
        assert res["mode"] == expect, f"「{q_text}」mode={res['mode']} 預期 {expect}"
        outs.append(f"{q_text[:6]}→{res['mode']}")
    return " / ".join(outs)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  RAG Pipeline 分段診斷")
    print("=" * 62)
    ctx = {}
    for fn in [t_env, t_raw, t_client, t_laws, t_manual,
               t_point, t_embed, t_search, t_search2, t_pre, t_answer]:
        fn(ctx)

    print("\n" + "=" * 62)
    fails = [(n, d) for n, ok, d in RESULTS if not ok]
    if not fails:
        print("🎉 全部通過 — pipeline 沒有斷點")
    else:
        print(f"🔴 斷點:第一個失敗 = 「{fails[0][0]}」")
        print(f"   細節:{fails[0][1]}")
        print("   → 這一段之前的環節都是好的,問題就在這一段(或它依賴的服務)")
