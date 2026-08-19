"""
beta_health_check.py — Beta 服務健康檢查(外部視角,不依賴瀏覽器)
檢查:Streamlit server 心跳 / Qdrant cloud / OpenAI / Google Sheets
副作用(是特性):打 Qdrant = 重置免費層閒置計時器(保活)
用法:
  python beta_health_check.py
  (讀 .env 的 QDRANT_URL/QDRANT_API_KEY/OPENAI_API_KEY;
   Sheets 憑證讀 .streamlit/secrets.toml 的 [gcp_service_account] 與 SHEET_ID)
"""
import os
import sys
import traceback

import httpx
from dotenv import load_dotenv

load_dotenv()

APP_URL = "https://industrial-innovation-rag-adi.streamlit.app"
RESULTS = []


def check(name):
    def deco(fn):
        def wrapper():
            try:
                detail = fn()
                RESULTS.append((name, True, detail))
                print(f"✅ [{name}] {detail}")
            except Exception:
                err = traceback.format_exc().strip().splitlines()[-1]
                RESULTS.append((name, False, err))
                print(f"❌ [{name}] {err}")
        return wrapper
    return deco


@check("1. Streamlit server 心跳")
def c_streamlit():
    # /_stcore/health 是 Streamlit 的健康端點,回 "ok" 代表 server process 活著
    r = httpx.get(f"{APP_URL}/~/+/_stcore/health", timeout=20, follow_redirects=True)
    assert r.status_code == 200 and "ok" in r.text.lower(), \
        f"HTTP {r.status_code}: {r.text[:80]}"
    return f"server 活著(health=ok)"


@check("2. Streamlit 首頁載入")
def c_streamlit_page():
    r = httpx.get(APP_URL, timeout=20, follow_redirects=True)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    asleep = "gone to sleep" in r.text.lower() or "wake" in r.text.lower()
    return "⚠️ 頁面顯示休眠中(要按喚醒)" if asleep else "頁面正常(HTTP 200)"


@check("3. Qdrant cloud(順便保活)")
def c_qdrant():
    url = os.getenv("QDRANT_URL", "")
    key = os.getenv("QDRANT_API_KEY", "")
    assert url.startswith("https"), "QDRANT_URL 未設定(檢查 .env)"
    r = httpx.get(f"{url.rstrip('/')}/collections",
                  headers={"api-key": key}, timeout=20)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:100]}"
    names = [c["name"] for c in r.json()["result"]["collections"]]
    assert "laws_hybrid" in names and "hybrid_manual" in names, f"collections={names}"
    return f"collections={names}(閒置計時器已重置)"


@check("4. OpenAI API")
def c_openai():
    from openai import OpenAI
    v = OpenAI().embeddings.create(model="text-embedding-3-small",
                                   input="健康檢查").data[0].embedding
    assert len(v) == 1536
    return "embedding OK"


@check("5. Google Sheets 落地層")
def c_sheets():
    import tomllib
    import gspread
    from google.oauth2.service_account import Credentials
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    creds = Credentials.from_service_account_info(
        secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    sh = gspread.authorize(creds).open_by_key(secrets["SHEET_ID"]).sheet1
    header = sh.row_values(1)
    assert "event" in header, f"表頭異常:{header[:5]}"
    return f"可讀寫,表頭 {len(header)} 欄,目前 {len(sh.get_all_values())-1} 筆資料"


if __name__ == "__main__":
    print("=" * 60)
    print("  Beta 服務健康檢查")
    print("=" * 60)
    for fn in [c_streamlit, c_streamlit_page, c_qdrant, c_openai, c_sheets]:
        fn()
    print("=" * 60)
    fails = [n for n, ok, _ in RESULTS if not ok]
    if not fails:
        print("🎉 全部健康 — 可以放心發連結 / 繼續 beta")
        sys.exit(0)
    print(f"🔴 異常:{', '.join(fails)}")
    sys.exit(1)