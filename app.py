"""
app.py — 租稅優惠法規 AI 問答|Beta 測試前端(Streamlit)
v5(內部試營運):公務帳號登入(@adi.gov.tw)、法規依據過濾、正式版文案
執行:python -m streamlit run app.py
"""
import os
import streamlit as st

# ---- Secrets → 環境變數橋接(必須在 import rag_core 之前)----
for k in ("OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
    if k in st.secrets:
        os.environ[k] = st.secrets[k]

import json
import uuid
from datetime import datetime, timezone, timedelta

import gspread
from google.oauth2.service_account import Credentials

import rag_core

TW = timezone(timedelta(hours=8))
LOG_PATH = "logs/beta_log.jsonl"
ALLOWED_DOMAIN = "@adi.gov.tw"
os.makedirs("logs", exist_ok=True)

st.set_page_config(page_title="租稅優惠法規 AI 問答", page_icon="🖍️",
                   layout="centered")

st.markdown("""
<style>
:root { --ink:#22303c; --archive:#35558A; --marker:#F5D76E; }
.stApp { background:#F7F8F6; }
h1, h2, h3 { font-family:"Noto Serif TC","PMingLiU",serif; color:var(--ink); }
.disclaimer { background:#fff; border-left:4px solid var(--marker);
  padding:.6rem .9rem; font-size:.85rem; color:#5a6672; border-radius:0 6px 6px 0; }
.badge { display:inline-block; background:var(--archive); color:#fff;
  font-size:.72rem; padding:.15rem .55rem; border-radius:999px; margin-right:.4rem; }
</style>
""", unsafe_allow_html=True)

# ---- 落地層:Google Sheets(主)+ 本地 JSONL(備援)----
SHEET_COLS = ["t", "sid", "event", "turn_id", "role", "query", "translated",
              "mode", "services", "confidence", "question_type", "chunks",
              "answer", "thumbs", "note"]


@st.cache_resource
def _get_sheet():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(st.secrets["SHEET_ID"]).sheet1


def log_row(row: dict):
    # 備援:本地 JSONL(雲端重啟會清空,但寫了不虧)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # 主要落地:Google Sheets(失敗不擋使用者)
    try:
        pre = row.get("preprocess") or {}
        flat = {**row,
                "services": ",".join(pre.get("services", [])),
                "confidence": pre.get("confidence", ""),
                "question_type": pre.get("question_type", ""),
                "chunks": " | ".join(f"{c.get('law_code','')} {c.get('article','')}"
                                     for c in row.get("chunks", []))}
        _get_sheet().append_row(
            [str(flat.get(k, ""))[:4000] for k in SHEET_COLS],
            value_input_option="RAW")
    except Exception as e:
        print(f"Sheets 寫入失敗(本地備援已存):{e}", flush=True)


# ---- 免費額度使用量(依 Sheets 今日 qa 筆數推估)----
#   每題呼叫:回答生成 ×1、問題判斷 ×1、語意向量 ×1(三個獨立額度池)
QUOTA_POOLS = [
    ("回答生成", 500, "gemini-3.5-flash-lite"),
    ("問題判斷", 500, "gemini-3.1-flash-lite"),
    ("語意向量", 1000, "gemini-embedding-001"),
]


@st.cache_data(ttl=180, show_spinner=False)
def _today_qa_count():
    """今日問答題數(每 3 分鐘更新一次,避免頻繁讀表)"""
    try:
        rows = _get_sheet().get_all_values()
        today = datetime.now(TW).strftime("%Y-%m-%d")
        return sum(1 for r in rows[1:]
                   if len(r) > 2 and r[0].startswith(today) and r[2] == "qa")
    except Exception:
        return None


def _log_note(turn_id):
    """意見欄按 Enter 時觸發:單獨落地一筆 feedback_note"""
    note = st.session_state.get(f"note_{turn_id}", "").strip()
    if note:
        log_row({"t": datetime.now(TW).isoformat(),
                 "sid": st.session_state.session_id,
                 "event": "feedback_note", "turn_id": turn_id, "note": note})
        st.toast("意見已記下,謝謝!")


# ---- Session 初始化 ----
ss = st.session_state
ss.setdefault("entered", False)
ss.setdefault("session_id", str(uuid.uuid4())[:8])
ss.setdefault("history", [])      # LLM 用 [{"role","content"}]
ss.setdefault("display", [])      # UI 用 [{"who","text","meta","turn_id"}]

# ---- 第一關:公務帳號登入 ----
if not st.user.is_logged_in:
    st.title("🖍️ 租稅優惠法規 AI 問答")
    st.caption("數位產業署|內部試營運")
    st.markdown('<div class="disclaimer">本系統為<b>實驗性內部工具</b>。'
                '回答由 AI 依法規資料生成,僅供同仁參考,'
                '對外正式意見仍以主管機關規定與函釋為準。</div>',
                unsafe_allow_html=True)
    st.write("")
    st.button("🔐 使用公務帳號登入(@adi.gov.tw)",
              use_container_width=True, on_click=st.login)
    st.stop()

# ---- 第二關:網域檢查 ----
user_email = (st.user.email or "").lower()
if not user_email.endswith(ALLOWED_DOMAIN):
    st.error(f"本系統僅限數位產業署公務帳號({ALLOWED_DOMAIN})使用。\n\n"
             f"目前登入:{user_email}")
    st.button("登出並更換帳號", on_click=st.logout)
    st.stop()

# ---- 進場紀錄(每 session 一次)----
if not ss.get("entered"):
    ss.entered = True
    log_row({"t": datetime.now(TW).isoformat(), "sid": ss.session_id,
             "event": "enter", "role": user_email})

# ---- 主畫面 ----
st.title("🖍️ 租稅優惠法規 AI 問答")
st.markdown('<span style="font-size:.8rem;color:#8a94a0">回答僅供參考,'
            '以主管機關正式規定為準</span>', unsafe_allow_html=True)

with st.sidebar:
    st.caption(f"👤 {user_email}")
    st.subheader("可以問什麼?")
    st.markdown("- 研發/設備投資抵減\n- 中小企業研發抵減\n- 個人投資新創減除\n"
                "- 所得基本稅額條例高風險新創\n\n例:「研發投抵哪天前要交件?」\n「申請書要蓋什麼章?」")
    if st.button("🔄 換個話題(清空對話)"):
        ss.history, ss.display = [], []
        st.rerun()
    if st.button("登出"):
        st.logout()

    # ===== 今日免費額度 =====
    st.divider()
    st.caption("📊 今日免費額度")
    used = _today_qa_count()
    if used is None:
        st.caption("(暫時無法取得使用量)")
    else:
        for label, limit, model in QUOTA_POOLS:
            pct = min(used / limit, 1.0)
            st.progress(pct, text=f"{label} {used}/{limit}")
        st.caption(f"今日已回答 {used} 題 · 每日 0 時(太平洋時間)重置")

# ---- 歷史訊息 ----
for m in ss.display:
    with st.chat_message("user" if m["who"] == "user" else "assistant"):
        st.markdown(m["text"])
        if m["who"] == "assistant":
            law_chunks = [c for c in m.get("meta", {}).get("chunks", [])
                          if c.get("source") and c["source"] != "作業手冊"]
            if law_chunks:
                with st.expander("📎 這則回答的法規依據"):
                    for c in law_chunks:
                        st.markdown(f"**{c['source']} {c['article']}**"
                                    f"(相似度 {c['score']})\n\n{c['text'][:120]}…")
            fb = st.feedback("thumbs", key=f"fb_{m['turn_id']}")
            if fb is not None and not ss.get(f"logged_{m['turn_id']}"):
                log_row({"t": datetime.now(TW).isoformat(), "sid": ss.session_id,
                         "event": "feedback", "turn_id": m["turn_id"],
                         "thumbs": "up" if fb == 1 else "down"})
                ss[f"logged_{m['turn_id']}"] = True
                st.toast("已收到回饋,謝謝!")
            st.text_input("(選填)哪裡不對,或你原本期待什麼?",
                          key=f"note_{m['turn_id']}",
                          label_visibility="collapsed",
                          placeholder="(選填)哪裡不對,或你原本期待什麼?按 Enter 送出",
                          on_change=_log_note, args=(m["turn_id"],))

# ---- 輸入 ----
if q := st.chat_input("輸入你的問題…"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("查找法規中…"):
            try:
                res = rag_core.answer(q, ss.history)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                res = {"mode": "error",
                       "text": f"系統暫時出了點問題,請再試一次。({type(e).__name__})",
                       "preprocess": {}, "chunks": [], "translated": q}
        st.markdown(res["text"])

    turn_id = str(uuid.uuid4())[:8]
    ss.display.append({"who": "user", "text": q, "turn_id": turn_id + "u"})
    ss.display.append({"who": "assistant", "text": res["text"],
                       "meta": {"chunks": res["chunks"]}, "turn_id": turn_id})
    ss.history.append({"role": "user", "content": q})
    ss.history.append({"role": "assistant", "content": res["text"]})

    log_row({"t": datetime.now(TW).isoformat(), "sid": ss.session_id,
             "event": "qa", "turn_id": turn_id, "role": user_email,
             "query": q, "translated": res.get("translated"),
             "mode": res["mode"], "preprocess": res.get("preprocess"),
             "chunks": [{k: c[k] for k in ("law_code", "article", "score")}
                        for c in res.get("chunks", [])],
             "answer": res["text"]})
    st.rerun()