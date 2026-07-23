"""
app.py — 租稅優惠法規 AI 問答|Beta 測試前端(Streamlit)
v4:log 落地改 Google Sheets(本地 JSONL 為備援)、診斷工具藏 ADMIN_CODE
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
os.makedirs("logs", exist_ok=True)

st.set_page_config(page_title="租稅優惠法規 AI 問答|Beta", page_icon="🖍️",
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
        print(f"Sheets 寫入失敗(本地備援已存):{e}")


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
ss.setdefault("authed", False)
ss.setdefault("role", None)
ss.setdefault("session_id", str(uuid.uuid4())[:8])
ss.setdefault("history", [])      # LLM 用 [{"role","content"}]
ss.setdefault("display", [])      # UI 用 [{"who","text","meta","turn_id"}]

# ---- 入場:通行碼 + 身分 ----
if not ss.authed:
    st.title("🖍️ 租稅優惠法規 AI 問答")
    st.caption("Beta 測試版|個人研究專案")
    st.markdown('<div class="disclaimer">本系統為<b>個人研究專案</b>之實驗工具,'
                '非任何政府機關官方服務。回答由 AI 依法規資料生成,僅供參考,'
                '實際申請請以主管機關正式規定與函釋為準。</div>',
                unsafe_allow_html=True)
    st.write("")
    code = st.text_input("測試通行碼", type="password")
    role = st.radio("請選擇最符合您背景的身分(影響測試分析,不影響回答)",
                    ["做過這項業務", "公務員(非本業務)", "一般人"], index=None)
    if st.button("進入測試", use_container_width=True):
        if code != st.secrets.get("BETA_CODE", "beta2026"):
            st.error("通行碼不正確。請向邀請你的人確認。")
        elif role is None:
            st.warning("請先選擇身分。")
        else:
            ss.authed, ss.role = True, role
            log_row({"t": datetime.now(TW).isoformat(), "sid": ss.session_id,
                     "event": "enter", "role": role})
            st.rerun()
    st.stop()

# ---- 主畫面 ----
st.title("🖍️ 租稅優惠法規 AI 問答")
st.markdown(f'<span class="badge">{ss.role}</span>'
            '<span style="font-size:.8rem;color:#8a94a0">回答僅供參考,'
            '以主管機關正式規定為準</span>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("可以問什麼?")
    st.markdown("- 研發/設備投資抵減\n- 中小企業研發抵減\n- 個人投資新創減除\n"
                "- 高風險新創認定\n\n例:「研發投抵哪天前要交件?」\n「申請書要蓋什麼章?」")
    if st.button("🔄 換個話題(清空對話)"):
        ss.history, ss.display = [], []
        st.rerun()

    # ===== 開發者工具(需 ADMIN_CODE,測試者看不到)=====
    st.divider()
    admin_code = st.text_input("⚙️", type="password",
                               label_visibility="collapsed", placeholder="")
    if admin_code and admin_code == st.secrets.get("ADMIN_CODE", ""):
        st.caption("🛠 開發者工具")
        if st.button("🔧 連線診斷"):
            import traceback
            st.code(f"QDRANT_URL = {os.getenv('QDRANT_URL')!r}")
            try:
                names = [c.name for c in rag_core.qdrant.get_collections().collections]
                st.success(f"Qdrant 連線成功:{names}")
            except Exception:
                st.error("Qdrant 連線失敗:")
                st.code(traceback.format_exc())
        if st.button("📊 Sheets 落地測試"):
            import traceback
            try:
                log_row({"t": datetime.now(TW).isoformat(), "sid": ss.session_id,
                         "event": "sheets_test", "turn_id": "test",
                         "note": "手動測試寫入"})
                st.success("已嘗試寫入一筆 sheets_test,開表確認")
            except Exception:
                st.code(traceback.format_exc())
        dbg_q = st.text_input("🔬 單題路由診斷(只看翻譯+前處理)")
        if dbg_q:
            from query_translator import translate_query
            from preprocessor import preprocess
            t = translate_query(dbg_q)
            st.code(f"翻譯: {dbg_q} → {t}")
            st.json(preprocess(t, []))

# ---- 歷史訊息 ----
for m in ss.display:
    with st.chat_message("user" if m["who"] == "user" else "assistant"):
        st.markdown(m["text"])
        if m["who"] == "assistant":
            if m.get("meta", {}).get("chunks"):
                with st.expander("📎 這則回答的依據(檢索到的條文)"):
                    for c in m["meta"]["chunks"]:
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
             "event": "qa", "turn_id": turn_id, "role": ss.role,
             "query": q, "translated": res.get("translated"),
             "mode": res["mode"], "preprocess": res.get("preprocess"),
             "chunks": [{k: c[k] for k in ("law_code", "article", "score")}
                        for c in res.get("chunks", [])],
             "answer": res["text"]})
    st.rerun()
