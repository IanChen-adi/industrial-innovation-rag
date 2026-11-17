"""
api.py — 法規問答 HTTP API（給 adi-portal 的 AI 協同助理呼叫）

與 app.py（Streamlit 網頁）共用 rag_core，回答邏輯與檢索結果完全一致。

端點
  GET  /           健康檢查
  POST /ask        {question, history?} → {mode, text, sources:[{source, law_code, article, score}]}

驗證
  呼叫端必須帶 X-API-Key 標頭，值等於環境變數 RAG_API_KEY（共用密鑰）。
  未設定 RAG_API_KEY 時一律拒絕（fail closed）。

環境變數（部署時設定，不進版控）
  RAG_API_KEY、OPENAI_API_KEY、GEMINI_API_KEY、QDRANT_URL、QDRANT_API_KEY、
  ANSWER_PROVIDER、EMBED_PROVIDER、PREPROC_PROVIDER（與 Streamlit Secrets 相同）

本機執行：uvicorn api:app --port 8080
"""
import hmac
import logging
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import rag_core

log = logging.getLogger("rag-api")
app = FastAPI(title="ADI 產創法規問答 API", docs_url=None, redoc_url=None, openapi_url=None)


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=12)


def _check_key(x_api_key: str | None) -> None:
    expected = os.getenv("RAG_API_KEY", "")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/")
def health():
    return {"ok": True, "service": "rag-api"}


@app.post("/ask")
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    history = [{"role": t.role, "content": t.content}
               for t in req.history if t.role in ("user", "assistant")][-6:]
    try:
        res = rag_core.answer(req.question.strip(), history)
    except Exception as e:  # 細節只寫 log，不回給呼叫端
        log.exception("rag answer failed")
        raise HTTPException(status_code=500, detail=f"rag_error:{type(e).__name__}")
    sources = [{k: c.get(k) for k in ("source", "law_code", "article", "score")}
               for c in res.get("chunks", [])]
    return {"mode": res.get("mode", "answer"), "text": res.get("text", ""), "sources": sources}
