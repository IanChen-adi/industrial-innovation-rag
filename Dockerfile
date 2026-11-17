# RAG 法規問答 API（api.py）— Cloud Run
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
# 只複製 API 需要的程式與資料（金鑰、.env、Streamlit secrets 一律不進映像）
COPY api.py rag_core.py preprocessor.py query_translator.py sparse_utils.py custom_dict.txt ./
ENV PORT=8080
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
