#!/usr/bin/env bash
# 部署法規問答 API（api.py）到 Cloud Run。相容 macOS 內建 bash 3.2。
# 用法：./deploy_api.sh        （需先建立 .env.api.yaml，見 .env.api.example.yaml）
set -euo pipefail

PROJECT="${PROJECT:-investsys}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-rag-api}"
ENV_FILE="${ENV_FILE:-.env.api.yaml}"
cd "$(dirname "$0")"

[ -f "${ENV_FILE}" ] || { echo "找不到 ${ENV_FILE}：請複製 .env.api.example.yaml 為 ${ENV_FILE} 並填入金鑰"; exit 1; }
grep -q '請產生' "${ENV_FILE}" && { echo "${ENV_FILE} 的 RAG_API_KEY 還是範本文字，請換成隨機字串"; exit 1; }

echo "### 部署 ${SERVICE}（${PROJECT} / ${REGION}）"
gcloud run deploy "${SERVICE}" --source . --project "${PROJECT}" --region "${REGION}" \
  --allow-unauthenticated --env-vars-file "${ENV_FILE}" \
  --memory 1Gi --min-instances 0 --max-instances 2 --quiet
echo "### 完成。服務網址："
gcloud run services describe "${SERVICE}" --project "${PROJECT}" --region "${REGION}" --format 'value(status.url)'
