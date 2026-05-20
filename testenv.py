from dotenv import load_dotenv
import os

# 從 .env 載入環境變數
load_dotenv()

# 讀取 OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# 確認讀到了(但不要印出完整 key)
if api_key:
    print(f"✅ 讀到 key 了,長度 {len(api_key)},開頭 {api_key[:7]}...")
else:
    print("❌ 讀不到 key,檢查 .env 是否存在")