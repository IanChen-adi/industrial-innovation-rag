from dotenv import load_dotenv
from openai import OpenAI

# 1. 載入 .env(會自動拿到 OPENAI_API_KEY)
load_dotenv()

# 2. 建立 OpenAI client
client = OpenAI()

# 3. 呼叫 embedding API
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="第 5 條 研究發展支出包含薪資、消耗性器材等費用。"
)

# 4. 拿出第一個結果的向量
vector = response.data[0].embedding

# 5. 看結果的「形狀」
print(f"✅ embedding 成功")
print(f"   向量維度: {len(vector)}")
print(f"   前 5 個數字: {vector[:5]}")
print(f"   資料型態: {type(vector[0]).__name__}")