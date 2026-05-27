import jieba
from collections import Counter
from qdrant_client.models import SparseVector

# 載入你的領域詞典（一次就好）
jieba.load_userdict("custom_dict.txt")


def text_to_sparse(text: str) -> SparseVector:
    """
    把一段文字轉成 sparse vector（B1 簡化版：只用 TF 詞頻）

    流程:
      1. jieba 斷詞
      2. 過濾空白/單字元雜訊
      3. 每個詞 hash 成固定 id
      4. 用詞頻當權重
    """
    # 1. 斷詞
    words = [w for w in jieba.cut(text) if w.strip()]

    # 2. 算詞頻（TF）
    tf = Counter(words)

    # 3. 每個詞 → hash id（用 Python 內建 hash 取絕對值，限制範圍避免過大）
    indices = []
    values = []
    for word, count in tf.items():
        word_id = abs(hash(word)) % (2**31)   # 轉成 31 位元內的正整數
        indices.append(word_id)
        values.append(float(count))           # B1: 權重 = 詞頻

    return SparseVector(indices=indices, values=values)


# --- 自我測試 ---
if __name__ == "__main__":
    tests = [
        "我要申請高風險新創認定",
        "研發投資抵減的申請條件",
    ]
    for t in tests:
        sv = text_to_sparse(t)
        print(f"文字: {t}")
        print(f"  indices: {sv.indices}")
        print(f"  values:  {sv.values}")
        print()