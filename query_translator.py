"""
Query 翻譯 + 服務識別：
1. 翻譯：縮寫/口語 → 法規正式用語
2. 識別：根據 query 內容，決定要查哪些 law_code（用於 metadata filter）
"""

# === 翻譯對應表（縮寫 → 正式詞）===
TRANSLATION_MAP = {
    # 研發
    "研發投抵": "研發投資抵減",
    "中小企抵減": "中小企業研發投資抵減",
    "中小企研發": "中小企業研發投資抵減",
    "投抵": "投資抵減",
    # 設備（5 大類）
    "智機投抵": "智慧機械投資抵減",
    "智機抵減": "智慧機械投資抵減",
    "5G投抵":   "5G投資抵減",
    "資安投抵": "資安投資抵減",
    "AI投抵":   "人工智慧投資抵減",
    "節能投抵": "節能減碳投資抵減",
}


# === 服務 → law_code 對應表 ===
# 偵測順序：長詞優先（避免短詞先匹配）
SERVICE_TO_LAW = {
    # 研發類
    "中小企業研發投資抵減": ["05_中小企業研發", "01_產創研發"],
    "研發投資抵減":         ["01_產創研發"],
    "研發抵減":             ["01_產創研發"],

    # 設備類（5 大細項 + 統稱）
    "智慧機械投資抵減": ["02_設備"],
    "5G投資抵減":       ["02_設備"],
    "資安投資抵減":     ["02_設備"],
    "人工智慧投資抵減": ["02_設備"],
    "節能減碳投資抵減": ["02_設備"],
    "設備投資抵減":     ["02_設備"],
    "設備抵減":         ["02_設備"],

    # 個人投資新創（03）
    "個人投資新創減除": ["03_個人投資新創減除"],
    "個人投資新創":     ["03_個人投資新創減除"],

    # 高風險新創（注意:全名只指 04;只說「認定」涵蓋 03+04 → 之後要反問）
    "所得基本稅額高風險新創認定": ["04_所得基本稅額高風險新創認定"],
    "高風險新創認定":             ["03_個人投資新創減除", "04_所得基本稅額高風險新創認定"],
    "高風險新創":                 ["03_個人投資新創減除", "04_所得基本稅額高風險新創認定"],
}

# === law_code → 官方申請網址 ===
# === law_code → 官方申請網址(已逐一驗證/待確認標注)===
LAW_TO_URL = {
    "01_產創研發": {
        "名稱": "研發投資抵減",
        "url": {"一般認定(研發活動審查認定)": "https://moda.gov.tw/ADI/services/apply-serivces/regulations-for-industrial-innovation/1575",   # ✅已驗證
                "專案認定": "https://moda.gov.tw/ADI/services/apply-serivces/regulations-for-industrial-innovation/1573"},                    # ✅已驗證
    },
    "02_設備": {
        "名稱": "設備投資抵減(智慧機械/5G/資安/AI/節能減碳)",
        "url": {"線上申辦系統": "https://ipd.nat.gov.tw/taxcredit/"},   # ⚠️請你點開確認
    },
    "03_個人投資新創減除": {
        "名稱": "個人投資新創減除",
        "url": {"申請入口": "https://moda.gov.tw/ADI/services/apply-serivces/regulations-for-industrial-innovation/1565"},   # ✅已驗證
    },
    "04_所得基本稅額高風險新創認定": {
        "名稱": "所得基本稅額高風險新創認定",
        "url": {"申請入口": "https://moda.gov.tw/ADI/services/apply-serivces/service-4/13086"},   # ⚠️請你點開確認
    },
    "05_中小企業研發": {
        "名稱": "中小企業研發投資抵減",
        "url": {"一般認定(研發活動審查認定)": "https://moda.gov.tw/ADI/services/apply-serivces/smb-rd-investment-incentive/1579",   # ⚠️請你點開確認
                "專案認定": "https://moda.gov.tw/ADI/services/apply-serivces/smb-rd-investment-incentive/1577"},                    # ⚠️請你點開確認
    },
}


def get_service_urls(law_codes) -> str:
    """根據 law_codes 回傳給 LLM 的網址段落;無對應時回空字串"""
    if not law_codes:
        return ""
    lines = ["【相關服務官方網址】"]
    for code in law_codes:
        if code in LAW_TO_URL:
            info = LAW_TO_URL[code]
            for label, u in info["url"].items():
                lines.append(f"- {info['名稱']}({label}):{u}")
    return "\n".join(lines) if len(lines) > 1 else ""

def translate_query(query: str) -> str:
    """翻譯 query：縮寫 → 正式詞，最長詞優先避免子字串誤替換"""
    sorted_keys = sorted(TRANSLATION_MAP.keys(), key=len, reverse=True)
    result = query
    for short in sorted_keys:
        if short in result:
            result = result.replace(short, TRANSLATION_MAP[short])
    return result


def identify_services(query: str) -> list:
    """
    根據 query 內容，回傳要查的 law_code 清單。
    若 query 沒有匹配任何服務 → 回傳 None（代表不加 filter，全庫查）
    """
    sorted_services = sorted(SERVICE_TO_LAW.keys(), key=len, reverse=True)

    matched_laws = []
    matched_services = []
    for service in sorted_services:
        if service in query:
            # 避免同一個 query 被多個重疊服務重複加 law（例如「研發投資抵減」已被匹配，就不再匹配「研發抵減」）
            if not any(service in s for s in matched_services):
                matched_services.append(service)
                for law in SERVICE_TO_LAW[service]:
                    if law not in matched_laws:
                        matched_laws.append(law)

    return matched_laws if matched_laws else None


# --- 自我測試 ---
if __name__ == "__main__":
    tests = [
        "我研發投抵哪一天以前要交件",                   # → 研發投資抵減 → 01
        "我要申請智機投抵",                              # → 智慧機械投資抵減 → 02
        "資安投抵的條件是什麼",                          # → 02
        "中小企抵減和研發投抵差在哪",                    # → 兩種都提到，05+01
        "高風險新創認定怎麼申請",                        # → 03+04（沒講基本稅額，兩個都包含）
        "所得基本稅額高風險新創認定的條件",              # → 只 04
        "我朋友投資的事可以省稅嗎",                       # → 無匹配 → None（全庫查）
    ]
    for t in tests:
        translated = translate_query(t)
        services = identify_services(translated)
        print(f"原:    {t}")
        print(f"譯:    {translated}")
        print(f"law:   {services}")
        print()