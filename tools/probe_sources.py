"""Phase 1 探路：確認 FinMind 免費版、證交所/櫃買、Supabase 在 GitHub Actions 上是否可用。
只印出狀態、筆數與欄位名稱，不印任何金鑰。研究用途，非投資建議。"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

FM_TOKEN = os.environ.get("FINMIND_TOKEN", "")
SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")


def get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001
        return None, repr(e).encode()


def finmind(dataset, **params):
    q = urllib.parse.urlencode({"dataset": dataset, **params})
    h = {"User-Agent": "Mozilla/5.0"}
    if FM_TOKEN:
        h["Authorization"] = f"Bearer {FM_TOKEN}"
    code, body = get("https://api.finmindtrade.com/api/v4/data?" + q, h)
    try:
        j = json.loads(body)
    except Exception:  # noqa: BLE001
        print(f"  {dataset} {params}: HTTP {code} 非 JSON: {body[:150]!r}")
        return
    rows = j.get("data") or []
    cols = list(rows[0].keys()) if rows else []
    print(f"  {dataset} {params}: HTTP {code} status={j.get('status')} msg={j.get('msg')!r} 筆數={len(rows)}")
    if rows:
        print(f"    欄位={cols}")
        print(f"    首筆={rows[0]}")


print("== FinMind 使用額度 ==")
code, body = get("https://api.web.finmindtrade.com/v2/user_info",
                 {"Authorization": f"Bearer {FM_TOKEN}", "User-Agent": "Mozilla/5.0"})
print(f"  token 已設定={bool(FM_TOKEN)} HTTP {code} {body[:300]!r}")

print("== FinMind 各資料集（免費版能不能用）==")
finmind("TaiwanStockInfo")
finmind("TaiwanStockTradingDate")
finmind("TaiwanStockPrice", data_id="2330", start_date="2018-01-01", end_date="2018-01-31")
finmind("TaiwanStockPrice", start_date="2024-01-03", end_date="2024-01-03")  # 不指定股票：整個市場某一天
finmind("TaiwanStockPriceAdj", data_id="2330", start_date="2018-01-01", end_date="2018-01-31")
finmind("TaiwanStockInstitutionalInvestorsBuySell", data_id="2330", start_date="2018-01-01", end_date="2018-01-31")
finmind("TaiwanStockMarginPurchaseShortSale", data_id="2330", start_date="2018-01-01", end_date="2018-01-31")
finmind("TaiwanStockMonthRevenue", data_id="2330", start_date="2018-01-01", end_date="2018-12-31")
finmind("TaiwanStockDividendResult", data_id="2330", start_date="2018-01-01", end_date="2019-12-31")
finmind("TaiwanStockDelisting")
finmind("TaiwanStockPrice", data_id="1409", start_date="2018-01-01", end_date="2018-01-31")  # 抽樣：較舊個股

print("== 證交所 / 櫃買 ==")
for name, url in {
    "TWSE 全市場日行情": "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date=20240103&type=ALLBUT0999&response=json",
    "TWSE 三大法人": "https://www.twse.com.tw/rwd/zh/fund/T86?date=20240103&selectType=ALLBUT0999&response=json",
    "TWSE 融資融券": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date=20240103&selectType=STOCK&response=json",
    "TWSE 除權息": "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate=20240101&endDate=20240131&response=json",
    "TPEx 上櫃日行情": "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d=113/01/03&o=json",
}.items():
    code, body = get(url)
    print(f"  {name}: HTTP {code} 大小={len(body)} 開頭={body[:100]!r}")

print("== Supabase ==")
if SB_URL and SB_KEY:
    code, body = get(SB_URL + "/storage/v1/bucket", {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
    try:
        names = [(b.get("name"), b.get("public")) for b in json.loads(body)]
    except Exception:  # noqa: BLE001
        names = body[:200]
    print(f"  Storage buckets: HTTP {code} {names}")
else:
    print("  SUPABASE_URL / SUPABASE_SECRET_KEY 未設定")
