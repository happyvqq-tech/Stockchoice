"""Phase 1 探路 #2：補驗疑點。只印狀態、筆數與欄位，不印任何金鑰。研究用途，非投資建議。"""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

FM_TOKEN = os.environ.get("FINMIND_TOKEN", "")
SB_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SB_KEY = os.environ.get("SUPABASE_SECRET_KEY", "").strip()


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
    h = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {FM_TOKEN}"}
    code, body = get("https://api.finmindtrade.com/api/v4/data?" + q, h)
    try:
        j = json.loads(body)
    except Exception:  # noqa: BLE001
        print(f"  {dataset} {params}: HTTP {code} 非 JSON")
        return []
    rows = j.get("data") or []
    dates = (rows[0].get("date"), rows[-1].get("date")) if rows else None
    print(f"  {dataset} {params}: HTTP {code} msg={j.get('msg')!r} 筆數={len(rows)} 日期範圍={dates}")
    return rows


print("== FinMind：長區間、下市股、股票分類 ==")
finmind("TaiwanStockPrice", data_id="2330", start_date="2018-01-01", end_date="2026-10-12")
finmind("TaiwanStockInstitutionalInvestorsBuySell", data_id="2330", start_date="2018-01-01", end_date="2026-10-12")
finmind("TaiwanStockMarginPurchaseShortSale", data_id="2330", start_date="2018-01-01", end_date="2026-10-12")
finmind("TaiwanStockMonthRevenue", data_id="2330", start_date="2018-01-01", end_date="2026-10-12")
finmind("TaiwanStockDividendResult", data_id="2330", start_date="2018-01-01", end_date="2026-10-12")
info = finmind("TaiwanStockInfo")
cats = {}
for r in info:
    cats[(r["type"], r["industry_category"])] = cats.get((r["type"], r["industry_category"]), 0) + 1
print("  類別統計(type, industry)前40:", sorted(cats.items(), key=lambda x: -x[1])[:40])
print("  不重複代號數=", len({r["stock_id"] for r in info}), " 4碼代號數=",
      len({r["stock_id"] for r in info if re.fullmatch(r"\d{4}", r["stock_id"])}))
delist = finmind("TaiwanStockDelisting")
four = [r for r in delist if re.fullmatch(r"\d{4}", r["stock_id"])]
print("  下市名單 4碼代號數=", len(four), " 範例=", four[:3])
for r in four[:2]:
    finmind("TaiwanStockPrice", data_id=r["stock_id"], start_date="2018-01-01", end_date="2026-10-12")

print("== 櫃買日期格式 ==")
for label, url in {
    "d=113/01/03": "https://www.tpex.org.tw/web/stock/aftertrading/daily_close_quotes/stk_quote_result.php?l=zh-tw&d=113/01/03&o=json",
    "date=2024/01/03": "https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date=2024%2F01%2F03&type=EW&response=json",
    "三大法人 date=2024/01/03": "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&date=2024%2F01%2F03&response=json",
    "融資融券 date=2024/01/03": "https://www.tpex.org.tw/www/zh-tw/margin/balance?date=2024%2F01%2F03&response=json",
}.items():
    code, body = get(url)
    m = re.search(rb'"date":"?(\d+)', body)
    print(f"  {label}: HTTP {code} 大小={len(body)} 回傳date={m.group(1) if m else None}")

print("== Supabase ==")
if not (SB_URL and SB_KEY):
    print("  SUPABASE_URL / SUPABASE_SECRET_KEY 未設定")
else:
    p = urllib.parse.urlparse(SB_URL)
    print(f"  網址格式: scheme={p.scheme} 結尾路徑={p.path!r} 是否 supabase.co={p.netloc.endswith('.supabase.co')}")
    base = f"{p.scheme}://{p.netloc}"
    code, body = get(base + "/storage/v1/bucket", {"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
    try:
        names = [(b.get("name"), "public" if b.get("public") else "private") for b in json.loads(body)]
    except Exception:  # noqa: BLE001
        names = body[:200]
    print(f"  Storage buckets: HTTP {code} {names}")
