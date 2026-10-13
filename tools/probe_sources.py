"""Phase 1 探路 #3：FinMind 免費版有沒有「分割 / 減資 / 面額變更」資料集。只印狀態與筆數。研究用途，非投資建議。"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

TOKEN = os.environ.get("FINMIND_TOKEN", "")


def q(dataset, **params):
    url = "https://api.finmindtrade.com/api/v4/data?" + urllib.parse.urlencode({"dataset": dataset, **params})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {TOKEN}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            code, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read()
    try:
        j = json.loads(body)
    except Exception:  # noqa: BLE001
        print(f"  {dataset} {params}: HTTP {code} 非 JSON")
        return
    rows = j.get("data") or []
    print(f"  {dataset} {params}: HTTP {code} msg={j.get('msg')!r} 筆數={len(rows)}")
    for r in rows[:3]:
        print("    ", r)
    if rows:
        print("     欄位=", list(rows[0].keys()), " 日期範圍=", rows[0].get("date"), "~", rows[-1].get("date"))


print("== 分割 / 減資 / 面額變更（免費版？）==")
for ds in ("TaiwanStockSplitPrice", "TaiwanStockCapitalReductionReferencePrice", "TaiwanStockParValueChange"):
    q(ds, data_id="0050", start_date="2018-01-01", end_date="2026-10-12")
    q(ds, start_date="2025-06-01", end_date="2025-06-30")  # 不指定股票（可能是付費）
print("== 對照：還原後仍有大跳動的個股，看是否有減資/分割 ==")
for sid in ("4949", "3095", "1529", "2364"):
    for ds in ("TaiwanStockCapitalReductionReferencePrice", "TaiwanStockSplitPrice"):
        q(ds, data_id=sid, start_date="2018-01-01", end_date="2026-10-12")
