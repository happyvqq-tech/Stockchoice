"""抽樣驗收：把我們存的 9 筆日線，與證交所／櫃買中心官方資料逐欄比對。只印結果，不印金鑰。研究用途，非投資建議。"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

# 我們資料庫裡的值（代號, 日期, 開, 高, 低, 收, 成交股數）
OURS = [
    ("2408", "2019-11-05", 70.8, 73, 70.4, 73, 51107655),
    ("2408", "2021-01-28", 79.9, 81.5, 78.7, 79.4, 19387911),
    ("2408", "2025-05-05", 35.75, 36.05, 33.5, 34.25, 46975609),
    ("3406", "2018-10-23", 220, 228.5, 218.5, 220, 7301988),
    ("3406", "2021-12-14", 548, 551, 529, 532, 2659200),
    ("3406", "2024-09-30", 489, 494.5, 485.5, 486, 1407451),
    ("6182", "2018-09-13", 39, 39.9, 37.7, 37.7, 22300446),
    ("6182", "2023-03-15", 45.5, 46.55, 45.25, 45.95, 8310551),
    ("6182", "2025-12-18", 31.6, 32.95, 30.2, 30.95, 52322869),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"_error": repr(e)}


def num(x):
    return float(str(x).replace(",", "").strip())


def twse(code, date):
    y, m, d = date.split("-")
    j = get(f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date={y}{m}01&stockNo={code}&response=json")
    roc = f"{int(y) - 1911}/{m}/{d}"
    for row in j.get("data", []):
        if row[0].strip() == roc:  # 日期,成交股數,成交金額,開,高,低,收,漲跌,筆數
            return dict(open=num(row[3]), high=num(row[4]), low=num(row[5]), close=num(row[6]), volume=num(row[1]))
    return {"_error": j.get("_error") or j.get("stat") or "查無該日"}


def tpex(code, date):
    y, m, d = date.split("-")
    j = get("https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?" + urllib.parse.urlencode({"date": f"{y}/{m}/{d}", "type": "EW", "response": "json"}))
    for t in j.get("tables", []):
        fields = t.get("fields", [])
        for row in t.get("data", []):
            if str(row[0]).strip() == code:
                f = {k: i for i, k in enumerate(fields)}

                def pick(*names):
                    for n in names:
                        for k, i in f.items():
                            if n in k:
                                return num(row[i])
                    raise KeyError(names)
                return dict(open=pick("開盤"), high=pick("最高"), low=pick("最低"), close=pick("收盤"), volume=pick("成交股數"))
    return {"_error": j.get("_error") or "查無該日或該股", "_date": j.get("date")}


ok_all = True
print("== 官方資料 vs 我們的資料 ==")
for code, date, o, h, l, c, v in OURS:
    off = (twse if code in ("2408", "3406") else tpex)(code, date)
    if "_error" in off:
        print(f"{code} {date}: 官方資料取得失敗 -> {off}")
        ok_all = False
        continue
    mine = dict(open=o, high=h, low=l, close=c, volume=v)
    diffs = []
    for k in ("open", "high", "low", "close"):
        if abs(off[k] - mine[k]) > 1e-6:
            diffs.append(f"{k}: 官方 {off[k]} / 我們 {mine[k]}")
    if abs(off["volume"] - mine["volume"]) > 0.5:
        diffs.append(f"volume: 官方 {off['volume']:.0f} / 我們 {mine['volume']}")
    ok_all &= not diffs
    print(f"{code} {date}: {'✅ 完全一致' if not diffs else '❌ ' + '；'.join(diffs)}  官方={off}")
print("== 結論 ==", "9 筆全部與官方一致" if ok_all else "有不一致或取得失敗，請看上方明細")
