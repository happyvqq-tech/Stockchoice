"""台股 adapter：FinMind 免費版（每小時 600 次）。
- 全市場單日、還原價為付費資料，不使用；還原價由除權息資料自行計算（見 adjust.py）。
- 事件資料依規格 5.3 轉成「可取得日」available_date。
研究用途，非投資建議。"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .base import BAR_COLUMNS, MarketAdapter

API = "https://api.finmindtrade.com/api/v4/data"
EXCLUDE_INDUSTRY = ("ETF", "ETN", "存託憑證", "創新板", "受益", "權證", "所有證券", "Index", "大盤")
BENCHMARK = "0050"


class RateLimited(Exception):
    pass


class FinMindClient:
    """FinMind HTTP 客戶端：自備節流（預設約 580 次/小時），遇上限丟 RateLimited。"""

    def __init__(self, token, min_interval=6.2, fetch=None, sleep=time.sleep, clock=time.monotonic):
        self.token, self.min_interval = token, min_interval
        self._fetch, self._sleep, self._clock = fetch or self._http, sleep, clock
        self._last = None

    def _http(self, url, headers):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def query(self, dataset, retries=3, **params):
        q = urllib.parse.urlencode({"dataset": dataset, **params})
        h = {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {self.token}"}
        for attempt in range(retries):
            if self._last is not None:
                wait = self.min_interval - (self._clock() - self._last)
                if wait > 0:
                    self._sleep(wait)
            self._last = self._clock()
            code, body = self._fetch(f"{API}?{q}", h)
            try:
                j = json.loads(body)
            except Exception:  # noqa: BLE001
                j = {"status": code, "msg": body[:100].decode("utf-8", "ignore") if isinstance(body, bytes) else str(body)}
            msg = str(j.get("msg", ""))
            if code == 402 or "upper limit" in msg.lower() or "reach" in msg.lower() and "limit" in msg.lower():
                raise RateLimited(msg)
            if code == 200 and j.get("status") == 200:
                return pd.DataFrame(j.get("data") or [])
            if code in (400, 401, 403):
                raise RuntimeError(f"FinMind {dataset} {params}: HTTP {code} {msg}")
            self._sleep(5 * (attempt + 1))
        raise RuntimeError(f"FinMind {dataset} {params}: 重試後仍失敗")


def next_trading_day(dates, calendar):
    """每個日期 → 其後的下一個交易日（規格 5.3：t 日收盤後公布的資料，t+1 起可用）。"""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    idx = cal.searchsorted(pd.DatetimeIndex(pd.to_datetime(dates)), side="right")
    out = [cal[i] if i < len(cal) else pd.NaT for i in idx]
    return pd.DatetimeIndex(out)


def revenue_available_date(year, month):
    """月營收保守可用日：次月 11 日（FinMind 沒有實際公布日）。"""
    y, m = int(year), int(month)
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return pd.Timestamp(year=y, month=m, day=11)


class TWAdapter(MarketAdapter):
    market = "TW"

    def __init__(self, client, calendar=None):
        self.c = client
        self._calendar = calendar

    def get_universe(self):
        info = self.c.query("TaiwanStockInfo")
        info = info[info["type"].isin(["twse", "tpex"])]
        info = info[info["stock_id"].str.fullmatch(r"\d{4}")]
        info = info[~info["stock_id"].str.startswith("0")]
        pat = "|".join(map(re.escape, EXCLUDE_INDUSTRY))
        info = info[~info["industry_category"].fillna("").str.contains(pat)]
        info = info.sort_values("date").drop_duplicates("stock_id", keep="last")
        return pd.DataFrame({"symbol": info["stock_id"], "name": info["stock_name"],
                             "industry": info["industry_category"], "exchange": info["type"]}).reset_index(drop=True)

    def get_delisted(self):
        d = self.c.query("TaiwanStockDelisting")
        return d[d["stock_id"].str.fullmatch(r"\d{4}") & ~d["stock_id"].str.startswith("0")].reset_index(drop=True)

    def get_calendar(self):
        d = self.c.query("TaiwanStockTradingDate")
        return pd.DataFrame({"date": pd.to_datetime(d["date"])})

    def get_daily_bars(self, symbol, start, end):
        d = self.c.query("TaiwanStockPrice", data_id=symbol, start_date=start, end_date=end)
        if d.empty:
            return pd.DataFrame(columns=BAR_COLUMNS)
        out = pd.DataFrame({
            "date": pd.to_datetime(d["date"]), "market": "TW", "symbol": d["stock_id"],
            "open": d["open"], "high": d["max"], "low": d["min"], "close": d["close"],
            "volume": d["Trading_Volume"], "turnover": d["Trading_money"]})
        return out[BAR_COLUMNS]

    def get_events(self, kind, symbol, start, end):
        if kind == "institutional":
            d = self.c.query("TaiwanStockInstitutionalInvestorsBuySell", data_id=symbol, start_date=start, end_date=end)
            return self._institutional(d)
        if kind == "margin":
            d = self.c.query("TaiwanStockMarginPurchaseShortSale", data_id=symbol, start_date=start, end_date=end)
            return self._margin(d)
        if kind == "revenue":
            d = self.c.query("TaiwanStockMonthRevenue", data_id=symbol, start_date=start, end_date=end)
            return self._revenue(d)
        if kind == "dividend":
            d = self.c.query("TaiwanStockDividendResult", data_id=symbol, start_date=start, end_date=end)
            return self._dividend(d)
        if kind == "reduction":
            d = self.c.query("TaiwanStockCapitalReductionReferencePrice", data_id=symbol, start_date=start, end_date=end)
            return self._reduction(d)
        raise ValueError(kind)

    def get_splits(self, start, end):
        """全市場分割／面額變更（FinMind 免費版可不指定股票，依年度分段查）。"""
        out = []
        for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            a, b = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")), min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
            out.append(self._split(self.c.query("TaiwanStockSplitPrice", start_date=a.strftime("%Y-%m-%d"), end_date=b.strftime("%Y-%m-%d"))))
        out = [o for o in out if len(o)]
        return pd.concat(out, ignore_index=True) if out else self._split(pd.DataFrame())

    # ---- 轉換（純函式，可單元測試）----
    def _cal(self):
        if self._calendar is None:
            self._calendar = self.get_calendar()["date"]
        return self._calendar

    def _institutional(self, d):
        cols = ["date", "symbol", "foreign_net", "trust_net", "dealer_net", "available_date"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        d = d.assign(net=d["buy"] - d["sell"], date=pd.to_datetime(d["date"]))
        grp = {"Foreign_Investor": "foreign_net", "Foreign_Dealer_Self": "foreign_net",
               "Investment_Trust": "trust_net", "Dealer_self": "dealer_net",
               "Dealer_Hedging": "dealer_net", "Dealer": "dealer_net"}
        d["g"] = d["name"].map(grp)
        p = d.dropna(subset=["g"]).pivot_table(index=["date", "stock_id"], columns="g", values="net", aggfunc="sum").reset_index()
        for c in ("foreign_net", "trust_net", "dealer_net"):
            if c not in p:
                p[c] = 0.0
        p = p.rename(columns={"stock_id": "symbol"})
        p["available_date"] = next_trading_day(p["date"], self._cal())
        return p[cols].sort_values("date").reset_index(drop=True)

    def _margin(self, d):
        cols = ["date", "symbol", "margin_balance", "short_balance", "available_date"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        p = pd.DataFrame({"date": pd.to_datetime(d["date"]), "symbol": d["stock_id"],
                          "margin_balance": d["MarginPurchaseTodayBalance"], "short_balance": d["ShortSaleTodayBalance"]})
        p["available_date"] = next_trading_day(p["date"], self._cal())
        return p.sort_values("date").reset_index(drop=True)

    def _revenue(self, d):
        cols = ["revenue_year", "revenue_month", "symbol", "revenue", "available_date"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        p = pd.DataFrame({"revenue_year": d["revenue_year"], "revenue_month": d["revenue_month"],
                          "symbol": d["stock_id"], "revenue": d["revenue"]})
        p["available_date"] = [revenue_available_date(y, m) for y, m in zip(p["revenue_year"], p["revenue_month"])]
        return p.sort_values("available_date").reset_index(drop=True)

    def _dividend(self, d):
        cols = ["date", "symbol", "before_price", "after_price"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        p = pd.DataFrame({"date": pd.to_datetime(d["date"]), "symbol": d["stock_id"],
                          "before_price": d["before_price"], "after_price": d["after_price"]})
        return p.sort_values("date").reset_index(drop=True)

    def _split(self, d):
        cols = ["date", "symbol", "before_price", "after_price"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame({"date": pd.to_datetime(d["date"]), "symbol": d["stock_id"],
                             "before_price": d["before_price"], "after_price": d["after_price"]}).sort_values("date").reset_index(drop=True)

    def _reduction(self, d):
        """減資：最後交易日收盤 → 恢復買賣參考價（彌補虧損的減資，股價會跳升，但股東價值不變，還原時視為中性）。"""
        cols = ["date", "symbol", "before_price", "after_price"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame({"date": pd.to_datetime(d["date"]), "symbol": d["stock_id"],
                             "before_price": d["ClosingPriceonTheLastTradingDay"],
                             "after_price": d["PostReductionReferencePrice"]}).sort_values("date").reset_index(drop=True)
