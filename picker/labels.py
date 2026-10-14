"""標籤（規格第 7 節）：t 日收盤後產生訊號 → t+1 開盤進場 → t+hold_days 收盤出場。
出場日若跌停或停牌 → 順延到下一個可賣出日（最多 5 個交易日）。研究用途，非投資建議。"""
import numpy as np
import pandas as pd

from picker.costs import round_trip

LIMIT = 0.10
DEFER_MAX = 5


def is_limit_up(open_, prev_close):
    return open_ >= prev_close * (1 + LIMIT - 0.005)


def is_limit_down(close, prev_close):
    return close <= prev_close * (1 - LIMIT + 0.005)


def price_grids(bars, calendar):
    """把日線轉成「交易日 × 股票」的表。adj 價用於報酬，raw 價用於漲跌停判斷。"""
    b = bars[(bars["close"] > 0) & (bars["open"] > 0)].drop_duplicates(["date", "symbol"])
    k = b["adj_close"] / b["close"]
    b = b.assign(adj_open=b["open"] * k)
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(calendar)))

    def g(col):
        return b.pivot(index="date", columns="symbol", values=col).reindex(cal)
    return {"adj_open": g("adj_open"), "adj_close": g("adj_close"), "open": g("open"), "close": g("close")}


def add_labels(panel, grids, bench_symbol, cfg, hold_days=None):
    """為面板每一列加上 entry_date / exit_date / fwd_ret（扣成本）/ fwd_excess / buyable。"""
    h = hold_days or cfg["hold_days"]
    cost = round_trip(cfg["costs"][cfg["primary_market"]])
    cal = grids["adj_close"].index
    ao, ac, rc, ro = grids["adj_open"], grids["adj_close"], grids["close"], grids["open"]
    prev = rc.shift(1)
    sellable = ~(rc.isna() | is_limit_down(rc, prev))
    ac_sell = ac.where(sellable).bfill(limit=DEFER_MAX)  # 跌停／停牌 → 往後找第一個賣得掉的收盤價
    entry = ao.shift(-1)
    buyable = ~(is_limit_up(ro, rc.shift(1)).shift(-1).fillna(True).astype(bool)) & entry.notna()
    exit_ = ac_sell.shift(-h)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    out = panel[["date", "symbol"]].copy()
    di = pos.reindex(out["date"]).values
    ci = {s: j for j, s in enumerate(ac.columns)}
    cj = out["symbol"].map(ci).values
    E, X, B = entry.values, exit_.values, buyable.values
    out["fwd_entry"] = E[di, cj]
    out["fwd_exit"] = X[di, cj]
    out["buyable"] = B[di, cj]
    out["entry_date"] = pd.Series(cal).shift(-1).reindex(di).values
    out["exit_date"] = pd.Series(cal).shift(-h).reindex(di).values
    out["fwd_gross"] = out["fwd_exit"] / out["fwd_entry"] - 1
    out["fwd_ret"] = out["fwd_gross"] - cost
    be, bx = E[:, ci[bench_symbol]], ac.shift(-h).values[:, ci[bench_symbol]]
    out["bench_ret"] = bx[di] / be[di] - 1
    out["fwd_excess"] = out["fwd_ret"] - out["bench_ret"]
    return out.drop(columns=["fwd_entry", "fwd_exit"])


def add_train_label(df, top_pct):
    """label = 1：當日股票池內 fwd_excess 排名前 top_pct。"""
    r = df.groupby("date")["fwd_excess"].rank(pct=True, ascending=False)
    df = df.copy()
    df["label"] = np.where(df["fwd_excess"].isna(), np.nan, (r <= top_pct).astype(float))
    return df
