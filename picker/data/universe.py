"""動態股票池（規格 5.2）：每月第一個交易日，以「該日之前」60 個交易日平均成交金額排名取前 N。
只用該日以前的資料，避免未來資訊。研究用途，非投資建議。"""
import pandas as pd


def monthly_universe(bars, calendar, top_n=300, lookback=60, min_days=20, exclude=()):
    """bars: 所有股票日線（date, symbol, turnover）；calendar: 交易日。
    回傳 month_start, symbol, rank, avg_turnover。"""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    bars = bars[~bars["symbol"].isin(exclude)]
    piv = bars.pivot_table(index="date", columns="symbol", values="turnover", aggfunc="last")
    piv = piv.reindex(cal[(cal >= piv.index.min()) & (cal <= piv.index.max())])
    firsts = pd.Series(piv.index, index=piv.index).groupby([piv.index.year, piv.index.month]).first()
    rows = []
    for d in firsts:
        pos = piv.index.get_loc(d)
        if pos < lookback:
            continue  # 回看期不足 60 個交易日
        win = piv.iloc[pos - lookback:pos]  # 嚴格早於 d
        cnt = win.notna().sum()
        avg = win.mean().where(cnt >= min_days).dropna().sort_values(ascending=False).head(top_n)
        for r, (sym, v) in enumerate(avg.items(), 1):
            rows.append((d, sym, r, float(v)))
    return pd.DataFrame(rows, columns=["month_start", "symbol", "rank", "avg_turnover"])
