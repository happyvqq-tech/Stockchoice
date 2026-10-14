"""特徵 v1（規格第 6 節）＋ 截面排名標準化。
所有特徵只用 t 日收盤（含）以前的資料；事件資料用「可取得日」對齊。研究用途，非投資建議。"""
import numpy as np
import pandas as pd

FEATURE_NAMES = {
    "ret_5": "5日報酬", "ret_20": "20日報酬", "ret_60": "60日報酬",
    "rel_strength_20": "相對大盤20日強度", "rsi_14": "RSI(14)",
    "close_to_ma20": "股價相對月線", "close_to_ma60": "股價相對季線", "ma20_slope": "月線斜率",
    "close_to_high20": "接近20日高點", "dist_52w_high": "距52週高點",
    "vol_ratio_20": "量能放大倍數", "atr14_pct": "波動度(ATR)", "bb_width_pctile": "布林帶寬壓縮度",
    "turnover_60": "60日平均成交金額",
    "foreign_net_5d": "外資5日買超", "trust_net_5d": "投信5日買超", "trust_consec_days": "投信連續買超天數",
    "margin_chg_5d": "融資5日變化", "rev_yoy": "月營收年增率", "rev_yoy_new_high": "月營收創12個月新高",
    "bench_above_ma60": "大盤在季線上", "bench_ret_20": "大盤20日報酬",
}
RANKED = [k for k in FEATURE_NAMES if k not in ("rev_yoy_new_high", "bench_above_ma60", "bench_ret_20", "trust_consec_days")]
FEATURES = list(FEATURE_NAMES)


def _pctile_last(x):
    return (x[:-1] < x[-1]).mean()


def bench_features(bench):
    """bench: 基準指數日線（date, adj_close）。回傳 date, bench_above_ma60, bench_ret_20, bench_ret_5。"""
    b = bench.sort_values("date").drop_duplicates("date").set_index("date")["adj_close"]
    out = pd.DataFrame({"bench_ret_20": b.pct_change(20), "bench_ret_5": b.pct_change(5)})
    ma = b.rolling(60).mean()
    out["bench_above_ma60"] = (b > ma).astype(float).where(ma.notna())
    return out.reset_index()


def _align(ev, dates, cols):
    """把事件資料（含 available_date）對齊到交易日：t 日只看得到 available_date ≤ t 的。"""
    if ev is None or len(ev) == 0:
        return pd.DataFrame(index=dates, columns=cols, dtype=float)
    g = ev.groupby("available_date")[cols].sum()
    g = g[~g.index.duplicated()]
    full = g.reindex(g.index.union(dates)).sort_index()
    return full.reindex(dates)


def _revenue_features(rev, dates):
    """每個交易日只用 available_date ≤ 該日的最新月營收。"""
    if rev is None or len(rev) == 0:
        return pd.DataFrame({"rev_yoy": np.nan, "rev_yoy_new_high": np.nan}, index=dates)
    r = rev.sort_values(["revenue_year", "revenue_month"]).drop_duplicates(["revenue_year", "revenue_month"]).copy()
    key = r["revenue_year"] * 12 + r["revenue_month"]
    s = pd.Series(r["revenue"].values, index=key.values)
    prev = s.reindex(key.values - 12).values
    r["rev_yoy"] = np.where(prev > 0, r["revenue"].values / prev - 1, np.nan)
    full = s.reindex(np.arange(key.min(), key.max() + 1))
    hi = full.rolling(12, min_periods=12).max().reindex(key.values).values
    r["rev_yoy_new_high"] = np.where(np.isnan(hi), np.nan, (r["revenue"].values >= hi).astype(float))
    r = r.sort_values("available_date")[["available_date", "rev_yoy", "rev_yoy_new_high"]]
    left = pd.DataFrame({"date": dates})
    m = pd.merge_asof(left, r, left_on="date", right_on="available_date", direction="backward")
    return m.set_index("date")[["rev_yoy", "rev_yoy_new_high"]]


def symbol_features(bars, bench, inst=None, margin=None, rev=None):
    """單一股票的原始特徵。bars 需含 date, open, high, low, close, volume, turnover, adj_close。"""
    b = bars.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    b = b[(b["close"] > 0) & (b["open"] > 0)].reset_index(drop=True)
    dates = pd.DatetimeIndex(b["date"])
    f = pd.DataFrame(index=dates)
    c = pd.Series(b["adj_close"].values, index=dates)
    k = c / b["close"].values  # 還原係數，同樣套在高低價
    hi = pd.Series(b["high"].values * k.values, index=dates)
    lo = pd.Series(b["low"].values * k.values, index=dates)
    vol = pd.Series(b["volume"].values.astype(float), index=dates)
    f["ret_5"], f["ret_20"], f["ret_60"] = c.pct_change(5), c.pct_change(20), c.pct_change(60)
    bf = bench.set_index("date")
    f["rel_strength_20"] = f["ret_20"] - bf["bench_ret_20"].reindex(dates)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    f["close_to_ma20"], f["close_to_ma60"] = c / ma20 - 1, c / ma60 - 1
    f["ma20_slope"] = ma20 / ma20.shift(5) - 1
    f["close_to_high20"] = c / hi.rolling(20).max()
    f["dist_52w_high"] = c / hi.rolling(252, min_periods=120).max() - 1
    f["vol_ratio_20"] = vol / vol.rolling(20).mean().replace(0, np.nan)
    pc = c.shift(1)
    tr = pd.concat([hi - lo, (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.rolling(14).mean() / c
    bbw = 4 * c.rolling(20).std() / ma20
    f["bb_width_pctile"] = bbw.rolling(120, min_periods=60).apply(_pctile_last, raw=True)
    f["turnover_60"] = pd.Series(b["turnover"].values, index=dates).rolling(60).mean()
    # 籌碼：資料在 available_date 才看得到（=次一交易日），所以只會落後、不會偷看
    ia = _align(inst, dates, ["foreign_net", "trust_net"])
    vsum5 = vol.shift(1).rolling(5).sum().replace(0, np.nan)  # 與法人資料同樣延後一日，口徑一致
    f["foreign_net_5d"] = ia["foreign_net"].rolling(5, min_periods=3).sum() / vsum5
    f["trust_net_5d"] = ia["trust_net"].rolling(5, min_periods=3).sum() / vsum5
    pos = (ia["trust_net"] > 0).astype(int).where(ia["trust_net"].notna())
    grp = (pos != 1).cumsum()
    f["trust_consec_days"] = pos.fillna(0).groupby(grp).cumsum().where(ia["trust_net"].notna())
    if margin is not None and len(margin):
        m = margin.drop_duplicates("available_date").set_index("available_date")["margin_balance"]
        m = m.reindex(m.index.union(dates)).sort_index().ffill().reindex(dates)
        f["margin_chg_5d"] = m / m.shift(5).replace(0, np.nan) - 1
    else:
        f["margin_chg_5d"] = np.nan
    f[["rev_yoy", "rev_yoy_new_high"]] = _revenue_features(rev, dates).values
    f["bench_above_ma60"] = bf["bench_above_ma60"].reindex(dates)
    f["bench_ret_20"] = bf["bench_ret_20"].reindex(dates)
    f = f.replace([np.inf, -np.inf], np.nan)
    f.index.name = "date"
    return f.reset_index()


def rank_normalize(panel):
    """每個交易日、在股票池內，把連續特徵換成 0~1 的排名百分位（只看同一天，不看別天）。"""
    out = panel.copy()
    g = out.groupby("date")
    for c in RANKED:
        out[c] = g[c].rank(pct=True)
    return out


def in_universe(panel, universe):
    """只留下「當月股票池」內的股票。股票池每月第一個交易日決定，該月整月沿用。"""
    u = universe[["month_start", "symbol"]].copy()
    months = pd.DatetimeIndex(np.sort(u["month_start"].unique()))
    idx = months.searchsorted(pd.DatetimeIndex(panel["date"]), side="right") - 1
    p = panel[idx >= 0].copy()
    p["month_start"] = months[idx[idx >= 0]]
    return p.merge(u, on=["month_start", "symbol"], how="inner").drop(columns="month_start")


def build_panel(bars, universe, bench_symbol, events=None, symbols=None):
    """bars: 全部日線（已含 adj_close）；events: {"institutional": df, "margin": df, "revenue": df}（含 symbol 欄）。
    回傳原始特徵面板（date, symbol, 特徵…），只含各月股票池成員。"""
    events = events or {}
    bench = bars[bars["symbol"] == bench_symbol]
    bf = bench_features(bench)
    by = {k: dict(tuple(v.groupby("symbol"))) for k, v in events.items() if v is not None and len(v)}
    syms = symbols if symbols is not None else sorted(universe["symbol"].unique())
    parts = []
    for s, b in bars[bars["symbol"].isin(syms)].groupby("symbol"):
        f = symbol_features(b, bf, by.get("institutional", {}).get(s), by.get("margin", {}).get(s),
                            by.get("revenue", {}).get(s))
        f.insert(1, "symbol", s)
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    return in_universe(panel, universe)
