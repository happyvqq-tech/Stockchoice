import numpy as np
import pandas as pd
import pytest

from picker.backtest.baselines import momentum_score, random_score
from picker.backtest.metrics import portfolio_metrics, signal_quality
from picker.backtest.portfolio import Grid, run_backtest
from picker.backtest.walkforward import make_windows, run_walkforward
from picker.config import load_config
from picker.costs import buy_cost, round_trip, sell_cost
from picker.data.adjust import add_adj_close
from picker.features.build import FEATURES, build_panel, rank_normalize, symbol_features, bench_features
from picker.guard import HoldoutError, assert_no_holdout
from picker.labels import add_labels, price_grids
from picker.models.rule_score import RULE_FEATURES, rule_score

CFG = load_config()


def make_bars(n_sym=12, n_days=320, seed=1):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for s in ["0050"] + [f"{1101 + i}" for i in range(n_sym)]:
        r = rng.normal(0.0004, 0.012, n_days)
        c = 50 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        hi, lo = np.maximum(o, c) * 1.005, np.minimum(o, c) * 0.995
        v = rng.integers(1000, 5000, n_days) * 1000
        rows.append(pd.DataFrame({"date": cal, "market": "TW", "symbol": s, "open": o, "high": hi, "low": lo,
                                  "close": c, "volume": v, "turnover": v * c}))
    bars = pd.concat(rows, ignore_index=True)
    bars["adj_close"] = bars["close"]
    return bars, cal


def make_universe(bars, cal):
    syms = [s for s in bars["symbol"].unique() if s != "0050"]
    ms = pd.Series(cal).groupby([cal.year, cal.month]).first().values
    return pd.DataFrame([(m, s) for m in ms for s in syms], columns=["month_start", "symbol"])


def test_cost_known_cases():
    c = CFG["costs"]["TW"]
    assert buy_cost(c) == pytest.approx(0.001425 + 0.001)
    assert sell_cost(c) == pytest.approx(0.001425 + 0.003 + 0.001)
    assert round_trip(c) == pytest.approx(0.00785)


def test_holdout_guard():
    cfg = {"holdout_start": "2026-04-01"}
    assert_no_holdout(pd.to_datetime(["2026-03-31"]), cfg)
    with pytest.raises(HoldoutError):
        assert_no_holdout(pd.to_datetime(["2026-04-01"]), cfg)
    assert_no_holdout(pd.to_datetime(["2026-05-01"]), cfg, final_exam=True)


def test_truncation_no_future_data():
    bars, cal = make_bars()
    bench = bench_features(bars[bars["symbol"] == "0050"])
    b = bars[bars["symbol"] == "1101"]
    inst = pd.DataFrame({"date": cal, "foreign_net": np.arange(len(cal)) * 1.0, "trust_net": np.sin(np.arange(len(cal))) * 1000,
                         "available_date": cal.shift(1, freq="B")})
    margin = pd.DataFrame({"date": cal, "margin_balance": np.arange(len(cal)) + 100.0, "available_date": cal.shift(1, freq="B")})
    rev = pd.DataFrame({"revenue_year": [2019] * 12 + [2020] * 8, "revenue_month": list(range(1, 13)) + list(range(1, 9)),
                        "revenue": np.arange(20) * 10 + 100.0})
    rev["available_date"] = [pd.Timestamp(y + (m == 12), m % 12 + 1, 11) for y, m in zip(rev.revenue_year, rev.revenue_month)]
    full = symbol_features(b, bench, inst, margin, rev).set_index("date")
    for t in [cal[150], cal[200], cal[290]]:
        cut = symbol_features(b[b["date"] <= t], bench[bench["date"] <= t], inst[inst["available_date"] <= t],
                              margin[margin["available_date"] <= t], rev[rev["available_date"] <= t]).set_index("date")
        pd.testing.assert_series_equal(cut.loc[t, FEATURES].astype(float), full.loc[t, FEATURES].astype(float), check_names=False)
    assert full[FEATURES].notna().any().all()


def test_label_uses_next_open_and_hold_close():
    bars, cal = make_bars()
    uni = make_universe(bars, cal)
    panel = build_panel(bars, uni, "0050")
    grids = price_grids(bars, cal)
    cfg = dict(CFG, hold_days=10)
    lab = add_labels(panel, grids, "0050", cfg)
    r = lab[lab["symbol"] == "1105"].iloc[100]
    i = list(cal).index(r["date"])
    b = bars[bars["symbol"] == "1105"].set_index("date")
    exp = b["close"].iloc[i + 10] / b["open"].iloc[i + 1] - 1 - round_trip(cfg["costs"]["TW"])
    assert r["fwd_ret"] == pytest.approx(exp)
    assert r["entry_date"] == cal[i + 1] and r["exit_date"] == cal[i + 10]
    be = bars[bars["symbol"] == "0050"].set_index("date")
    assert r["bench_ret"] == pytest.approx(be["close"].iloc[i + 10] / be["open"].iloc[i + 1] - 1)


def test_limit_up_entry_not_buyable_and_limit_down_exit_deferred():
    bars, cal = make_bars()
    i = 100
    m = (bars["symbol"] == "1103")
    d = bars[m].index
    bars.loc[d[i + 1], "open"] = bars.loc[d[i], "close"] * 1.10  # 隔日開盤漲停
    bars.loc[d[i + 3], "close"] = bars.loc[d[i + 2], "close"] * 0.90  # 出場日跌停（h=2 時 exit = i+2... 用 h=3）
    bars["adj_close"] = bars["close"]
    uni = make_universe(bars, cal)
    panel = build_panel(bars, uni, "0050")
    lab = add_labels(panel, price_grids(bars, cal), "0050", dict(CFG, hold_days=3))
    row = lab[(lab["symbol"] == "1103") & (lab["date"] == cal[i])].iloc[0]
    assert not row["buyable"]
    row2 = lab[(lab["symbol"] == "1103") & (lab["date"] == cal[i + 1])].iloc[0]
    b = bars[m].set_index("date")
    # 訊號日 i+1 → 出場日 i+4？不：此處驗證 i 之前的訊號：出場日 i+3 跌停 → 順延到 i+4 收盤
    row3 = lab[(lab["symbol"] == "1103") & (lab["date"] == cal[i])].iloc[0]
    assert row3["exit_date"] == cal[i + 3]
    exp = b["close"].iloc[i + 4] / b["open"].iloc[i + 1] - 1 - round_trip(CFG["costs"]["TW"])
    assert row3["fwd_ret"] == pytest.approx(exp)


def test_rank_normalize_is_per_day():
    bars, cal = make_bars()
    panel = build_panel(bars, make_universe(bars, cal), "0050")
    rk = rank_normalize(panel)
    g = rk.dropna(subset=["ret_20"]).groupby("date")["ret_20"]
    assert (g.max() <= 1).all() and (g.min() > 0).all()


def _scored(seed=3, n_days=320):
    bars, cal = make_bars(n_sym=14, n_days=n_days, seed=seed)
    uni = make_universe(bars, cal)
    rng = np.random.default_rng(seed)
    syms = [s for s in bars["symbol"].unique() if s != "0050"]
    inst = pd.concat([pd.DataFrame({"date": cal, "symbol": s, "foreign_net": rng.normal(0, 1e5, len(cal)),
                                    "trust_net": rng.normal(0, 1e4, len(cal)), "available_date": cal.shift(1, freq="B")}) for s in syms])
    rev = pd.concat([pd.DataFrame({"revenue_year": 2019, "revenue_month": range(1, 13), "symbol": s,
                                   "revenue": rng.uniform(80, 120, 12),
                                   "available_date": [pd.Timestamp(2019 + (m == 12), m % 12 + 1, 11) for m in range(1, 13)]})
                     for s in syms] + [pd.DataFrame({"revenue_year": 2020, "revenue_month": range(1, 13), "symbol": s,
                                   "revenue": rng.uniform(80, 120, 12),
                                   "available_date": [pd.Timestamp(2020 + (m == 12), m % 12 + 1, 11) for m in range(1, 13)]})
                     for s in syms])
    panel = build_panel(bars, uni, "0050", events={"institutional": inst, "revenue": rev})
    rk = rank_normalize(panel)
    rk["score"] = rule_score(rk)
    return bars, cal, panel, rk


def test_rule_score_needs_enough_features():
    _, _, _, rk = _scored()
    assert rk["score"].notna().any()
    assert set(RULE_FEATURES) <= set(rk.columns)
    assert (rk["score"].dropna().between(0, 1)).all()


def test_backtest_engine_buffer_reduces_turnover_and_costs_apply():
    bars, cal, panel, rk = _scored()
    grid = Grid(price_grids(bars, cal))
    cfg = dict(CFG, top_k=5, hold_days=10)
    ind = {s: ("A" if int(s) % 2 else "B") for s in rk["symbol"].unique()}
    sc = rk[["date", "symbol", "score", "bench_above_ma60"]].copy()
    sc["score"] = np.random.default_rng(0).random(len(sc))  # 隨機分數：沒有緩衝時換手很高
    P1, T1 = run_backtest(sc, grid, cfg, "0050", industry=ind, buffer_rank=5, max_per_industry=5, use_regime=False)
    P2, T2 = run_backtest(sc, grid, cfg, "0050", industry=ind, buffer_rank=12, max_per_industry=5, use_regime=False)
    assert len(P1) > 5 and P1["turnover"].mean() > P2["turnover"].mean()
    assert (P1["n_held"] <= 5).all()
    m = portfolio_metrics(P1, T1, 10)
    assert m["max_drawdown"] <= 0 and "sharpe" in m


def test_industry_cap_and_bear_top5():
    bars, cal, panel, rk = _scored()
    grid = Grid(price_grids(bars, cal))
    cfg = dict(CFG, top_k=6, top_k_bear=3, hold_days=10)
    ind = {s: "A" for s in rk["symbol"].unique()}
    sc = rk[["date", "symbol", "score", "bench_above_ma60"]].copy()
    sc["score"] = np.random.default_rng(1).random(len(sc))
    P, _ = run_backtest(sc, grid, cfg, "0050", industry=ind, max_per_industry=2, use_regime=False)
    assert (P["n_held"] <= 2).all()  # 全部同產業 → 最多 2 檔
    sc["bench_above_ma60"] = 0.0
    P, _ = run_backtest(sc, grid, cfg, "0050", industry={}, use_regime=True)
    assert (P["n_held"] <= 3).all() and P["bear"].all()


def test_walkforward_guard_and_windows():
    bars, cal, panel, rk = _scored(n_days=320)
    grid = Grid(price_grids(bars, cal))
    cfg = dict(CFG, holdout_start=str(cal[200].date()), walkforward=dict(train_months=3, test_months=2, step_months=2, purge_days=10))
    sc = rk[["date", "symbol", "score", "bench_above_ma60"]]
    wins = make_windows(cal[0], cal[199], cfg)
    assert wins and all(w[3] < pd.Timestamp(cfg["holdout_start"]) for w in wins)
    with pytest.raises(HoldoutError):
        run_walkforward(sc, grid, cfg, "0050", wins)  # 資料含期末考期間 → 必須報錯
    cut = grid  # 截掉期末考後可執行
    g2 = Grid({k: v.loc[: cal[199]] for k, v in price_grids(bars, cal).items()})
    out = run_walkforward(sc[sc["date"] <= cal[199]], g2, cfg, "0050", wins)
    assert out
    run_walkforward(sc, grid, cfg, "0050", wins, final_exam=True)


def test_signal_quality_detects_informative_score():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": np.repeat(pd.date_range("2021-01-01", periods=30), 50), "symbol": list(range(50)) * 30})
    df["fwd_excess"] = rng.normal(0, 0.05, len(df))
    df["score"] = df["fwd_excess"] + rng.normal(0, 0.02, len(df))
    q = signal_quality(df)
    assert q["ic"] > 0.5 and q["top_mean_excess"] > 0.05
    df["score"] = rng.random(len(df))
    assert abs(signal_quality(df)["ic"]) < 0.15


def test_engine_return_accounting_single_stock():
    """手算：只買一檔（top_k=1），第一期報酬 = 下期開盤/進場開盤 − 1 − 買進成本（1 檔 = 全部資金）。"""
    cal = pd.bdate_range("2021-01-01", periods=40)
    def mk(sym, op):
        return pd.DataFrame({"date": cal, "market": "TW", "symbol": sym, "open": op, "high": op, "low": op, "close": op,
                             "volume": 1000, "turnover": 1e6, "adj_close": op})
    a = 100 * 1.01 ** np.arange(40)
    bars = pd.concat([mk("0050", np.full(40, 50.0)), mk("2330", a)], ignore_index=True)
    grid = Grid(price_grids(bars, cal))
    sc = pd.DataFrame({"date": cal, "symbol": "2330", "score": 1.0, "bench_above_ma60": 1.0})
    cfg = dict(CFG, top_k=1, hold_days=10, buffer_rank=30)
    P, T = run_backtest(sc, grid, cfg, "0050")
    c = CFG["costs"]["TW"]
    exp = a[11] / a[1] - 1 - buy_cost(c)  # 訊號日 0 → 第 1 日開盤買；訊號日 10 → 第 11 日開盤估值
    assert P["ret"].iloc[0] == pytest.approx(exp)
    assert P["sells"].iloc[0] == 0 and P["bench"].iloc[0] == pytest.approx(0.0)
    assert P["ret"].iloc[1] == pytest.approx(a[21] / a[11] - 1)  # 續抱不再付成本
