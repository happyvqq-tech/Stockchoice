"""Walk-forward 視窗與期末考保護（規格第 10 節）。研究用途，非投資建議。"""
import pandas as pd

from picker.backtest.portfolio import run_backtest
from picker.guard import assert_no_holdout


def make_windows(first_date, last_date, cfg):
    """訓練 train_months、測試 test_months、每次前進 step_months。回傳 (train_start, train_end, test_start, test_end)；
    測試窗口必須完整落在 last_date 之前。RuleScore 不需要訓練，但視窗與之後的 LightGBM 完全相同，方便直接比較。"""
    w = cfg["walkforward"]
    out, ts = [], pd.Timestamp(first_date)
    while True:
        te_start = ts + pd.DateOffset(months=w["train_months"])
        te_end = te_start + pd.DateOffset(months=w["test_months"])
        if te_end > pd.Timestamp(last_date):
            break
        out.append((ts, te_start - pd.Timedelta(days=1), te_start, te_end - pd.Timedelta(days=1)))
        ts = ts + pd.DateOffset(months=w["step_months"])
    return out


def run_walkforward(scored, grid, cfg, bench_symbol, windows, final_exam=False, **kw):
    """每個測試窗口各跑一次回測（持股從空手開始）。回傳每輪的 (window, periods, trades)。"""
    assert_no_holdout(scored["date"], cfg, final_exam, "評估資料")
    assert_no_holdout([grid.cal.max()], cfg, final_exam, "價格資料")
    rounds = []
    for (a, b, c, d) in windows:
        s = scored[(scored["date"] >= c) & (scored["date"] <= d)]
        if s.empty:
            continue
        P, T = run_backtest(s, grid, cfg, bench_symbol, **kw)
        assert_no_holdout(P["end_date"] if len(P) else [], cfg, final_exam, "回測期間")
        rounds.append(((c, d), P, T))
    return rounds
