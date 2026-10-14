"""指標（規格 11.1）。研究用途，非投資建議。"""
import numpy as np
import pandas as pd


def signal_quality(df, top_n=10):
    """df: date, symbol, score, fwd_excess（訊號品質：每天前 top_n 名的平均超額報酬、IC）。窗口會重疊，只用來衡量選股能力。"""
    d = df.dropna(subset=["score", "fwd_excess"])
    if d.empty:
        return {"days": 0}
    top = d.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(top_n)
    daily = top.groupby("date")["fwd_excess"].mean()
    ic = d.groupby("date").apply(lambda g: g["score"].rank().corr(g["fwd_excess"].rank()) if len(g) > 20 else np.nan,
                                 include_groups=False)
    x = top["fwd_excess"]
    win, lose = x[x > 0], x[x <= 0]
    wr = len(win) / len(x)
    aw, al = (win.mean() if len(win) else 0.0), (-lose.mean() if len(lose) else 0.0)
    return {"days": int(len(daily)), "top_mean_excess": float(daily.mean()), "win_rate": float(wr),
            "payoff": float(aw / al) if al else float("nan"), "expectancy": float(wr * aw - (1 - wr) * al),
            "ic": float(ic.mean()), "ic_positive_days": float((ic.dropna() > 0).mean())}


def portfolio_metrics(P, trades, hold_days):
    if P is None or len(P) < 2:
        return {"periods": 0}
    ppy = 252 / hold_days
    n = len(P)
    eq = (1 + P["ret"]).cumprod()
    ann = eq.iloc[-1] ** (ppy / n) - 1
    bench = (1 + P["bench"]).cumprod()
    bann = bench.iloc[-1] ** (ppy / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sd = P["ret"].std()
    out = {"periods": int(n), "ann_return": float(ann), "bench_ann_return": float(bann), "ann_excess": float(ann - bann),
           "max_drawdown": float(mdd), "sharpe": float(P["ret"].mean() / sd * np.sqrt(ppy)) if sd else float("nan"),
           "turnover_per_period": float(P["turnover"].mean()), "turnover_annual": float(P["turnover"].mean() * ppy),
           "bear_share": float(P["bear"].mean()), "avg_held": float(P["n_held"].mean()),
           "period_excess_mean": float((P["ret"] - P["bench"]).mean()),
           "positive_period_share": float((P["ret"] > 0).mean())}
    if trades is not None and len(trades):
        x = trades["ret"]
        win, lose = x[x > 0], x[x <= 0]
        wr = len(win) / len(x)
        aw, al = (win.mean() if len(win) else 0.0), (-lose.mean() if len(lose) else 0.0)
        out.update({"trades": int(len(x)), "trade_win_rate": float(wr),
                    "trade_payoff": float(aw / al) if al else float("nan"), "trade_expectancy": float(wr * aw - (1 - wr) * al)})
    return out
