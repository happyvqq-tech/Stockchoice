"""RuleScore walk-forward 報告（Phase 2）。資料從 Supabase（或本機快取）讀，結果寫 experiments/runs/。
用法：python -m picker.run_rulescore [--final-exam]
研究用途，非投資建議。"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from picker.backtest.baselines import momentum_score, random_score
from picker.backtest.metrics import portfolio_metrics, signal_quality
from picker.backtest.portfolio import Grid, run_backtest
from picker.backtest.walkforward import make_windows, run_walkforward
from picker.config import load_config
from picker.data.adjust import add_adj_close
from picker.data.pipeline import _load_events, load_adjustments, load_all_bars, log
from picker.data.store import get_store
from picker.features.build import FEATURE_NAMES, build_panel, rank_normalize
from picker.guard import assert_no_holdout
from picker.labels import add_labels, price_grids
from picker.models.rule_score import rule_score

DISCLAIMER = "研究用途，非投資建議"


def load_inputs(store, cfg):
    bars = load_all_bars(store)
    bars = bars[(bars["close"] > 0) & (bars["open"] > 0)]
    adj = load_adjustments(store)
    ad = dict(tuple(adj.groupby("symbol")))
    bars = pd.concat([add_adj_close(b, ad.get(s)) for s, b in bars.groupby("symbol")], ignore_index=True)
    events = {k: _load_events(store, k) for k in ("institutional", "margin", "revenue")}
    uni = store.read("universe/universe.parquet")
    info = store.read("meta/universe_info.parquet")
    cal = store.read("meta/calendar.parquet")["date"]
    return bars, events, uni, info, pd.DatetimeIndex(pd.to_datetime(cal))


def dev_cut(bars, events, cal, cfg, final_exam):
    """開發期間：價格與事件資料一律截到 holdout_start 之前，讓期末考資料根本進不來。"""
    if final_exam:
        return bars, events, cal
    h = pd.Timestamp(cfg["holdout_start"])
    bars = bars[bars["date"] < h]
    events = {k: (v[v["available_date"] < h] if len(v) else v) for k, v in events.items()}
    return bars, events, cal[cal < h]


def summarize(rounds, hold_days):
    Ps = [P for _, P, _ in rounds if len(P)]
    Ts = [T for _, _, T in rounds if len(T)]
    allP = pd.concat(Ps, ignore_index=True) if Ps else pd.DataFrame()
    allT = pd.concat(Ts, ignore_index=True) if Ts else pd.DataFrame()
    m = portfolio_metrics(allP, allT, hold_days)
    per_round = [portfolio_metrics(P, T, hold_days) for _, P, T in rounds if len(P) > 1]
    m["rounds"] = len(per_round)
    ex = [(1 + P["ret"]).prod() - (1 + P["bench"]).prod() for _, P, _ in rounds if len(P) > 1]
    m["positive_rounds"] = float(np.mean([e > 0 for e in ex])) if ex else float("nan")
    m["round_ret_mean"] = float(np.mean([(1 + P["ret"]).prod() - 1 for _, P, _ in rounds if len(P)])) if Ps else float("nan")
    return m, per_round


def fmt(m, keys):
    def f(k, v):
        if isinstance(v, float) and not np.isnan(v):
            return f"{v:.1%}" if k not in ("sharpe", "ic", "payoff", "trade_payoff") else f"{v:.2f}"
        return str(v)
    return " | ".join(f(k, m.get(k, float("nan"))) for k in keys)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-exam", action="store_true", help="允許使用期末考期間（須記錄在實驗日誌）")
    ap.add_argument("--name", default="rulescore_v1")
    ap.add_argument("--random-draws", type=int, default=100)
    a = ap.parse_args(argv)
    cfg = load_config()
    store = get_store(cfg)
    log("讀取資料…")
    bars, events, uni, info, cal = load_inputs(store, cfg)
    bars, events, cal = dev_cut(bars, events, cal, cfg, a.final_exam)
    bench = cfg["benchmark"][cfg["primary_market"]]
    log(f"日線 {len(bars):,} 列；建立特徵…")
    panel = build_panel(bars, uni, bench, events)
    rk = rank_normalize(panel)
    rk["score"] = rule_score(rk)
    grids = price_grids(bars, cal)
    grid = Grid(grids)
    industry = dict(zip(info["symbol"], info["industry"]))
    first = rk["date"].min()
    wins = make_windows(first, cal.max(), cfg)
    assert_no_holdout(rk["date"], cfg, a.final_exam, "特徵面板")
    log(f"特徵面板 {len(rk):,} 列；walk-forward {len(wins)} 輪")
    sc = rk[["date", "symbol", "score", "bench_above_ma60", "ret_20"]].copy()
    first_test, last_test = wins[0][2], wins[-1][3]

    variants = {  # 一次只改一處（相對於 main）
        "main（10日、緩衝30、紅綠燈、同業≤3）": dict(),
        "改持有20日": dict(hold_days=20),
        "不用緩衝區": dict(buffer_rank=cfg["top_k"]),
        "不用大盤紅綠燈": dict(use_regime=False),
        "不限同產業": dict(max_per_industry=99),
    }
    results, detail = {}, {}
    for name, kw in variants.items():
        h = kw.get("hold_days", cfg["hold_days"])
        rounds = run_walkforward(sc, grid, cfg, bench, wins, a.final_exam, industry=industry, **kw)
        # 偏移敏感度：換股起點錯開 0..h-1 天
        offs = [summarize(run_walkforward(sc, grid, cfg, bench, wins, a.final_exam, industry=industry, offset=o, **kw), h)[0]
                for o in range(0, h, max(1, h // 4))]
        m, per_round = summarize(rounds, h)
        m["ann_excess_offsets"] = [round(o.get("ann_excess", float("nan")), 4) for o in offs]
        results[name], detail[name] = m, per_round
        log(f"{name}: 年化 {m.get('ann_return', float('nan')):.1%} 超額 {m.get('ann_excess', float('nan')):.1%}")

    # 基準：純動能、隨機（與 main 同一套交易規則，只換分數）
    base = {}
    mom = sc.assign(score=momentum_score(sc))
    base["純動能（ret_20 前10）"] = summarize(run_walkforward(mom, grid, cfg, bench, wins, a.final_exam, industry=industry), cfg["hold_days"])[0]
    rnd = []
    for sd in range(a.random_draws):
        r = sc.assign(score=random_score(sc, sd))
        rnd.append(summarize(run_walkforward(r, grid, cfg, bench, wins, a.final_exam, industry=industry), cfg["hold_days"])[0])
    keys = ["ann_return", "ann_excess", "max_drawdown", "sharpe", "turnover_annual", "trade_expectancy", "positive_rounds"]
    base[f"隨機（{a.random_draws}次平均）"] = {k: float(np.nanmean([x.get(k, np.nan) for x in rnd])) for k in keys}
    main_m = results[list(variants)[0]]
    bench_row = {"ann_return": main_m.get("bench_ann_return"), "ann_excess": 0.0}

    # 訊號品質：每天前10名（窗口重疊）
    lab = add_labels(rk, grids, bench, cfg)
    q = rk[["date", "symbol", "score"]].merge(lab[["date", "symbol", "fwd_excess", "buyable"]], on=["date", "symbol"])
    q = q[(q["date"] >= first_test) & (q["date"] <= last_test)]
    assert_no_holdout(q["date"], cfg, a.final_exam, "訊號品質")
    sq = signal_quality(q)
    sq_mom = signal_quality(q.merge(sc[["date", "symbol", "ret_20"]], on=["date", "symbol"]).assign(score=lambda d: d["ret_20"]))
    sq_all = float(q["fwd_excess"].mean())

    out = Path("experiments/runs") / f"{date.today():%Y%m%d}_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"variants": results, "baselines": base, "signal": sq, "signal_momentum": sq_mom,
                                                  "windows": [[str(x.date()) for x in w] for w in wins]},
                                                 ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    cols = ["ann_return", "ann_excess", "max_drawdown", "sharpe", "turnover_annual", "trade_win_rate", "trade_payoff",
            "trade_expectancy", "positive_rounds"]
    hdr = "| 方案 | 年化報酬 | 年化超額(對0050) | 最大回撤 | Sharpe | 年換手 | 勝率 | 盈虧比 | 每筆期望值 | 贏0050的輪次 |\n|---|---|---|---|---|---|---|---|---|---|\n"
    L = [f"# RuleScore v1 walk-forward 報告（{date.today()}）", "",
         f"**{DISCLAIMER}。** 測試期 {first_test.date()} ~ {last_test.date()}，共 {len(wins)} 輪（每輪 3 個月）。"
         f"期末考（{cfg['holdout_start']} 之後）完全沒有使用。", "",
         "> ⚠️ 免費資料沒有 2023-04 之後下市的股票，且有 79 檔查無價格 → 存活者偏差，**以下結果偏樂觀**。", "",
         "## 1. 主要結果與各項規則的影響", hdr]
    for n, m in results.items():
        L.append(f"| {n} | " + fmt(m, cols) + " |")
    L += ["", "## 2. 與三個基準比較（同一套交易規則，只換選股分數）", hdr,
          f"| RuleScore（main） | " + fmt(main_m, cols) + " |"]
    for n, m in base.items():
        L.append(f"| {n} | " + fmt(m, cols) + " |")
    L.append(f"| 0050 買進持有 | {bench_row['ann_return']:.1%} | 0.0% | - | - | - | - | - | - | - |")
    L += ["", "## 3. 訊號品質（每天前10名，窗口重疊，衡量選股能力）", "",
          "| 項目 | RuleScore | 純動能 |", "|---|---|---|",
          f"| 前10名平均超額報酬（扣成本、10日） | {sq['top_mean_excess']:.2%} | {sq_mom['top_mean_excess']:.2%} |",
          f"| 勝率 | {sq['win_rate']:.1%} | {sq_mom['win_rate']:.1%} |",
          f"| 盈虧比 | {sq['payoff']:.2f} | {sq_mom['payoff']:.2f} |",
          f"| 每筆期望值 | {sq['expectancy']:.2%} | {sq_mom['expectancy']:.2%} |",
          f"| IC（每日 Spearman 平均） | {sq['ic']:.3f} | {sq_mom['ic']:.3f} |",
          f"| IC 為正的天數比例 | {sq['ic_positive_days']:.1%} | {sq_mom['ic_positive_days']:.1%} |",
          f"| 全體股票平均超額報酬（參考） | {sq_all:.2%} | |", "",
          "## 4. 換股起點敏感度（年化超額，起點錯開）", ""]
    for n, m in results.items():
        L.append(f"- {n}: {m['ann_excess_offsets']}")
    L += ["", "## 5. 每輪（main）", "", "| 測試期 | 本輪報酬 | 0050同期 | 超額 |", "|---|---|---|---|"]
    mn = list(variants)[0]
    rounds_main = run_walkforward(sc, grid, cfg, bench, wins, a.final_exam, industry=industry)
    for (w, P, _) in rounds_main:
        if len(P):
            r, b = (1 + P["ret"]).prod() - 1, (1 + P["bench"]).prod() - 1
            L.append(f"| {w[0].date()} ~ {w[1].date()} | {r:.1%} | {b:.1%} | {r - b:+.1%} |")
    L += ["", "## 6. 特徵名稱對照", "", "、".join(f"{k}＝{v}" for k, v in FEATURE_NAMES.items()), "",
          f"_{DISCLAIMER}_"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    log(f"已寫入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
