"""Phase 3：LightGBM walk-forward，並依規格 11.2 與 RuleScore 比較。
用法：python -m picker.run_lgbm [--final-exam]    研究用途，非投資建議。"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from picker.backtest.baselines import momentum_score, random_score
from picker.backtest.metrics import signal_quality
from picker.backtest.portfolio import Grid
from picker.backtest.walkforward import make_windows, run_walkforward
from picker.config import load_config
from picker.data.pipeline import log
from picker.data.store import get_store
from picker.features.build import FEATURE_NAMES, FEATURES, build_panel, rank_normalize
from picker.guard import assert_no_holdout
from picker.labels import add_labels, add_train_label, price_grids
from picker.models.lgbm import fit_window, predict, reasons
from picker.models.rule_score import rule_score
from picker.run_rulescore import DISCLAIMER, dev_cut, fmt, load_inputs, summarize

COLS = ["ann_return", "ann_excess", "max_drawdown", "sharpe", "turnover_annual", "trade_win_rate", "trade_payoff",
        "trade_expectancy", "positive_rounds"]
HDR = ("| 方案 | 年化報酬 | 年化超額(對0050) | 最大回撤 | Sharpe | 年換手 | 勝率 | 盈虧比 | 每筆期望值 | 贏0050的輪次 |\n"
       "|---|---|---|---|---|---|---|---|---|---|\n")


def round_excess(rounds):
    return {w: (1 + P["ret"]).prod() - (1 + P["bench"]).prod() for w, P, _ in rounds if len(P) > 1}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-exam", action="store_true")
    ap.add_argument("--name", default="lgbm_v1")
    ap.add_argument("--random-draws", type=int, default=30)
    a = ap.parse_args(argv)
    cfg = load_config()
    store = get_store(cfg)
    log("讀取資料…")
    bars, events, uni, info, cal = load_inputs(store, cfg)
    bars, events, cal = dev_cut(bars, events, cal, cfg, a.final_exam)
    bench = cfg["benchmark"][cfg["primary_market"]]
    panel = build_panel(bars, uni, bench, events)
    rk = rank_normalize(panel)
    rk["rule"] = rule_score(rk)
    grids = price_grids(bars, cal)
    grid = Grid(grids)
    industry = dict(zip(info["symbol"], info["industry"]))
    assert_no_holdout(rk["date"], cfg, a.final_exam, "特徵面板")
    lab = add_labels(rk, grids, bench, cfg)
    df = rk.merge(lab[["date", "symbol", "exit_date", "fwd_excess"]], on=["date", "symbol"])
    df = add_train_label(df, cfg["label_top_pct"])
    wins = make_windows(df["date"].min(), cal.max(), cfg)
    log(f"面板 {len(df):,} 列；walk-forward {len(wins)} 輪；開始訓練…")

    parts, imps, per_round_info, last_model, last_test = [], [], [], None, None
    for (a0, b0, c0, d0) in wins:
        dtrain = df[df["date"] >= a0]
        m, inf = fit_window(dtrain, cfg, c0)
        if m is None:
            log(f"{c0.date()}: 資料不足，略過")
            continue
        test = df[(df["date"] >= c0) & (df["date"] <= d0)]
        assert_no_holdout(test["date"], cfg, a.final_exam, "測試期")
        parts.append(pd.DataFrame({"date": test["date"], "symbol": test["symbol"], "score": predict(m, test),
                                   "bench_above_ma60": test["bench_above_ma60"]}))
        imps.append(inf["importance"].rename(c0))
        per_round_info.append((c0, d0, inf["fit_rows"], inf["best_iter"], inf["importance"].nlargest(3).index.tolist()))
        last_model, last_test = m, test
        log(f"{c0.date()}~{d0.date()} 訓練 {inf['fit_rows']:,} 列，最佳樹數 {inf['best_iter']}")
    sc_l = pd.concat(parts, ignore_index=True)
    sc_r = df[["date", "symbol", "bench_above_ma60"]].assign(score=df["rule"])
    sc_m = df[["date", "symbol", "bench_above_ma60"]].assign(score=momentum_score(df))
    kw = dict(industry=industry)
    rl = run_walkforward(sc_l, grid, cfg, bench, wins, a.final_exam, **kw)
    rr = run_walkforward(sc_r, grid, cfg, bench, wins, a.final_exam, **kw)
    ml, _ = summarize(rl, cfg["hold_days"])
    mr, _ = summarize(rr, cfg["hold_days"])
    mm, _ = summarize(run_walkforward(sc_m, grid, cfg, bench, wins, a.final_exam, **kw), cfg["hold_days"])
    rnd = []
    for sd in range(a.random_draws):
        r = df[["date", "symbol", "bench_above_ma60"]].assign(score=random_score(df, sd))
        rnd.append(summarize(run_walkforward(r, grid, cfg, bench, wins, a.final_exam, **kw), cfg["hold_days"])[0])
    mrand = {k: float(np.nanmean([x.get(k, np.nan) for x in rnd])) for k in COLS}

    # 規格 11.2 採用規則
    el, er = round_excess(rl), round_excess(rr)
    common = [w for w in el if w in er]
    better = float(np.mean([el[w] > er[w] for w in common])) if common else float("nan")
    c1 = ml.get("trade_expectancy", -9) > mr.get("trade_expectancy", 9)
    c2 = better >= 0.6
    c3 = ml.get("ann_excess", -9) > mrand["ann_excess"] and ml.get("ann_excess", -9) > mm.get("ann_excess", 9)
    adopt = bool(c1 and c2 and c3)

    # 訊號品質
    keep = df[["date", "symbol", "fwd_excess"]]
    qs = {}
    for name, s in (("LightGBM", sc_l), ("RuleScore", sc_r.merge(sc_l[["date", "symbol"]], on=["date", "symbol"]))):
        q = s[["date", "symbol", "score"]].merge(keep, on=["date", "symbol"])
        assert_no_holdout(q["date"], cfg, a.final_exam, "訊號品質")
        qs[name] = signal_quality(q)
    imp = pd.concat(imps, axis=1)
    imp_mean = imp.mean(axis=1).sort_values(ascending=False)

    out = Path("experiments/runs") / f"{date.today():%Y%m%d}_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"lgbm": ml, "rulescore": mr, "momentum": mm, "random": mrand, "signal": qs,
                                                  "adopt": {"expectancy_higher": c1, "rounds_better": better, "beats_baselines": c3,
                                                            "adopt": adopt}}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    last_day = last_test["date"].max()
    picks = last_test[last_test["date"] == last_day].assign(score=lambda x: predict(last_model, x))
    picks = picks.sort_values("score", ascending=False).head(10)
    picks = picks.assign(reason=reasons(last_model, picks))
    L = [f"# LightGBM v1 vs RuleScore walk-forward 報告（{date.today()}）", "",
         f"**{DISCLAIMER}。** 測試期 {wins[0][2].date()} ~ {wins[-1][3].date()}，{len(per_round_info)} 輪訓練；期末考完全沒有使用。",
         "> ⚠️ 存活者偏差（免費資料缺 2023-04 後下市股）→ 結果偏樂觀；處置／注意／全額交割股尚未排除。", "",
         "## 採用判定（規格 11.2，三項都要成立）", "",
         f"1. 每筆期望值高於 RuleScore：{'✅' if c1 else '❌'}（LightGBM {ml.get('trade_expectancy', float('nan')):.2%} vs RuleScore {mr.get('trade_expectancy', float('nan')):.2%}）",
         f"2. ≥60% 輪次比 RuleScore 好：{'✅' if c2 else '❌'}（{better:.0%}，共 {len(common)} 輪）",
         f"3. 同時勝過隨機與純動能：{'✅' if c3 else '❌'}",
         f"**結論：{'符合採用規則' if adopt else '不符合採用規則，維持 RuleScore 為現行版本'}**", "",
         "## 1. 績效比較（同一套交易規則，只換選股分數）", "", HDR.rstrip("\n"),
         f"| LightGBM | {fmt(ml, COLS)} |", f"| RuleScore | {fmt(mr, COLS)} |", f"| 純動能 | {fmt(mm, COLS)} |",
         f"| 隨機（{a.random_draws}次平均） | {fmt(mrand, COLS)} |", f"| 0050 買進持有 | {ml.get('bench_ann_return', float('nan')):.1%} | 0.0% | - | - | - | - | - | - | - |",
         "", "## 2. 訊號品質（每天前10名）", "", "| 項目 | LightGBM | RuleScore |", "|---|---|---|"]
    for k, nm, f in (("top_mean_excess", "前10名平均超額（扣成本，10日）", "{:.2%}"), ("win_rate", "勝率", "{:.1%}"), ("payoff", "盈虧比", "{:.2f}"),
                     ("expectancy", "每筆期望值", "{:.2%}"), ("ic", "IC", "{:.3f}"), ("ic_positive_days", "IC為正的天數比例", "{:.1%}")):
        L.append(f"| {nm} | {f.format(qs['LightGBM'][k])} | {f.format(qs['RuleScore'][k])} |")
    L += ["", "## 3. 特徵重要性（各輪平均，gain）", "", "| 特徵 | 重要性 |", "|---|---|"]
    L += [f"| {FEATURE_NAMES[k]} | {v:.1%} |" for k, v in imp_mean.head(12).items()]
    L += ["", "## 4. 每輪（訓練列數、最佳樹數、前三大特徵、本輪超額）", "", "| 測試期 | 訓練列數 | 樹數 | 前三大特徵 | LightGBM 超額 | RuleScore 超額 |", "|---|---|---|---|---|---|"]
    for c0, d0, n, bi, top3 in per_round_info:
        w = (c0, d0)
        L.append(f"| {c0.date()} ~ {d0.date()} | {n:,} | {bi} | {'、'.join(FEATURE_NAMES[x] for x in top3)} | "
                 f"{el.get(w, float('nan')):+.1%} | {er.get(w, float('nan')):+.1%} |")
    L += ["", f"## 5. 選股理由示範（最後一個測試日 {last_day.date()} 的前 10 名，pred_contrib）", "", "| 代號 | 分數 | 理由 |", "|---|---|---|"]
    L += [f"| {r.symbol} | {r.score:.3f} | {r.reason} |" for r in picks.itertuples()]
    L += ["", "> 分數是模型排序用的數字，**不是上漲機率**。", "", f"_{DISCLAIMER}_"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    log(f"已寫入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
