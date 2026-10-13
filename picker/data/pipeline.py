"""歷史資料下載管線（可中斷續傳）。每個階段都先看 store 裡已有什麼，只補缺的。
用法：python -m picker.data.pipeline --stage all --budget-minutes 330
研究用途，非投資建議。"""
import argparse
import faulthandler
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from picker.config import load_config
from picker.data.adapters.tw import BENCHMARK, FinMindClient, RateLimited, TWAdapter
from picker.data.adjust import add_adj_close, combine_adjustments
from picker.data.quality import check_bars, render_report
from picker.data.store import get_store
from picker.data.universe import monthly_universe

KINDS = ("institutional", "margin", "revenue", "dividend", "reduction")


class Budget:
    def __init__(self, minutes):
        self.end = time.monotonic() + minutes * 60

    def left(self):
        return self.end - time.monotonic()

    def ok(self):
        return self.left() > 90


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def call(fn, budget, *args, **kw):
    """執行一次 FinMind 呼叫；碰到每小時上限就睡 5 分鐘再試，預算用完回傳 None。"""
    while budget.ok():
        try:
            return fn(*args, **kw)
        except RateLimited as e:
            log("已達 FinMind 每小時上限，等 5 分鐘：", e)
            time.sleep(min(300, max(1, budget.left() - 60)))
    return None


def stage_meta(ad, store, budget):
    if store.exists("meta/universe_info.parquet") and store.exists("meta/calendar.parquet"):
        return True
    uni = call(ad.get_universe, budget)
    cal = call(ad.get_calendar, budget)
    dl = call(ad.get_delisted, budget)
    if uni is None or cal is None or dl is None:
        return False
    store.write("meta/universe_info.parquet", uni)
    store.write("meta/calendar.parquet", cal)
    store.write("meta/delisted.parquet", dl)
    log(f"meta：候選股 {len(uni)} 檔、交易日 {len(cal)} 天、下市名單 {len(dl)} 檔")
    return True


def candidate_symbols(store, start):
    uni = store.read("meta/universe_info.parquet")["symbol"].tolist()
    dl = store.read("meta/delisted.parquet")
    dl = dl[pd.to_datetime(dl["date"]) >= pd.Timestamp(start)]["stock_id"].tolist()
    return list(dict.fromkeys([BENCHMARK] + uni + dl))


def stage_prices(ad, store, budget, cfg, today):
    start = str(cfg["data_start"])
    syms = candidate_symbols(store, start)
    done = {k.split("/")[-1][:-8] for k in store.list("bars")}
    todo = [s for s in syms if s not in done]
    log(f"prices：候選 {len(syms)} 檔，已完成 {len(done)}，待抓 {len(todo)}")
    for i, s in enumerate(todo, 1):
        if not budget.ok():
            return False
        df = call(ad.get_daily_bars, budget, s, start, today)
        if df is None:
            return False
        store.write(f"bars/{s}.parquet", df)  # 空表也存，代表「查過了，免費版沒有資料」
        if i % 100 == 0:
            log(f"prices：本輪已抓 {i}/{len(todo)}")
    return True


def load_all_bars(store, workers=8):
    """讀回全部日線。雲端儲存一檔一檔讀很慢，所以用多執行緒並行讀。"""
    keys = store.list("bars")
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(store.read, keys))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stage_universe(store, cfg):
    if store.exists("universe/universe.parquet"):  # 已算過就直接用，不必每一輪都重讀全部日線
        uni = store.read("universe/universe.parquet")
        log(f"universe：沿用既有結果（{uni['month_start'].nunique()} 個月、{uni['symbol'].nunique()} 檔）")
        return sorted(set(uni["symbol"]) | {BENCHMARK})
    u = cfg["universe"]["TW"]
    bars = load_all_bars(store)
    cal = store.read("meta/calendar.parquet")["date"]
    uni = monthly_universe(bars, cal, u["top_n_by_turnover"], u["lookback_days"], exclude=(BENCHMARK,))
    store.write("universe/universe.parquet", uni)
    log(f"universe：{uni['month_start'].nunique()} 個月、涉及 {uni['symbol'].nunique()} 檔不同股票")
    return sorted(set(uni["symbol"]) | {BENCHMARK})


def stage_splits(ad, store, budget, cfg, today):
    """全市場分割／面額變更（只要約 9 次請求）。"""
    if store.exists("events/split_all.parquet"):
        return True
    df = call(ad.get_splits, budget, str(cfg["data_start"]), today)
    if df is None:
        return False
    store.write("events/split_all.parquet", df)
    log(f"splits：分割／面額變更 {len(df)} 筆")
    return True


def stage_details(ad, store, budget, cfg, today, symbols):
    start = str(cfg["data_start"])
    have = {k for k in store.list("events")}
    todo = [(k, s) for s in symbols for k in KINDS if f"events/{k}/{s}.parquet" not in have]
    log(f"details：需要 {len(symbols)} 檔 × {len(KINDS)} 類，待抓 {len(todo)} 次請求")
    for i, (k, s) in enumerate(todo, 1):
        if not budget.ok():
            return False
        df = call(ad.get_events, budget, k, s, start, today)
        if df is None:
            return False
        store.write(f"events/{k}/{s}.parquet", df)
        if i % 100 == 0:
            log(f"details：本輪已抓 {i}/{len(todo)}")
    return True


def _load_events(store, kind, workers=8):
    keys = store.list(f"events/{kind}")
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(store.read, keys) if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_adjustments(store):
    """還原用事件：除權息 ＋ 分割／面額變更 ＋ 減資。"""
    splits = store.read("events/split_all.parquet") if store.exists("events/split_all.parquet") else None
    return combine_adjustments(_load_events(store, "dividend"), splits, _load_events(store, "reduction"))


def stage_quality(store, out="data_quality_report.md"):
    t0 = time.monotonic()
    bars = load_all_bars(store)
    log(f"quality：讀完日線 {len(bars)} 列（{time.monotonic() - t0:.0f} 秒）")
    cal = store.read("meta/calendar.parquet")["date"]
    divs = load_adjustments(store)
    log(f"quality：讀完還原事件 {len(divs)} 列（累計 {time.monotonic() - t0:.0f} 秒）")
    res = check_bars(bars, cal, divs)
    log(f"quality：檢查完成（累計 {time.monotonic() - t0:.0f} 秒）")
    extra = ""
    n_empty = len(store.list("bars")) - bars["symbol"].nunique()
    extra += f"## 查無日線資料的股票\n{n_empty} 檔（多半是下市股：免費版可能不提供，回測結果可能偏樂觀）\n"
    open(out, "w", encoding="utf-8").write(render_report(res, bars["symbol"].nunique(), len(bars), extra))
    log("quality：已輸出", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["meta", "prices", "universe", "details", "quality", "all"])
    ap.add_argument("--budget-minutes", type=float, default=330)
    ap.add_argument("--hang-guard-minutes", type=float, default=0,
                    help="大於 0 時：超過這個時間仍未結束就印出卡在哪一行並中止（除錯用）")
    a = ap.parse_args(argv)
    if a.hang_guard_minutes > 0:
        faulthandler.dump_traceback_later(a.hang_guard_minutes * 60, exit=True)
    cfg, budget, today = load_config(), Budget(a.budget_minutes), date.today().isoformat()
    store = get_store(cfg)
    ad = TWAdapter(FinMindClient(os.environ["FINMIND_TOKEN"]))
    complete = False
    if stage_meta(ad, store, budget):
        ad._calendar = store.read("meta/calendar.parquet")["date"]
        ok = True
        if a.stage in ("prices", "all"):
            ok = stage_prices(ad, store, budget, cfg, today)
        if ok and a.stage in ("details", "all"):
            ok = stage_splits(ad, store, budget, cfg, today)
        if ok and a.stage in ("universe", "details", "all"):
            syms = stage_universe(store, cfg)
            if a.stage in ("details", "all"):
                ok = stage_details(ad, store, budget, cfg, today, syms)
        if ok and a.stage in ("quality", "all"):
            stage_quality(store)
        complete = bool(ok) and a.stage in ("all", "quality")
    log("完成" if complete else "尚未完成，等待下一輪續傳")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        open(gh, "a").write(f"complete={'true' if complete else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
