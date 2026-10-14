"""回測引擎（規格第 9 節＋ 18 節）：不重疊換股、等權重、含成本與漲跌停、緩衝區、大盤紅綠燈、同產業上限。
每 hold_days 個交易日產生一次訊號；訊號日收盤後排名 → 隔日開盤買賣。報酬用還原價。
研究用途，非投資建議。"""
import numpy as np
import pandas as pd

from picker.costs import buy_cost, sell_cost
from picker.labels import DEFER_MAX, is_limit_down, is_limit_up


class Grid:
    """價格查詢小幫手（交易日 × 股票的 numpy 陣列）。"""
    def __init__(self, grids):
        self.cal = grids["adj_close"].index
        self.pos = {d: i for i, d in enumerate(self.cal)}
        self.col = {s: j for j, s in enumerate(grids["adj_close"].columns)}
        self.ao = grids["adj_open"].values
        self.ro = grids["open"].values
        self.rc = grids["close"].values
        self.ao_ff = grids["adj_open"].ffill().values  # 下市／停牌：沿用最後價格（偏樂觀，報告要註明）

    def buy(self, sym, i):
        """訊號日位置 i → 隔日(i+1)開盤買進價；買不到（無價／開盤漲停）回傳 NaN。"""
        j = self.col.get(sym)
        if j is None or i + 1 >= len(self.cal):
            return np.nan
        o, pc = self.ro[i + 1, j], self.rc[i, j]
        if np.isnan(self.ao[i + 1, j]) or (not np.isnan(pc) and is_limit_up(o, pc)):
            return np.nan
        return self.ao[i + 1, j]

    def mark(self, sym, i):
        """訊號日位置 i 的隔日開盤估值價（持有中的股票，不管能不能買）。"""
        j = self.col.get(sym)
        if j is None or i + 1 >= len(self.cal):
            return np.nan
        return self.ao_ff[i + 1, j]

    def sell(self, sym, i):
        """訊號日位置 i → 隔日開盤賣出；開盤跌停／無價 → 順延，最多 DEFER_MAX 個交易日。"""
        j = self.col.get(sym)
        if j is None:
            return np.nan, i + 1
        k = i + 1
        for step in range(DEFER_MAX + 1):
            k = i + 1 + step
            if k >= len(self.cal):
                break
            o, pc = self.ro[k, j], self.rc[k - 1, j]
            if not np.isnan(self.ao[k, j]) and not (not np.isnan(pc) and is_limit_down(o, pc)):
                return self.ao[k, j], k
        return self.ao_ff[min(k, len(self.cal) - 1), j], min(k, len(self.cal) - 1)


def run_backtest(scored, grid, cfg, bench_symbol, hold_days=None, offset=0, industry=None, excluded=None,
                 top_k=None, buffer_rank=None, use_regime=True, max_per_industry=None):
    """scored: date, symbol, score, bench_above_ma60 的面板（每列都是股票池內、當天有資料者）。
    回傳 (periods, trades)。每期報酬 = 該期持股等權重報酬 − 買賣成本（只有真的買賣才收）。"""
    h = hold_days or cfg["hold_days"]
    K = top_k or cfg["top_k"]
    buf = buffer_rank or cfg["buffer_rank"]
    cap = max_per_industry or cfg["max_per_industry"]
    cost = cfg["costs"][cfg["primary_market"]]
    bc, sc = buy_cost(cost), sell_cost(cost)
    industry = industry or {}
    dates = np.sort(scored["date"].unique())
    by = {d: g for d, g in scored.groupby("date")}
    bj = grid.col[bench_symbol]
    held = {}  # symbol -> [本期起點價, 進場價, 進場日]
    periods, trades = [], []
    cur = None  # 正在進行的一期
    for d in dates[offset::h]:
        i = grid.pos[pd.Timestamp(d)]
        if i + 1 >= len(grid.cal):
            break
        g = by[d]
        g = g[g["score"].notna()]
        if excluded is not None:
            g = g[~g["symbol"].isin(excluded(pd.Timestamp(d)))]
        g = g.sort_values("score", ascending=False)
        rank = {sym: r for r, sym in enumerate(g["symbol"], 1)}
        regime = g["bench_above_ma60"].iloc[0] if len(g) else np.nan
        bear = bool(use_regime and regime == 0)
        N = min(cfg["top_k_bear"], K) if bear else K
        keep = sorted([x for x in held if rank.get(x, 10**9) <= buf], key=lambda x: rank[x])[:N]
        drop = [x for x in held if x not in keep]
        # 1) 結算上一期：被剔除者隔日開盤賣出，續抱者隔日開盤估值
        if cur is not None:
            rets = []
            for x, (p0, pe, de) in held.items():
                if x in drop:
                    p1, _ = grid.sell(x, i)
                else:
                    p1 = grid.mark(x, i)
                r = p1 / p0 - 1 if p0 and not np.isnan(p1) else 0.0
                rets.append(r - (sc if x in drop else 0.0))
                if x in drop:
                    pe_ret = (p1 / pe - 1) if pe and not np.isnan(p1) else 0.0
                    trades.append({"symbol": x, "entry": de, "exit": grid.cal[i + 1], "ret": pe_ret - bc - sc})
                held[x] = [p1 if x not in drop else p1, pe, de]
            w = 1.0 / cur["N"]
            cur["ret"] = w * sum(rets) - w * bc * cur["buys"]
            cur["sells"] = len(drop)
            cur["end_date"] = grid.cal[i + 1]
            b0, b1 = grid.ao_ff[cur["i0"] + 1, bj], grid.ao_ff[i + 1, bj]
            cur["bench"] = b1 / b0 - 1
            periods.append(cur)
            for x in drop:
                del held[x]
        # 2) 選出下一期持股：續抱者 ＋ 依排名補滿（同產業上限）
        cnt = {}
        for x in keep:
            cnt[industry.get(x)] = cnt.get(industry.get(x), 0) + 1
        picked = list(keep)
        for sym in g["symbol"]:
            if len(picked) >= N:
                break
            if sym in held:
                continue
            ind = industry.get(sym)
            if ind is not None and cnt.get(ind, 0) >= cap:
                continue
            cnt[ind] = cnt.get(ind, 0) + 1
            picked.append(sym)
        added = 0
        for sym in picked:
            if sym in held:
                continue
            p = grid.buy(sym, i)
            if np.isnan(p):
                continue  # 開盤漲停買不到，名額空著不遞補
            held[sym] = [p, p, grid.cal[i + 1]]
            added += 1
        for x in keep:  # 續抱者下一期起點＝本次估值價
            held[x][0] = grid.mark(x, i)
        cur = {"signal_date": pd.Timestamp(d), "start_date": grid.cal[i + 1], "i0": i, "N": N,
               "n_held": len(held), "buys": added, "bear": bear}
    # 最後一期還沒結算 → 丟掉（不假設未來）；但還在手上的部位以最後估值價結清，計入交易統計
    for x, (p0, pe, de) in held.items():
        if cur is not None and pe and p0 and not np.isnan(p0):
            trades.append({"symbol": x, "entry": de, "exit": cur["start_date"], "ret": p0 / pe - 1 - bc - sc})
    P = pd.DataFrame(periods)
    if len(P):
        P["turnover"] = (P["buys"] + P["sells"]) / 2 / K
        P["equity"] = (1 + P["ret"]).cumprod()
        P = P.drop(columns=["i0"])
    return P, pd.DataFrame(trades, columns=["symbol", "entry", "exit", "ret"])
