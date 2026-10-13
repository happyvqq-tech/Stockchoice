"""資料品質檢查（規格 5.5），輸出 data_quality_report.md。研究用途，非投資建議。
重點：價格為 0 的列視為「當天無資料」排除；並用還原價重算漲跌幅，驗證除權息還原是否正確。"""
import pandas as pd

from picker.data.adjust import add_adj_close

LIMIT = 0.105  # 台股單日漲跌幅 10%（含升降單位誤差）
SPLIT_LIKE = 0.30  # 還原後仍超過 30%：疑似分割、減資、面額變更等未涵蓋事件


def _top(df, n=30):
    return df.head(n).reset_index(drop=True)


def check_bars(bars, calendar, dividends=None):
    cal = pd.DatetimeIndex(pd.to_datetime(calendar))
    res = {}
    res["duplicates"] = _top(bars[bars.duplicated(["symbol", "date"], keep=False)][["symbol", "date"]], 50)
    res["n_duplicates"] = int(bars.duplicated(["symbol", "date"]).sum())
    res["not_in_calendar"] = int((~bars["date"].isin(cal)).sum())

    zero = (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
    res["n_nonpositive"] = int(zero.sum())
    res["nonpositive_by_symbol"] = (bars[zero].groupby("symbol").size().sort_values(ascending=False)
                                    .rename("零價格列數").reset_index().head(20))
    res["n_nonpositive_symbols"] = int(bars[zero]["symbol"].nunique())

    b = bars[~zero].sort_values(["symbol", "date"]).copy()  # 零價格列＝無資料，排除
    b["ret"] = b.groupby("symbol")["close"].pct_change()
    raw_big = b[b["ret"].abs() > LIMIT]
    res["n_raw_big"] = len(raw_big)

    if dividends is not None and len(dividends) and {"symbol", "date", "before_price", "after_price"} <= set(dividends.columns):
        parts = []
        for sym, g in b.groupby("symbol"):
            d = dividends[dividends["symbol"] == sym] if "symbol" in dividends else None
            parts.append(add_adj_close(g, d))
        b = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"])
        b["adj_ret"] = b.groupby("symbol")["adj_close"].pct_change()
        adj_big = b[b["adj_ret"].abs() > LIMIT]
        res["n_adj_big"] = len(adj_big)
        res["n_split_like"] = int((b["adj_ret"].abs() > SPLIT_LIKE).sum())
        res["split_like"] = _top(b[b["adj_ret"].abs() > SPLIT_LIKE][["symbol", "date", "close", "adj_ret"]]
                                 .sort_values("adj_ret", key=abs, ascending=False), 30)
        res["split_like_by_symbol"] = (b[b["adj_ret"].abs() > SPLIT_LIKE].groupby("symbol").size()
                                       .sort_values(ascending=False).rename("次數").reset_index().head(15))
        res["bench"] = b[(b["symbol"] == "0050") & (b["adj_ret"].abs() > 0.08)][["date", "close", "adj_ret"]].head(10)
    else:
        res["n_adj_big"] = None
    res["big_moves"] = _top(raw_big[["symbol", "date", "close", "ret"]].sort_values("ret", key=abs, ascending=False))

    gaps = []
    for sym, g in bars.groupby("symbol"):
        span = cal[(cal >= g["date"].min()) & (cal <= g["date"].max())]
        gaps.append((sym, len(span) - g["date"].nunique(), len(span)))
    gp = pd.DataFrame(gaps, columns=["symbol", "missing_days", "span_days"])
    gp["missing_pct"] = gp["missing_days"] / gp["span_days"].where(gp["span_days"] > 0)
    res["gaps"] = _top(gp.sort_values("missing_pct", ascending=False))
    res["median_missing_pct"] = float(gp["missing_pct"].median()) if len(gp) else 0.0
    res["coverage"] = bars.groupby(bars["date"].dt.year)["symbol"].nunique().rename("檔數").reset_index().rename(columns={"date": "年"})
    return res


def render_report(res, n_symbols, n_rows, extra=""):
    def tbl(df):
        return df.to_markdown(index=False) if df is not None and len(df) else "（無）"
    adj = res["n_adj_big"]
    lines = [
        "# 資料品質報告", "", "> 研究用途，非投資建議。", "",
        f"- 股票檔數：{n_symbols}　日線總列數：{n_rows}",
        f"- 重複列：{res['n_duplicates']}　不在交易日曆內的日期：{res['not_in_calendar']}",
        f"- **價格為 0 的列：{res['n_nonpositive']}（涉及 {res['n_nonpositive_symbols']} 檔）**——視為「當天無資料」，後續一律排除",
        f"- 個股缺漏交易日比例中位數：{res['median_missing_pct']:.2%}（含停牌）",
        f"- 單日漲跌超過 {LIMIT:.1%}（原始收盤價，已排除零價格列）：{res['n_raw_big']} 筆",
    ]
    if adj is not None:
        lines += [
            f"- 同樣條件，**改用還原價**：{adj} 筆（還原有效的話應大幅下降，剩下的多為新上市前幾天、處置/恢復交易）",
            f"- 還原後仍超過 {SPLIT_LIKE:.0%} 的跳動：**{res['n_split_like']} 筆**（疑似股票分割、減資、面額變更，除權息資料沒涵蓋）",
        ]
    lines += ["", "## 各年度有資料的檔數", tbl(res["coverage"]), "",
              "## 零價格列最多的個股（前 20）", tbl(res["nonpositive_by_symbol"]), ""]
    if adj is not None:
        lines += ["## 還原後仍有巨大跳動的個股（前 15）", tbl(res["split_like_by_symbol"]), "",
                  "## 還原後最大跳動的紀錄（前 30）", tbl(res["split_like"]), "",
                  "## 0050 還原後單日超過 8% 的日子", tbl(res["bench"]), ""]
    lines += ["## 缺漏最多的個股（前 30）", tbl(res["gaps"]), "",
              "## 原始價格單日漲跌最大的紀錄（前 30）", tbl(res["big_moves"]), "",
              "## 重複列（前 50）", tbl(res["duplicates"]), "", extra]
    return "\n".join(lines)
