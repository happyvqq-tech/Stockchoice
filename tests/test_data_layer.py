"""資料層測試：還原價、可取得日、股票池防未來資訊（截斷測試）、儲存層、FinMind 客戶端。"""
import json

import pandas as pd
import pytest

from picker.data.adapters.tw import FinMindClient, RateLimited, TWAdapter, next_trading_day, revenue_available_date
from picker.data.adjust import add_adj_close
from picker.data.store import LocalStore
from picker.data.universe import monthly_universe

CAL = pd.bdate_range("2018-01-01", "2018-12-31")


def test_adj_close_only_adjusts_before_ex_date():
    bars = pd.DataFrame({"date": pd.to_datetime(["2018-06-22", "2018-06-25", "2018-06-26"]), "close": [227.5, 219.5, 220.0]})
    div = pd.DataFrame({"date": pd.to_datetime(["2018-06-25"]), "before_price": [227.5], "after_price": [219.5]})
    out = add_adj_close(bars, div)
    assert out["adj_close"].iloc[0] == pytest.approx(219.5)  # 除權息前一天被還原成 219.5
    assert out["adj_close"].iloc[1] == 219.5 and out["adj_close"].iloc[2] == 220.0
    # 還原後，除權息日的報酬 = 0%（價格連續），不是 -3.5%
    assert out["adj_close"].iloc[1] / out["adj_close"].iloc[0] - 1 == pytest.approx(0.0)


def test_adj_close_without_dividends_is_close():
    bars = pd.DataFrame({"date": pd.to_datetime(["2018-01-02"]), "close": [10.0]})
    assert add_adj_close(bars, None)["adj_close"].iloc[0] == 10.0


def test_next_trading_day_skips_weekend():
    out = next_trading_day(pd.to_datetime(["2018-01-05"]), CAL)  # 週五 → 下週一
    assert out[0] == pd.Timestamp("2018-01-08")


def test_revenue_available_date_is_11th_of_next_month():
    assert revenue_available_date(2017, 12) == pd.Timestamp("2018-01-11")
    assert revenue_available_date(2018, 3) == pd.Timestamp("2018-04-11")


def test_institutional_available_next_day_and_aggregates():
    raw = pd.DataFrame([
        {"date": "2018-01-02", "stock_id": "2330", "buy": 100, "sell": 40, "name": "Foreign_Investor"},
        {"date": "2018-01-02", "stock_id": "2330", "buy": 10, "sell": 0, "name": "Foreign_Dealer_Self"},
        {"date": "2018-01-02", "stock_id": "2330", "buy": 5, "sell": 20, "name": "Investment_Trust"},
        {"date": "2018-01-02", "stock_id": "2330", "buy": 1, "sell": 2, "name": "Dealer_self"},
    ])
    out = TWAdapter(None, calendar=CAL)._institutional(raw)
    r = out.iloc[0]
    assert (r["foreign_net"], r["trust_net"], r["dealer_net"]) == (70, -15, -1)
    assert r["available_date"] == pd.Timestamp("2018-01-03")


def _synthetic_bars(days, n=8):
    rows = []
    for s in range(n):
        for d in days:
            rows.append({"date": d, "symbol": f"S{s}", "turnover": 1000.0 * (s + 1) + (d - days[0]).days})
    return pd.DataFrame(rows)


def test_universe_ranks_by_turnover_and_uses_only_prior_days():
    days = pd.bdate_range("2018-01-01", "2018-08-31")
    u = monthly_universe(_synthetic_bars(days), days, top_n=3, lookback=20, min_days=10)
    first = u[u["month_start"] == u["month_start"].min()]
    assert list(first.sort_values("rank")["symbol"]) == ["S7", "S6", "S5"]


def test_universe_truncation_no_lookahead():
    """截斷測試：把某個月第一個交易日（含）之後的資料全部砍掉，該月的股票池必須完全相同。"""
    days = pd.bdate_range("2018-01-01", "2018-08-31")
    bars = _synthetic_bars(days)
    # 砍掉最後一個月第一個交易日之後的資料，之前各月的股票池不得改變
    full = monthly_universe(bars, days, top_n=3, lookback=20, min_days=10)
    d = full["month_start"].max()
    cut = bars[bars["date"] < d]
    part = monthly_universe(cut, days[days < d], top_n=3, lookback=20, min_days=10)
    a = full[full["month_start"] < d].reset_index(drop=True)
    b = part[part["month_start"] < d].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # 且當天的排名只看前 20 個交易日：d 當天的資料被竄改不影響 d 的名單
    tampered = bars.copy()
    tampered.loc[(tampered["date"] >= d) & (tampered["symbol"] == "S0"), "turnover"] = 1e12
    t = monthly_universe(tampered, days, top_n=3, lookback=20, min_days=10)
    pd.testing.assert_frame_equal(full[full["month_start"] == d].reset_index(drop=True),
                                  t[t["month_start"] == d].reset_index(drop=True))


def test_local_store_roundtrip(tmp_path):
    s = LocalStore(tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    s.write("bars/2330.parquet", df)
    assert s.exists("bars/2330.parquet") and not s.exists("bars/9999.parquet")
    assert s.list("bars") == ["bars/2330.parquet"]
    pd.testing.assert_frame_equal(s.read("bars/2330.parquet"), df)


def _client(responses):
    it = iter(responses)
    return FinMindClient("t", min_interval=0, fetch=lambda url, h: next(it), sleep=lambda s: None)


def test_finmind_client_parses_and_detects_rate_limit():
    ok = (200, json.dumps({"status": 200, "msg": "success", "data": [{"a": 1}]}).encode())
    assert len(_client([ok]).query("X")) == 1
    limit = (402, json.dumps({"status": 402, "msg": "Requests reach the upper limit"}).encode())
    with pytest.raises(RateLimited):
        _client([limit]).query("X")
    paid = (400, json.dumps({"status": 400, "msg": "Your level is register"}).encode())
    with pytest.raises(RuntimeError):
        _client([paid]).query("X")


def test_universe_filter_keeps_only_common_stocks():
    class C:
        def query(self, dataset, **kw):
            return pd.DataFrame([
                {"industry_category": "半導體業", "stock_id": "2330", "stock_name": "台積電", "type": "twse", "date": "2020-01-01"},
                {"industry_category": "ETF", "stock_id": "0056", "stock_name": "高股息", "type": "twse", "date": "2020-01-01"},
                {"industry_category": "電子工業", "stock_id": "2882A", "stock_name": "特別股", "type": "twse", "date": "2020-01-01"},
                {"industry_category": "生技醫療業", "stock_id": "1234", "stock_name": "興櫃", "type": "emerging", "date": "2020-01-01"},
                {"industry_category": "存託憑證", "stock_id": "9103", "stock_name": "美德醫療-DR", "type": "twse", "date": "2020-01-01"},
            ])
    assert list(TWAdapter(C()).get_universe()["symbol"]) == ["2330"]


def test_supabase_store_list_recurses_into_folders():
    """回歸測試：events/ 底下是子資料夾，列表必須往下找，否則續傳會以為什麼都沒抓過。"""
    from picker.data.store import SupabaseStore
    tree = {
        "events": [{"name": "institutional", "id": None}, {"name": "margin", "id": None}],
        "events/institutional": [{"name": "2330.parquet", "id": "a"}, {"name": "2317.parquet", "id": "b"}],
        "events/margin": [{"name": "2330.parquet", "id": "c"}],
    }
    st = SupabaseStore("https://x.supabase.co/rest/v1", "k")
    st._req = lambda m, path, data=None, headers=None, timeout=0: (200, json.dumps(tree[json.loads(data)["prefix"]]).encode())
    assert st.list("events") == ["events/institutional/2317.parquet", "events/institutional/2330.parquet", "events/margin/2330.parquet"]
