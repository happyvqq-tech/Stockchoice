"""用假 adapter 跑完整管線：各階段、續傳（已完成的不重抓）、品質報告。"""
import pandas as pd

from picker.config import load_config
from picker.data import pipeline as pl
from picker.data.adapters.base import BAR_COLUMNS, MarketAdapter
from picker.data.store import LocalStore

DAYS = pd.bdate_range("2018-01-01", "2018-09-28")


class FakeAdapter(MarketAdapter):
    market = "TW"

    def __init__(self):
        self.calls = []
        self._calendar = DAYS

    def get_universe(self):
        return pd.DataFrame({"symbol": ["1101", "2330"], "name": ["a", "b"], "industry": ["x", "y"], "exchange": ["twse"] * 2})

    def get_delisted(self):
        return pd.DataFrame({"date": ["2019-01-01"], "stock_id": ["1409"], "stock_name": ["gone"]})

    def get_calendar(self):
        return pd.DataFrame({"date": DAYS})

    def get_daily_bars(self, symbol, start, end):
        self.calls.append(("bars", symbol))
        if symbol == "1409":
            return pd.DataFrame(columns=BAR_COLUMNS)
        base = {"1101": 1, "2330": 5, "0050": 3}[symbol]
        return pd.DataFrame({"date": DAYS, "market": "TW", "symbol": symbol, "open": 10.0, "high": 11.0, "low": 9.0,
                             "close": 10.0, "volume": 1000, "turnover": 1e6 * base})

    def get_events(self, kind, symbol, start, end):
        self.calls.append((kind, symbol))
        return pd.DataFrame({"symbol": [symbol]})


def test_full_pipeline_and_resume(tmp_path):
    cfg, store, ad = load_config(), LocalStore(tmp_path), FakeAdapter()
    cfg["data_start"] = "2018-01-01"
    b = pl.Budget(60)
    assert pl.stage_meta(ad, store, b)
    assert pl.stage_prices(ad, store, b, cfg, "2018-09-28")
    assert {c[1] for c in ad.calls} == {"0050", "1101", "2330", "1409"}
    n = len(ad.calls)
    assert pl.stage_prices(ad, store, b, cfg, "2018-09-28") and len(ad.calls) == n  # 續傳：不重抓
    syms = pl.stage_universe(store, cfg)
    assert set(syms) == {"1101", "2330", "0050"}  # 空資料的 1409 不進股票池；基準必備
    assert pl.stage_universe(store, cfg) == syms  # 第二次直接沿用既有結果
    assert pl.stage_details(ad, store, b, cfg, "2018-09-28", syms)
    assert len(store.list("events/institutional")) == 3
    out = tmp_path / "q.md"
    pl.stage_quality(store, str(out))
    txt = out.read_text(encoding="utf-8")
    assert "資料品質報告" in txt and "1 檔" in txt and "價格為 0 的列：0" in txt


def test_budget_exhausted_stops_without_losing_progress(tmp_path):
    cfg, store, ad = load_config(), LocalStore(tmp_path), FakeAdapter()
    cfg["data_start"] = "2018-01-01"
    pl.stage_meta(ad, store, pl.Budget(60))
    assert pl.stage_prices(ad, store, pl.Budget(0), cfg, "2018-09-28") is False  # 預算 0：什麼都不抓、回報未完成
    assert store.list("bars") == []
