"""市場無關的資料來源統一介面。每個市場（TW、之後的 US、CN）實作一個 adapter。
研究用途，非投資建議。"""
from abc import ABC, abstractmethod

import pandas as pd

BAR_COLUMNS = ["date", "market", "symbol", "open", "high", "low", "close", "volume", "turnover"]


class MarketAdapter(ABC):
    market: str

    @abstractmethod
    def get_universe(self) -> pd.DataFrame:
        """可交易普通股清單：symbol, name, industry, exchange。"""

    @abstractmethod
    def get_calendar(self) -> pd.DataFrame:
        """交易日曆：date（datetime64）。"""

    @abstractmethod
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """原始（未還原）日線，欄位見 BAR_COLUMNS。無資料回傳空表（欄位仍在）。"""

    @abstractmethod
    def get_events(self, kind: str, symbol: str, start: str, end: str) -> pd.DataFrame:
        """事件類資料，每列帶 available_date（最早可用於決策的日期）。
        kind: institutional / margin / revenue / dividend。"""

    def get_splits(self, start: str, end: str) -> pd.DataFrame:
        """全市場的股票分割／面額變更事件：date, symbol, before_price, after_price。預設無資料。"""
        return pd.DataFrame(columns=["date", "symbol", "before_price", "after_price"])
