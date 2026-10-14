"""交易成本（規格 9.2）。數字都從 config 讀。研究用途，非投資建議。"""


def buy_cost(c):
    """買進一次的成本比率（手續費 ＋ 滑價）。c = config["costs"]["TW"]"""
    return c["commission"] * c.get("commission_discount", 1.0) + c["slippage"]


def sell_cost(c):
    """賣出一次的成本比率（手續費 ＋ 證交稅 ＋ 滑價）。"""
    return c["commission"] * c.get("commission_discount", 1.0) + c["sell_tax"] + c["slippage"]


def round_trip(c):
    return buy_cost(c) + sell_cost(c)
