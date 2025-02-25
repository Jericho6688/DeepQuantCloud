import backtrader as bt
# 最简策略类，没有操作，仅能跑起来
class easy(bt.Strategy):
    params = (('data_name', ''),)

    def next(self):
        return

