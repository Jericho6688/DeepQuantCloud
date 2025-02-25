import backtrader as bt
import numpy as np

class Signal_open(bt.Indicator):
    lines = ('Signal_open',)
    params = (('ma_long', 10), ('ma_short', 5),)

    def __init__(self):
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.ma_long)
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.ma_short)
        self.Signal_indic = bt.indicators.CrossOver(self.ma_short, self.ma_long)

    def next(self):
        if self.Signal_indic > 0:
            self.l.Signal_open[0] = 1  # 买入信号
        elif self.Signal_indic < 0:
            self.l.Signal_open[0] = -1  # 卖出信号
        else:
            self.l.Signal_open[0] = 0  # 无信号





class Signal_close(bt.Indicator):
    lines = ('Signal_close',)
    params = (('ma_long', 10), ('ma_short', 5),)

    def __init__(self):
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.ma_long)
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.ma_short)
        self.Signal_indic = bt.indicators.CrossOver(self.ma_short, self.ma_long)

    def next(self):
        if self.Signal_indic > 0:
            self.l.Signal_close[0] = 1  # 买入信号
        elif self.Signal_indic < 0:
            self.l.Signal_close[0] = -1  # 卖出信号
        else:
            self.l.Signal_close[0] = 0  # 无信号




class ma_crossover_signal_10_5(bt.Strategy):
    params = (
        ('stop_loss_pct', 0.05),
        ('data_name', ''),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.data = self.getdatabyname(self.p.data_name)
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.stop_price = None

        self.signal_open = Signal_open(self.data).l.Signal_open
        self.signal_close = Signal_close(self.data).l.Signal_close

        # 使用信号指标
        # self.ma_crossover_signal = MaCrossoverSignal(self.data, ma_long=self.p.ma_long, ma_short=self.p.ma_short)


    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY , {self.p.data_name}, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.stop_price = self.buyprice * (1 - self.p.stop_loss_pct)  # 设置止损价

                self.log(f"CASH: {self.broker.get_cash():.2f}")  # 显示现金余额
                self.log(f"VALUE: {self.broker.get_value():.2f}")  # 显示账户总价值


            else:  # 卖单
                self.log(f'SELL , {self.p.data_name},Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.stop_price = None  # 重置止损价

                self.log(f"CASH: {self.broker.get_cash():.2f}")  # 显示现金余额
                self.log(f"VALUE: {self.broker.get_value():.2f}")  # 显示账户总价值

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None


    def next(self):
        if self.order:
            return

        pos = self.getpositionbyname(self.p.data_name)

        if not pos:  # 没有持仓
            if self.signal_open[0] == 1:  # 使用信号指标判断买入
                self.order = self.buy(data=self.data)
        else:  # 持有仓位
            if self.signal_close[0] == -1 or (self.stop_price is not None and self.data.close[0] < self.stop_price):  # 使用信号指标判断卖出
                self.order = self.sell(data=self.data)