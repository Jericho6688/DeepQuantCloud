import backtrader as bt

class ma_crossover_60_20(bt.Strategy):
    params = (
        ('ma_long', 60),
        ('ma_short',20),
        ('stop_loss_pct', 0.05),
        ('data_name','')
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.data = self.getdatabyname(self.p.data_name)  # 获取指定数据
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.stop_price = None

        # 定义指标 (使用 self.data)
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.ma_long)
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.ma_short)
        self.crossover_long = bt.indicators.CrossOver(self.data.close, self.ma_long)
        self.crossover_short = bt.indicators.CrossOver(self.data.close, self.ma_short)

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

        #  使用 getpositionbyname 获取指定数据的持仓信息
        pos = self.getpositionbyname(self.p.data_name)

        if not pos:  # 没有持有指定股票的仓位
            if self.crossover_long > 0:  # 价格上穿 ma_long
                #self.log(f'BUY CREATE, {self.data.close[0]:.2f}')
                self.order = self.buy(data=self.data) # 明确指定 data


        else:  # 持有指定股票的仓位
            if (self.crossover_short < 0) or (self.stop_price is not None and self.data.close[0] < self.stop_price):
                #self.log(f'SELL CREATE, {self.data.close[0]:.2f}')
                self.order = self.sell(data=self.data) # 明确指定 data