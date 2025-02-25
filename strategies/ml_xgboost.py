


import backtrader as bt
import numpy as np
import joblib  # 或其他合适的模型加载库
import xgboost as xgb

# ... 加载您的模型 ...

model_path = "/app/model/model_xgboost.joblib" # 将 'your_model_path.pkl' 替换为模型的实际路径
model = joblib.load(model_path)  # 使用 joblib 或其他合适的库加载模型
'''
# 另一种读取模型的方法
model = xgb.Booster()
model.load_model("/app/model/model_xgboost.json")
'''
class Signal_open(bt.Indicator):
    lines = ('signal_open',)
    params = (('data_length', 60),)

    def next(self):
        if len(self.data) >= self.p.data_length + 1:  # 需要额外 60 天数据。len(self.data) 代表喂到next里的数据长度
                                                      # 当前的ohlsv加上前60天的cv，125个特征值，61天数据
            # 获取当天的 OHLCV
            today_ohlcv = [
                self.data.open[0],
                self.data.high[0],
                self.data.low[0],
                self.data.close[0],
                self.data.volume[0]
            ]

            # 获取前 60 天的收盘价和成交量
            past_close = [self.data.close[-j] for j in range(1, self.p.data_length + 1)]
            past_volume = [self.data.volume[-j] for j in range(1, self.p.data_length + 1)]

            # 组合数据
            data_inputs_np = np.array(today_ohlcv + past_close + past_volume).reshape(1, -1)

            data_inputs = xgb.DMatrix(data_inputs_np)
            # 调用您的深度学习模型进行预测
            try:
                signal_indic_flot = model.predict(data_inputs)[0] # 返回值是数组，虽然只有一个值，但也要用[0]取出来
                signal_indic = np.round(signal_indic_flot).astype(int) # 返回值是概率，小数，要取整形成标签
                # 将类别转换为交易信号 (根据你的模型输出调整)
                if signal_indic == 0:
                    signal_indic = -1
                elif signal_indic == 1:
                    signal_indic = 0
                elif signal_indic == 2:
                    signal_indic = 1
                else:
                    print(f"Error predicting signal ：{signal_indic}")
                    # signal_indic = 0


            except Exception as e:
                print(f"Error predicting signal: {e}")
                signal_indic = 0

            self.l.signal_open[0] = signal_indic




class Signal_close(bt.Indicator):
    lines = ('signal_close',)
    params = (('ma_long', 10), ('ma_short', 5),)

    def __init__(self):
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.p.ma_long)
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.p.ma_short)
        self.signal_indic = bt.indicators.CrossOver(self.ma_short, self.ma_long)

    def next(self):
        if self.signal_indic > 0:
            self.l.signal_close[0] = 1  # 买入信号
        elif self.signal_indic < 0:
            self.l.signal_close[0] = -1  # 卖出信号
        else:
            self.l.signal_close[0] = 0  # 无信号




class ml_xgboost(bt.Strategy):
    params = (
        ('stop_loss_pct', 0.005),
        ('stop_earn_pct', 0.05),
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
        self.stop_loss_price = None
        self.stop_earn_price = None

        self.signal_open = Signal_open(self.data).l.signal_open
        self.signal_close = Signal_close(self.data).l.signal_close


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
                self.stop_loss_price = self.buyprice * (1 - self.p.stop_loss_pct)  # 设置止损价
                self.stop_earn_price = self.buyprice * (1 + self.p.stop_earn_pct)  # 设置止盈价

                self.log(f"CASH: {self.broker.get_cash():.2f}")  # 显示现金余额
                self.log(f"VALUE: {self.broker.get_value():.2f}")  # 显示账户总价值


            else:  # 卖单
                self.log(f'SELL , {self.p.data_name},Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.stop_loss_price = None  # 重置止损价
                self.stop_earn_price = None  # 重置止盈价

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
            # if self.signal_close[0] == -1 or (self.stop_loss_price is not None and self.data.close[0] < self.stop_loss_price) or (self.stop_earn_price is not None and self.data.close[0] > self.stop_earn_price):  # 使用信号指标判断卖出
            #     self.order = self.sell(data=self.data)
            if (self.stop_loss_price is not None and self.data.close[0] < self.stop_loss_price) or (self.stop_earn_price is not None and self.data.close[0] > self.stop_earn_price):  # 使用信号指标判断卖出
                self.order = self.sell(data=self.data)