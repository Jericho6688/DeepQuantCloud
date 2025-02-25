

import backtrader as bt
import numpy as np
import torch
import torch.nn as nn


# 定义 LSTM 模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, expected_length):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.expected_length = expected_length

    def forward(self, x):

        # 模型可接收的 数据长度 与 模型自身训练时用的 数据长度 不一样的输入，lstm模型有自己匹配长度的功能，但是会影响性能。这里加一个检验inputs长度功能
        if x.shape[1] != self.expected_length:
            raise ValueError(f"Input sequence length must be {self.expected_length}, but got {x.shape[1]}")

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class Signal_open(bt.Indicator):
    lines = ('signal_open',)
    params = (('data_length', 60),
              ('expected_length', 60),
              ('model_path', '/app/model/model_lstm.pth'),
              ('input_dim', 2),
              ('hidden_dim', 64),
              ('output_dim', 3),
              ('device', 'cpu')
              )

    def __init__(self):

        # 加载 LSTM 模型并加载权重
        self.model = LSTMModel(self.p.input_dim, self.p.hidden_dim, self.p.output_dim, self.p.expected_length)
        state_dict = torch.load(self.p.model_path, map_location=self.p.device, weights_only=True)  # 加载 state_dict
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def next(self):
        if len(self.data) >= self.p.data_length + 1:

            # ... 数据准备 ...
            data_inputs = np.array([
                [self.data.close[-j], self.data.volume[-j]]
                for j in range(0, self.p.data_length + 0)
            ])

            # 调用 LSTM 模型进行预测
            try:
                inputs = torch.tensor(data_inputs, dtype=torch.float32).unsqueeze(0) # .unsqueeze(0) 表示只输入的1组时间序列，batch_size = 1.训练的时候一般用很多组。
                with torch.no_grad():
                    output = self.model(inputs) # 获取模型输出
                    signal_indic = output.argmax(dim=1).item() # 获取预测类别

                    # 将类别转换为交易信号 (根据你的模型输出调整)
                    if signal_indic == 0:
                        signal_indic = -1
                    elif signal_indic == 1:
                        signal_indic = 0
                    elif signal_indic == 2:
                        signal_indic = 1
                    else:
                      signal_indic = 0
           

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




class ml_lstm(bt.Strategy):
    params = (
        ('stop_loss_pct', 0.01),
        ('stop_earn_pct', 0.50),
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

# 信号交易，是用next生成两个信号列，l.Signal_open 与 l.Signal_close ，
# next每调用一次，索引0的数据就会指向当前bar，其余数据索引顺势向前移动一位，就是0变成-1，-1变-2，以此类推
# 当前bar会计算出两个signal，一个开仓一个平仓，分别插入两个信号数列，并且新插入的信号引用为当前值[0]，之前的信号索引就会变成负数。
# 仓位判断函数会看，没仓位时候，读取开仓信号，有仓位时候，读取平仓信号。