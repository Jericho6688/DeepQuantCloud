import backtrader as bt
import pandas as pd
import psycopg2
import importlib
import os
import matplotlib.pyplot as plt
import yaml
from math import sqrt

CONFIG_FILE = "backtester1.yaml"
columns_to_fetch = ['date', 'open', 'high', 'low', 'close', 'volume']


# Configuration Loading Function
def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


# Data Fetching Function
def get_data(table_name, start_date, end_date, columns):
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            database=os.environ.get("NEW_DB_NAME"),
            user=os.environ.get("NEW_DB_USER"),
            password=os.environ.get("NEW_DB_PASSWORD"),
        )
        cur = conn.cursor()

        query = f"""
            SELECT {', '.join(columns)}
            FROM {table_name}
            WHERE date >= %s AND date <= %s
            ORDER BY date;
        """
        cur.execute(query, (start_date, end_date))
        data = cur.fetchall()
        cur.close()
        conn.close()

        df = pd.DataFrame(data, columns=columns)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.columns = [col.lower() for col in df.columns]
        return df

    except Exception as e:
        print(f"Error getting data for {table_name}: {e}")
        return None


# Strategy Loading Function
def load_strategies(config):
    strategies = {}
    for strategy_config in config.get("strategies", []):
        strategy_name = strategy_config.get("name")
        if strategy_name:
            try:
                module_name = strategy_name
                class_name = strategy_name
                strategy_module = importlib.import_module(f"strategies.{module_name}")
                strategies[strategy_name] = getattr(strategy_module, class_name)
            except (ModuleNotFoundError, AttributeError, ValueError) as e:
                print(f"Error importing strategy {strategy_name}: {e}")
    return strategies


if __name__ == "__main__":
    # Load Configuration
    config = load_config(CONFIG_FILE)
    if config is None:
        exit()

    # Extract Backtest and Strategy Configurations
    backtest_config = config.get("backtest")
    strategies_config = config.get("strategies")
    if not backtest_config or not strategies_config:
        print("Error: Invalid config file format.")
        exit()

    # Retrieve Backtest Parameters
    total_capital = backtest_config.get("cash", 100000.0)
    commission = backtest_config.get("commission", 0.001)
    start_date = backtest_config["start_date"]
    end_date = backtest_config["end_date"]
    start_dt = pd.to_datetime(start_date)

    # Load Strategy Classes
    strategy_classes = load_strategies(config)
    strategy_results = {}
    strategy_equity_curves = {}

    # Iterate Through Strategies
    strategy_index = 1
    for strat_conf in strategies_config:
        strategy_name = strat_conf.get("name")
        symbol = strat_conf.get("symbol")
        percents = strat_conf.get("percents", 20)
        allocation = total_capital * percents / 100.0

        # Fetch Data for the Current Symbol
        data_df = get_data(symbol, start_date, end_date, columns_to_fetch)
        if data_df is None:
            print(f"Data for symbol {symbol} is None, skipping.")
            continue

        # Create Cerebro Instance and Set Broker Parameters
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(allocation)
        cerebro.broker.setcommission(commission=commission)

        # Add Data Feed to Cerebro
        data = bt.feeds.PandasData(dataname=data_df, name=symbol)
        cerebro.adddata(data, name=symbol)

        # Add Strategy to Cerebro
        strategy_class = strategy_classes.get(strategy_name)
        if strategy_class is None:
            print(f"Strategy {strategy_name} not found, skipping {symbol}.")
            continue
        cerebro.addstrategy(strategy_class, data_name=symbol)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=percents)

        # Add Analyzers to Cerebro
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', annualize=True, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)

        # Run Backtest
        print(f"\nRunning strategy {strategy_name} for {symbol} with allocation {allocation:.2f} ...")
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        print(f"Final Portfolio Value for {symbol} : {final_value:.2f}")

        # Extract Analyzer Results
        strat = results[0]
        analyzers = strat.analyzers
        sharpe = analyzers.sharpe_ratio.get_analysis()
        drawdown = analyzers.drawdown.get_analysis()
        time_return = analyzers.time_return.get_analysis()

        # Generate Equity Curve
        tr_series = pd.Series(time_return).sort_index()
        equity_curve = allocation * ((tr_series + 1).cumprod())
        if equity_curve.index.min() > start_dt:
            equity_curve = pd.concat([pd.Series({start_dt: allocation}), equity_curve])
            equity_curve = equity_curve.sort_index()
        strategy_equity_curves[symbol] = equity_curve

        # Store Strategy Results
        strategy_results[symbol] = {
            "strategy": strategy_name,
            "allocation": allocation,
            "final_value": final_value,
            "profit": final_value - allocation,
            "sharpe": sharpe,
            "drawdown": drawdown,
            "time_return": time_return,
            "trade_analyzer": analyzers.trade_analyzer.get_analysis(),
        }

        # Plot and Save Strategy Chart
        figs = cerebro.plot(style='candlestick', iplot=False)
        if figs:
            for i, fig in enumerate(figs[0]):
                chart_filename = f"strategy_chart_{strategy_index}.png"
                fig.savefig(chart_filename)
                print(f"Strategy chart saved as {chart_filename}")
                plt.close(fig)
        strategy_index += 1

    # Calculate Overall Results
    total_allocated = sum(item["allocation"] for item in strategy_results.values())
    total_final = sum(item["final_value"] for item in strategy_results.values())
    unallocated_capital = total_capital - total_allocated
    overall_final = total_final + unallocated_capital

    # Print Comprehensive Backtest Report
    print("\n=== Comprehensive Backtest Report ===")
    print(f"Total Initial Capital: {total_capital:.2f}")
    print(f"Allocated Capital: {total_allocated:.2f}")
    print(f"Final Capital from Strategies: {total_final:.2f}")
    print(f"Unallocated Capital: {unallocated_capital:.2f}")
    print(f"Overall Final Capital: {overall_final:.2f}")
    print(f"Total Profit/Loss: {overall_final - total_capital:.2f}")

    # Print Individual Strategy Reports
    for symbol, res in strategy_results.items():
        print(f"\n--- Report for {symbol} ({res['strategy']}) ---")
        print(f"Allocation: {res['allocation']:.2f}")
        print(f"Final Portfolio Value: {res['final_value']:.2f}")
        print(f"Profit/Loss: {res['profit']:.2f}")
        if res["sharpe"] and 'sharperatio' in res["sharpe"] and res["sharpe"]['sharperatio'] is not None:
            print(f"Sharpe Ratio: {res['sharpe']['sharperatio']:.4f}")
        else:
            print("Sharpe Ratio: Unable to calculate")
        if res["drawdown"] and 'max' in res["drawdown"] and 'drawdown' in res["drawdown"]["max"]:
            print(f"Maximum Drawdown: {res['drawdown']['max']['drawdown']:.2f}%")
        else:
            print("Maximum Drawdown: Unable to calculate")
        if res["time_return"]:
            tr_series = pd.Series(res["time_return"]).sort_index()
            total_return = (tr_series + 1).prod() - 1
            years = len(tr_series) / 252
            annualized_return = (1 + total_return) ** (1 / years) - 1
            print(f"Annualized Return: {annualized_return * 100:.2f}%")
        else:
            print("Annualized Return: Unable to calculate")

    # Generate and Plot Global Equity Curve
    if strategy_equity_curves:
        equity_df = pd.concat(strategy_equity_curves, axis=1)
        full_index = pd.date_range(start=start_dt, end=equity_df.index.max(), freq='B')
        equity_df = equity_df.reindex(full_index)
        equity_df = equity_df.sort_index().ffill()
        equity_df['Global'] = equity_df.sum(axis=1) + unallocated_capital

        global_equity = equity_df['Global']
        global_returns = global_equity.pct_change().dropna()

        total_return = (global_equity.iloc[-1] / global_equity.iloc[0]) - 1
        trading_days = len(global_returns)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1

        sharpe_ratio = (global_returns.mean() / global_returns.std()) * sqrt(252) if global_returns.std() else None

        roll_max = global_equity.cummax()
        daily_drawdown = (roll_max - global_equity) / roll_max
        max_drawdown = daily_drawdown.max() * 100

        plt.figure(figsize=(12, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        plt.plot(global_equity.index, global_equity, color="#00FFFF", label="Global Capital", linewidth=2)
        plt.title("Global Capital Curve", color="#00FFFF", fontsize=16)
        plt.xlabel("Date", color="#00FFFF", fontsize=14)
        plt.ylabel("Capital", color="#00FFFF", fontsize=14)
        plt.legend(facecolor="#000000", edgecolor="#00FFFF")
        plt.tick_params(colors="#00FFFF")
        global_chart_filename = "global_capital_curve.png"
        plt.savefig(global_chart_filename, facecolor='#000000')
        plt.show()
        print(f"\nGlobal capital curve saved as {global_chart_filename}")

        # Print Global Account Performance
        print("\n=== Global Account Performance ===")
        print(f"Overall Annualized Return: {annualized_return * 100:.2f}%")
        if sharpe_ratio is not None:
            print(f"Overall Sharpe Ratio: {sharpe_ratio:.4f}")
        else:
            print("Overall Sharpe Ratio: Unable to calculate")
        print(f"Overall Maximum Drawdown: {max_drawdown:.2f}%")