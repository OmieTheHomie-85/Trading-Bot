"""
main.py — ties everything together.

Workflow:
  1. Load historical price data (from Alpaca or a CSV for testing)
  2. Run the optimizer to find the best strategy + params
  3. Hand the winner to AlpacaBot for live/paper trading

Usage:
  python main.py --mode optimize          # just optimize, print results
  python main.py --mode trade             # optimize then start trading
  python main.py --mode backtest          # backtest all strategies and compare
"""

import argparse
import pandas as pd
import yfinance as yf  # pip install yfinance  (free, good for testing)

import config
from strategy import MomentumStrategy, RSIStrategy, DualMAStrategy
from backtester import Backtester
from optimizer import Optimizer

from config import *

SYMBOLS = symbols


def load_data(symbols: list[str], period: str = "5y") -> dict[str, pd.Series]:
    """Download historical closing prices via yfinance (free, no API key needed)."""
    print(f"Downloading {period} of data for {symbols}...")
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)
    closes = raw["Close"]
    return {sym: closes[sym].dropna() for sym in symbols}


def run_backtest(price_data: dict[str, pd.Series]):
    """Compare all three strategies side by side."""
    strategies = [
        MomentumStrategy(ma_window=50, momentum_window=90, top_n=3),
        RSIStrategy(rsi_period=14, oversold=35, overbought=70, top_n=3),
        DualMAStrategy(fast_window=20, slow_window=50, top_n=3),
    ]
    bt = Backtester(price_data, commission=0.001, rebalance_freq="W")
    print("\n=== Backtest Results ===\n")
    for s in strategies:
        result = bt.run(s)
        print(result.summary())
        print("-" * 40)


def run_optimize(price_data: dict[str, pd.Series]):
    """Walk-forward optimize and print winner."""
    opt = Optimizer(price_data, train_frac=0.7)
    results = opt.optimize()
    print("\n=== Optimization Results (out-of-sample, ranked by Sharpe) ===\n")
    for r in results:
        print(r.summary())
        print("-" * 40)


def run_trade(price_data: dict[str, pd.Series]):
    """Optimize, pick best strategy, start live paper trading."""
    from alpaca_bot import AlpacaBot

    opt = Optimizer(price_data, train_frac=0.7)
    best = opt.best_strategy()

    bot = AlpacaBot(
        api_key=KEY,
        secret_key=SECRET,
        strategy=best,
        paper=True,
    )
    bot.run(symbols=SYMBOLS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["backtest", "optimize", "trade"],
        default="optimize",
    )
    args = parser.parse_args()

    price_data = load_data(SYMBOLS, period="5y")

    if args.mode == "backtest":
        run_backtest(price_data)
    elif args.mode == "optimize":
        run_optimize(price_data)
    elif args.mode == "trade":
        run_trade(price_data)