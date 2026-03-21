import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from strategy import BaseStrategy


@dataclass
class BacktestResult:
    strategy_name: str
    total_return: float        # e.g. 0.25 = 25%
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float        # e.g. -0.15 = -15%
    win_rate: float            # fraction of days with positive return
    num_trades: int
    equity_curve: pd.Series    # daily portfolio value

    def summary(self) -> str:
        return (
            f"Strategy:          {self.strategy_name}\n"
            f"Total return:      {self.total_return*100:.2f}%\n"
            f"Annualized return: {self.annualized_return*100:.2f}%\n"
            f"Sharpe ratio:      {self.sharpe_ratio:.3f}\n"
            f"Max drawdown:      {self.max_drawdown*100:.2f}%\n"
            f"Win rate:          {self.win_rate*100:.1f}%\n"
            f"Num trades:        {self.num_trades}\n"
        )


class Backtester:
    """
    Simulates a strategy against historical price data day by day.

    Args:
        price_data:   {symbol: pd.Series of daily closing prices (DatetimeIndex)}
        initial_cash: Starting portfolio value in dollars
        commission:   Cost per trade as a fraction (e.g. 0.001 = 0.1%)
        rebalance_freq: How often to rebalance — 'D' daily, 'W' weekly, 'ME' monthly
    """

    def __init__(
        self,
        price_data: dict[str, pd.Series],
        initial_cash: float = 10_000,
        commission: float = 0.001,
        rebalance_freq: str = "W",
    ):
        self.price_data = price_data
        self.initial_cash = initial_cash
        self.commission = commission
        self.rebalance_freq = rebalance_freq
        self._align_data()

    def _align_data(self):
        """Align all series to a common date index."""
        df = pd.DataFrame(self.price_data).dropna()
        self.prices_df = df
        self.dates = df.index

    def _get_rebalance_indices(self, lookback: int) -> set:
        """
        Return the integer positions (in self.dates) that are rebalance days.
        We resample the full price dataframe and then find the nearest
        available trading day, skipping any that fall inside the lookback warmup.
        """
        # Last trading day of each period
        period_end_dates = self.prices_df.resample(self.rebalance_freq).last().index
        rebalance_idx = set()
        for d in period_end_dates:
            # Find the position of this date (or the next available one)
            pos = self.dates.searchsorted(d)
            if pos < len(self.dates) and pos >= lookback:
                rebalance_idx.add(pos)
        return rebalance_idx

    def run(self, strategy: BaseStrategy, lookback: int = 60) -> BacktestResult:
        """
        Run the backtest. lookback = number of days of history fed to the strategy.
        """
        cash = self.initial_cash
        holdings: dict[str, float] = {}   # symbol -> number of shares
        equity_curve = []
        num_trades = 0
        rebalance_indices = self._get_rebalance_indices(lookback)

        for i, date in enumerate(self.dates):
            current_prices = self.prices_df.loc[date]

            # Portfolio value today
            portfolio_value = cash + sum(
                holdings.get(sym, 0) * current_prices.get(sym, 0)
                for sym in holdings
            )
            equity_curve.append(portfolio_value)

            # Rebalance?
            if i not in rebalance_indices:
                continue

            # Feed historical window to strategy
            window = self.prices_df.iloc[i - lookback: i]
            price_window = {sym: window[sym] for sym in window.columns}

            target_weights = strategy.target_weights(price_window)

            # Compute target dollar values
            target_values = {
                sym: portfolio_value * w for sym, w in target_weights.items()
            }

            # Sell positions not in target
            for sym in list(holdings.keys()):
                if sym not in target_values and holdings[sym] > 0:
                    proceeds = holdings[sym] * current_prices[sym]
                    cash += proceeds * (1 - self.commission)
                    holdings[sym] = 0
                    num_trades += 1

            # Buy / adjust positions
            for sym, target_val in target_values.items():
                current_val = holdings.get(sym, 0) * current_prices.get(sym, 0)
                diff = target_val - current_val
                price = current_prices.get(sym, 0)
                if price == 0:
                    continue
                shares_delta = diff / price
                cost = abs(shares_delta * price) * self.commission
                if diff > 10:  # $10 minimum
                    holdings[sym] = holdings.get(sym, 0) + shares_delta
                    cash -= shares_delta * price + cost
                    num_trades += 1
                elif diff < -10:
                    holdings[sym] = holdings.get(sym, 0) + shares_delta
                    cash -= shares_delta * price - cost
                    num_trades += 1

        equity = pd.Series(equity_curve, index=self.dates)
        return self._compute_metrics(strategy.name, equity, num_trades)

    def _compute_metrics(
        self, name: str, equity: pd.Series, num_trades: int
    ) -> BacktestResult:
        daily_returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years = len(equity) / 252
        annualized = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe (risk-free = 0 for simplicity)
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() > 0
            else 0.0
        )

        # Max drawdown
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        win_rate = (daily_returns > 0).mean()

        return BacktestResult(
            strategy_name=name,
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=num_trades,
            equity_curve=equity,
        )