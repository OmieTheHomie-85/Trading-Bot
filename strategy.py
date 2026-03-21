import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    @abstractmethod
    def target_weights(self, price_data: dict[str, pd.Series]) -> dict[str, float]:
        """
        Given price history for each symbol, return target portfolio weights.
        Weights should sum to <= 1.0. Empty dict = hold cash.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────
# Strategy 1: Momentum + Moving Average (yours)
# ──────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    def __init__(self, ma_window=50, momentum_window=90, top_n=3):
        self.ma_window = ma_window
        self.momentum_window = momentum_window  # fixed bug: was hardcoded to 90
        self.top_n = top_n

    def price_above_ma(self, prices: pd.Series) -> bool:
        if len(prices) < self.ma_window:
            return False
        ma = prices.rolling(self.ma_window).mean().iloc[-1]
        return prices.iloc[-1] > ma

    def momentum(self, prices: pd.Series) -> float:
        if len(prices) < self.momentum_window + 1:
            return float("-inf")
        current = prices.iloc[-1]
        old = prices.iloc[-(self.momentum_window + 1)]
        if old == 0:
            return float("-inf")
        return (current / old) - 1

    def target_weights(self, price_data: dict[str, pd.Series]) -> dict[str, float]:
        candidates = []
        for symbol, prices in price_data.items():
            if self.price_above_ma(prices):
                score = self.momentum(prices)
                candidates.append((symbol, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in candidates[: self.top_n]]
        if not selected:
            return {}
        weight = 1.0 / len(selected)
        return {s: weight for s in selected}


# ──────────────────────────────────────────────
# Strategy 2: RSI mean-reversion
# ──────────────────────────────────────────────

class RSIStrategy(BaseStrategy):
    """
    Buy assets that are oversold (RSI < oversold_threshold).
    Sell / avoid assets that are overbought (RSI > overbought_threshold).
    Equal-weight the top_n most oversold qualifying assets.
    """

    def __init__(self, rsi_period=14, oversold=35, overbought=70, top_n=3):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.top_n = top_n

    def rsi(self, prices: pd.Series) -> float:
        if len(prices) < self.rsi_period + 1:
            return 50.0  # neutral fallback
        delta = prices.diff().dropna()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean().iloc[-1]
        if loss == 0:
            return 100.0
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def target_weights(self, price_data: dict[str, pd.Series]) -> dict[str, float]:
        candidates = []
        for symbol, prices in price_data.items():
            r = self.rsi(prices)
            if r < self.oversold:
                candidates.append((symbol, r))  # lower RSI = more oversold = higher priority
        candidates.sort(key=lambda x: x[1])  # most oversold first
        selected = [s for s, _ in candidates[: self.top_n]]
        if not selected:
            return {}
        weight = 1.0 / len(selected)
        return {s: weight for s in selected}


# ──────────────────────────────────────────────
# Strategy 3: Dual moving average crossover
# ──────────────────────────────────────────────

class DualMAStrategy(BaseStrategy):
    """
    Buy when the fast MA crosses above the slow MA (golden cross).
    Hold cash when fast MA is below slow MA (death cross).
    """

    def __init__(self, fast_window=20, slow_window=50, top_n=3):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.top_n = top_n

    def crossover_score(self, prices: pd.Series) -> float | None:
        if len(prices) < self.slow_window:
            return None
        fast_ma = prices.rolling(self.fast_window).mean().iloc[-1]
        slow_ma = prices.rolling(self.slow_window).mean().iloc[-1]
        if fast_ma <= slow_ma:
            return None  # death cross — skip
        # Score = how far fast is above slow (as a fraction)
        return (fast_ma - slow_ma) / slow_ma

    def target_weights(self, price_data: dict[str, pd.Series]) -> dict[str, float]:
        candidates = []
        for symbol, prices in price_data.items():
            score = self.crossover_score(prices)
            if score is not None:
                candidates.append((symbol, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in candidates[: self.top_n]]
        if not selected:
            return {}
        weight = 1.0 / len(selected)
        return {s: weight for s in selected}