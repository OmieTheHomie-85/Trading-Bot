import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


SYMBOLS = ["SPY", "QQQ", "VTI", "IWN", "EFA", "EEM", "TLT", "GLD"]


class AlpacaBot:
    def __init__(self, api_key: str, secret_key: str, strategy, paper: bool = True):
        """
        Args:
            api_key:    Alpaca API key
            secret_key: Alpaca secret key
            strategy:   An instance of your Strategy class
            paper:      True = paper trading, False = live trading
        """
        self.strategy = strategy
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_price_data(self, symbols: list[str], lookback_days: int = 120) -> dict[str, pd.Series]:
        """Fetch daily closing prices for each symbol."""
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=lookback_days)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )

        bars = self.data_client.get_stock_bars(request).df

        price_data = {}
        for symbol in symbols:
            try:
                symbol_bars = bars.xs(symbol, level="symbol") if isinstance(bars.index, pd.MultiIndex) else bars
                price_data[symbol] = symbol_bars["close"].sort_index()
            except KeyError:
                logger.warning(f"No data returned for {symbol}, skipping.")

        return price_data

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> float:
        """Return total account equity."""
        account = self.trading_client.get_account()
        return float(account.equity)

    def get_current_positions(self) -> dict[str, float]:
        """Return {symbol: market_value} for all open positions."""
        positions = self.trading_client.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, symbol: str, qty: float, side: OrderSide):
        """Place a fractional market order."""
        if qty <= 0:
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        result = self.trading_client.submit_order(order)
        logger.info(f"Order placed: {side.value.upper()} {qty:.4f} {symbol} | id={result.id}")
        return result

    def close_position(self, symbol: str):
        """Close an entire position in a symbol."""
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self, target_weights: dict[str, float]):
        """
        Rebalance the portfolio to match target_weights.
        target_weights: {symbol: fraction_of_equity}, e.g. {"SPY": 0.5, "TLT": 0.5}
        """
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_current_positions()

        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"Target weights: {target_weights}")

        # Symbols to exit (held but not in new target)
        for symbol in current_positions:
            if symbol not in target_weights:
                logger.info(f"Exiting position: {symbol}")
                self.close_position(symbol)

        # Allow orders to settle briefly
        time.sleep(1)

        # Buy/increase positions
        for symbol, weight in target_weights.items():
            target_value = portfolio_value * weight
            current_value = current_positions.get(symbol, 0.0)
            diff = target_value - current_value

            # Get current price for qty calculation
            price_data = self.get_price_data([symbol], lookback_days=5)
            if symbol not in price_data or price_data[symbol].empty:
                logger.warning(f"Could not get price for {symbol}, skipping.")
                continue

            price = price_data[symbol].iloc[-1]
            qty = abs(diff) / price

            if diff > 50:  # $50 minimum to avoid tiny orders
                self.place_order(symbol, qty, OrderSide.BUY)
            elif diff < -50:
                self.place_order(symbol, qty, OrderSide.SELL)
            else:
                logger.info(f"{symbol}: within tolerance, no order needed.")

    # ------------------------------------------------------------------
    # Market hours check
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        clock = self.trading_client.get_clock()
        return clock.is_open

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, symbols: list[str] = SYMBOLS, check_interval_minutes: int = 60):
        """
        Main bot loop. Runs during market hours, checks every `check_interval_minutes`.
        """
        logger.info("Bot started.")
        while True:
            try:
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting 10 minutes...")
                    time.sleep(600)
                    continue

                logger.info("Market is open. Running strategy...")

                price_data = self.get_price_data(symbols)
                target_weights = self.strategy.target_weights(price_data)

                if not target_weights:
                    logger.info("No assets selected by strategy. Holding cash.")
                else:
                    self.rebalance(target_weights)

                logger.info(f"Sleeping {check_interval_minutes} minutes until next check.")
                time.sleep(check_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # wait a minute and retry


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    from strategy import Strategy  # your Strategy class

    API_KEY = "your_api_key_here"
    SECRET_KEY = "your_secret_key_here"

    strategy = Strategy(ma_window=50, momentum_window=90, top_n=3)
    bot = AlpacaBot(api_key=API_KEY, secret_key=SECRET_KEY, strategy=strategy, paper=True)
    bot.run(symbols=SYMBOLS)