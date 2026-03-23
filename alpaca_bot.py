import os
import time
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYMBOLS = ["SPY", "QQQ", "VTI", "IWN", "EFA", "EEM", "TLT", "GLD"]


class Discord:
    """Sends messages to a Discord channel via webhook."""

    def __init__(self, webhook_url: str | None):
        self.webhook_url = webhook_url

    def send(self, message: str):
        if not self.webhook_url:
            return
        try:
            requests.post(self.webhook_url, json={"content": message}, timeout=5)
        except Exception as e:
            logger.warning(f"Discord notification failed: {e}")

    def trade(self, side: str, qty: float, symbol: str, value: float):
        emoji = "🟢" if side == "BUY" else "🔴"
        self.send(f"{emoji} **{side}** `{symbol}` — {qty:.4f} shares (${value:,.2f})")

    def rebalance_summary(self, portfolio_value: float, weights: dict):
        lines = ["📊 **Rebalance**", f"Portfolio: **${portfolio_value:,.2f}**", ""]
        for sym, w in weights.items():
            lines.append(f"  `{sym}` → {w * 100:.0f}%")
        self.send("\n".join(lines))

    def holding_cash(self):
        self.send("💤 **Holding cash** — no assets passed the strategy filter this cycle.")

    def error(self, message: str):
        self.send(f"🚨 **Error:** {message}")

    def started(self, strategy_name: str):
        self.send(f"🤖 **Bot started** — running `{strategy_name}`")

    def market_status(self, is_open: bool):
        if is_open:
            self.send("🔔 **Market open** — starting trading cycle.")
        else:
            self.send("🔕 **Market closed** — bot is waiting.")


class AlpacaBot:
    def __init__(self, api_key: str, secret_key: str, strategy, paper: bool = True):
        self.strategy = strategy
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.discord = Discord(os.getenv("DISCORD_WEBHOOK_URL"))

    # ------------------------------------------------------------------
    # Data — uses yfinance (free, no subscription needed)
    # ------------------------------------------------------------------

    def get_price_data(self, symbols: list[str], lookback_days: int = 120) -> dict[str, pd.Series]:
        """Fetch daily closing prices via yfinance — free, no Alpaca subscription needed."""
        logger.info(f"Fetching price data for {symbols} via yfinance...")
        period = f"{lookback_days}d"
        try:
            raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)
            closes = raw["Close"]
            return {sym: closes[sym].dropna() for sym in symbols if sym in closes.columns}
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            return {}

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest closing price for a single symbol."""
        data = yf.download(symbol, period="5d", auto_adjust=True, progress=False)
        if data.empty:
            return 0.0
        return float(data["Close"].iloc[-1])

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def get_portfolio_value(self) -> float:
        return float(self.trading_client.get_account().equity)

    def get_current_positions(self) -> dict[str, float]:
        positions = self.trading_client.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def place_order(self, symbol: str, qty: float, side: OrderSide):
        if qty <= 0:
            return
        order = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        result = self.trading_client.submit_order(order)
        price = self.get_latest_price(symbol)
        logger.info(f"Order placed: {side.value.upper()} {qty:.4f} {symbol} | id={result.id}")
        self.discord.trade(side.value.upper(), qty, symbol, qty * price)
        return result

    def close_position(self, symbol: str):
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            self.discord.send(f"🔴 **SELL** `{symbol}` — full position closed")
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            self.discord.error(f"Failed to close {symbol}: {e}")

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance(self, target_weights: dict[str, float]):
        portfolio_value = self.get_portfolio_value()
        current_positions = self.get_current_positions()
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        self.discord.rebalance_summary(portfolio_value, target_weights)

        # Exit positions not in new target
        for symbol in current_positions:
            if symbol not in target_weights:
                self.close_position(symbol)

        time.sleep(1)

        # Buy/adjust positions
        for symbol, weight in target_weights.items():
            target_value = portfolio_value * weight
            current_value = current_positions.get(symbol, 0.0)
            diff = target_value - current_value
            price = self.get_latest_price(symbol)
            if price == 0:
                logger.warning(f"Could not get price for {symbol}, skipping.")
                continue
            qty = abs(diff) / price
            if diff > 50:
                self.place_order(symbol, qty, OrderSide.BUY)
            elif diff < -50:
                self.place_order(symbol, qty, OrderSide.SELL)
            else:
                logger.info(f"{symbol}: within tolerance, no order needed.")

    # ------------------------------------------------------------------
    # Market hours check
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        return self.trading_client.get_clock().is_open

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, symbols: list[str] = SYMBOLS, check_interval_minutes: int = 60):
        logger.info("Bot started.")
        self.discord.started(self.strategy.__class__.__name__)
        market_was_open = None

        while True:
            try:
                market_open = self.is_market_open()

                if market_open != market_was_open:
                    self.discord.market_status(market_open)
                    market_was_open = market_open

                if not market_open:
                    logger.info("Market is closed. Waiting 10 minutes...")
                    time.sleep(600)
                    continue

                logger.info("Market is open. Running strategy...")
                price_data = self.get_price_data(symbols)
                target_weights = self.strategy.target_weights(price_data)

                if not target_weights:
                    logger.info("No assets selected by strategy. Holding cash.")
                    self.discord.holding_cash()
                else:
                    self.rebalance(target_weights)

                logger.info(f"Sleeping {check_interval_minutes} minutes until next check.")
                time.sleep(check_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.discord.error(str(e))
                time.sleep(60)