"""
Trading Strategies Module

Implements signal-generating strategies used in quantitative research.

Each strategy follows the same interface:
    generate_signals(prices) → pd.Series of {-1, 0, +1}

Financial Intuition:
- Signal +1 = go long (buy), -1 = go short (sell), 0 = flat (no position)
- Strategies are pure signal generators — the backtester handles execution
- Keep strategies stateless so they're easy to test and combine
"""

import numpy as np
import pandas as pd
from typing import Optional


class Strategies:
    """
    Collection of standard quantitative trading strategies.
    """

    # ------------------------------------------------------------------
    # Moving Average Crossover
    # ------------------------------------------------------------------

    @staticmethod
    def sma_crossover(
        prices: pd.Series,
        fast: int = 20,
        slow: int = 50
    ) -> pd.Series:
        """
        Simple Moving Average crossover strategy.

        Logic:
        - BUY  (+1) when fast SMA crosses ABOVE slow SMA (uptrend starting)
        - SELL (-1) when fast SMA crosses BELOW slow SMA (downtrend starting)

        This is one of the most classic trend-following signals.
        Works best in trending markets; whipsaws in choppy markets.

        Args:
            prices: Price series (typically Adj Close)
            fast:   Short lookback window in days (e.g. 20)
            slow:   Long lookback window in days  (e.g. 50)

        Returns:
            Signal series: +1 (long), -1 (short), 0 (no position)
        """
        sma_fast = prices.rolling(fast).mean()
        sma_slow = prices.rolling(slow).mean()

        signal = pd.Series(0, index=prices.index)
        signal[sma_fast > sma_slow] = 1
        signal[sma_fast < sma_slow] = -1

        return signal

    # ------------------------------------------------------------------
    # RSI Mean-Reversion
    # ------------------------------------------------------------------

    @staticmethod
    def rsi_strategy(
        prices: pd.Series,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70
    ) -> pd.Series:
        """
        RSI (Relative Strength Index) mean-reversion strategy.

        RSI measures momentum on a 0–100 scale:
        - RSI < 30: Asset is OVERSOLD → expect a bounce up → BUY signal
        - RSI > 70: Asset is OVERBOUGHT → expect a pullback → SELL signal

        Math:
            RSI = 100 - 100 / (1 + RS)
            RS  = average_gain / average_loss  over `period` days

        Args:
            prices:     Price series
            period:     RSI lookback (standard = 14)
            oversold:   RSI threshold to go long  (default 30)
            overbought: RSI threshold to go short (default 70)

        Returns:
            Signal series
        """
        delta = prices.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=prices.index)
        signal[rsi < oversold]   =  1
        signal[rsi > overbought] = -1

        return signal

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Bands mean-reversion strategy.

        Bands are drawn at ±`num_std` standard deviations around a rolling mean.
        Price tends to revert toward the middle band.

        Logic:
        - Price below lower band → BUY  (+1)  (oversold)
        - Price above upper band → SELL (-1)  (overbought)
        - Price between bands   → FLAT  (0)

        Args:
            prices:  Price series
            window:  Rolling window for mean and std (default 20)
            num_std: Band width in standard deviations (default 2)

        Returns:
            Signal series
        """
        sma   = prices.rolling(window).mean()
        std   = prices.rolling(window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std

        signal = pd.Series(0, index=prices.index)
        signal[prices < lower] =  1
        signal[prices > upper] = -1

        return signal

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    @staticmethod
    def momentum(
        prices: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Price momentum strategy.

        Simple idea: assets that have risen recently tend to keep rising
        (momentum effect, documented across markets since the 1990s).

        Logic:
        - Price higher than `lookback` days ago → BUY  (+1)
        - Price lower  than `lookback` days ago → SELL (-1)

        Args:
            prices:   Price series
            lookback: Number of days for momentum window

        Returns:
            Signal series
        """
        past_price = prices.shift(lookback)

        signal = pd.Series(0, index=prices.index)
        signal[prices > past_price] =  1
        signal[prices < past_price] = -1

        return signal


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.data.loader import MarketDataLoader

    loader = MarketDataLoader()
    data   = loader.fetch_data('AAPL', '2023-01-01', '2024-01-01')
    prices = loader.get_price_series(data, price_type='Close')

    strat = Strategies()

    sma_sig  = strat.sma_crossover(prices)
    rsi_sig  = strat.rsi_strategy(prices)
    bb_sig   = strat.bollinger_bands(prices)
    mom_sig  = strat.momentum(prices)

    print("Signal counts (last 30 days):")
    for name, sig in [('SMA Cross', sma_sig), ('RSI', rsi_sig),
                      ('Bollinger', bb_sig), ('Momentum', mom_sig)]:
        tail = sig.tail(30)
        print(f"  {name:12s}  Long={( tail==1).sum():3d}  "
              f"Short={(tail==-1).sum():3d}  Flat={(tail==0).sum():3d}")
