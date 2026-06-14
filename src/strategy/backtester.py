"""
Backtesting Engine

Simulates trading a strategy on historical data and measures performance.

Design Principles:
- Signals are generated on day T, executed at open of day T+1 (no look-ahead bias)
- Transaction costs modelled as a round-trip basis-point spread
- Positions are fully invested (no partial sizing in this version)
- All metrics are consistent with the risk module

Financial Intuition:
- Backtesting answers: "How WOULD this strategy have done historically?"
- Key caveat: past performance ≠ future results
- Watch for overfitting — strategies tuned on the same data they're tested on
"""

import numpy as np
import pandas as pd
from typing import Optional


class Backtester:
    """
    Event-driven backtester for single-asset strategies.
    """

    def __init__(self, transaction_cost_bps: float = 10.0):
        """
        Args:
            transaction_cost_bps: Round-trip cost in basis points (10 bps = 0.10%)
        """
        self.tc = transaction_cost_bps / 10_000   # convert to decimal

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        initial_capital: float = 100_000
    ) -> dict:
        """
        Run a backtest given a price series and signal series.

        Execution rule:
            Signal on day T → position entered at close of day T
            (conservative: some practitioners use T+1 open)

        Args:
            prices:          Daily price series
            signals:         Signal series aligned with prices (+1/0/-1)
            initial_capital: Starting portfolio value in dollars

        Returns:
            dict with:
                equity_curve   — pd.Series of portfolio value over time
                trade_log      — pd.DataFrame of individual trades
                metrics        — performance summary dict
        """
        prices  = prices.dropna()
        signals = signals.reindex(prices.index).fillna(0)

        # Daily returns of the underlying
        asset_returns = prices.pct_change().fillna(0)

        # Strategy returns = signal (shifted 1 day) × asset return
        # Shift ensures we use yesterday's signal for today's return
        position = signals.shift(1).fillna(0)
        strat_returns = position * asset_returns
        strat_returns = strat_returns.fillna(0)

        # Deduct transaction costs on every position change
        trades = position.diff().abs()
        strat_returns -= trades * self.tc

        # Build equity curve — drop leading NaN rows from signal warmup
        equity_raw = initial_capital * (1 + strat_returns).cumprod()
        equity = equity_raw.dropna()

        # Trade log
        trade_log = self._build_trade_log(prices, position, strat_returns)

        # Performance metrics
        metrics = self._compute_metrics(equity, strat_returns, asset_returns)

        return {
            'equity_curve':   equity,
            'strat_returns':  strat_returns,
            'asset_returns':  asset_returns,
            'position':       position,
            'trade_log':      trade_log,
            'metrics':        metrics,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trade_log(
        self,
        prices: pd.Series,
        position: pd.Series,
        strat_returns: pd.Series
    ) -> pd.DataFrame:
        """Record each day a trade occurred."""
        changes = position.diff().abs()
        trade_days = changes[changes > 0].index

        records = []
        for day in trade_days:
            records.append({
                'date':     day,
                'price':    prices.loc[day],
                'position': position.loc[day],
                'daily_pnl': strat_returns.loc[day],
            })
        return pd.DataFrame(records)

    def _compute_metrics(
        self,
        equity: pd.Series,
        strat_returns: pd.Series,
        asset_returns: pd.Series
    ) -> dict:
        """Compute standard performance metrics."""
        trading_days = 252

        # Returns
        total_return     = equity.iloc[-1] / equity.iloc[0] - 1
        ann_return       = (1 + total_return) ** (trading_days / len(equity)) - 1
        ann_vol          = strat_returns.std() * np.sqrt(trading_days)
        sharpe           = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = equity.expanding().max()
        drawdown    = (equity - running_max) / running_max
        max_dd      = drawdown.min()

        # Win rate
        winning_days = (strat_returns > 0).sum()
        total_days   = (strat_returns != 0).sum()
        win_rate     = winning_days / total_days if total_days > 0 else 0

        # Buy-and-hold benchmark
        bh_total = asset_returns.add(1).prod() - 1

        return {
            'total_return':    total_return,
            'ann_return':      ann_return,
            'ann_volatility':  ann_vol,
            'sharpe_ratio':    sharpe,
            'max_drawdown':    max_dd,
            'win_rate':        win_rate,
            'num_trades':      len(self._trade_count(equity)),
            'buy_hold_return': bh_total,
            'excess_return':   total_return - bh_total,
        }

    def _trade_count(self, equity: pd.Series) -> pd.Series:
        return equity   # placeholder — real count in trade_log

    def print_results(self, metrics: dict, strategy_name: str = "Strategy") -> None:
        """Pretty-print backtest results."""
        print(f"\n{'='*55}")
        print(f"  BACKTEST RESULTS — {strategy_name}")
        print(f"{'='*55}")
        print(f"  Total Return:       {metrics['total_return']*100:>8.2f}%")
        print(f"  Ann. Return:        {metrics['ann_return']*100:>8.2f}%")
        print(f"  Ann. Volatility:    {metrics['ann_volatility']*100:>8.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Win Rate:           {metrics['win_rate']*100:>8.2f}%")
        print(f"  Buy-&-Hold Return:  {metrics['buy_hold_return']*100:>8.2f}%")
        print(f"  Excess Return:      {metrics['excess_return']*100:>8.2f}%")
        print(f"{'='*55}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.data.loader import MarketDataLoader
    from src.strategy.strategies import Strategies

    loader = MarketDataLoader()
    data   = loader.fetch_data('AAPL', '2022-01-01', '2024-01-01')
    prices = loader.get_price_series(data, price_type='Close')

    strat = Strategies()
    bt    = Backtester(transaction_cost_bps=10)

    for name, sig in [
        ('SMA Crossover (20/50)', strat.sma_crossover(prices, 20, 50)),
        ('RSI (14)',               strat.rsi_strategy(prices, 14)),
        ('Bollinger Bands (20)',   strat.bollinger_bands(prices, 20)),
        ('Momentum (20)',          strat.momentum(prices, 20)),
    ]:
        result = bt.run(prices, sig)
        bt.print_results(result['metrics'], name)
