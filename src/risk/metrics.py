"""
Risk Metrics Module

Computes fundamental risk measures used in quantitative finance.

Financial Intuition:
- Returns tell us how much we made/lost
- Volatility tells us how uncertain those returns are
- High volatility = high risk = higher potential reward (or loss)

This module is the foundation for everything else we'll build.
"""

import pandas as pd
import numpy as np
from typing import Optional


class RiskMetrics:
    """
    Computes risk and return metrics for financial time series.
    
    Production Note:
    In real quant shops, these metrics run on thousands of securities daily.
    We optimize for clarity here, but in production you'd vectorize everything.
    """
    
    @staticmethod
    def simple_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate simple (arithmetic) returns.
        
        Math Intuition:
        Simple return = (P_t - P_{t-1}) / P_{t-1}
        
        This tells us: "What percentage did the price change?"
        
        Example:
        - Stock goes from $100 to $110: return = (110-100)/100 = 0.10 = 10%
        - Stock goes from $110 to $100: return = (100-110)/110 = -0.091 = -9.1%
        
        Financial Context:
        - Simple returns are intuitive (easy to understand)
        - But they're NOT additive across time
        - If you gain 10% then lose 10%, you don't break even!
          (100 * 1.10 * 0.90 = 99, not 100)
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of simple returns (same length, first value is NaN)
        """
        # pct_change() is equivalent to: (prices - prices.shift(1)) / prices.shift(1)
        returns = prices.pct_change()
        return returns
    
    @staticmethod
    def log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate logarithmic (continuous) returns.
        
        Math Intuition:
        Log return = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
        
        Why use logarithms?
        
        1. TIME ADDITIVITY:
           - Simple returns: R_total = (1+R1)*(1+R2)*(1+R3) - 1  [messy!]
           - Log returns: R_total = R1 + R2 + R3  [clean!]
        
        2. SYMMETRY:
           - Gain 10% then lose 10% ≠ break even (simple returns)
           - Gain ln(1.1) then lose ln(0.9) ≈ break even (log returns)
        
        3. STATISTICAL PROPERTIES:
           - Log returns are closer to normally distributed
           - Makes statistical models (like GARCH) work better
        
        Financial Context:
        - Quants prefer log returns for modeling
        - But traders/investors think in simple returns
        - Always clarify which you're using!
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of log returns
        """
        # Method 1: Using numpy log
        log_returns = np.log(prices / prices.shift(1))
        
        # Method 2 (equivalent): Difference of logs
        # log_returns = np.log(prices).diff()
        
        return log_returns
    
    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 21,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate rolling (moving) volatility.
        
        Math Intuition:
        Volatility = Standard Deviation of Returns
        
        Standard deviation measures "spread" around the mean:
        - σ = sqrt( (1/N) * Σ(r_i - r_mean)^2 )
        
        Why rolling?
        - Volatility changes over time (not constant!)
        - Recent data is more relevant than old data
        - Window size = how much history to use
        
        Financial Context:
        - Low volatility: Stable, boring stock (utilities, consumer staples)
        - High volatility: Risky, exciting stock (tech startups, crypto)
        - Volatility clustering: High vol periods follow high vol periods
        
        Annualization:
        - Daily volatility is small (~0.01 or 1%)
        - Annual volatility is easier to interpret (~0.15 or 15%)
        - Conversion: σ_annual = σ_daily * sqrt(252)
        - Why sqrt(252)? Because variance scales with time, std dev scales with sqrt(time)
        - 252 = typical trading days per year
        
        Args:
            returns: Series of returns (simple or log)
            window: Number of periods to use (21 days ≈ 1 month)
            annualize: If True, convert to annual volatility
            
        Returns:
            Series of rolling volatility
        """
        # Calculate rolling standard deviation
        vol = returns.rolling(window=window).std()
        
        # Annualize if requested
        if annualize:
            # Assume daily data with 252 trading days per year
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def realized_volatility(
        returns: pd.Series,
        annualize: bool = True
    ) -> float:
        """
        Calculate realized (historical) volatility over the entire period.
        
        This is the "single number" volatility of the entire dataset.
        
        Args:
            returns: Series of returns
            annualize: If True, convert to annual volatility
            
        Returns:
            Single volatility value
        """
        vol = returns.std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    @staticmethod
    def drawdown(prices: pd.Series) -> pd.Series:
        """
        Calculate drawdown: decline from peak.
        
        Math Intuition:
        Drawdown_t = (Price_t - Peak_t) / Peak_t
        
        where Peak_t = maximum price seen up to time t
        
        Financial Context:
        - Drawdown measures "pain" - how much you'd have lost buying at the peak
        - Example: Stock was $100, now $70 → drawdown = -30%
        - Maximum drawdown (MDD) is the worst possible loss over the period
        - MDD is key for risk management: "What's the worst that could happen?"
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of drawdowns (always ≤ 0)
        """
        # Calculate running maximum (peak)
        running_max = prices.expanding().max()
        
        # Drawdown = (current price - peak) / peak
        drawdown = (prices - running_max) / running_max
        
        return drawdown
    
    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown: worst peak-to-trough decline.
        
        This is a single number summarizing worst-case loss.
        
        Returns:
            Maximum drawdown (negative number)
        """
        dd = RiskMetrics.drawdown(prices)
        return dd.min()
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio: risk-adjusted return.
        
        Math Intuition:
        Sharpe Ratio = (Mean Return - Risk Free Rate) / Volatility
        
        This tells us: "How much return do we get per unit of risk?"
        
        Financial Context:
        - Sharpe > 1: Good (getting more than 1% return for each 1% of risk)
        - Sharpe > 2: Excellent (very efficient risk-taking)
        - Sharpe < 0: Bad (losing money or underperforming risk-free rate)
        
        Example:
        - Strategy A: 15% return, 10% volatility → Sharpe = 1.5
        - Strategy B: 20% return, 20% volatility → Sharpe = 1.0
        - Strategy A is better! (More efficient use of risk)
        
        Risk-free rate:
        - Usually US Treasury bills (~0-5%)
        - Represents "safe" alternative to investing
        - We default to 0 for simplicity
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            annualize: If True, compute annualized Sharpe
            
        Returns:
            Sharpe ratio (scalar)
        """
        # Calculate excess returns (above risk-free rate)
        if annualize:
            # Convert annual risk-free rate to daily
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        else:
            daily_rf = risk_free_rate
        
        excess_returns = returns - daily_rf
        
        # Sharpe = mean / std
        sharpe = excess_returns.mean() / excess_returns.std()
        
        # Annualize if requested
        if annualize:
            sharpe = sharpe * np.sqrt(252)
        
        return sharpe
    
    @staticmethod
    def summary_statistics(
        prices: pd.Series,
        returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compute comprehensive summary statistics.
        
        This gives you a quick "health check" of your data.
        
        Args:
            prices: Price series
            returns: Return series (computed if not provided)
            
        Returns:
            DataFrame with summary statistics
        """
        if returns is None:
            returns = RiskMetrics.log_returns(prices)
        
        # Remove NaN values for calculations
        returns_clean = returns.dropna()
        
        stats = {
            'Start Date': prices.index[0],
            'End Date': prices.index[-1],
            'Total Days': len(prices),
            'Start Price': prices.iloc[0],
            'End Price': prices.iloc[-1],
            'Total Return (%)': 100 * (prices.iloc[-1] / prices.iloc[0] - 1),
            'Annualized Return (%)': 100 * (returns_clean.mean() * 252),
            'Annualized Volatility (%)': 100 * (returns_clean.std() * np.sqrt(252)),
            'Sharpe Ratio': RiskMetrics.sharpe_ratio(returns_clean),
            'Max Drawdown (%)': 100 * RiskMetrics.max_drawdown(prices),
            'Min Daily Return (%)': 100 * returns_clean.min(),
            'Max Daily Return (%)': 100 * returns_clean.max(),
        }
        
        return pd.DataFrame(stats, index=['Value']).T


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import data module
    sys.path.append(str(Path(__file__).parent.parent))
    from data.loader import MarketDataLoader
    
    # Load data
    loader = MarketDataLoader()
    data = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
    prices = data['Adj Close']
    
    # Initialize risk metrics
    rm = RiskMetrics()
    
    # Compute returns
    print("\n=== Returns Comparison ===")
    simple_ret = rm.simple_returns(prices)
    log_ret = rm.log_returns(prices)
    
    print(f"Simple returns - First 5 days:")
    print(simple_ret.head())
    print(f"\nLog returns - First 5 days:")
    print(log_ret.head())
    print(f"\nDifference (should be small for small returns):")
    print((simple_ret - log_ret).head())
    
    # Compute volatility
    print("\n=== Volatility ===")
    vol_21d = rm.rolling_volatility(log_ret, window=21)
    vol_63d = rm.rolling_volatility(log_ret, window=63)
    
    print(f"21-day rolling volatility (annualized):")
    print(vol_21d.tail())
    print(f"\n63-day rolling volatility (annualized):")
    print(vol_63d.tail())
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    summary = rm.summary_statistics(prices, log_ret)
    print(summary)