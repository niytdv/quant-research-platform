"""
Financial Metrics Module
Calculates returns, volatility, moving averages, and risk metrics
"""

import pandas as pd
import numpy as np

class FinancialMetrics:
    """
    Calculate financial metrics and indicators
    """
    
    def __init__(self):
        pass
    
    # ========== RETURNS ==========
    
    def calculate_simple_returns(self, df, column='Close'):
        """
        Calculate simple returns
        
        Formula: R_t = (P_t - P_{t-1}) / P_{t-1}
        
        Args:
            df (pd.DataFrame): Price data
            column (str): Price column to use
        
        Returns:
            pd.Series: Simple returns
        """
        returns = df[column].pct_change()
        return returns
    
    def calculate_log_returns(self, df, column='Close'):
        """
        Calculate logarithmic returns
        
        Formula: r_t = ln(P_t / P_{t-1})
        
        Better for:
        - Compounding over time
        - Statistical properties (more normal distribution)
        
        Args:
            df (pd.DataFrame): Price data
            column (str): Price column to use
        
        Returns:
            pd.Series: Log returns
        """
        log_returns = np.log(df[column] / df[column].shift(1))
        return log_returns
    
    def calculate_cumulative_returns(self, returns):
        """
        Calculate cumulative returns (total performance)
        
        Formula: (1 + r_1) * (1 + r_2) * ... * (1 + r_n) - 1
        
        Args:
            returns (pd.Series): Daily returns
        
        Returns:
            pd.Series: Cumulative returns
        """
        cumulative = (1 + returns).cumprod() - 1
        return cumulative
    
    # ========== VOLATILITY ==========
    
    def calculate_volatility(self, returns, window=20):
        """
        Calculate rolling volatility (standard deviation of returns)
        
        Formula: σ = sqrt( (1/N) * Σ(R_i - R_mean)² )
        
        Args:
            returns (pd.Series): Return series
            window (int): Rolling window size (default 20 days)
        
        Returns:
            pd.Series: Rolling volatility
        """
        volatility = returns.rolling(window=window).std()
        return volatility
    
    def annualize_volatility(self, daily_volatility):
        """
        Annualize daily volatility
        
        Formula: σ_annual = σ_daily * sqrt(252)
        252 = typical trading days per year
        
        Args:
            daily_volatility (float or pd.Series): Daily volatility
        
        Returns:
            float or pd.Series: Annualized volatility
        """
        return daily_volatility * np.sqrt(252)
    
    # ========== MOVING AVERAGES ==========
    
    def calculate_sma(self, df, column='Close', window=20):
        """
        Calculate Simple Moving Average
        
        Formula: SMA_t = (1/n) * Σ(P_{t-i}) for i=0 to n-1
        
        Args:
            df (pd.DataFrame): Price data
            column (str): Column to calculate SMA on
            window (int): Window size
        
        Returns:
            pd.Series: Simple moving average
        """
        sma = df[column].rolling(window=window).mean()
        return sma
    
    def calculate_ema(self, df, column='Close', window=20):
        """
        Calculate Exponential Moving Average
        
        Formula: EMA_t = α * P_t + (1-α) * EMA_{t-1}
        where α = 2/(n+1)
        
        EMA gives more weight to recent prices
        
        Args:
            df (pd.DataFrame): Price data
            column (str): Column to calculate EMA on
            window (int): Window size
        
        Returns:
            pd.Series: Exponential moving average
        """
        ema = df[column].ewm(span=window, adjust=False).mean()
        return ema
    
    # ========== RISK METRICS ==========
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        """
        Calculate Sharpe Ratio (risk-adjusted return)
        
        Formula: Sharpe = (R_portfolio - R_risk_free) / σ_portfolio
        
        Interpretation:
        - > 1: Good risk-adjusted returns
        - > 2: Very good
        - > 3: Excellent
        
        Args:
            returns (pd.Series): Return series
            risk_free_rate (float): Risk-free rate (annualized)
        
        Returns:
            float: Sharpe ratio
        """
        # Annualize returns
        avg_return = returns.mean() * 252
        
        # Annualize volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe
        if volatility == 0:
            return 0
        
        sharpe = (avg_return - risk_free_rate) / volatility
        return sharpe
    
    def calculate_max_drawdown(self, returns):
        """
        Calculate Maximum Drawdown (largest peak-to-trough decline)
        
        Formula: 
        Drawdown_t = (Peak_value - Current_value) / Peak_value
        Max_DD = max(all drawdowns)
        
        Args:
            returns (pd.Series): Return series
        
        Returns:
            dict: Contains max_drawdown, peak_date, trough_date
        """
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Find dates
        max_dd_date = drawdown.idxmin()
        peak_date = running_max[:max_dd_date].idxmax()
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'peak_date': peak_date,
            'trough_date': max_dd_date,
            'drawdown_series': drawdown
        }
    
    def calculate_all_metrics(self, df, column='Close'):
        """
        Calculate all metrics for a stock
        
        Args:
            df (pd.DataFrame): Price data
            column (str): Price column
        
        Returns:
            dict: All calculated metrics
        """
        # Returns
        simple_returns = self.calculate_simple_returns(df, column)
        log_returns = self.calculate_log_returns(df, column)
        cumulative_returns = self.calculate_cumulative_returns(simple_returns)
        
        # Volatility
        daily_vol = simple_returns.std()
        annual_vol = self.annualize_volatility(daily_vol)
        rolling_vol = self.calculate_volatility(simple_returns, window=20)
        
        # Moving averages
        sma_20 = self.calculate_sma(df, column, window=20)
        sma_50 = self.calculate_sma(df, column, window=50)
        ema_20 = self.calculate_ema(df, column, window=20)
        
        # Risk metrics
        sharpe = self.calculate_sharpe_ratio(simple_returns)
        max_dd_info = self.calculate_max_drawdown(simple_returns)
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1],
            'annualized_return': simple_returns.mean() * 252,
            'daily_volatility': daily_vol,
            'annualized_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd_info['max_drawdown_pct'],
            'num_trading_days': len(df),
            'price_start': df[column].iloc[0],
            'price_end': df[column].iloc[-1],
            'price_high': df[column].max(),
            'price_low': df[column].min()
        }
        
        return metrics


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    aapl = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Calculate metrics
    metrics_calc = FinancialMetrics()
    
    print("\n--- Calculating Returns ---")
    simple_ret = metrics_calc.calculate_simple_returns(aapl)
    print(f"Average daily return: {simple_ret.mean():.4f} ({simple_ret.mean()*100:.2f}%)")
    
    print("\n--- Calculating Volatility ---")
    daily_vol = simple_ret.std()
    annual_vol = metrics_calc.annualize_volatility(daily_vol)
    print(f"Daily volatility: {daily_vol:.4f}")
    print(f"Annualized volatility: {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    
    print("\n--- All Metrics ---")
    all_metrics = metrics_calc.calculate_all_metrics(aapl)
    for key, value in all_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")