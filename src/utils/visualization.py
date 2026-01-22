"""
Visualization Utilities

Creates publication-quality charts for financial data analysis.

Design Philosophy:
- Clean, professional plots (suitable for presentations)
- Informative titles and labels
- Consistent color schemes
- Multiple subplots for comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


# Set clean style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class FinancialVisualizer:
    """
    Creates standardized visualizations for financial data.
    """
    
    @staticmethod
    def plot_price_and_returns(
        prices: pd.Series,
        returns: pd.Series,
        title: str = "Price and Returns Analysis"
    ) -> plt.Figure:
        """
        Create a 2-panel plot: prices and returns.
        
        This is the most basic visualization every quant uses.
        
        Args:
            prices: Price series
            returns: Return series
            title: Main title for the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Panel 1: Price chart
        axes[0].plot(prices.index, prices.values, linewidth=1.5, color='navy')
        axes[0].set_title(f'{title} - Price', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Returns
        axes[1].plot(returns.index, returns.values, linewidth=0.8, color='darkgreen', alpha=0.7)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_title('Daily Returns', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Return', fontsize=10)
        axes[1].set_xlabel('Date', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_volatility(
        returns: pd.Series,
        vol_windows: List[int] = [21, 63, 252],
        title: str = "Rolling Volatility Analysis"
    ) -> plt.Figure:
        """
        Plot multiple rolling volatility windows.
        
        Financial Context:
        - 21 days ≈ 1 month (short-term)
        - 63 days ≈ 3 months (medium-term)
        - 252 days ≈ 1 year (long-term)
        
        Args:
            returns: Return series
            vol_windows: List of window sizes in days
            title: Main title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, window in enumerate(vol_windows):
            vol = returns.rolling(window=window).std() * np.sqrt(252)
            ax.plot(
                vol.index,
                vol.values,
                label=f'{window}-day',
                linewidth=1.5,
                color=colors[i % len(colors)],
                alpha=0.8
            )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Annualized Volatility', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_drawdown(
        prices: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> plt.Figure:
        """
        Visualize drawdown over time.
        
        This shows periods of loss and recovery.
        
        Args:
            prices: Price series
            title: Main title
            
        Returns:
            Matplotlib figure
        """
        # Calculate drawdown
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Panel 1: Price with peaks
        axes[0].plot(prices.index, prices.values, label='Price', linewidth=1.5, color='navy')
        axes[0].plot(running_max.index, running_max.values, label='Peak',
                    linewidth=1.5, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title(f'{title} - Price and Peaks', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Drawdown
        axes[1].fill_between(drawdown.index, 0, drawdown.values,
                             color='red', alpha=0.3, label='Drawdown')
        axes[1].plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        axes[1].set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=10)
        axes[1].set_xlabel('Date', fontsize=10)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        axes[1].grid(True, alpha=0.3)
        
        # Annotate max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        axes[1].annotate(
            f'Max DD: {max_dd_val:.2%}',
            xy=(max_dd_idx, max_dd_val),
            xytext=(max_dd_idx, max_dd_val - 0.1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=10,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_return_distribution(
        returns: pd.Series,
        title: str = "Return Distribution"
    ) -> plt.Figure:
        """
        Plot histogram and Q-Q plot of returns.
        
        Financial Context:
        - Normal distribution is a common assumption (often wrong!)
        - Fat tails = more extreme events than normal distribution predicts
        - Q-Q plot shows if data is normal (points on diagonal = normal)
        
        Args:
            returns: Return series
            title: Main title
            
        Returns:
            Matplotlib figure
        """
        returns_clean = returns.dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Histogram with normal distribution overlay
        axes[0].hist(returns_clean, bins=50, density=True, alpha=0.7,
                    color='navy', edgecolor='black')
        
        # Overlay normal distribution
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
        axes[0].plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2)),
                    'r-', linewidth=2, label='Normal Distribution')
        
        axes[0].set_title(f'{title} - Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Return', fontsize=10)
        axes[0].set_ylabel('Density', fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Q-Q plot
        from scipy import stats
        stats.probplot(returns_clean, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_comprehensive_analysis(
        prices: pd.Series,
        returns: pd.Series,
        ticker: str = "Asset"
    ) -> plt.Figure:
        """
        Create a comprehensive 4-panel analysis dashboard.
        
        This is your "one-stop-shop" visualization.
        
        Args:
            prices: Price series
            returns: Return series
            ticker: Asset name for title
            
        Returns:
            Matplotlib figure with 4 subplots
        """
        returns_clean = returns.dropna()
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # Panel 1: Price chart
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(prices.index, prices.values, linewidth=1.5, color='navy')
        ax1.set_title(f'{ticker} - Price History', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Returns
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(returns.index, returns.values, linewidth=0.8, color='darkgreen', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_title('Daily Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Rolling volatility
        ax3 = fig.add_subplot(gs[1, 1])
        vol_21 = returns.rolling(21).std() * np.sqrt(252)
        vol_63 = returns.rolling(63).std() * np.sqrt(252)
        ax3.plot(vol_21.index, vol_21.values, label='21-day', linewidth=1.5, color='blue')
        ax3.plot(vol_63.index, vol_63.values, label='63-day', linewidth=1.5, color='red')
        ax3.set_title('Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Volatility', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Return distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(returns_clean, bins=50, density=True, alpha=0.7,
                color='navy', edgecolor='black')
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        x = np.linspace(returns_clean.min(), returns_clean.max(), 100)
        ax4.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2)),
                'r-', linewidth=2, label='Normal')
        ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Return', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Drawdown
        ax5 = fig.add_subplot(gs[2, 1])
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        ax5.fill_between(drawdown.index, 0, drawdown.values,
                        color='red', alpha=0.3)
        ax5.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        ax5.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Date', fontsize=10)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax5.grid(True, alpha=0.3)
        
        return fig


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    from data.loader import MarketDataLoader
    from risk.metrics import RiskMetrics
    
    # Load data
    loader = MarketDataLoader()
    data = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
    prices = data['Adj Close']
    
    # Compute returns
    rm = RiskMetrics()
    returns = rm.log_returns(prices)
    
    # Create visualizations
    viz = FinancialVisualizer()
    
    print("Generating visualizations...")
    
    # Comprehensive dashboard
    fig = viz.plot_comprehensive_analysis(prices, returns, ticker='AAPL')
    plt.savefig('aapl_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comprehensive analysis to aapl_analysis.png")
    
    plt.show()