"""
Visualization Module
Create charts and plots for financial analysis
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class Visualizer:
    """
    Create visualizations for financial data
    """
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with plotting style
        """
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_price_history(self, df, ticker='Stock', column='Close', ma_windows=[20, 50]):
        """
        Plot price history with moving averages
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df.index, df[column], label='Price', linewidth=2, alpha=0.8)
        
        for window in ma_windows:
            ma = df[column].rolling(window=window).mean()
            ax.plot(df.index, ma, label=f'{window}-day MA', linewidth=1.5, alpha=0.7)
        
        ax.set_title(f'{ticker} Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    from metrics import FinancialMetrics
    
    # Load data
    loader = DataLoader()
    aapl = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Calculate returns
    metrics = FinancialMetrics()
    returns = metrics.calculate_simple_returns(aapl)
    
    # Create visualizations
    viz = Visualizer()
    
    print("Creating visualizations...")
    fig1 = viz.plot_price_history(aapl, ticker='AAPL')
    
    plt.show()
    print("Done! Close the plot window to continue.")