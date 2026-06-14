"""
Market Data Loader Module

This module handles fetching historical price data from financial APIs.
In production quant shops, this would connect to Bloomberg, Reuters, or proprietary databases.
We use Yahoo Finance as a free, reliable source for educational purposes.

Financial Intuition:
- We need OHLCV data: Open, High, Low, Close, Volume
- Adjusted Close accounts for stock splits and dividends (critical for accurate returns)
- We validate data quality before using it in models
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Union, List
from datetime import datetime, timedelta


class MarketDataLoader:
    """
    Fetches and stores historical market data.
    
    Design choice: We separate fetching from processing to make each step testable.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_data(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str = None,
        save_to_disk: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for given tickers.
        
        Financial Context:
        - We use 'Adj Close' instead of 'Close' to account for corporate actions
        - Example: If a stock splits 2-for-1, raw close prices would show a 50% drop
          but adjusted close properly reflects that you still have the same value
        
        Args:
            tickers: Single ticker (e.g., 'AAPL') or list of tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
            save_to_disk: Whether to cache data locally
            
        Returns:
            DataFrame with columns: [Open, High, Low, Close, Adj Close, Volume]
            Index: DatetimeIndex
            
        Example:
            >>> loader = MarketDataLoader()
            >>> data = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
        """
        # Default to today if end_date not specified
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Ensure tickers is a list
        if isinstance(tickers, str):
            tickers = [tickers]
            
        print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
        
        # Download data using yfinance
        # group_by='ticker' is important when fetching multiple tickers
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by='ticker' if len(tickers) > 1 else 'column'
        )
        
        # yfinance >= 0.2 always returns a MultiIndex (Price, Ticker).
        # Flatten it for single-ticker fetches so callers get plain column names.
        if isinstance(data.columns, pd.MultiIndex):
            if len(tickers) == 1:
                # Drop the ticker level → columns become ['Close', 'Open', ...]
                data.columns = data.columns.droplevel('Ticker')
            # For multi-ticker keep the MultiIndex as-is

        # Basic validation
        if data.empty:
            raise ValueError(f"No data fetched for {tickers}. Check ticker symbols and date range.")
        
        # Save to disk for reproducibility
        if save_to_disk:
            filename = f"{'_'.join(tickers)}_{start_date}_{end_date}.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath)
            print(f"Data saved to {filepath}")
        
        print(f"OK: Fetched {len(data)} rows of data")
        return data
    
    def load_from_disk(self, filename: str) -> pd.DataFrame:
        """
        Load previously saved data from disk.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with parsed datetime index
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"OK: Loaded {len(data)} rows from {filepath}")
        return data
    
    def get_price_series(
        self,
        data: pd.DataFrame,
        ticker: str = None,
        price_type: str = 'Adj Close'
    ) -> pd.Series:
        """
        Extract a single price series from OHLCV data.
        
        Financial Context:
        - Most quant models use 'Adj Close' for return calculations
        - 'Close' is the raw closing price
        - 'Open', 'High', 'Low' are useful for intraday strategies
        
        Args:
            data: OHLCV DataFrame
            ticker: Ticker symbol (required for multi-ticker data)
            price_type: One of ['Open', 'High', 'Low', 'Close', 'Adj Close']
            
        Returns:
            Series of prices with datetime index
        """
        # Handle single ticker data
        if ticker is None:
            if price_type in data.columns:
                return data[price_type].dropna()
            else:
                raise ValueError(f"Price type '{price_type}' not found in data")
        
        # Handle multi-ticker data
        if (ticker, price_type) in data.columns:
            return data[(ticker, price_type)].dropna()
        else:
            raise ValueError(f"Ticker '{ticker}' or price type '{price_type}' not found")


# Example usage and testing
if __name__ == "__main__":
    # Initialize loader
    loader = MarketDataLoader()
    
    # Fetch data for a single stock
    print("\n=== Example 1: Single ticker ===")
    aapl_data = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
    print(aapl_data.head())
    print(f"\nData shape: {aapl_data.shape}")
    print(f"Columns: {aapl_data.columns.tolist()}")
    
    # Fetch data for multiple stocks
    print("\n=== Example 2: Multiple tickers ===")
    tech_data = loader.fetch_data(
        ['AAPL', 'MSFT', 'GOOGL'],
        '2022-01-01',
        '2023-12-31'
    )
    print(tech_data.head())
    
    # Extract price series
    print("\n=== Example 3: Extract price series ===")
    aapl_prices = loader.get_price_series(aapl_data, price_type='Adj Close')
    print(aapl_prices.head())
    print(f"\nPrice range: ${aapl_prices.min():.2f} - ${aapl_prices.max():.2f}")