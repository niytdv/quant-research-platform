"""
Data Loader Module
Fetches stock market data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import os

class DataLoader:
    """
    Handles fetching and storing market data
    """
    
    def __init__(self, data_dir='data/raw/'):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Directory to save raw data
        """
        self.data_dir = data_dir
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data for a single ticker
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            print(f"Fetching data for {ticker}...")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Check if data was returned
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                return None
            
            # Add ticker column
            df['Ticker'] = ticker
            
            print(f"Successfully fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, tickers, start_date, end_date):
        """
        Fetch data for multiple stocks
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        data_dict = {}
        
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, start_date, end_date)
            if df is not None:
                data_dict[ticker] = df
        
        print(f"\nSuccessfully fetched data for {len(data_dict)} stocks")
        return data_dict
    
    def save_data(self, df, ticker):
        """
        Save DataFrame to CSV file
        
        Args:
            df (pd.DataFrame): Data to save
            ticker (str): Stock ticker (used for filename)
        """
        filepath = os.path.join(self.data_dir, f"{ticker}_raw.csv")
        df.to_csv(filepath)
        print(f"Saved data to {filepath}")
    
    def load_data(self, ticker):
        """
        Load previously saved data from CSV
        
        Args:
            ticker (str): Stock ticker
            
        Returns:
            pd.DataFrame: Loaded stock data
        """
        filepath = os.path.join(self.data_dir, f"{ticker}_raw.csv")
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data from {filepath}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None


# Example usage (you can test this directly)
if __name__ == "__main__":
    # Create loader instance
    loader = DataLoader()
    
    # Test with Apple stock
    aapl_data = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    if aapl_data is not None:
        print("\n--- First 5 rows ---")
        print(aapl_data.head())
        
        print("\n--- Data Info ---")
        print(aapl_data.info())
        
        # Save the data
        loader.save_data(aapl_data, 'AAPL')