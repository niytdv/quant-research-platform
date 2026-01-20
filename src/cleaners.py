"""
Data Cleaning Module
Handles missing data, outliers, and data alignment
"""

import pandas as pd
import numpy as np

class DataCleaner:
    """
    Cleans and preprocesses financial data
    """
    
    def __init__(self):
        pass
    
    def handle_missing_data(self, df, method='ffill'):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input data with potential NaN values
            method (str): 'ffill' (forward fill), 'bfill' (backward fill), 
                         'drop' (remove rows), 'interpolate' (linear interpolation)
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        
        df_clean = df.copy()
        
        if method == 'ffill':
            # Forward fill: use previous valid value
            df_clean = df_clean.fillna(method='ffill')
        elif method == 'bfill':
            # Backward fill: use next valid value
            df_clean = df_clean.fillna(method='bfill')
        elif method == 'drop':
            # Remove rows with any missing values
            df_clean = df_clean.dropna()
        elif method == 'interpolate':
            # Linear interpolation between values
            df_clean = df_clean.interpolate(method='linear')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def detect_outliers(self, df, column='Close', threshold=3):
        """
        Detect outliers using Z-score method
        
        Formula: z = (x - mean) / std_dev
        If |z| > threshold, it's an outlier
        
        Args:
            df (pd.DataFrame): Input data
            column (str): Column to check for outliers
            threshold (float): Z-score threshold (default 3 = 99.7% of data)
        
        Returns:
            pd.DataFrame: Data with outlier information
        """
        df_copy = df.copy()
        
        # Calculate z-scores
        mean = df_copy[column].mean()
        std = df_copy[column].std()
        df_copy['z_score'] = (df_copy[column] - mean) / std
        
        # Mark outliers
        df_copy['is_outlier'] = np.abs(df_copy['z_score']) > threshold
        
        num_outliers = df_copy['is_outlier'].sum()
        print(f"Found {num_outliers} outliers in {column} (threshold={threshold})")
        
        return df_copy
    
    def remove_outliers(self, df, column='Close', threshold=3):
        """
        Remove outliers from dataset
        
        Args:
            df (pd.DataFrame): Input data
            column (str): Column to check
            threshold (float): Z-score threshold
        
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        df_with_outliers = self.detect_outliers(df, column, threshold)
        df_clean = df_with_outliers[~df_with_outliers['is_outlier']].copy()
        
        # Drop helper columns
        df_clean = df_clean.drop(columns=['z_score', 'is_outlier'])
        
        print(f"Removed {len(df) - len(df_clean)} outlier rows")
        return df_clean
    
    def align_dataframes(self, df_dict):
        """
        Align multiple stock dataframes to same dates
        
        Args:
            df_dict (dict): Dictionary of {ticker: dataframe}
        
        Returns:
            dict: Aligned dataframes with same date range
        """
        print(f"Aligning {len(df_dict)} dataframes...")
        
        # Find common date range
        all_dates = []
        for ticker, df in df_dict.items():
            all_dates.append(set(df.index))
        
        # Get intersection of all dates
        common_dates = set.intersection(*all_dates)
        print(f"Common trading days: {len(common_dates)}")
        
        # Filter each dataframe to common dates
        aligned_dict = {}
        for ticker, df in df_dict.items():
            aligned_dict[ticker] = df.loc[df.index.isin(common_dates)].sort_index()
        
        return aligned_dict
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the data
        
        Args:
            df (pd.DataFrame): Input data
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'num_rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'missing_values': df.isnull().sum().to_dict(),
            'price_range': {
                'min': df['Low'].min(),
                'max': df['High'].max(),
                'avg_close': df['Close'].mean()
            }
        }
        return summary


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load some data
    loader = DataLoader()
    aapl = loader.fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
    
    # Clean it
    cleaner = DataCleaner()
    
    print("\n--- Handling Missing Data ---")
    aapl_clean = cleaner.handle_missing_data(aapl, method='ffill')
    
    print("\n--- Detecting Outliers ---")
    aapl_outliers = cleaner.detect_outliers(aapl_clean, column='Close', threshold=3)
    
    print("\n--- Data Summary ---")
    summary = cleaner.get_data_summary(aapl_clean)
    for key, value in summary.items():
        print(f"{key}: {value}")