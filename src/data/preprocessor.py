"""
Data Preprocessing Module

Handles data cleaning, validation, and transformation.

Financial Intuition:
- Missing data can cause biased results (e.g., stocks halted due to news)
- Outliers might be real (market crashes) or errors (data glitches)
- We need to decide: forward-fill, drop, or interpolate?

Math Intuition:
- Forward-fill: Assume price stays constant (reasonable for short gaps)
- Linear interpolation: Assume smooth price movement (less realistic)
- Drop: Conservative but loses information
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessor:
    """
    Cleans and validates financial time series data.
    
    Production Note: In real quant shops, you'd also check for:
    - Look-ahead bias (using future data)
    - Survivorship bias (only studying stocks that survived)
    - Corporate actions (mergers, delistings)
    """
    
    @staticmethod
    def check_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify missing values in the dataset.
        
        Financial Context:
        - Market holidays cause missing data (expected)
        - Data provider outages cause missing data (problematic)
        - Halted stocks cause missing data (important signal!)
        
        Returns:
            DataFrame showing count and percentage of missing values per column
        """
        missing_count = data.isnull().sum()
        missing_pct = 100 * missing_count / len(data)
        
        missing_info = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        })
        
        return missing_info[missing_info['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
    
    @staticmethod
    def handle_missing_data(
        data: pd.DataFrame,
        method: str = 'ffill',
        limit: int = 5
    ) -> pd.DataFrame:
        """
        Handle missing values in price data.
        
        Financial Context:
        - Forward-fill (ffill): "Last known price" - standard for daily data
        - Backward-fill (bfill): Rarely used (introduces look-ahead bias)
        - Interpolate: Can create artificial prices
        
        Args:
            data: Input DataFrame
            method: 'ffill' (forward fill), 'bfill', 'interpolate', or 'drop'
            limit: Maximum number of consecutive NaNs to fill
            
        Returns:
            Cleaned DataFrame
            
        Design Choice:
        We limit forward-fill to avoid assuming prices stay constant for weeks.
        If a stock has no data for 5+ days, something unusual happened.
        """
        data_clean = data.copy()
        
        if method == 'ffill':
            # Forward fill with limit
            data_clean = data_clean.fillna(method='ffill', limit=limit)
            
        elif method == 'bfill':
            # Backward fill (use cautiously - can introduce look-ahead bias!)
            data_clean = data_clean.fillna(method='bfill', limit=limit)
            
        elif method == 'interpolate':
            # Linear interpolation
            data_clean = data_clean.interpolate(method='linear', limit=limit)
            
        elif method == 'drop':
            # Drop any rows with missing values
            data_clean = data_clean.dropna()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Report on cleaning
        rows_before = len(data)
        rows_after = len(data_clean)
        print(f"Missing data handling ({method}):")
        print(f"  Rows before: {rows_before}")
        print(f"  Rows after: {rows_after}")
        print(f"  Rows dropped: {rows_before - rows_after}")
        
        return data_clean
    
    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect outliers in a price or return series.
        
        Math Intuition:
        
        1. IQR Method (Interquartile Range):
           - Find Q1 (25th percentile) and Q3 (75th percentile)
           - IQR = Q3 - Q1
           - Outliers: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
           - This is robust to extreme values
        
        2. Z-Score Method:
           - Z-score = (value - mean) / std_dev
           - Outliers: |Z-score| > threshold (typically 3)
           - Assumes normal distribution (often violated in finance!)
        
        Financial Context:
        - In returns: outliers might be flash crashes, earnings surprises
        - In prices: check for stock splits, data errors
        - Don't blindly remove outliers - they might be real market events!
        
        Args:
            series: Price or return series
            method: 'iqr' or 'zscore'
            threshold: Z-score threshold (only used for zscore method)
            
        Returns:
            Tuple of (outlier_mask, outlier_values)
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            # Z-score = (x - mean) / std
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_mask = z_scores > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_values = series[outlier_mask]
        
        print(f"Outlier detection ({method}):")
        print(f"  Total values: {len(series)}")
        print(f"  Outliers found: {outlier_mask.sum()}")
        print(f"  Outlier percentage: {100 * outlier_mask.sum() / len(series):.2f}%")
        
        if len(outlier_values) > 0:
            print(f"  Min outlier: {outlier_values.min():.4f}")
            print(f"  Max outlier: {outlier_values.max():.4f}")
        
        return outlier_mask, outlier_values
    
    @staticmethod
    def validate_data_quality(data: pd.DataFrame) -> dict:
        """
        Run comprehensive data quality checks.
        
        Checks:
        1. No duplicate dates
        2. Chronological order
        3. Reasonable price ranges (> 0)
        4. No extreme gaps in dates
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'issues': []
        }
        
        # Check 1: Duplicate dates
        if data.index.duplicated().any():
            results['is_valid'] = False
            results['issues'].append("Duplicate dates found")
        
        # Check 2: Chronological order
        if not data.index.is_monotonic_increasing:
            results['is_valid'] = False
            results['issues'].append("Data not in chronological order")
        
        # Check 3: Positive prices
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (data[col] <= 0).any():
                results['is_valid'] = False
                results['issues'].append(f"Non-positive values in {col}")
        
        # Check 4: Large date gaps (>7 days for daily data)
        date_diffs = data.index.to_series().diff()
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        if len(large_gaps) > 0:
            results['issues'].append(
                f"Found {len(large_gaps)} gaps larger than 7 days"
            )
        
        return results


# Example usage
if __name__ == "__main__":
    from loader import MarketDataLoader
    
    # Load some data
    loader = MarketDataLoader()
    data = loader.fetch_data('AAPL', '2020-01-01', '2023-12-31')
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    print("\n=== Data Quality Check ===")
    quality_report = preprocessor.validate_data_quality(data)
    print(f"Valid: {quality_report['is_valid']}")
    if quality_report['issues']:
        print("Issues found:")
        for issue in quality_report['issues']:
            print(f"  - {issue}")
    
    print("\n=== Missing Data Analysis ===")
    missing_info = preprocessor.check_missing_data(data)
    if len(missing_info) > 0:
        print(missing_info)
    else:
        print("No missing data found!")
    
    # Extract Adj Close and check for outliers
    prices = data['Adj Close']
    returns = prices.pct_change()
    
    print("\n=== Outlier Detection in Returns ===")
    outlier_mask, outlier_values = preprocessor.detect_outliers(
        returns.dropna(),
        method='iqr'
    )
    
    if len(outlier_values) > 0:
        print("\nExtreme return dates:")
        print(outlier_values.sort_values())