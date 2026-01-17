"""
Configuration file for the Quant Research Platform
"""

# Data settings
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2024-01-01'

# Stock tickers for testing
TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY']

# Data source
DATA_SOURCE = 'yahoo'  # Using Yahoo Finance

# File paths
DATA_DIR = 'data/'
CLEANED_DATA_DIR = 'data/cleaned/'
RAW_DATA_DIR = 'data/raw/'