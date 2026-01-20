import unittest
import pandas as pd
import numpy as np
from src.metrics import FinancialMetrics

class TestFinancialMetrics(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        self.metrics = FinancialMetrics()
        
        # Create simple test data
        dates = pd.date_range('2023-01-01', periods=100)
        prices = [100 + i for i in range(100)]  # Linear growth
        self.test_df = pd.DataFrame({'Close': prices}, index=dates)
    
    def test_simple_returns(self):
        """Test simple returns calculation"""
        returns = self.metrics.calculate_simple_returns(self.test_df)
        
        # First return should be NaN
        self.assertTrue(pd.isna(returns.iloc[0]))
        
        # Returns should be positive for increasing prices
        self.assertTrue((returns[1:] > 0).all())
    
    def test_log_returns(self):
        """Test log returns calculation"""
        log_returns = self.metrics.calculate_log_returns(self.test_df)
        
        # First return should be NaN
        self.assertTrue(pd.isna(log_returns.iloc[0]))
        
        # Log returns should exist
        self.assertFalse(log_returns[1:].isna().all())
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = self.metrics.calculate_simple_returns(self.test_df)
        sharpe = self.metrics.calculate_sharpe_ratio(returns)
        
        # Sharpe should be a number
        self.assertIsInstance(sharpe, (int, float))
        
        # For positive returns, Sharpe should be positive
        self.assertGreater(sharpe, 0)

if __name__ == '__main__':
    unittest.main()