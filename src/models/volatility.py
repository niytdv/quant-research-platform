"""
Volatility Modeling Module

GARCH and EWMA models for dynamic volatility forecasting.

Financial Intuition:
- Volatility is NOT constant — it clusters (high vol follows high vol)
- GARCH captures this "volatility clustering" explicitly
- EWMA is simpler but still adapts to recent market conditions
- Forecasting volatility is essential for option pricing and risk management
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class VolatilityModeler:
    """
    Fits and forecasts volatility using GARCH and EWMA models.

    GARCH(1,1) equation:
        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    where:
        ω  = long-run variance baseline
        α  = sensitivity to recent shocks (ARCH term)
        β  = persistence of past variance (GARCH term)
        α + β < 1 means variance is mean-reverting (desirable)
    """

    def fit_garch(
        self,
        returns: pd.Series,
        p: int = 1,
        q: int = 1,
        forecast_horizon: int = 10
    ) -> dict:
        """
        Fit a GARCH(p,q) model and produce a volatility forecast.

        Args:
            returns: Daily log or simple returns
            p: ARCH order (lagged squared residuals)
            q: GARCH order (lagged variance)
            forecast_horizon: Number of days to forecast ahead

        Returns:
            dict with keys:
                model_result       — fitted arch Result object
                conditional_vol    — in-sample annualized volatility (pd.Series)
                forecast_vol       — out-of-sample annualized forecast (np.array)
                persistence        — α + β (how long shocks last)
                aic, bic           — model selection criteria
        """
        returns_clean = returns.dropna()

        # arch library expects returns scaled to percentage points
        returns_pct = returns_clean * 100

        model = arch_model(returns_pct, vol='Garch', p=p, q=q, rescale=False)
        result = model.fit(disp='off')

        # Convert conditional volatility back to decimal and annualize
        cond_vol = result.conditional_volatility / 100 * np.sqrt(252)

        # Forecast
        fc = result.forecast(horizon=forecast_horizon, reindex=False)
        forecast_vol = np.sqrt(fc.variance.values[-1]) / 100 * np.sqrt(252)

        return {
            'model_result': result,
            'conditional_vol': cond_vol,
            'forecast_vol': forecast_vol,
            'persistence': result.params.get('alpha[1]', 0) + result.params.get('beta[1]', 0),
            'aic': result.aic,
            'bic': result.bic,
        }

    def ewma_volatility(
        self,
        returns: pd.Series,
        span: int = 20
    ) -> pd.Series:
        """
        Exponential Weighted Moving Average volatility.

        Simpler than GARCH — weights recent observations more heavily.
        RiskMetrics uses span=94 for daily data (λ ≈ 0.94).

        Args:
            returns: Daily returns
            span: Controls decay speed (higher = more history used)

        Returns:
            Annualized EWMA volatility series
        """
        ewma_var = returns.ewm(span=span, adjust=False).var()
        return np.sqrt(ewma_var) * np.sqrt(252)

    def compare_models(self, returns: pd.Series) -> dict:
        """
        Compare GARCH vs EWMA vs simple historical volatility.

        Args:
            returns: Daily returns

        Returns:
            dict with current volatility estimate from each model
        """
        hist_vol = returns.std() * np.sqrt(252)

        ewma = self.ewma_volatility(returns)
        current_ewma = ewma.iloc[-1]

        garch = self.fit_garch(returns)
        current_garch = garch['conditional_vol'].iloc[-1]

        return {
            'historical': float(hist_vol),
            'ewma': float(current_ewma),
            'garch': float(current_garch),
            'formatted': {
                'Historical': f"{float(hist_vol) * 100:.2f}%",
                'EWMA':       f"{float(current_ewma) * 100:.2f}%",
                'GARCH':      f"{float(current_garch) * 100:.2f}%",
            }
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.data.loader import MarketDataLoader
    from src.risk.metrics import RiskMetrics

    loader = MarketDataLoader()
    rm = RiskMetrics()
    modeler = VolatilityModeler()

    data = loader.fetch_data('AAPL', '2023-01-01', '2024-01-01')
    prices = loader.get_price_series(data, price_type='Close')
    returns = rm.log_returns(prices)

    print("=" * 60)
    print("VOLATILITY MODELING — AAPL 2023")
    print("=" * 60)

    comparison = modeler.compare_models(returns)
    print("\nModel Comparison (current annualized vol):")
    for name, val in comparison['formatted'].items():
        print(f"  {name}: {val}")

    garch = modeler.fit_garch(returns, forecast_horizon=10)
    print(f"\nGARCH(1,1) diagnostics:")
    print(f"  Persistence (α+β): {garch['persistence']:.4f}")
    print(f"  AIC: {garch['aic']:.2f}  |  BIC: {garch['bic']:.2f}")
    print(f"\n10-day volatility forecast:")
    for i, v in enumerate(garch['forecast_vol'], 1):
        print(f"  Day {i:>2}: {v * 100:.2f}%")
