# Quant Research Platform

A modular Python platform for quantitative financial analysis — built around real hedge fund workflows. Fetches market data, cleans it, computes risk/return metrics, and generates professional visualizations.

---

## Project Status

| Phase | Module | Status |
|-------|--------|--------|
| Phase 1 | Data pipeline (loader + preprocessor) | ✅ Complete |
| Phase 1 | Risk & return metrics | ✅ Complete |
| Phase 1 | Visualization utilities | ✅ Complete |
| Phase 2 | Volatility modeling (GARCH) | 🔲 Planned |
| Phase 2 | Options pricing (Black-Scholes) | 🔲 Planned |
| Phase 2 | Strategy backtesting engine | 🔲 Planned |
| Phase 2 | Portfolio optimization | 🔲 Planned |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/quant-research-platform.git
cd quant-research-platform

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch notebooks
jupyter notebook
```

---

## Project Structure

```
quant-research-platform/
├── src/
│   ├── data/
│   │   ├── loader.py          # Fetch OHLCV data from Yahoo Finance
│   │   └── preprocessor.py    # Clean, validate, detect outliers
│   ├── risk/
│   │   └── metrics.py         # Returns, volatility, Sharpe, drawdown
│   ├── utils/
│   │   └── visualization.py   # Price charts, vol plots, dashboards
│   ├── models/                # (Phase 2) Volatility & options models
│   ├── strategy/              # (Phase 2) Backtesting engine
│   └── portfolio/             # (Phase 2) Optimizer, regime, stress test
├── notebooks/
│   ├── phase1_data_exploration.ipynb
│   ├── phase2_strategy_research.ipynb
│   └── phase3_risk_management.ipynb
├── data/
│   └── raw/                   # Cached CSV downloads
├── tests/
│   ├── test_data_loader.py
│   └── test_risk_metrics.py
├── requirements.txt
└── setup.py
```

---

## What's Built

### Data Pipeline (`src/data/`)

**`MarketDataLoader`** — wraps `yfinance` to fetch historical OHLCV data.
- Single or multi-ticker downloads
- Auto-saves to `data/raw/` as CSV for reproducibility
- Extracts individual price series (Adj Close by default)

```python
from src.data.loader import MarketDataLoader

loader = MarketDataLoader()
data = loader.fetch_data(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-01-01')
prices = loader.get_price_series(data, ticker='AAPL')
```

**`DataPreprocessor`** — cleans and validates time series data.
- Missing data handling: forward-fill, backward-fill, interpolation, or drop
- Outlier detection: IQR method or Z-score method
- Data quality checks: duplicate dates, non-chronological order, negative prices, large date gaps

```python
from src.data.preprocessor import DataPreprocessor

pre = DataPreprocessor()
clean_data = pre.handle_missing_data(data, method='ffill', limit=5)
outlier_mask, outliers = pre.detect_outliers(returns, method='iqr')
quality = pre.validate_data_quality(clean_data)
```

---

### Risk & Return Metrics (`src/risk/metrics.py`)

**`RiskMetrics`** — core quantitative calculations.

| Method | What it computes |
|--------|-----------------|
| `simple_returns()` | Arithmetic daily returns: `(P_t - P_{t-1}) / P_{t-1}` |
| `log_returns()` | Logarithmic returns: `ln(P_t / P_{t-1})` — time-additive, used in models |
| `rolling_volatility()` | Moving std dev of returns, annualized by `× sqrt(252)` |
| `realized_volatility()` | Single-period annualized volatility |
| `drawdown()` | Running decline from peak: `(P_t - max(P)) / max(P)` |
| `max_drawdown()` | Worst peak-to-trough loss over the full period |
| `sharpe_ratio()` | `(mean_return - risk_free_rate) / std_dev`, annualized |
| `summary_statistics()` | Full stats table: dates, total/annual return, vol, Sharpe, MDD |

```python
from src.risk.metrics import RiskMetrics

rm = RiskMetrics()
returns = rm.log_returns(prices)
vol = rm.rolling_volatility(returns, window=21)   # 1-month rolling vol
sharpe = rm.sharpe_ratio(returns, risk_free_rate=0.04)
mdd = rm.max_drawdown(prices)
summary = rm.summary_statistics(prices, returns)
```

---

### Visualizations (`src/utils/visualization.py`)

**`FinancialVisualizer`** — publication-quality charts using matplotlib + seaborn.

| Method | Output |
|--------|--------|
| `plot_price_and_returns()` | 2-panel: price history + daily returns |
| `plot_volatility()` | Overlaid rolling vol for multiple windows (21d, 63d, 252d) |
| `plot_drawdown()` | Price with running peak + drawdown fill chart, annotated MDD |
| `plot_return_distribution()` | Histogram with normal overlay + Q-Q plot |
| `plot_comprehensive_analysis()` | 5-panel dashboard: all of the above in one figure |

```python
from src.utils.visualization import FinancialVisualizer

viz = FinancialVisualizer()
fig = viz.plot_comprehensive_analysis(prices, returns, ticker='AAPL')
fig.savefig('aapl_dashboard.png', dpi=150, bbox_inches='tight')
```

---

## Key Concepts

**Why Adjusted Close?**
Raw close prices show artificial drops on stock split dates. Adjusted Close corrects for splits and dividends so return calculations are accurate.

**Simple vs Log Returns**
Simple returns are intuitive for reporting. Log returns are time-additive (`R_total = R1 + R2 + ...`) and better behaved statistically — preferred for modeling.

**Annualizing Volatility**
Daily volatility scales by `sqrt(252)` (trading days per year) because variance is additive over time, so standard deviation scales with the square root of time.

**Sharpe Ratio**
Measures return per unit of risk. A Sharpe above 1.0 is acceptable; above 2.0 is excellent. It lets you compare strategies with different return and risk levels on a level playing field.

**Max Drawdown**
The worst peak-to-trough loss in the period. A key risk management number — it answers "what's the worst that could have happened if I bought at the peak?"

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Time series data manipulation |
| `numpy` | Numerical calculations |
| `yfinance` | Market data from Yahoo Finance |
| `matplotlib` | Charting |
| `seaborn` | Chart styling |
| `scipy` | Q-Q plots, statistical tests |
| `statsmodels` | (Phase 2) GARCH, regression |
| `pytest` | Unit testing |

---

## Running Tests

```bash
pytest tests/
```

---

## Phase 2 Roadmap

- **Volatility modeling** — GARCH(1,1) for volatility forecasting
- **Options pricing** — Black-Scholes model with Greeks
- **Backtesting engine** — Event-driven strategy testing with realistic transaction costs
- **Portfolio optimization** — Mean-variance efficient frontier (Markowitz)
- **Regime detection** — Hidden Markov Model for bull/bear market regimes
- **Stress testing** — Scenario analysis and historical shock simulation

---

## License

MIT License — built as a quantitative finance learning project.
