# Quant Research Platform

A modular Python platform for quantitative financial analysis — built around real hedge fund workflows. Fetches live market data, measures risk, models volatility, prices options, backtests trading strategies, and presents everything in an interactive web dashboard.

---

## Project Status

| Phase | Module | Status |
|-------|--------|--------|
| Phase 1 | Data pipeline (loader + preprocessor) | Complete |
| Phase 1 | Risk & return metrics | Complete |
| Phase 1 | Visualization utilities | Complete |
| Phase 2 | Volatility modeling — GARCH + EWMA | Complete |
| Phase 2 | Options pricing — Black-Scholes + Monte Carlo | Complete |
| Phase 2 | Strategy backtesting engine | Complete |
| Phase 2 | Streamlit web dashboard | Complete |
| Phase 3 | Portfolio optimization (Markowitz) | Planned |
| Phase 3 | Regime detection (HMM) | Planned |
| Phase 3 | Stress testing | Planned |

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

# Launch the dashboard
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
quant-research-platform/
├── app.py                         # Streamlit web dashboard
├── src/
│   ├── data/
│   │   ├── loader.py              # Fetch OHLCV data from Yahoo Finance
│   │   └── preprocessor.py       # Clean, validate, detect outliers
│   ├── risk/
│   │   └── metrics.py            # Returns, volatility, Sharpe, drawdown
│   ├── models/
│   │   ├── volatility.py         # GARCH(1,1) and EWMA volatility models
│   │   └── options.py            # Black-Scholes, Monte Carlo, Greeks, IV
│   ├── strategy/
│   │   ├── strategies.py         # SMA crossover, RSI, Bollinger, Momentum
│   │   └── backtester.py         # Event-driven backtesting engine
│   ├── portfolio/                 # (Phase 3) Optimizer, regime, stress test
│   └── utils/
│       └── visualization.py      # Matplotlib charts (price, vol, drawdown)
├── notebooks/
│   ├── phase1_data_exploration.ipynb
│   ├── phase2_strategy_research.ipynb
│   └── phase3_risk_management.ipynb
├── data/
│   └── raw/                       # Cached CSV downloads
├── tests/
│   ├── test_data_loader.py
│   └── test_risk_metrics.py
└── requirements.txt
```

---

## What's Built

### Data Pipeline — `src/data/`

**`MarketDataLoader`** pulls historical OHLCV data from Yahoo Finance via `yfinance`.

- Single or multi-ticker downloads with automatic local caching to `data/raw/`
- Handles yfinance's MultiIndex column format automatically
- Extracts clean price series by type (Close, Adj Close, Open, etc.)

```python
from src.data.loader import MarketDataLoader

loader = MarketDataLoader()
data   = loader.fetch_data('AAPL', '2023-01-01', '2024-01-01')
prices = loader.get_price_series(data, price_type='Close')
```

**`DataPreprocessor`** cleans and validates time series before analysis.

- Missing data: forward-fill, backward-fill, interpolation, or drop
- Outlier detection: IQR method or Z-score method
- Quality checks: duplicate dates, non-chronological order, negative prices, large gaps

---

### Risk & Return Metrics — `src/risk/metrics.py`

**`RiskMetrics`** — the core calculation engine everything else builds on.

| Method | What it computes |
|--------|-----------------|
| `simple_returns()` | `(P_t - P_{t-1}) / P_{t-1}` |
| `log_returns()` | `ln(P_t / P_{t-1})` — time-additive, preferred for modeling |
| `rolling_volatility()` | Moving std dev, annualized by `× sqrt(252)` |
| `realized_volatility()` | Full-period annualized volatility |
| `drawdown()` | Running decline from peak: `(P_t - max(P)) / max(P)` |
| `max_drawdown()` | Worst peak-to-trough loss |
| `sharpe_ratio()` | `(mean_excess_return / std_dev) × sqrt(252)` |
| `summary_statistics()` | Full stats table |

```python
from src.risk.metrics import RiskMetrics

rm      = RiskMetrics()
returns = rm.log_returns(prices)
sharpe  = rm.sharpe_ratio(returns, risk_free_rate=0.04)
mdd     = rm.max_drawdown(prices)
```

**Sample output on AAPL, MSFT, GOOGL (2023):**

| Ticker | Return | Volatility | Sharpe | Max Drawdown |
|--------|--------|-----------|--------|-------------|
| AAPL   | +54.8% | 19.9%     | 2.22   | -14.9%      |
| MSFT   | +58.4% | 25.0%     | 1.86   | -13.0%      |
| GOOGL  | +56.7% | 30.5%     | 1.49   | -17.3%      |

---

### Volatility Models — `src/models/volatility.py`

**`VolatilityModeler`** captures the fact that volatility is not constant — it clusters.

- **GARCH(1,1)** — fits a model where today's variance depends on yesterday's shock and yesterday's variance: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`. Produces in-sample conditional volatility and multi-day forecasts.
- **EWMA** — exponentially weighted moving average volatility, simpler but adapts to recent market conditions (used by RiskMetrics / JPMorgan).
- `compare_models()` — side-by-side comparison of GARCH vs EWMA vs simple historical vol.

```python
from src.models.volatility import VolatilityModeler

modeler  = VolatilityModeler()
garch    = modeler.fit_garch(returns, forecast_horizon=20)
forecast = garch['forecast_vol']       # 20-day annualized vol forecast
ewma     = modeler.ewma_volatility(returns, span=20)
comp     = modeler.compare_models(returns)
```

---

### Option Pricing — `src/models/options.py`

**`OptionPricer`** prices European options using two methods and computes all sensitivities.

**Black-Scholes (analytical)**
```
C = S·N(d1) - K·e^{-rT}·N(d2)
```
Exact formula, instantaneous, assumes constant volatility.

**Monte Carlo simulation**
Generates thousands of random price paths using Geometric Brownian Motion and averages the discounted payoff. Returns price + 95% confidence interval.

**Greeks**

| Greek | Measures |
|-------|---------|
| Delta | Price change per $1 move in stock |
| Gamma | Rate of change of delta |
| Vega  | Price change per 1% move in volatility |
| Theta | Daily time decay |
| Rho   | Price change per 1% move in interest rate |

**Implied Volatility** — inverts Black-Scholes to back out the market's implied vol from an observed option price.

```python
from src.models.options import OptionPricer

p       = OptionPricer(risk_free_rate=0.04)
call    = p.black_scholes_call(S=180, K=185, T=30/365, sigma=0.28)
greeks  = p.greeks(S=180, K=185, T=30/365, sigma=0.28, option_type='call')
mc      = p.monte_carlo_price(S=180, K=185, T=30/365, sigma=0.28, option_type='call')
iv      = p.implied_volatility(market_price=4.10, S=180, K=185, T=30/365)
```

---

### Trading Strategies — `src/strategy/strategies.py`

Four signal-generating strategies, each returning `+1` (long), `-1` (short), or `0` (flat).

| Strategy | Logic |
|----------|-------|
| `sma_crossover(fast, slow)` | Buy when short MA crosses above long MA (trend following) |
| `rsi_strategy(period)` | Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought) |
| `bollinger_bands(window)` | Buy below lower band, sell above upper band (mean reversion) |
| `momentum(lookback)` | Buy what's risen, sell what's fallen over the lookback window |

---

### Backtesting Engine — `src/strategy/backtester.py`

**`Backtester`** simulates trading a strategy on historical data with realistic costs.

- Signal on day T is executed at close of day T (conservative)
- Configurable round-trip transaction costs in basis points
- Returns equity curve, full trade log, and a complete metrics dict

```python
from src.strategy.backtester import Backtester
from src.strategy.strategies import Strategies

bt     = Backtester(transaction_cost_bps=10)
strat  = Strategies()
result = bt.run(prices, strat.sma_crossover(prices, fast=20, slow=50))

print(result['metrics'])
# total_return, ann_return, ann_volatility, sharpe_ratio,
# max_drawdown, win_rate, buy_hold_return, excess_return
```

**Sample backtest — AAPL 2022–2024 (10 bps costs):**

| Strategy | Return | Sharpe | Max DD | Win Rate |
|----------|--------|--------|--------|---------|
| SMA 20/50 | -39.6% | -0.82 | -47.9% | 44% |
| RSI-14    | +13.0% |  0.47 | -13.3% | 43% |
| Bollinger | -0.9%  | -0.07 |  -6.0% | 31% |
| Momentum  | +7.0%  |  0.12 | -30.6% | 47% |

---

### Web Dashboard — `app.py`

Built with **Streamlit + Plotly**. Run it with `python -m streamlit run app.py`.

| View | Content |
|------|---------|
| Overview | KPI cards, performance table, cumulative return chart |
| Detailed Metrics | Per-stock cards, return histogram + Q-Q plot, stats table |
| Visualizations | Price history, drawdown chart, correlation heatmap, bar comparisons |
| Volatility Models | GARCH vs EWMA vs historical, 20-day forecast table |
| Backtesting | Equity curves vs buy & hold, strategy metrics, tunable sliders |
| Option Pricer | Live BS + MC pricing, Greeks, sensitivity heatmap |

---

## Key Concepts

**Adjusted Close vs Close**
Raw close prices drop artificially on stock split dates. Adjusted Close corrects for splits and dividends so return calculations reflect what an investor actually experienced.

**Log Returns vs Simple Returns**
Simple returns are intuitive for reporting. Log returns are time-additive — `R_total = R1 + R2 + ...` — and have better statistical properties, which is why quants use them in models.

**Annualizing with sqrt(252)**
Daily variance is additive over time, so daily standard deviation (volatility) scales by `sqrt(252)` — the number of trading days per year — to get the annual figure.

**Sharpe Ratio**
Return earned per unit of risk taken. Above 1.0 is decent; above 2.0 is excellent. The key use is comparison — a strategy returning 30% with 30% vol has the same Sharpe as one returning 10% with 10% vol.

**GARCH Persistence**
The sum `α + β` in GARCH(1,1). When close to 1.0 (typical for equities), volatility shocks take a long time to decay — today's turbulence predicts tomorrow's turbulence.

**Max Drawdown**
The worst peak-to-trough decline in the period. If you had bought at exactly the worst moment, this is your loss. Portfolio managers care more about this than average returns.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Time series data manipulation |
| `numpy` | Numerical calculations |
| `yfinance` | Live market data from Yahoo Finance |
| `scipy` | Statistics, Q-Q plots, optimization (implied vol) |
| `arch` | GARCH volatility modeling |
| `matplotlib` / `seaborn` | Static charts |
| `plotly` | Interactive charts in the dashboard |
| `streamlit` | Web dashboard framework |
| `statsmodels` | (Phase 3) Regression, additional time series |
| `pytest` | Unit testing |

---

## Running Tests

```bash
pytest tests/
```

---

## License

MIT — built as a quantitative finance learning project.
