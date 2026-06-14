"""
Full end-to-end test of every code path used in app.py.
Run this directly: python debug_app.py
"""
import sys, io, contextlib, traceback
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

PASS = []
FAIL = []

def check(name, fn):
    try:
        result = fn()
        PASS.append(name)
        print(f"  PASS  {name}")
        return result
    except Exception as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return None

print("="*60)
print("  IMPORT CHECKS")
print("="*60)

from src.data.loader      import MarketDataLoader
from src.risk.metrics     import RiskMetrics
from src.models.volatility import VolatilityModeler
from src.models.options   import OptionPricer
from src.strategy.strategies import Strategies
from src.strategy.backtester import Backtester
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
print("  All imports OK")

loader   = MarketDataLoader()
rm       = RiskMetrics()
modeler  = VolatilityModeler()
pricer   = OptionPricer()
strat    = Strategies()
bt       = Backtester(transaction_cost_bps=10)

print()
print("="*60)
print("  DATA FETCH")
print("="*60)

def do_fetch():
    tickers = ('AAPL', 'MSFT', 'GOOGL')
    result = {}
    for ticker in tickers:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw = yf.download(ticker, start='2023-01-01', end='2024-01-01',
                              progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"Empty data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel('Ticker')
        result[ticker] = raw['Close'].dropna()
    return result

prices_dict = check("fetch data", do_fetch)
if prices_dict is None:
    print("Cannot continue without data")
    sys.exit(1)

returns_dict = {t: rm.log_returns(p).dropna() for t, p in prices_dict.items()}

print()
print("="*60)
print("  METRICS")
print("="*60)

def do_metrics():
    rows = []
    for ticker, prices in prices_dict.items():
        returns = returns_dict[ticker]
        rows.append({
            "Ticker":        ticker,
            "Total Return":  f"{(prices.iloc[-1]/prices.iloc[0]-1)*100:+.2f}%",
            "Volatility":    f"{rm.realized_volatility(returns)*100:.2f}%",
            "Sharpe Ratio":  f"{rm.sharpe_ratio(returns):.2f}",
            "Max Drawdown":  f"{rm.max_drawdown(prices)*100:.2f}%",
            "_total_ret_raw": float(prices.iloc[-1]/prices.iloc[0]-1),
            "_sharpe_raw":    float(rm.sharpe_ratio(returns)),
            "_mdd_raw":       float(rm.max_drawdown(prices)),
            "_vol_raw":       float(rm.realized_volatility(returns)),
        })
    return rows

all_metrics = check("compute metrics", do_metrics)

print()
print("="*60)
print("  VISUALIZATIONS")
print("="*60)

def do_cumulative():
    fig = go.Figure()
    for ticker, prices in prices_dict.items():
        cum = (prices / prices.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=cum.index.tolist(), y=cum.values.tolist(),
                                  mode="lines", name=ticker))
    return fig
check("cumulative returns chart", do_cumulative)

def do_correlation():
    ret_df = pd.DataFrame({t: returns_dict[t] for t in prices_dict}).dropna()
    corr = ret_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                    zmin=-1, zmax=1)
    return fig
check("correlation heatmap", do_correlation)

def do_drawdown():
    ticker = 'AAPL'
    prices = prices_dict[ticker]
    running_max = prices.expanding().max()
    drawdown = (prices - running_max) / running_max * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=prices.index.tolist(), y=prices.values.tolist(),
                              mode="lines", name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index.tolist(), y=drawdown.values.tolist(),
                              fill="tozeroy", name="Drawdown"), row=2, col=1)
    return fig
check("drawdown chart", do_drawdown)

def do_distribution():
    returns = returns_dict['AAPL']
    hist_vals = returns.values
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Histogram(x=hist_vals.tolist(), nbinsx=50,
                               histnorm="probability density"), row=1, col=1)
    mu, sigma = hist_vals.mean(), hist_vals.std()
    x_norm = np.linspace(hist_vals.min(), hist_vals.max(), 200)
    y_norm = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm-mu)/sigma)**2)
    fig.add_trace(go.Scatter(x=x_norm.tolist(), y=y_norm.tolist(), mode="lines"), row=1, col=1)
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(hist_vals)
    fig.add_trace(go.Scatter(x=osm.tolist(), y=osr.tolist(), mode="markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[float(osm[0]), float(osm[-1])],
                              y=[slope*osm[0]+intercept, slope*osm[-1]+intercept],
                              mode="lines"), row=1, col=2)
    return fig
check("return distribution", do_distribution)

print()
print("="*60)
print("  VOLATILITY MODELS")
print("="*60)

def do_volatility():
    returns = returns_dict['AAPL']
    comp  = modeler.compare_models(returns)
    garch = modeler.fit_garch(returns, forecast_horizon=20)
    ewma  = modeler.ewma_volatility(returns, span=20)
    print(f"    Historical: {comp['historical']*100:.2f}%  GARCH: {comp['garch']*100:.2f}%  EWMA: {comp['ewma']*100:.2f}%")
    print(f"    Persistence: {garch['persistence']:.4f}  AIC: {garch['aic']:.1f}")
    return garch
check("volatility models", do_volatility)

print()
print("="*60)
print("  BACKTESTING")
print("="*60)

def do_backtest():
    prices = prices_dict['AAPL']
    strategies = {
        'SMA 20/50':   strat.sma_crossover(prices, 20, 50),
        'RSI-14':      strat.rsi_strategy(prices, 14),
        'Bollinger-20':strat.bollinger_bands(prices, 20),
        'Momentum-20': strat.momentum(prices, 20),
    }
    results = {}
    for name, sig in strategies.items():
        res = bt.run(prices, sig)
        m = res['metrics']
        print(f"    {name}: Return={m['total_return']*100:+.1f}%  Sharpe={m['sharpe_ratio']:.2f}")
        results[name] = res
    # equity curve chart
    fig = go.Figure()
    for name, res in results.items():
        eq = res['equity_curve']
        fig.add_trace(go.Scatter(x=eq.index.tolist(), y=eq.values.tolist(),
                                  mode="lines", name=name))
    return fig
check("backtesting", do_backtest)

print()
print("="*60)
print("  OPTION PRICER")
print("="*60)

def do_options():
    S, K, T, sigma = 180.0, 185.0, 30/365, 0.28
    p = OptionPricer(risk_free_rate=0.04)
    call_bs = p.black_scholes_call(S, K, T, sigma)
    put_bs  = p.black_scholes_put(S, K, T, sigma)
    call_mc = p.monte_carlo_price(S, K, T, sigma, 'call', 10_000)
    put_mc  = p.monte_carlo_price(S, K, T, sigma, 'put',  10_000)
    g_call  = p.greeks(S, K, T, sigma, 'call')
    g_put   = p.greeks(S, K, T, sigma, 'put')
    iv      = p.implied_volatility(call_bs + 0.10, S, K, T, 'call')
    print(f"    Call BS={call_bs:.3f}  Put BS={put_bs:.3f}")
    print(f"    Call MC={call_mc['price']:.3f}  Put MC={put_mc['price']:.3f}")
    print(f"    Delta={g_call['delta']:.4f}  Vega={g_call['vega']:.4f}  Theta={g_call['theta']:.4f}")
    print(f"    IV={iv*100:.2f}%")
    # sensitivity heatmap
    S_range     = np.linspace(S*0.80, S*1.20, 10)
    sigma_range = np.linspace(0.10, 0.60, 8)
    heat = np.array([[p.black_scholes_call(s, K, T, sig) for s in S_range]
                      for sig in sigma_range])
    fig = px.imshow(heat,
                    x=[f"${v:.0f}" for v in S_range],
                    y=[f"{v*100:.0f}%" for v in sigma_range],
                    color_continuous_scale="Viridis", text_auto=".2f")
    return fig
check("option pricer", do_options)

print()
print("="*60)
print(f"  RESULTS:  {len(PASS)} passed,  {len(FAIL)} failed")
print("="*60)
if FAIL:
    print("FAILURES:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
else:
    print("  All checks passed - app.py should work correctly")
