import sys
sys.path.insert(0, '.')

from src.data.loader import MarketDataLoader
from src.risk.metrics import RiskMetrics
from src.models.volatility import VolatilityModeler
from src.models.options import OptionPricer
from src.strategy.strategies import Strategies
from src.strategy.backtester import Backtester

print('Loading data...')
loader = MarketDataLoader()
rm = RiskMetrics()

data    = loader.fetch_data('AAPL', '2022-01-01', '2024-01-01', save_to_disk=False)
prices  = loader.get_price_series(data, price_type='Close')
returns = rm.log_returns(prices)

# ── Volatility ───────────────────────────────────────────────────────
print()
print('=' * 55)
print('  VOLATILITY MODELS')
print('=' * 55)
modeler = VolatilityModeler()
comp = modeler.compare_models(returns)
for name, val in comp['formatted'].items():
    print(f'  {name:<12}: {val}')

garch = modeler.fit_garch(returns, forecast_horizon=5)
print(f'  Persistence: {garch["persistence"]:.4f}')
print(f'  5-day forecast:')
for i, v in enumerate(garch['forecast_vol'], 1):
    print(f'    Day {i}: {v*100:.2f}%')

# ── Options ──────────────────────────────────────────────────────────
print()
print('=' * 55)
print('  OPTION PRICING  (S=180, K=185, T=30d, sigma=28%)')
print('=' * 55)
pricer = OptionPricer(risk_free_rate=0.04)
S, K, T, sigma = 180, 185, 30 / 365, 0.28
summary = pricer.price_summary(S, K, T, sigma)
print(summary.to_string())

iv = pricer.implied_volatility(
    pricer.black_scholes_call(S, K, T, sigma) + 0.10, S, K, T, 'call'
)
print(f'  Implied Vol (market = BS + $0.10): {iv*100:.2f}%')

# ── Backtesting ──────────────────────────────────────────────────────
print()
print('=' * 55)
print('  BACKTESTING — AAPL 2022-2024')
print('=' * 55)
strat = Strategies()
bt    = Backtester(transaction_cost_bps=10)

for name, sig in [
    ('SMA 20/50',    strat.sma_crossover(prices, 20, 50)),
    ('RSI-14',       strat.rsi_strategy(prices, 14)),
    ('Bollinger-20', strat.bollinger_bands(prices, 20)),
    ('Momentum-20',  strat.momentum(prices, 20)),
]:
    m = bt.run(prices, sig)['metrics']
    print(
        f'  {name:<14}  Return={m["total_return"]*100:+6.1f}%  '
        f'Sharpe={m["sharpe_ratio"]:5.2f}  '
        f'MaxDD={m["max_drawdown"]*100:5.1f}%  '
        f'WinRate={m["win_rate"]*100:.0f}%'
    )

print()
print('All Phase 2 modules operational!')
