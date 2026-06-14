"""Simulate exactly what app.py fetch() does."""
import sys, io, contextlib
sys.path.insert(0, '.')
import pandas as pd
import yfinance as yf

tickers = ('AAPL', 'MSFT', 'GOOGL')
start, end = '2023-01-01', '2024-01-01'

result = {}
for ticker in tickers:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        print(f"EMPTY for {ticker}")
        continue

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel('Ticker')

    print(f"{ticker}: columns={raw.columns.tolist()}, shape={raw.shape}")
    series = raw['Close'].dropna()
    print(f"  series len={len(series)}, last={series.iloc[-1]:.2f}")
    result[ticker] = series

print("\nAll OK:", list(result.keys()))
