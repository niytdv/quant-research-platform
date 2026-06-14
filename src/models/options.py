"""
Option Pricing Module

Black-Scholes analytical pricing and Monte Carlo simulation.

Financial Intuition:
- An option gives the right (not obligation) to buy/sell at a fixed price
- Call option: right to BUY at strike K → profitable when stock rises above K
- Put option:  right to SELL at strike K → profitable when stock falls below K

Black-Scholes Assumptions:
- Stock price follows Geometric Brownian Motion
- Constant volatility (violated in practice — hence GARCH)
- No dividends, no transaction costs
- Continuous trading possible

The Greeks tell us HOW MUCH the option price changes when inputs change:
- Delta:  sensitivity to stock price (most important for hedging)
- Gamma:  rate of change of delta (convexity)
- Vega:   sensitivity to volatility (crucial for vol trading)
- Theta:  daily time decay (options lose value each day)
- Rho:    sensitivity to interest rates
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional


class OptionPricer:
    """
    Price European options using Black-Scholes and Monte Carlo.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate, e.g. 0.04 for 4%
        """
        self.r = risk_free_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _d1_d2(self, S: float, K: float, T: float, sigma: float, r: float):
        """Compute d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # ------------------------------------------------------------------
    # Black-Scholes pricing
    # ------------------------------------------------------------------

    def black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: Optional[float] = None
    ) -> float:
        """
        Black-Scholes call price.

        C = S·N(d1) - K·e^{-rT}·N(d2)

        Args:
            S:     Current stock price
            K:     Strike price
            T:     Time to expiry in years (e.g. 30/365)
            sigma: Annualized volatility (e.g. 0.25 for 25%)
            r:     Risk-free rate (defaults to self.r)

        Returns:
            Call option price
        """
        r = r if r is not None else self.r
        d1, d2 = self._d1_d2(S, K, T, sigma, r)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def black_scholes_put(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: Optional[float] = None
    ) -> float:
        """
        Black-Scholes put price via put-call parity.

        P = C - S + K·e^{-rT}

        Args: same as black_scholes_call

        Returns:
            Put option price
        """
        r = r if r is not None else self.r
        call = self.black_scholes_call(S, K, T, sigma, r)
        return call - S + K * np.exp(-r * T)

    # ------------------------------------------------------------------
    # Monte Carlo pricing
    # ------------------------------------------------------------------

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        simulations: int = 10_000,
        r: Optional[float] = None,
        seed: int = 42
    ) -> dict:
        """
        Price an option via Monte Carlo simulation.

        Simulates `simulations` terminal stock price paths using GBM:
            S_T = S · exp[(r - σ²/2)T + σ√T · Z]   where Z ~ N(0,1)

        Args:
            S:           Current stock price
            K:           Strike price
            T:           Time to expiry in years
            sigma:       Annualized volatility
            option_type: 'call' or 'put'
            simulations: Number of random paths
            r:           Risk-free rate
            seed:        Random seed for reproducibility

        Returns:
            dict with price, std_error, and 95% confidence interval
        """
        r = r if r is not None else self.r
        rng = np.random.default_rng(seed)

        Z = rng.standard_normal(simulations)
        S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

        if option_type == 'call':
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)

        discounted = np.exp(-r * T) * payoff
        price = discounted.mean()
        se = discounted.std() / np.sqrt(simulations)

        return {
            'price': price,
            'std_error': se,
            'ci_lower': price - 1.96 * se,
            'ci_upper': price + 1.96 * se,
        }

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str = 'call',
        r: Optional[float] = None
    ) -> dict:
        """
        Compute all five Greeks for a European option.

        Delta  — dC/dS   How much option moves per $1 stock move
        Gamma  — d²C/dS² Rate of change of delta (same for call and put)
        Vega   — dC/dσ   Sensitivity to 1% vol change (divided by 100)
        Theta  — dC/dt   Daily time decay (divided by 365)
        Rho    — dC/dr   Sensitivity to 1% rate change (divided by 100)

        Args:
            S, K, T, sigma, option_type, r: standard option parameters

        Returns:
            dict of {delta, gamma, vega, theta, rho}
        """
        r = r if r is not None else self.r
        d1, d2 = self._d1_d2(S, K, T, sigma, r)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega  = S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% vol move

        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'vega':  vega,
            'theta': theta,
            'rho':   rho,
        }

    # ------------------------------------------------------------------
    # Implied Volatility
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        option_type: str = 'call',
        r: Optional[float] = None
    ) -> Optional[float]:
        """
        Back out implied volatility from a market option price.

        Uses Brent's method to solve: BS_price(σ) = market_price

        Args:
            market_price: Observed market price of the option
            S, K, T, option_type, r: standard option parameters

        Returns:
            Implied volatility, or None if no solution found
        """
        r = r if r is not None else self.r

        def objective(sigma):
            if option_type == 'call':
                return self.black_scholes_call(S, K, T, sigma, r) - market_price
            return self.black_scholes_put(S, K, T, sigma, r) - market_price

        try:
            return brentq(objective, 1e-4, 5.0)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Convenience: full pricing summary
    # ------------------------------------------------------------------

    def price_summary(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        r: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Full pricing table for both call and put.

        Returns:
            DataFrame with BS price, MC price, and Greeks for each option type
        """
        r = r if r is not None else self.r
        rows = []
        for otype in ('call', 'put'):
            bs = (self.black_scholes_call if otype == 'call'
                  else self.black_scholes_put)(S, K, T, sigma, r)
            mc = self.monte_carlo_price(S, K, T, sigma, otype)
            g  = self.greeks(S, K, T, sigma, otype, r)
            rows.append({
                'Type':        otype.capitalize(),
                'BS Price':    round(bs, 4),
                'MC Price':    round(mc['price'], 4),
                'Delta':       round(g['delta'], 4),
                'Gamma':       round(g['gamma'], 6),
                'Vega':        round(g['vega'], 4),
                'Theta':       round(g['theta'], 4),
                'Rho':         round(g['rho'], 4),
            })
        return pd.DataFrame(rows).set_index('Type')


if __name__ == "__main__":
    pricer = OptionPricer(risk_free_rate=0.04)

    S, K, T, sigma = 180, 185, 30 / 365, 0.28

    print("=" * 60)
    print("OPTION PRICING — AAPL-style example")
    print("=" * 60)
    print(f"S={S}, K={K}, T={T*365:.0f}d, σ={sigma*100:.0f}%\n")

    summary = pricer.price_summary(S, K, T, sigma)
    print(summary.to_string())

    iv = pricer.implied_volatility(
        pricer.black_scholes_call(S, K, T, sigma) + 0.10,
        S, K, T, 'call'
    )
    print(f"\nImplied vol (market price = BS + $0.10): {iv * 100:.2f}%")
