# ◈ quantedge — Derivatives Intelligence Engine

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Press
Enter (blank) on the one-time email prompt to skip it.

## What it does

### 🔍 Search
Type a company name OR ticker — including NSE-listed Indian equities (e.g.
"Reliance", "ICICI Bank", "TCS", or raw tickers like `RELIANCE.NS`). Multiple
matches show a "Did you mean:" picker.

### 📡 Multi-source data layer
Yahoo Finance is tried first (broadest coverage + options chains). If it
fails or returns thin data — common for some NSE tickers — the engine
automatically falls back to **NSE India's public API** for live quotes,
historical prices, and option chains, and finally to **Stooq** for
historical prices as a last resort. The active data source is always shown
under the ticker header.

### 📈 Tab 1 — Pricing & Volatility
Historical / EWMA / GARCH(1,1) volatility, an option priced 3 ways (BSM,
300-step American binomial tree, 50k-path Monte Carlo) against the live
market price, and first- **and second-order** Greeks (delta, gamma, vega,
theta, rho, plus vanna, volga, charm).

### 🌋 Tab 2 — Vol Surface & Forecasting
- Random Forest ML volatility forecaster (lag-feature based), benchmarked
  against GARCH with an out-of-sample R²
- Model-free implied volatility (CBOE VIX-style variance-weighted average
  across all OTM strikes) vs single-strike ATM implied vol
- Full 3D implied volatility surface (strike × maturity) across up to 6
  expiries
- 2D implied volatility smile for the selected expiry

### 🧬 Tab 3 — Advanced Models
Heston (stochastic volatility, Monte Carlo) and Merton (jump-diffusion,
Monte Carlo) option pricing, compared against flat-vol BSM — demonstrates
*why* the volatility smile exists, since BSM's constant-vol assumption
can't produce it but these models can.

### 🧮 Tab 4 — Strategy Builder
Pick a preset multi-leg strategy (straddle, strangle, spreads, iron condor,
butterfly, covered call, protective put, etc.), see the priced legs, an
interactive payoff-at-expiry diagram with breakevens, and aggregated
portfolio Greeks (including vanna/volga/charm) across the whole position.

### 🛡️ Tab 5 — Risk & Hedging
Dynamic delta-hedge simulation (configurable rehedge frequency and
transaction costs), parametric (delta-normal) and historical-simulation
VaR/CVaR at your chosen confidence level, and a full-repricing scenario
stress test (spot ±5%/±10%, vol spike/crush, rate +100bps).

### 📊 Tab 6 — Backtest & Track Record
- A historical volatility-signal backtest using a variance-swap-style proxy
  (forecast vol vs subsequently realized vol) with **walk-forward
  validation** — the signal threshold is calibrated on the first half of
  history and tested out-of-sample on the second half — reporting Sharpe,
  Sortino, Calmar, max drawdown, and win rate.
- A **track record**: every recommendation the engine generates is logged
  to a local SQLite database (`optika_track_record.db`) so you can look
  back at past calls.

### 🎯 Tab 7 — Final Recommendation
Separate, transparent rule-based verdicts for **Options** (implied-vs-
forecast vol gap, trend, theta decay, moneyness) and **Futures** (cost-of-
carry contango/backwardation, market mispricing/arbitrage check, trend) —
each showing exactly which inputs drove the call.

## Known limitations (worth stating explicitly in a write-up)

- Data is "on-demand live" (refreshed each run), not a real-time tick stream
- Options chains only cover *currently listed* contracts — no historical
  chains, which is why the backtest uses a price-based volatility proxy
  instead of real historical option P&L
- NSE India's API can rate-limit or block requests without a proper browser
  session; the fallback is best-effort and may occasionally fail — if so,
  try the Yahoo-native ticker or wait and retry
- The ML volatility forecaster needs `scikit-learn`; GARCH needs `arch` —
  both degrade gracefully to simpler estimators if not installed
- The recommendation engine is intentionally rule-based (not a fitted
  statistical/ML model) so every input to the verdict is visible and
  explainable — it is not investment advice
