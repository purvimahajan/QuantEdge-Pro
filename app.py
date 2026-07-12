"""
Optika — Derivatives Intelligence Engine
====================================================================
Search any stock (name or ticker, including NSE-listed Indian equities),
and this engine will:
  1. pull live spot/rate/dividend/options-chain data (Yahoo Finance,
     with NSE India and Stooq as automatic fallbacks)
  2. estimate volatility: historical, EWMA, GARCH, ML-forecast (Random
     Forest), model-free (VIX-style) implied vol, and a full 3D vol surface
  3. price options 3 classical ways (BSM, binomial tree, Monte Carlo) plus
     2 advanced stochastic models (Heston stochastic vol, Merton jump-diffusion)
  4. compute first- AND second-order Greeks (delta/gamma/vega/theta/rho,
     vanna/volga/charm), aggregate them across a multi-leg strategy, and
     simulate dynamic delta-hedging
  5. build and visualize multi-leg option strategies (straddles, spreads,
     condors, butterflies) with payoff diagrams
  6. quantify risk: parametric & historical VaR/CVaR, scenario stress tests
  7. backtest a volatility trading signal historically with walk-forward
     validation and full performance stats (Sharpe, Sortino, Calmar, drawdown)
  8. check futures/forward fair value via cost-of-carry
  9. log every recommendation to a local track record so you can see how
     the engine's own calls would have played out
  10. produce transparent, rule-based OPTIONS and FUTURES trade signals

Run with:
    pip install -r requirements.txt
    streamlit run app.py

NOTE ON DATA: this is "on-demand live" analysis (refreshed each time you
run it), not tick-level real-time streaming. Yahoo Finance is the primary
source; for NSE-listed Indian equities where Yahoo's feed is thin or
unavailable, the engine automatically falls back to NSE India's public API,
and then to Stooq, for historical prices. Every recommendation is a
transparent, rule-based educational signal — NOT financial advice.
"""

import sqlite3
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# 0. TICKER SEARCH (company name -> ticker symbol)
# =============================================================================

COMMON_NAME_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "tesla": "TSLA", "meta": "META", "facebook": "META",
    "netflix": "NFLX", "nvidia": "NVDA", "amd": "AMD", "intel": "INTC",
    "s&p 500": "^GSPC", "sp500": "^GSPC", "nasdaq": "^IXIC", "dow jones": "^DJI",
    "spy": "SPY", "gold": "GC=F", "crude oil": "CL=F", "bitcoin": "BTC-USD",
    "berkshire": "BRK-B", "jpmorgan": "JPM", "jp morgan": "JPM", "visa": "V",
    "walmart": "WMT", "disney": "DIS", "coca cola": "KO", "coca-cola": "KO",
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS", "infosys": "INFY.NS",
    "icici bank": "ICICIBANK.NS", "hdfc bank": "HDFCBANK.NS", "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS", "sbi": "SBIN.NS", "state bank of india": "SBIN.NS",
    "wipro": "WIPRO.NS", "nifty": "^NSEI", "nifty 50": "^NSEI", "sensex": "^BSESN",
    "bank nifty": "^NSEBANK", "adani": "ADANIENT.NS", "adani enterprises": "ADANIENT.NS",
    "bajaj finance": "BAJFINANCE.NS", "bajaj auto": "BAJAJ-AUTO.NS", "itc": "ITC.NS",
    "larsen": "LT.NS", "l&t": "LT.NS", "maruti": "MARUTI.NS", "maruti suzuki": "MARUTI.NS",
    "axis bank": "AXISBANK.NS", "kotak": "KOTAKBANK.NS", "kotak bank": "KOTAKBANK.NS",
    "hindustan unilever": "HINDUNILVR.NS", "hul": "HINDUNILVR.NS",
    "bharti airtel": "BHARTIARTL.NS", "airtel": "BHARTIARTL.NS",
    "hcl tech": "HCLTECH.NS", "hcl technologies": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS", "sun pharma": "SUNPHARMA.NS",
    "titan": "TITAN.NS", "asian paints": "ASIANPAINT.NS", "ultratech": "ULTRACEMCO.NS",
    "nestle india": "NESTLEIND.NS", "ntpc": "NTPC.NS", "ongc": "ONGC.NS",
    "power grid": "POWERGRID.NS", "coal india": "COALINDIA.NS", "hindalco": "HINDALCO.NS",
    "cipla": "CIPLA.NS", "dr reddy": "DRREDDY.NS", "grasim": "GRASIM.NS",
    "eicher motors": "EICHERMOT.NS", "britannia": "BRITANNIA.NS", "divis lab": "DIVISLAB.NS",
    "shree cement": "SHREECEM.NS", "hero motocorp": "HEROMOTOCO.NS",
}


@st.cache_data(ttl=3600, show_spinner=False)
def search_ticker(query: str):
    """
    Resolve a free-text query (company name OR ticker) into a list of
    (symbol, display_name) candidates. Curated matches (esp. major Indian
    large-caps) are checked FIRST and always ranked at the top, since
    Yahoo's fuzzy search can otherwise match a generic word like "Reliance"
    to an unrelated NYSE small-cap instead of Reliance Industries. Yahoo
    Finance search results are appended after as additional options, then
    a raw-ticker guess as the final fallback.
    """
    query = query.strip()
    if not query:
        return []

    results = []
    key = query.lower()

    # 1. Curated matches first (highest priority, always shown first)
    seen_symbols = set()
    for name, symbol in COMMON_NAME_MAP.items():
        if key == name or key in name or name in key:
            if symbol not in seen_symbols:
                results.append((symbol, f"✓ {name.title()} ({symbol})"))
                seen_symbols.add(symbol)

    # 2. Yahoo Finance live search, appended after curated matches
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        data = resp.json()
        for q in data.get("quotes", []):
            symbol = q.get("symbol")
            name = q.get("shortname") or q.get("longname") or symbol
            exch = q.get("exchDisp", "")
            if symbol and symbol not in seen_symbols:
                results.append((symbol, f"{name} ({symbol}) — {exch}"))
                seen_symbols.add(symbol)
    except Exception:
        pass

    # 3. Last resort: treat the raw query as a ticker
    if not results:
        guess = query.upper().replace(" ", "")
        results.append((guess, f"{guess} (unverified — typed as-is)"))

    return results


# =============================================================================
# 1. DATA LAYER — multi-source: Yahoo Finance -> NSE India -> Stooq
# =============================================================================

def _nse_session(symbol: str = ""):
    """
    NSE India's API blocks bare requests without a realistic browser session.
    The reliable pattern (used by most NSE-scraping libraries) is:
      1. Hit the homepage to get initial cookies
      2. Hit the actual option-chain/quote PAGE (not the API) for the
         symbol-specific cookies NSE's bot-detection expects
      3. Only then call the JSON API endpoint
    Even with this, NSE may still block requests from outside India or from
    cloud/datacenter IPs -- that's a real limitation of the free public
    source, not something headers alone can always fix.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest",
    })
    try:
        s.get("https://www.nseindia.com", timeout=6)
        page = "option-chain" if not symbol else f"get-quotes/equity?symbol={symbol}"
        s.headers["Referer"] = "https://www.nseindia.com/"
        s.get(f"https://www.nseindia.com/{page}", timeout=6)
    except Exception:
        pass
    return s


def _nse_response_status(r) -> str:
    """Classifies an NSE response as OK / blocked / empty, since a 200 status can still be a block page."""
    if r.status_code in (401, 403, 429):
        return "blocked_status"
    ctype = r.headers.get("Content-Type", "")
    if "json" not in ctype.lower() or r.text.strip().startswith(("<", "<!DOCTYPE")):
        return "blocked_html"  # got a webpage/challenge instead of JSON -> bot-blocked
    return "ok"


def fetch_nse_quote(symbol: str):
    """Live quote fallback from NSE India for a bare symbol (no .NS suffix)."""
    try:
        s = _nse_session(symbol)
        r = s.get(f"https://www.nseindia.com/api/quote-equity?symbol={symbol}", timeout=6)
        if _nse_response_status(r) != "ok":
            return None
        data = r.json()
        price = data.get("priceInfo", {}).get("lastPrice")
        return float(price) if price else None
    except Exception:
        return None


def fetch_nse_history(symbol: str, days: int = 730):
    """Historical daily closes fallback from NSE India (bare symbol, no .NS)."""
    try:
        s = _nse_session(symbol)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        url = (f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}"
               f"&series=[%22EQ%22]&from={from_date.strftime('%d-%m-%Y')}&to={to_date.strftime('%d-%m-%Y')}")
        r = s.get(url, timeout=8)
        if _nse_response_status(r) != "ok":
            return None
        data = r.json().get("data", [])
        if not data:
            return None
        rows = [{"Date": pd.to_datetime(d["CH_TIMESTAMP"]), "Close": float(d["CH_CLOSING_PRICE"])} for d in data]
        df = pd.DataFrame(rows).sort_values("Date").set_index("Date")
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df
    except Exception:
        return None


def fetch_nse_option_chain(symbol: str):
    """
    Live option chain fallback from NSE India for F&O-eligible symbols.
    Returns (chain_df, expiry_dates, status_note) -- status_note always
    tells the truth about WHY it failed (blocked vs genuinely no contracts)
    rather than guessing.
    """
    try:
        s = _nse_session(symbol)
        r = s.get(f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}", timeout=8)
        status = _nse_response_status(r)
        if status == "blocked_status":
            return pd.DataFrame(), [], (f"NSE India returned HTTP {r.status_code} — the request was blocked "
                                         f"(common when calling from outside India or from a cloud/hosted IP)")
        if status == "blocked_html":
            return pd.DataFrame(), [], ("NSE India returned a webpage instead of data — its bot-detection "
                                         "blocked this request. This is a known limitation of NSE's free public "
                                         "API when accessed programmatically; it isn't specific to this stock.")
        data = r.json()
        records = data.get("records", {}).get("data", [])
        if not records:
            return pd.DataFrame(), [], (f"NSE responded but returned no contract data for '{symbol}' — "
                                         f"either it genuinely has no F&O listing, or NSE served a partial/"
                                         f"empty response (try again in a moment).")
        rows = []
        expiry_dates = data.get("records", {}).get("expiryDates", [])
        for rec in records:
            for side in ("CE", "PE"):
                leg = rec.get(side)
                if leg:
                    rows.append({
                        "strike": leg.get("strikePrice"),
                        "lastPrice": leg.get("lastPrice", 0),
                        "bid": leg.get("bidprice", 0),
                        "ask": leg.get("askPrice", 0),
                        "volume": leg.get("totalTradedVolume", 0),
                        "openInterest": leg.get("openInterest", 0),
                        "expiry": leg.get("expiryDate"),
                        "type": "call" if side == "CE" else "put",
                    })
        return pd.DataFrame(rows), expiry_dates, "OK"
    except Exception as e:
        return pd.DataFrame(), [], f"NSE India request failed ({type(e).__name__}) — likely blocked or unreachable from this network"


@st.cache_data(ttl=600, show_spinner=False)
def fetch_stooq_history(ticker: str):
    """Generic fallback for historical daily prices (best coverage for US/global tickers)."""
    try:
        sym = ticker.lower()
        if "." not in sym and not sym.startswith("^"):
            sym = sym + ".us"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        df = pd.read_csv(url)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def get_spot_and_history(ticker: str, period: str = "2y"):
    """
    Tries Yahoo Finance first (broadest coverage + options data). If that
    fails or returns empty (common for some NSE tickers), falls back to
    NSE India directly (for .NS symbols), then to Stooq as a last resort.
    Returns (spot, history_df, source_label).
    """
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if not hist.empty:
            hist = hist.copy()
            hist["log_return"] = np.log(hist["Close"] / hist["Close"].shift(1))
            return float(hist["Close"].iloc[-1]), hist, "Yahoo Finance"
    except Exception:
        pass

    if ticker.upper().endswith(".NS"):
        bare = ticker.upper().replace(".NS", "")
        nse_hist = fetch_nse_history(bare)
        if nse_hist is not None and not nse_hist.empty:
            return float(nse_hist["Close"].iloc[-1]), nse_hist, "NSE India (fallback)"

    stooq_hist = fetch_stooq_history(ticker)
    if stooq_hist is not None and not stooq_hist.empty:
        return float(stooq_hist["Close"].iloc[-1]), stooq_hist, "Stooq (fallback)"

    return None, None, None


@st.cache_data(ttl=900, show_spinner=False)
def get_risk_free_rate(ticker: str = "") -> float:
    """13-week T-bill (^IRX) for USD instruments; India 91-day T-bill proxy for .NS tickers."""
    try:
        if ticker.upper().endswith(".NS") or ticker.upper() in ("^NSEI", "^NSEBANK", "^BSESN"):
            return 0.065  # approximate Indian short-term risk-free rate proxy
        irx = yf.Ticker("^IRX").history(period="5d")
        rate = float(irx["Close"].iloc[-1]) / 100.0
        if rate <= 0 or rate > 0.25:
            return 0.045
        return rate
    except Exception:
        return 0.045


@st.cache_data(ttl=900, show_spinner=False)
def get_dividend_yield(ticker: str, spot: float) -> float:
    """Computed from trailing dividend rate / spot for reliability across exchanges."""
    try:
        info = yf.Ticker(ticker).info
        rate = info.get("trailingAnnualDividendRate")
        if rate and spot:
            dy = rate / spot
            if 0 <= dy < 0.5:
                return dy
        dy = info.get("dividendYield")
        if dy is None:
            return 0.0
        dy = dy if dy < 1 else dy / 100.0
        return dy if dy <= 0.5 else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=900, show_spinner=False)
def get_currency_symbol(ticker: str) -> str:
    symbols = {
        "USD": "$", "INR": "₹", "GBP": "£", "EUR": "€", "JPY": "¥",
        "HKD": "HK$", "CAD": "C$", "AUD": "A$", "CNY": "¥", "SGD": "S$",
        "CHF": "CHF ", "KRW": "₩",
    }
    if ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO"):
        return "₹"
    try:
        code = yf.Ticker(ticker).info.get("currency", "USD")
        return symbols.get(code, code + " ")
    except Exception:
        return "$"


@st.cache_data(ttl=900, show_spinner=False)
def get_expiries(ticker: str):
    """Tries Yahoo first; falls back to NSE India's option chain expiry list for .NS symbols. Returns (expiries, source, note)."""
    try:
        exp = list(yf.Ticker(ticker).options)
        if exp:
            return exp, "Yahoo Finance", "OK"
    except Exception:
        pass
    if ticker.upper().endswith(".NS"):
        bare = ticker.upper().replace(".NS", "")
        _, expiry_dates, note = fetch_nse_option_chain(bare)
        if expiry_dates:
            return expiry_dates, "NSE India (fallback)", "OK"
        return [], None, f"Yahoo Finance doesn't carry NSE options data (expected). NSE fallback: {note}"
    return [], None, "Yahoo Finance has no listed options for this ticker"


@st.cache_data(ttl=900, show_spinner=False)
def get_options_chain(ticker: str, expiry: str, source: str = "Yahoo Finance"):
    if source == "NSE India (fallback)":
        bare = ticker.upper().replace(".NS", "")
        full_chain, _, _ = fetch_nse_option_chain(bare)
        if full_chain.empty:
            return pd.DataFrame()
        return full_chain[full_chain["expiry"] == expiry].copy()
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        calls, puts = chain.calls.copy(), chain.puts.copy()
        calls["type"], puts["type"] = "call", "put"
        return pd.concat([calls, puts], ignore_index=True)
    except Exception:
        return pd.DataFrame()


# =============================================================================
# 2. PRICING LAYER — BSM, binomial, Monte Carlo, Heston, Merton jump-diffusion
# =============================================================================

def bsm_price(spot, strike, rate, div_yield, vol, tau, option_type="call") -> float:
    if tau <= 0 or vol <= 0:
        return max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    if option_type == "call":
        return spot * np.exp(-div_yield * tau) * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
    return strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * np.exp(-div_yield * tau) * norm.cdf(-d1)


def bsm_greeks(spot, strike, rate, div_yield, vol, tau, option_type="call") -> dict:
    """First-order Greeks: delta, gamma, vega, theta, rho."""
    if tau <= 0 or vol <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-div_yield * tau)
    disc_r = np.exp(-rate * tau)

    gamma = disc_q * pdf_d1 / (spot * vol * np.sqrt(tau))
    vega = spot * disc_q * pdf_d1 * np.sqrt(tau) / 100

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * np.sqrt(tau))
                 - rate * strike * disc_r * norm.cdf(d2)
                 + div_yield * spot * disc_q * norm.cdf(d1)) / 365
        rho = strike * tau * disc_r * norm.cdf(d2) / 100
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (-spot * disc_q * pdf_d1 * vol / (2 * np.sqrt(tau))
                 + rate * strike * disc_r * norm.cdf(-d2)
                 - div_yield * spot * disc_q * norm.cdf(-d1)) / 365
        rho = -strike * tau * disc_r * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def bsm_greeks_second_order(spot, strike, rate, div_yield, vol, tau, option_type="call") -> dict:
    """
    Second-order Greeks:
      vanna = d(delta)/d(vol)  -- how delta shifts as vol moves
      volga (vomma) = d(vega)/d(vol) -- how vega shifts as vol moves
      charm = d(delta)/d(time) -- how delta decays as time passes
    """
    if tau <= 0 or vol <= 0:
        return {"vanna": 0.0, "volga": 0.0, "charm": 0.0}
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-div_yield * tau)

    vanna = -disc_q * pdf_d1 * d2 / vol
    volga = spot * disc_q * pdf_d1 * np.sqrt(tau) * d1 * d2 / vol

    if option_type == "call":
        charm = (div_yield * disc_q * norm.cdf(d1)
                 - disc_q * pdf_d1 * (2 * (rate - div_yield) * tau - d2 * vol * np.sqrt(tau))
                 / (2 * tau * vol * np.sqrt(tau))) / 365
    else:
        charm = (-div_yield * disc_q * norm.cdf(-d1)
                 - disc_q * pdf_d1 * (2 * (rate - div_yield) * tau - d2 * vol * np.sqrt(tau))
                 / (2 * tau * vol * np.sqrt(tau))) / 365

    return {"vanna": vanna, "volga": volga / 100, "charm": charm}


def binomial_price(spot, strike, rate, div_yield, vol, tau, option_type="call",
                    style="european", steps=200) -> float:
    if tau <= 0 or vol <= 0:
        return max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)

    dt = tau / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((rate - div_yield) * dt) - d) / (u - d)
    disc = np.exp(-rate * dt)

    j = np.arange(steps + 1)
    prices = spot * (u ** (steps - j)) * (d ** j)
    values = np.maximum(prices - strike, 0.0) if option_type == "call" else np.maximum(strike - prices, 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if style == "american":
            j = np.arange(i + 1)
            prices_i = spot * (u ** (i - j)) * (d ** j)
            intrinsic = (np.maximum(prices_i - strike, 0.0) if option_type == "call"
                         else np.maximum(strike - prices_i, 0.0))
            values = np.maximum(values, intrinsic)

    return float(values[0])


def monte_carlo_price(spot, strike, rate, div_yield, vol, tau, option_type="call",
                       n_paths=50_000, seed=42):
    if tau <= 0 or vol <= 0:
        payoff = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
        return payoff, 0.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    terminal = spot * np.exp((rate - div_yield - 0.5 * vol ** 2) * tau + vol * np.sqrt(tau) * z)
    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    disc_payoff = np.exp(-rate * tau) * payoff
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))


def heston_mc_price(spot, strike, rate, div_yield, tau, option_type="call",
                     v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.6,
                     n_paths=20_000, n_steps=100, seed=42):
    """
    Heston stochastic-volatility model, priced via Monte Carlo (Euler
    discretization with full truncation to keep variance non-negative).
    rho < 0 (typical for equities) is what generates the volatility skew
    that flat-vol BSM cannot produce.
    """
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    s = np.full(n_paths, spot)
    v = np.full(n_paths, v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        zv = z1
        zs = rho * z1 + np.sqrt(max(1 - rho ** 2, 0)) * z2

        v_pos = np.maximum(v, 0)
        s = s * np.exp((rate - div_yield - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * zs)
        v = v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos * dt) * zv

    payoff = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)
    disc_payoff = np.exp(-rate * tau) * payoff
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))


def merton_jump_price(spot, strike, rate, div_yield, vol, tau, option_type="call",
                       jump_intensity=0.5, jump_mean=-0.05, jump_std=0.15,
                       n_paths=20_000, seed=42):
    """
    Merton jump-diffusion: GBM plus a compound Poisson jump component.
    jump_intensity: average jumps/year. jump_mean/std: log-jump size
    distribution (typically negative mean for equities -- crash risk).
    """
    rng = np.random.default_rng(seed)
    k = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
    drift = rate - div_yield - jump_intensity * k - 0.5 * vol ** 2

    n_jumps = rng.poisson(jump_intensity * tau, n_paths)
    jump_component = np.array([
        rng.normal(jump_mean, jump_std, nj).sum() if nj > 0 else 0.0
        for nj in n_jumps
    ])
    z = rng.standard_normal(n_paths)
    terminal = spot * np.exp(drift * tau + vol * np.sqrt(tau) * z + jump_component)

    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    disc_payoff = np.exp(-rate * tau) * payoff
    return float(disc_payoff.mean()), float(disc_payoff.std(ddof=1) / np.sqrt(n_paths))


# =============================================================================
# 3. VOLATILITY LAYER — historical, EWMA, GARCH, ML forecast, implied, surface
# =============================================================================

def historical_vol(log_returns: pd.Series, window: int = 252) -> float:
    r = log_returns.dropna().tail(window)
    return float(r.std() * np.sqrt(252)) if len(r) >= 10 else np.nan


def ewma_vol(log_returns: pd.Series, lam: float = 0.94) -> float:
    r = log_returns.dropna().values
    if len(r) < 10:
        return np.nan
    var = r[0] ** 2
    for ret in r[1:]:
        var = lam * var + (1 - lam) * ret ** 2
    return float(np.sqrt(var * 252))


def garch_forecast_vol(log_returns: pd.Series, horizon: int = 5) -> float:
    r = log_returns.dropna() * 100
    if len(r) < 100:
        return np.nan
    if not HAS_ARCH:
        return ewma_vol(log_returns)
    try:
        model = arch_model(r, vol="Garch", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")
        fcast = fit.forecast(horizon=horizon, reindex=False)
        variance_forecast = fcast.variance.values[-1].mean()
        daily_vol = np.sqrt(variance_forecast) / 100
        return float(daily_vol * np.sqrt(252))
    except Exception:
        return ewma_vol(log_returns)


def ml_vol_forecast(log_returns: pd.Series, window: int = 252):
    """
    Random Forest volatility forecaster: trains on lagged realized-vol,
    squared-return, and rolling-mean features to predict next-period
    realized volatility, then compares against GARCH out-of-sample.
    Returns (forecast_vol, in_sample_r2) or (nan, nan) if too little data
    or scikit-learn isn't installed.
    """
    if not HAS_SKLEARN:
        return np.nan, np.nan

    r = log_returns.dropna()
    if len(r) < 300:
        return np.nan, np.nan

    df = pd.DataFrame({"ret": r})
    df["realized_vol_5d"] = df["ret"].rolling(5).std() * np.sqrt(252)
    df["realized_vol_20d"] = df["ret"].rolling(20).std() * np.sqrt(252)
    df["sq_ret"] = df["ret"] ** 2
    df["target"] = df["ret"].rolling(20).std().shift(-20) * np.sqrt(252)  # forward-looking 20d realized vol
    df = df.dropna()
    if len(df) < 100:
        return np.nan, np.nan

    features = ["realized_vol_5d", "realized_vol_20d", "sq_ret"]
    X, y = df[features].values, df["target"].values
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test) if len(X_test) > 5 else np.nan

    latest_features = df[features].iloc[[-1]].values
    forecast = float(model.predict(latest_features)[0])
    return forecast, r2


def implied_vol(market_price, spot, strike, rate, div_yield, tau, option_type="call"):
    if tau <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
    if market_price < intrinsic:
        return np.nan

    def objective(vol):
        return bsm_price(spot, strike, rate, div_yield, vol, tau, option_type) - market_price

    try:
        return brentq(objective, 1e-4, 5.0, maxiter=200)
    except ValueError:
        return np.nan


def model_free_implied_vol(chain: pd.DataFrame, spot, rate, tau):
    """
    Approximates the CBOE VIX-style "model-free" implied volatility: a
    variance-weighted average across all OTM strikes at one expiry,
    rather than relying on a single ATM implied vol. More robust because
    it uses the whole strike range instead of one noisy quote.
    Formula (simplified CBOE methodology): sigma^2 = (2/tau) * sum[ (dK/K^2) * e^(r*tau) * Q(K) ]
    """
    if chain.empty or tau <= 0:
        return np.nan

    otm = chain[
        ((chain["type"] == "call") & (chain["strike"] >= spot)) |
        ((chain["type"] == "put") & (chain["strike"] < spot))
    ].copy()
    otm = otm[otm["lastPrice"] > 0].sort_values("strike")
    if len(otm) < 4:
        return np.nan

    strikes = otm["strike"].values
    prices = otm["lastPrice"].values
    dK = np.gradient(strikes)

    total = np.sum((dK / strikes ** 2) * np.exp(rate * tau) * prices)
    variance = (2 / tau) * total - (1 / tau) * (spot / strikes[np.argmin(np.abs(strikes - spot))] - 1) ** 2
    variance = max(variance, 1e-6)
    return float(np.sqrt(variance))


def build_vol_surface(ticker, expiries, spot, rate, div_yield, source, option_type="call", max_expiries=6):
    """Loops implied_vol() across strikes for several expiries to build a full surface DataFrame."""
    rows = []
    for expiry in expiries[:max_expiries]:
        chain = get_options_chain(ticker, expiry, source)
        if chain.empty:
            continue
        tau = max((parse_expiry_date(expiry) - datetime.now()).days, 1) / 365
        sub = chain[chain["type"] == option_type]
        for _, r in sub.iterrows():
            mp = r.get("lastPrice", np.nan)
            if pd.isna(mp) or mp <= 0:
                continue
            iv = implied_vol(mp, spot, r["strike"], rate, div_yield, tau, option_type)
            if not np.isnan(iv) and 0.01 < iv < 3:
                rows.append({"expiry": expiry, "tau": tau, "strike": r["strike"], "iv": iv})
    return pd.DataFrame(rows)


# =============================================================================
# 4. STOCHASTIC PATH SIMULATION (for hedging demo)
# =============================================================================

def simulate_gbm_path(spot, rate, div_yield, vol, tau, n_steps=60, seed=None):
    rng = np.random.default_rng(seed)
    dt = tau / n_steps
    z = rng.standard_normal(n_steps)
    log_returns = (rate - div_yield - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    return spot * np.exp(np.concatenate([[0], np.cumsum(log_returns)]))


# =============================================================================
# 5. HEDGING LAYER — dynamic delta hedge
# =============================================================================

def simulate_delta_hedge(price_path, strike, rate, div_yield, vol, tau_total,
                          option_type="call", rehedge_every=1, cost_bps=5.0):
    n = len(price_path)
    dt = tau_total / (n - 1)
    rows = []
    cash = bsm_price(price_path[0], strike, rate, div_yield, vol, tau_total, option_type)
    prev_delta = 0.0
    shares_held = 0.0

    for i in range(n):
        t_remaining = max(tau_total - i * dt, 1e-6)
        s = price_path[i]
        delta = bsm_greeks(s, strike, rate, div_yield, vol, t_remaining, option_type)["delta"]

        if i % rehedge_every == 0 or i == n - 1:
            trade = delta - prev_delta
            trade_cost = abs(trade) * s * (cost_bps / 10_000)
            cash -= trade * s + trade_cost
            shares_held += trade
            prev_delta = delta

        cash *= np.exp(rate * dt)
        rows.append({"step": i, "spot": s, "delta": delta, "cash": cash, "shares_held": shares_held})

    payoff = max(price_path[-1] - strike, 0.0) if option_type == "call" else max(strike - price_path[-1], 0.0)
    final_pnl = cash + shares_held * price_path[-1] - payoff
    return pd.DataFrame(rows), final_pnl


# =============================================================================
# 6. STRATEGY BUILDER — multi-leg payoff diagrams & portfolio Greeks
# =============================================================================

PRESET_STRATEGIES = {
    "Long Call": [{"type": "call", "position": 1, "offset": 0.0}],
    "Long Put": [{"type": "put", "position": 1, "offset": 0.0}],
    "Covered Call": [{"type": "call", "position": -1, "offset": 0.05}],
    "Protective Put": [{"type": "put", "position": 1, "offset": -0.05}],
    "Straddle": [{"type": "call", "position": 1, "offset": 0.0}, {"type": "put", "position": 1, "offset": 0.0}],
    "Strangle": [{"type": "call", "position": 1, "offset": 0.05}, {"type": "put", "position": 1, "offset": -0.05}],
    "Bull Call Spread": [{"type": "call", "position": 1, "offset": -0.02}, {"type": "call", "position": -1, "offset": 0.05}],
    "Bear Put Spread": [{"type": "put", "position": 1, "offset": 0.02}, {"type": "put", "position": -1, "offset": -0.05}],
    "Iron Condor": [
        {"type": "put", "position": -1, "offset": -0.05}, {"type": "put", "position": 1, "offset": -0.10},
        {"type": "call", "position": -1, "offset": 0.05}, {"type": "call", "position": 1, "offset": 0.10},
    ],
    "Butterfly (Call)": [
        {"type": "call", "position": 1, "offset": -0.05}, {"type": "call", "position": -2, "offset": 0.0},
        {"type": "call", "position": 1, "offset": 0.05},
    ],
}


def build_legs_from_preset(preset_name, spot, rate, div_yield, vol, tau):
    """Turns a preset's relative strike offsets into concrete legs with priced premiums."""
    legs = []
    for leg in PRESET_STRATEGIES[preset_name]:
        strike = round(spot * (1 + leg["offset"]), 2)
        premium = bsm_price(spot, strike, rate, div_yield, vol, tau, leg["type"])
        legs.append({"type": leg["type"], "strike": strike, "position": leg["position"], "premium": premium})
    return legs


def payoff_at_expiry(legs, spot_range):
    """Computes total P&L at expiry across a range of terminal spot prices."""
    total = np.zeros_like(spot_range)
    for leg in legs:
        intrinsic = (np.maximum(spot_range - leg["strike"], 0.0) if leg["type"] == "call"
                     else np.maximum(leg["strike"] - spot_range, 0.0))
        leg_pnl = leg["position"] * (intrinsic - leg["premium"])
        total += leg_pnl
    return total


def portfolio_greeks(legs, spot, rate, div_yield, vol, tau):
    """Aggregates first- and second-order Greeks across every leg, weighted by position size."""
    agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
           "vanna": 0.0, "volga": 0.0, "charm": 0.0}
    for leg in legs:
        g1 = bsm_greeks(spot, leg["strike"], rate, div_yield, vol, tau, leg["type"])
        g2 = bsm_greeks_second_order(spot, leg["strike"], rate, div_yield, vol, tau, leg["type"])
        for k in ["delta", "gamma", "vega", "theta", "rho"]:
            agg[k] += leg["position"] * g1[k]
        for k in ["vanna", "volga", "charm"]:
            agg[k] += leg["position"] * g2[k]
    return agg


# =============================================================================
# 7. RISK LAYER — VaR / CVaR and scenario stress testing
# =============================================================================

def var_cvar_parametric(portfolio_delta_dollar, vol, confidence=0.95, horizon_days=1):
    """Delta-normal parametric VaR/CVaR: approximates the position as a linear (delta) exposure."""
    daily_vol = vol / np.sqrt(252)
    z = norm.ppf(1 - confidence)
    var = -portfolio_delta_dollar * z * daily_vol * np.sqrt(horizon_days)
    cvar = portfolio_delta_dollar * daily_vol * np.sqrt(horizon_days) * norm.pdf(z) / (1 - confidence)
    return float(var), float(abs(cvar))


def var_cvar_historical(log_returns, portfolio_delta_dollar, confidence=0.95):
    """Historical-simulation VaR/CVaR: applies real historical daily return shocks to current delta exposure."""
    r = log_returns.dropna()
    if len(r) < 30:
        return np.nan, np.nan
    pnl_scenarios = portfolio_delta_dollar * r.values
    var = -np.percentile(pnl_scenarios, (1 - confidence) * 100)
    tail = pnl_scenarios[pnl_scenarios <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def stress_test(legs, spot, rate, div_yield, vol, tau):
    """Full repricing (not just Taylor approximation) of the whole leg book under shock scenarios."""
    scenarios = [
        ("Spot -10%", spot * 0.90, vol, rate),
        ("Spot -5%", spot * 0.95, vol, rate),
        ("Base case", spot, vol, rate),
        ("Spot +5%", spot * 1.05, vol, rate),
        ("Spot +10%", spot * 1.10, vol, rate),
        ("Vol +50% (spike)", spot, vol * 1.5, rate),
        ("Vol -30% (crush)", spot, vol * 0.7, rate),
        ("Rate +100bps", spot, vol, rate + 0.01),
    ]
    rows = []
    base_value = sum(leg["position"] * bsm_price(spot, leg["strike"], rate, div_yield, vol, tau, leg["type"])
                      for leg in legs)
    for label, s, v, r in scenarios:
        value = sum(leg["position"] * bsm_price(s, leg["strike"], r, div_yield, max(v, 1e-4), tau, leg["type"])
                    for leg in legs)
        rows.append({"scenario": label, "portfolio_value": value, "pnl_vs_base": value - base_value})
    return pd.DataFrame(rows)


# =============================================================================
# 8. BACKTEST LAYER — historical vol-signal backtest + walk-forward validation
# =============================================================================

def historical_vol_signal_backtest(hist: pd.DataFrame, forecast_window=20, realized_window=20,
                                    z_threshold=0.75, train_frac=0.5):
    """
    Since free historical options-chain data isn't available, this backtests
    a volatility-trading signal directly on price data using a variance-swap
    -style proxy: at each rebalance date, forecast near-term vol (EWMA), then
    compare it to the vol that was ACTUALLY realized over the following
    window. If forecast vol was priced "as if it were implied vol," selling
    vol when forecast > subsequent realized (and buying when forecast <
    realized) is profitable -- this proxies a real implied-vs-realized vol
    trade without needing paid historical options data.

    Walk-forward: the z-score threshold's mean/std are calibrated on the
    first `train_frac` of the data only, then applied out-of-sample on the
    remainder, so the signal isn't fit and tested on the same period.
    """
    df = hist.copy()
    df["fcast_vol"] = df["log_return"].rolling(forecast_window).std() * np.sqrt(252)
    df["realized_fwd_vol"] = df["log_return"].rolling(realized_window).std().shift(-realized_window) * np.sqrt(252)
    df["vol_gap"] = df["fcast_vol"] - df["realized_fwd_vol"]
    df = df.dropna(subset=["fcast_vol", "realized_fwd_vol", "vol_gap"])
    if len(df) < 50:
        return pd.DataFrame(), {}

    split = int(len(df) * train_frac)
    train_mean, train_std = df["vol_gap"].iloc[:split].mean(), df["vol_gap"].iloc[:split].std()
    test = df.iloc[split:].copy()
    if train_std == 0 or np.isnan(train_std):
        return pd.DataFrame(), {}

    test["z"] = (test["vol_gap"] - train_mean) / train_std
    test["signal"] = np.where(test["z"] > z_threshold, "sell_vol",
                        np.where(test["z"] < -z_threshold, "buy_vol", "flat"))

    # Payoff proxy: a variance-swap-like P&L = notional * (forecast_vol^2 - realized_vol^2)
    # Selling vol profits when forecast (what you "sold" implied at) > what actually realized.
    notional = 1.0
    test["pnl"] = np.where(
        test["signal"] == "sell_vol", notional * (test["fcast_vol"] ** 2 - test["realized_fwd_vol"] ** 2),
        np.where(test["signal"] == "buy_vol", notional * (test["realized_fwd_vol"] ** 2 - test["fcast_vol"] ** 2), 0.0)
    )

    stats = performance_stats(test["pnl"])
    return test[["fcast_vol", "realized_fwd_vol", "z", "signal", "pnl"]], stats


def performance_stats(pnl_series: pd.Series) -> dict:
    pnl = pnl_series.dropna()
    if len(pnl) < 5:
        return {}
    active = pnl[pnl != 0]
    mean, std = pnl.mean(), pnl.std()
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else np.nan
    downside = pnl[pnl < 0].std()
    sortino = (mean / downside) * np.sqrt(252) if downside and downside > 0 else np.nan
    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()
    calmar = (mean * 252) / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan
    return {
        "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "Max Drawdown": max_dd, "Win Rate": win_rate,
        "Total P&L (proxy units)": pnl.sum(), "Num Active Signals": len(active),
    }


# =============================================================================
# 9. TRACK RECORD DATABASE — logs every recommendation for later evaluation
# =============================================================================

DB_PATH = "optika_track_record.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, ticker TEXT, asset_class TEXT,
            verdict TEXT, score REAL, spot_at_call REAL
        )
    """)
    conn.commit()
    conn.close()


def log_recommendation(ticker, asset_class, verdict, score, spot):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO recommendations (timestamp, ticker, asset_class, verdict, score, spot_at_call) VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), ticker, asset_class, verdict, score, spot),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_recommendation_history(ticker=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM recommendations"
        params = ()
        if ticker:
            query += " WHERE ticker = ?"
            params = (ticker,)
        query += " ORDER BY timestamp DESC LIMIT 50"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# 10. FUTURES / FORWARD FAIR VALUE (cost of carry)
# =============================================================================

def theoretical_forward_price(spot, rate, div_yield, tau) -> float:
    return spot * np.exp((rate - div_yield) * tau)


def generate_summary_report(ticker, ccy, spot, rate, div_yield, source,
                             h_vol, e_vol, g_vol, iv, ml_vol, mf_iv,
                             have_options, expiry, strike, option_type, tau,
                             bsm_p, binom_p, mc_p, greeks1, greeks2,
                             heston_p, jump_p, hedge_pnl,
                             fwd, market_futures,
                             opt_verdict, opt_notes, fut_verdict, fut_notes,
                             bt_stats) -> str:
    """Compiles every computed result across all tabs into one plain-text/markdown report."""
    lines = []
    lines.append("=" * 70)
    lines.append("OPTIKA — ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Ticker: {ticker}")
    lines.append(f"Data source: {source}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("1. MARKET SNAPSHOT")
    lines.append("-" * 70)
    lines.append(f"Spot price:        {ccy}{spot:,.2f}")
    lines.append(f"Risk-free rate:    {rate:.2%}")
    lines.append(f"Dividend yield:    {div_yield:.2%}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("2. VOLATILITY ESTIMATES")
    lines.append("-" * 70)
    lines.append(f"Historical vol (1y):     {h_vol:.2%}" if not np.isnan(h_vol) else "Historical vol: N/A")
    lines.append(f"EWMA vol:                {e_vol:.2%}" if not np.isnan(e_vol) else "EWMA vol: N/A")
    lines.append(f"GARCH(1,1) forecast:     {g_vol:.2%}" if not np.isnan(g_vol) else "GARCH forecast: N/A")
    lines.append(f"ML (Random Forest) fcst: {ml_vol:.2%}" if not np.isnan(ml_vol) else "ML forecast: N/A")
    lines.append(f"Implied vol (ATM):       {iv:.2%}" if not np.isnan(iv) else "Implied vol: N/A")
    lines.append(f"Model-free IV (VIX-style): {mf_iv:.2%}" if not np.isnan(mf_iv) else "Model-free IV: N/A")
    lines.append("")

    if have_options:
        lines.append("-" * 70)
        lines.append("3. OPTION CONTRACT & PRICING")
        lines.append("-" * 70)
        lines.append(f"Expiry: {expiry}   Strike: {ccy}{strike:.2f}   Type: {option_type}   "
                      f"Days to expiry: {int(tau * 365)}")
        lines.append(f"Black-Scholes-Merton:     {ccy}{bsm_p:.2f}")
        lines.append(f"Binomial (American):      {ccy}{binom_p:.2f}")
        lines.append(f"Monte Carlo:              {ccy}{mc_p:.2f}")
        lines.append(f"Heston (stochastic vol):  {ccy}{heston_p:.2f}" if heston_p is not None else "")
        lines.append(f"Merton (jump-diffusion):  {ccy}{jump_p:.2f}" if jump_p is not None else "")
        lines.append("")
        lines.append("Greeks:")
        lines.append(f"  Delta: {greeks1['delta']:.4f}   Gamma: {greeks1['gamma']:.5f}   "
                      f"Vega: {greeks1['vega']:.4f}   Theta: {greeks1['theta']:.4f}   Rho: {greeks1['rho']:.4f}")
        lines.append(f"  Vanna: {greeks2['vanna']:.5f}   Volga: {greeks2['volga']:.5f}   Charm: {greeks2['charm']:.5f}")
        lines.append("")
        lines.append(f"Simulated delta-hedge P&L (selling & hedging, illustrative path): {ccy}{hedge_pnl:.2f}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("4. FUTURES / FORWARD FAIR VALUE")
    lines.append("-" * 70)
    lines.append(f"Theoretical forward price: {ccy}{fwd:.2f}")
    if market_futures and market_futures > 0:
        lines.append(f"Market futures quote:      {ccy}{market_futures:.2f}")
    lines.append("")

    if bt_stats:
        lines.append("-" * 70)
        lines.append("5. VOLATILITY-SIGNAL BACKTEST (walk-forward, proxy P&L)")
        lines.append("-" * 70)
        for k, v in bt_stats.items():
            lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("6. FINAL RECOMMENDATIONS")
    lines.append("-" * 70)
    if opt_verdict:
        lines.append(f"OPTIONS: {opt_verdict}")
        for n in opt_notes:
            lines.append(f"  - {n}")
        lines.append("")
    lines.append(f"FUTURES: {fut_verdict}")
    for n in fut_notes:
        lines.append(f"  - {n}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("DISCLAIMER: This report is generated by transparent, rule-based heuristics")
    lines.append("for educational purposes. It is NOT financial advice, does not account for")
    lines.append("individual risk tolerance or portfolio context, and should not be the sole")
    lines.append("basis for a real trade.")
    lines.append("=" * 70)

    return "\n".join(lines)


def parse_expiry_date(expiry_str: str) -> datetime:
    """Handles both Yahoo's 'YYYY-MM-DD' and NSE India's 'DD-Mon-YYYY' expiry formats."""
    s = str(expiry_str).strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:11] if fmt == "%d-%b-%Y" else s[:10], fmt)
        except ValueError:
            continue
    return datetime.now() + timedelta(days=30)  # safe fallback so the app never crashes on a bad date string


# =============================================================================
# UPLOADED OPTIONS CHAIN — CSV/Excel parsing with auto column-mapping
# =============================================================================

COLUMN_ALIASES = {
    "expiry": ["expiry", "expiry date", "expirydate", "exp date", "exp_date", "expiration"],
    "strike": ["strike", "strike price", "strikeprice", "strk"],
    "type": ["type", "option type", "opt type", "cp", "ce/pe", "instrument"],
    "lastPrice": ["lastprice", "last price", "ltp", "close", "price", "close price"],
    "bid": ["bid", "bid price", "bidprice", "bid qty x price", "buy price"],
    "ask": ["ask", "ask price", "askprice", "offer", "sell price"],
    "volume": ["volume", "vol", "traded volume", "totaltradedvolume", "contracts traded"],
    "openInterest": ["openinterest", "open interest", "oi"],
}


def read_uploaded_chain(uploaded_file) -> pd.DataFrame:
    """Reads a CSV or Excel file into a raw DataFrame, whatever its original column names are."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def auto_map_columns(columns) -> dict:
    """Best-effort match of uploaded column names to our required fields, so most files 'just work'."""
    cols_lower = {str(c).strip().lower(): c for c in columns}
    mapping = {}
    for field, aliases in COLUMN_ALIASES.items():
        found = None
        for alias in aliases:
            if alias in cols_lower:
                found = cols_lower[alias]
                break
        if not found:
            for alias in aliases:
                match = next((orig for low, orig in cols_lower.items() if alias in low), None)
                if match:
                    found = match
                    break
        mapping[field] = found
    return mapping


def normalize_uploaded_chain(raw_df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Builds our standard [expiry, strike, type, lastPrice, bid, ask, volume, openInterest] schema from any uploaded file."""
    out = pd.DataFrame()

    expiry_col = col_map.get("expiry")
    if expiry_col and expiry_col in raw_df.columns:
        parsed = pd.to_datetime(raw_df[expiry_col], errors="coerce")
        out["expiry"] = parsed.dt.strftime("%Y-%m-%d").fillna(raw_df[expiry_col].astype(str))
    else:
        out["expiry"] = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

    strike_col = col_map.get("strike")
    out["strike"] = pd.to_numeric(raw_df[strike_col], errors="coerce") if strike_col in raw_df.columns else np.nan

    type_col = col_map.get("type")
    if type_col and type_col in raw_df.columns:
        raw_type = raw_df[type_col].astype(str).str.strip().str.upper()
        type_map = {"CE": "call", "CALL": "call", "C": "call", "PE": "put", "PUT": "put", "P": "put"}
        out["type"] = raw_type.map(type_map).fillna(raw_type.str.lower())
    else:
        out["type"] = "call"

    for field in ["lastPrice", "bid", "ask", "volume", "openInterest"]:
        col = col_map.get(field)
        out[field] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0.0) if col in raw_df.columns else 0.0

    return out.dropna(subset=["strike"])


# =============================================================================
# 11. RECOMMENDATION ENGINE (rule-based, educational, transparent)
# =============================================================================

def generate_options_recommendation(iv, forecast_vol, hist_vol, trend_signal, moneyness, days_to_expiry):
    notes = []
    score = 0

    if not np.isnan(iv) and not np.isnan(forecast_vol) and forecast_vol > 0:
        vol_gap = (iv - forecast_vol) / forecast_vol
        if vol_gap > 0.15:
            score -= 2
            notes.append(f"Implied vol ({iv:.1%}) is {vol_gap:.0%} above the forecast ({forecast_vol:.1%}) "
                          f"— options look **rich**. Favors premium-selling strategies over buying options outright.")
        elif vol_gap < -0.15:
            score += 2
            notes.append(f"Implied vol ({iv:.1%}) is {abs(vol_gap):.0%} below the forecast ({forecast_vol:.1%}) "
                          f"— options look **cheap**. Favors premium-buying strategies over selling.")
        else:
            notes.append(f"Implied vol ({iv:.1%}) is roughly in line with the forecast ({forecast_vol:.1%}) "
                          f"— no strong mispricing signal either way.")
    else:
        notes.append("Not enough data to compare implied vs forecast volatility.")

    if trend_signal == "up":
        notes.append("Price is in a short-term uptrend (50-day MA above 200-day MA) — bullish directional bias.")
        score += 1
    elif trend_signal == "down":
        notes.append("Price is in a short-term downtrend (50-day MA below 200-day MA) — bearish directional bias.")
        score -= 1
    else:
        notes.append("No clear trend (50/200-day MAs are close together) — limited directional edge.")

    if days_to_expiry < 14:
        notes.append(f"Only {days_to_expiry} days to expiry — theta decay accelerates fast; "
                      f"cuts against buying premium, favors selling it.")
        score -= 1

    if abs(moneyness - 1) > 0.1:
        notes.append(f"Strike is {abs(moneyness - 1):.0%} away from spot — a directional, "
                      f"lower-probability bet rather than a pure volatility trade.")

    if score >= 2:
        verdict = "LEAN: Consider BUYING options (vol looks cheap / setup favors long premium)"
    elif score <= -2:
        verdict = "LEAN: Consider SELLING options / premium (vol looks rich / setup favors short premium)"
    else:
        verdict = "LEAN: NEUTRAL — no strong edge either way"

    return verdict, notes, score


def generate_futures_recommendation(spot, theoretical_fwd, market_futures, rate, div_yield,
                                     trend_signal, tau, ccy="$"):
    notes = []
    score = 0

    carry = rate - div_yield
    if carry > 0:
        notes.append(f"Carry is positive (rate {rate:.2%} > dividend yield {div_yield:.2%}) — **contango**: "
                      f"theoretical futures price ({ccy}{theoretical_fwd:.2f}) sits above spot ({ccy}{spot:.2f}).")
    else:
        notes.append(f"Carry is negative (dividend yield {div_yield:.2%} > rate {rate:.2%}) — **backwardation**: "
                      f"theoretical futures price ({ccy}{theoretical_fwd:.2f}) sits below spot ({ccy}{spot:.2f}).")

    arbitrage_flag = False
    if market_futures and market_futures > 0:
        mispricing = (market_futures - theoretical_fwd) / theoretical_fwd
        if abs(mispricing) > 0.005:
            arbitrage_flag = True
            if mispricing > 0:
                score += 2
                notes.append(f"Market futures price ({ccy}{market_futures:.2f}) is {mispricing:+.2%} above fair value "
                              f"— theoretically favors cash-and-carry (buy spot, sell futures), subject to real costs.")
            else:
                score -= 2
                notes.append(f"Market futures price ({ccy}{market_futures:.2f}) is {mispricing:+.2%} below fair value "
                              f"— theoretically favors reverse cash-and-carry (sell spot, buy futures).")
        else:
            notes.append(f"Market futures price ({ccy}{market_futures:.2f}) is close to fair value "
                          f"({mispricing:+.2%}) — no meaningful carry arbitrage.")
    else:
        notes.append("No market futures quote supplied — reflects directional lean only, not carry-arbitrage.")

    if trend_signal == "up":
        score += 1
        notes.append("Underlying uptrend (50>200 MA) — directional bias leans long futures.")
    elif trend_signal == "down":
        score -= 1
        notes.append("Underlying downtrend (50<200 MA) — directional bias leans short futures.")
    else:
        notes.append("No clear trend — limited directional edge for futures.")

    if arbitrage_flag:
        verdict = "ARBITRAGE LEAN: BUY futures (sell spot)" if score > 0 else "ARBITRAGE LEAN: SELL futures (buy spot)"
    elif score >= 2:
        verdict = "LEAN: Consider LONG futures"
    elif score <= -2:
        verdict = "LEAN: Consider SHORT futures"
    else:
        verdict = "LEAN: NEUTRAL — no strong carry or trend edge"

    return verdict, notes, score


# =============================================================================
# 12. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Optika | Derivatives Intelligence", layout="wide", page_icon="◈")
init_db()

# --- Custom theme: typography, spacing, card styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', -apple-system, sans-serif; }

.optika-hero {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 55%, #22315a 100%);
    padding: 34px 40px; border-radius: 16px; margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(15, 23, 41, 0.18);
}
.optika-hero h1 {
    color: #ffffff; margin: 0; font-size: 2.4rem; font-weight: 700;
    letter-spacing: -0.02em; display: flex; align-items: center; gap: 12px;
}
.optika-hero .logo-mark {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #5b8def 0%, #7c5cff 100%);
    font-size: 1.5rem; font-weight: 700; color: white;
}
.optika-hero p {
    color: #a9b6d4; margin: 10px 0 0 0; font-size: 1.02rem; max-width: 720px; line-height: 1.5;
}
.optika-badge {
    display: inline-block; background: rgba(124, 92, 255, 0.15); color: #b8a9ff;
    padding: 3px 11px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    margin-top: 14px; letter-spacing: 0.03em; border: 1px solid rgba(124, 92, 255, 0.3);
}
[data-testid="stMetric"] {
    background: #f8f9fc; border: 1px solid #e8eaf0; border-radius: 12px;
    padding: 14px 16px 10px 16px;
}
[data-testid="stMetricLabel"] { font-weight: 500; color: #5a6478; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 10px 16px; font-weight: 500;
}
footer, #MainMenu { visibility: hidden; }
.optika-footer {
    text-align: center; color: #9aa3b5; font-size: 0.82rem; margin-top: 48px;
    padding-top: 20px; border-top: 1px solid #eaecf0;
}

/* --- Section headers: consistent branded accent instead of plain markdown h3 --- */
h3 {
    font-weight: 700 !important; color: #1a1a2e !important; letter-spacing: -0.01em;
    padding-left: 12px; border-left: 4px solid #7c5cff; margin-top: 28px !important;
}
h4 { font-weight: 600 !important; color: #2a2f45 !important; }

/* --- Primary buttons: brand gradient instead of default red --- */
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: linear-gradient(135deg, #5b8def 0%, #7c5cff 100%) !important;
    border: none !important; font-weight: 600 !important; color: white !important;
    box-shadow: 0 2px 8px rgba(124, 92, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    box-shadow: 0 4px 14px rgba(124, 92, 255, 0.38) !important; transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important; font-weight: 500 !important;
}

/* --- Tabs: branded underline instead of default red --- */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #7c5cff !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: #7c5cff !important; }
.stTabs [data-baseweb="tab-border"] { background-color: #eaecf0 !important; }

/* --- Radio / segmented controls --- */
div[role="radiogroup"] label {
    background: #f8f9fc; border: 1px solid #e8eaf0; border-radius: 8px;
    padding: 6px 14px; margin-right: 6px !important;
}

/* --- Data editor / dataframe polish --- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border-radius: 10px; overflow: hidden; border: 1px solid #e8eaf0;
}

/* --- Slider accent --- */
.stSlider [role="slider"] { background-color: #7c5cff !important; }

/* --- Sidebar polish --- */
[data-testid="stSidebar"] { background: #fafbfd; border-right: 1px solid #eaecf0; }
[data-testid="stSidebar"] h3 { border-left: none; padding-left: 0; color: #7c5cff !important; }

/* --- Alert boxes: rounded, softer borders --- */
[data-testid="stAlert"] { border-radius: 10px; }

/* --- Selectbox / input focus accent --- */
.stSelectbox [data-baseweb="select"]:focus-within,
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #7c5cff !important; box-shadow: 0 0 0 1px #7c5cff !important;
}
</style>
""", unsafe_allow_html=True)

# --- Hero header ---
st.markdown("""
<div class="optika-hero">
    <h1><span class="logo-mark">Ø</span> Optika</h1>
    <p>A quantitative derivatives intelligence engine — live options &amp; futures pricing, volatility
    forecasting, hedging simulation, and risk analytics for global tickers and NSE-listed Indian equities.</p>
    <span class="optika-badge">EDUCATIONAL TOOL · NOT FINANCIAL ADVICE</span>
</div>
""", unsafe_allow_html=True)

# --- About / methodology sidebar ---
with st.sidebar:
    st.markdown("### ◈ About Optika")
    st.markdown(
        "Optika prices and risk-manages options and futures using classical and modern "
        "quantitative finance methods, cross-checked against live market data."
    )
    st.markdown("**Pricing models**")
    st.markdown("- Black-Scholes-Merton (closed-form)\n- Binomial tree (American exercise)\n"
                "- Monte Carlo simulation\n- Heston (stochastic volatility)\n- Merton (jump-diffusion)")
    st.markdown("**Volatility**")
    st.markdown("- Historical, EWMA, GARCH(1,1)\n- Random Forest ML forecast\n"
                "- Implied vol (Brent inversion)\n- Model-free (VIX-style) IV\n- Full 3D vol surface")
    st.markdown("**Risk & hedging**")
    st.markdown("- 1st & 2nd-order Greeks\n- Dynamic delta-hedging\n"
                "- Parametric & historical VaR/CVaR\n- Scenario stress testing")
    st.markdown("**Data sources**")
    st.markdown("Yahoo Finance (primary) → NSE India (fallback for Indian equities) → Stooq (fallback)")
    st.markdown("---")
    st.caption("Built with Python, Streamlit, SciPy, scikit-learn & Plotly. All recommendations are "
               "transparent rule-based heuristics for learning purposes — not investment advice.")

# --- Search bar (company name OR ticker) ---
col_search, col_go = st.columns([4, 1])
with col_search:
    query = st.text_input("🔍 Search by company name or ticker (e.g. Apple, Reliance, ICICI Bank, TCS, SPY)",
                           value="Reliance").strip()
with col_go:
    st.write("")
    st.write("")
    run = st.button("Run Analysis", type="primary")

ticker_input = None
if query:
    candidates = search_ticker(query)
    if len(candidates) > 1:
        labels = [c[1] for c in candidates]
        chosen_label = st.selectbox("Did you mean:", labels, index=0)
        ticker_input = candidates[labels.index(chosen_label)][0]
    elif candidates:
        ticker_input = candidates[0][0]
        st.caption(f"Matched to **{candidates[0][1]}**")

if run or ticker_input:
    with st.spinner("Fetching live data (Yahoo Finance, with NSE India / Stooq fallback)..."):
        spot, hist, source = get_spot_and_history(ticker_input)

    if spot is None:
        st.error(f"Couldn't find data for '{ticker_input}' from any source (Yahoo, NSE India, Stooq). "
                 f"Check the ticker/spelling and try again — for Indian stocks, try the NSE suffix "
                 f"e.g. `RELIANCE.NS`.")
        st.stop()

    rate = get_risk_free_rate(ticker_input)
    div_yield = get_dividend_yield(ticker_input, spot)
    ccy = get_currency_symbol(ticker_input)

    st.subheader(f"{ticker_input} — Spot: {ccy}{spot:,.2f}  |  Rate: {rate:.2%}  |  Div yield: {div_yield:.2%}")
    st.caption(f"📡 Data source: **{source}**" +
               (" — NSE India live fallback (limited history)" if "NSE" in source else ""))

    expiries, exp_source, exp_note = get_expiries(ticker_input)

    tabs = st.tabs(["📈 Pricing & Volatility", "🌋 Vol Surface & Forecasting", "🧬 Advanced Models",
                     "🧮 Strategy Builder", "🛡️ Risk & Hedging", "📊 Backtest & Track Record",
                     "🎯 Final Recommendation"])

    # Shared state across tabs
    have_options = bool(expiries)
    expiry = strike = option_type = tau = None
    h_vol = e_vol = g_vol = iv = np.nan
    vol_for_pricing = np.nan
    chain = pd.DataFrame()
    market_price = np.nan
    trend = "flat"

    hist["ma50"] = hist["Close"].rolling(50).mean()
    hist["ma200"] = hist["Close"].rolling(200).mean()
    valid_ma = hist.dropna(subset=["ma50", "ma200"])
    if len(valid_ma) > 0:
        trend = "up" if valid_ma["ma50"].iloc[-1] > valid_ma["ma200"].iloc[-1] else "down"

    h_vol = historical_vol(hist["log_return"])
    e_vol = ewma_vol(hist["log_return"])

    # Pre-declared with safe defaults so the report generator (after all tabs)
    # never hits a NameError, regardless of which code paths actually ran.
    ml_vol, ml_r2, mf_iv = np.nan, np.nan, np.nan
    bsm_p = binom_p = mc_p = np.nan
    greeks1 = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    greeks2 = {"vanna": 0.0, "volga": 0.0, "charm": 0.0}
    heston_p = jump_p = None
    hedge_pnl = None
    bt_stats = {}
    opt_verdict, opt_notes = None, []
    fut_verdict, fut_notes = None, []

    # ---------------- TAB 1: Pricing & Volatility ----------------
    with tabs[0]:
        st.markdown("#### 📝 Options Data Source")
        data_mode = st.radio(
            "Choose how options data is provided for this ticker",
            ["Auto-fetch (Yahoo Finance → NSE India)", "Upload CSV/Excel or enter manually"],
            index=1 if not have_options else 0,
            horizontal=True,
            key=f"data_mode_{ticker_input}",
            label_visibility="collapsed",
        )

        manual_chain_all = None
        if data_mode == "Upload CSV/Excel or enter manually":
            state_key = f"manual_chain_{ticker_input}"

            uploaded_file = st.file_uploader(
                "Upload an options chain (CSV or Excel — e.g. exported from your broker or NSE's website)",
                type=["csv", "xlsx", "xls"], key=f"uploader_{ticker_input}",
            )

            if uploaded_file is not None:
                try:
                    raw_df = read_uploaded_chain(uploaded_file)
                    col_map = auto_map_columns(raw_df.columns)

                    with st.expander("Column mapping (auto-detected — adjust if anything looks wrong)", expanded=False):
                        cols_available = ["(none)"] + list(raw_df.columns)
                        new_map = {}
                        for field in ["expiry", "strike", "type", "lastPrice", "bid", "ask", "volume", "openInterest"]:
                            default = col_map.get(field) or "(none)"
                            default_idx = cols_available.index(default) if default in cols_available else 0
                            chosen = st.selectbox(field, cols_available, index=default_idx, key=f"map_{field}_{ticker_input}")
                            new_map[field] = None if chosen == "(none)" else chosen
                        col_map = new_map

                    normalized = normalize_uploaded_chain(raw_df, col_map)
                    if normalized.empty:
                        st.error("Couldn't parse any valid rows (need at least a Strike column with numeric values). "
                                 "Check the column mapping above.")
                    else:
                        st.session_state[state_key] = normalized
                        st.success(f"Loaded {len(normalized)} contracts from '{uploaded_file.name}'. "
                                   f"Review/edit below before analyzing.")
                except Exception as e:
                    st.error(f"Couldn't read this file ({type(e).__name__}). Make sure it's a valid CSV or Excel file.")

            if state_key not in st.session_state:
                st.session_state[state_key] = pd.DataFrame({
                    "expiry": [(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")] * 2,
                    "strike": [round(spot * 0.95, 2), round(spot * 1.05, 2)],
                    "type": ["call", "put"],
                    "lastPrice": [0.0, 0.0],
                    "bid": [0.0, 0.0],
                    "ask": [0.0, 0.0],
                    "volume": [0, 0],
                    "openInterest": [0, 0],
                })

            st.caption("Review or edit the table below (uploaded rows are pre-filled here) — Strike, Type, "
                       "Expiry, and Last Price are required; Bid/Ask/Volume/OI are optional but improve accuracy.")
            edited = st.data_editor(
                st.session_state[state_key],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "expiry": st.column_config.TextColumn("Expiry (YYYY-MM-DD)"),
                    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "type": st.column_config.SelectboxColumn("Type", options=["call", "put"]),
                    "lastPrice": st.column_config.NumberColumn("Last Price", format="%.2f"),
                    "bid": st.column_config.NumberColumn("Bid", format="%.2f"),
                    "ask": st.column_config.NumberColumn("Ask", format="%.2f"),
                    "volume": st.column_config.NumberColumn("Volume"),
                    "openInterest": st.column_config.NumberColumn("Open Interest"),
                },
                key=f"editor_{ticker_input}",
            )
            st.session_state[state_key] = edited

            manual_chain_all = edited.dropna(subset=["strike", "type", "expiry"]).copy()
            manual_chain_all = manual_chain_all[manual_chain_all["strike"] > 0]
            if not manual_chain_all.empty:
                manual_chain_all["strike"] = manual_chain_all["strike"].astype(float)
                manual_chain_all["lastPrice"] = pd.to_numeric(manual_chain_all["lastPrice"], errors="coerce").fillna(0.0)
                manual_chain_all["expiry"] = manual_chain_all["expiry"].astype(str)
                expiries = sorted(manual_chain_all["expiry"].unique())
                exp_source = "Manual Entry"
                have_options = True
            else:
                have_options = False

        st.markdown("---")

        if not have_options:
            if data_mode == "Upload CSV/Excel or enter manually":
                st.info("Upload a file above, or add at least one valid row in the table (Strike > 0, Type, "
                       "Expiry, Last Price) to enable pricing.")
            else:
                msg = f"No options data available. {exp_note}."
                if ticker_input.upper().endswith(".BO"):
                    nse_alt = ticker_input.upper().replace(".BO", ".NS")
                    msg += f" Try **{nse_alt}** — NSE-listed large caps usually have active F&O data."
                elif ticker_input.upper().endswith(".NS") and "blocked" in exp_note.lower():
                    msg += (" This is a known limitation of NSE India's free public API — it actively blocks "
                            "scripted/automated requests, especially from outside India or from cloud-hosted "
                            "servers, regardless of the stock. Reliable NSE options data generally requires a "
                            "paid/authenticated source (e.g. a broker API like Zerodha Kite or Angel One SmartAPI). "
                            "**Use the Manual Entry option above instead** — enter the option prices you see on "
                            "your broker's app and Optika will price/hedge/analyze them normally.")
                st.warning(msg + " Showing price/volatility analysis only.")

            with st.spinner("Fitting GARCH(1,1)..."):
                g_vol = garch_forecast_vol(hist["log_return"])
            v1, v2, v3 = st.columns(3)
            v1.metric("Historical Vol (1y)", f"{h_vol:.1%}" if not np.isnan(h_vol) else "N/A")
            v2.metric("EWMA Vol", f"{e_vol:.1%}" if not np.isnan(e_vol) else "N/A")
            v3.metric("GARCH Forecast Vol", f"{g_vol:.1%}" if not np.isnan(g_vol) else "N/A")
            st.line_chart(hist["Close"])
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                expiry = st.selectbox("Expiry", expiries, index=min(2, len(expiries) - 1))

            if data_mode == "Upload CSV/Excel or enter manually":
                chain = manual_chain_all[manual_chain_all["expiry"] == str(expiry)].copy()
            else:
                chain = get_options_chain(ticker_input, expiry, exp_source)

            strikes = sorted(chain["strike"].unique()) if not chain.empty else []
            default_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot)) if strikes else 0
            with c2:
                strike = st.selectbox("Strike", strikes, index=default_idx) if strikes else st.number_input("Strike", value=float(spot))
            with c3:
                option_type = st.selectbox("Option type", ["call", "put"])

            expiry_str = str(expiry)[:11]
            tau = max((parse_expiry_date(expiry_str) - datetime.now()).days, 1) / 365

            with st.spinner("Fitting GARCH(1,1)..."):
                g_vol = garch_forecast_vol(hist["log_return"])

            row = chain[(chain["strike"] == strike) & (chain["type"] == option_type)]
            market_price = float(row["lastPrice"].iloc[0]) if not row.empty else np.nan
            iv = implied_vol(market_price, spot, strike, rate, div_yield, tau, option_type) if not np.isnan(market_price) else np.nan
            vol_for_pricing = iv if not np.isnan(iv) else h_vol

            st.markdown("### Volatility Estimates")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Historical Vol (1y)", f"{h_vol:.1%}" if not np.isnan(h_vol) else "N/A")
            v2.metric("EWMA Vol", f"{e_vol:.1%}" if not np.isnan(e_vol) else "N/A")
            v3.metric("GARCH Forecast Vol", f"{g_vol:.1%}" if not np.isnan(g_vol) else "N/A")
            v4.metric("Implied Vol (this option)", f"{iv:.1%}" if not np.isnan(iv) else "N/A")

            st.markdown("### Option Pricing — 3 Classical Methods")
            bsm_p = bsm_price(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)
            binom_p = binomial_price(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type, "american", 300)
            mc_p, mc_err = monte_carlo_price(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Black-Scholes-Merton", f"{ccy}{bsm_p:.2f}")
            p2.metric("Binomial Tree (American)", f"{ccy}{binom_p:.2f}")
            p3.metric("Monte Carlo", f"{ccy}{mc_p:.2f}", f"±{mc_err:.3f} SE")
            p4.metric("Market Price", f"{ccy}{market_price:.2f}" if not np.isnan(market_price) else "N/A")

            st.markdown("### Greeks — First & Second Order")
            greeks1 = bsm_greeks(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)
            greeks2 = bsm_greeks_second_order(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            gc1.metric("Delta", f"{greeks1['delta']:.3f}")
            gc2.metric("Gamma", f"{greeks1['gamma']:.4f}")
            gc3.metric("Vega (per 1% vol)", f"{greeks1['vega']:.3f}")
            gc4.metric("Theta (per day)", f"{greeks1['theta']:.3f}")
            gc5.metric("Rho (per 1% rate)", f"{greeks1['rho']:.3f}")
            gc6, gc7, gc8 = st.columns(3)
            gc6.metric("Vanna", f"{greeks2['vanna']:.4f}", help="d(Delta)/d(Vol) — how delta shifts as volatility moves")
            gc7.metric("Volga", f"{greeks2['volga']:.4f}", help="d(Vega)/d(Vol) — how vega shifts as volatility moves")
            gc8.metric("Charm", f"{greeks2['charm']:.4f}", help="d(Delta)/d(Time) — how delta decays as time passes")

            with st.expander("Show price history"):
                st.line_chart(hist["Close"])

    # ---------------- TAB 2: Vol Surface & Forecasting ----------------
    with tabs[1]:
        st.markdown("### Machine-Learning Volatility Forecast")
        if HAS_SKLEARN:
            ml_vol, ml_r2 = ml_vol_forecast(hist["log_return"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Random Forest Forecast Vol", f"{ml_vol:.1%}" if not np.isnan(ml_vol) else "N/A")
            mc2.metric("GARCH Forecast Vol", f"{g_vol:.1%}" if not np.isnan(g_vol) else "N/A")
            mc3.metric("ML Out-of-Sample R²", f"{ml_r2:.2f}" if not np.isnan(ml_r2) else "N/A")
            st.caption("R² compares the Random Forest's held-out prediction accuracy for 20-day-forward "
                       "realized volatility — closer to 1.0 means a better forecast; near 0 or negative "
                       "means it's barely beating a flat-average guess.")
        else:
            st.info("Install `scikit-learn` (`pip install scikit-learn`) to enable the ML volatility forecaster.")

        if have_options:
            st.markdown("### Model-Free Implied Volatility (VIX-style)")
            mf_iv = model_free_implied_vol(chain, spot, rate, tau)
            mf1, mf2 = st.columns(2)
            mf1.metric("Model-Free IV (this expiry)", f"{mf_iv:.1%}" if not np.isnan(mf_iv) else "N/A")
            mf2.metric("Single-Strike ATM Implied Vol", f"{iv:.1%}" if not np.isnan(iv) else "N/A")
            st.caption("Model-free IV uses a variance-weighted average across ALL out-of-the-money strikes "
                       "(the CBOE VIX methodology) rather than one strike's noisy quote — generally more robust.")

            st.markdown("### Full Implied Volatility Surface (Strike × Maturity)")
            if st.button("Build 3D Volatility Surface (may take a moment for auto-fetched data)"):
                with st.spinner("Building surface across expiries..."):
                    if data_mode == "Upload CSV/Excel or enter manually":
                        surf_rows = []
                        for _, r in manual_chain_all.iterrows():
                            mp = r.get("lastPrice", np.nan)
                            if pd.isna(mp) or mp <= 0:
                                continue
                            r_tau = max((parse_expiry_date(str(r["expiry"])[:11]) - datetime.now()).days, 1) / 365
                            r_iv = implied_vol(mp, spot, r["strike"], rate, div_yield, r_tau, r["type"])
                            if not np.isnan(r_iv) and 0.01 < r_iv < 3:
                                surf_rows.append({"expiry": r["expiry"], "tau": r_tau, "strike": r["strike"], "iv": r_iv})
                        surface_df = pd.DataFrame(surf_rows)
                    else:
                        surface_df = build_vol_surface(ticker_input, expiries, spot, rate, div_yield, exp_source, option_type)
                if not surface_df.empty and len(surface_df["tau"].unique()) > 1:
                    fig_surf = go.Figure(data=[go.Mesh3d(
                        x=surface_df["strike"], y=surface_df["tau"] * 365, z=surface_df["iv"],
                        intensity=surface_df["iv"], colorscale="Viridis", opacity=0.85,
                    )])
                    fig_surf.update_layout(
                        scene=dict(xaxis_title="Strike", yaxis_title="Days to Expiry", zaxis_title="Implied Vol"),
                        height=550, margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig_surf, use_container_width=True)
                elif not surface_df.empty:
                    st.info("Only one expiry has valid quotes — add rows for a second expiry to render a "
                            "true 3D surface (strike × maturity). Showing the smile below instead.")
                else:
                    st.info("Not enough valid quotes across expiries to build a surface for this ticker.")

            st.markdown("### Implied Volatility Smile (selected expiry)")
            smile_rows = []
            for _, r in chain[chain["type"] == option_type].iterrows():
                mp = r.get("lastPrice", np.nan)
                if pd.isna(mp) or mp <= 0:
                    continue
                iv_k = implied_vol(mp, spot, r["strike"], rate, div_yield, tau, option_type)
                if not np.isnan(iv_k) and 0.01 < iv_k < 3:
                    smile_rows.append({"strike": r["strike"], "iv": iv_k})
            if smile_rows:
                smile_df = pd.DataFrame(smile_rows).sort_values("strike")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=smile_df["strike"], y=smile_df["iv"], mode="lines+markers"))
                fig.add_vline(x=spot, line_dash="dash", annotation_text="Spot")
                fig.update_layout(xaxis_title="Strike", yaxis_title="Implied Vol", height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough live quotes to build a smile for this expiry.")
        else:
            st.info("No options chain available for this ticker — vol surface and smile need live options data.")

    # ---------------- TAB 3: Advanced Models (Heston / Jump-Diffusion) ----------------
    with tabs[2]:
        if have_options:
            st.markdown("### Stochastic-Volatility & Jump-Diffusion Pricing")
            st.caption("BSM assumes constant volatility and continuous price paths — both are false in "
                       "real markets. These models relax those assumptions and are what actually generate "
                       "the volatility smile you see in Tab 2.")

            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                heston_rho = st.slider("Heston: spot-vol correlation (ρ)", -0.9, 0.0, -0.6, 0.1,
                                        help="Negative for equities: vol tends to rise when price falls (leverage effect)")
            with hc2:
                heston_kappa = st.slider("Heston: mean-reversion speed (κ)", 0.5, 5.0, 2.0, 0.5)
            with hc3:
                heston_vol_of_vol = st.slider("Heston: vol-of-vol (σᵥ)", 0.1, 0.8, 0.3, 0.05)

            with st.spinner("Running Heston & Merton jump-diffusion Monte Carlo..."):
                heston_p, heston_err = heston_mc_price(
                    spot, strike, rate, div_yield, tau, option_type,
                    v0=vol_for_pricing ** 2, kappa=heston_kappa,
                    theta=vol_for_pricing ** 2, sigma_v=heston_vol_of_vol, rho=heston_rho,
                )
                jump_p, jump_err = merton_jump_price(
                    spot, strike, rate, div_yield, vol_for_pricing, tau, option_type,
                )

            bsm_p_adv = bsm_price(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)
            a1, a2, a3 = st.columns(3)
            a1.metric("BSM (flat vol)", f"{ccy}{bsm_p_adv:.2f}")
            a2.metric("Heston (stochastic vol)", f"{ccy}{heston_p:.2f}", f"±{heston_err:.3f} SE")
            a3.metric("Merton (jump-diffusion)", f"{ccy}{jump_p:.2f}", f"±{jump_err:.3f} SE")

            diff_heston = (heston_p - bsm_p_adv) / bsm_p_adv * 100 if bsm_p_adv > 0 else 0
            diff_jump = (jump_p - bsm_p_adv) / bsm_p_adv * 100 if bsm_p_adv > 0 else 0
            st.markdown(f"- Heston prices this option **{diff_heston:+.1f}%** vs BSM — the gap reflects "
                        f"the extra premium/discount from allowing volatility itself to be random and "
                        f"correlated with the spot move.")
            st.markdown(f"- Merton jump-diffusion prices it **{diff_jump:+.1f}%** vs BSM — the gap reflects "
                        f"crash/jump risk that continuous GBM paths cannot capture.")
            st.caption("If market price sits closer to Heston/Merton than to flat-vol BSM, that's evidence "
                       "the market is pricing in stochastic vol or jump risk — exactly why the smile exists.")
        else:
            st.info("No options chain available for this ticker — advanced model comparison needs a priced option.")

    # ---------------- TAB 4: Strategy Builder ----------------
    with tabs[3]:
        if have_options:
            st.markdown("### Multi-Leg Strategy Payoff Builder")
            preset = st.selectbox("Choose a strategy", list(PRESET_STRATEGIES.keys()))
            legs = build_legs_from_preset(preset, spot, rate, div_yield, vol_for_pricing, tau)

            leg_df = pd.DataFrame(legs)
            leg_df["premium"] = leg_df["premium"].round(2)
            st.dataframe(leg_df, use_container_width=True)

            spot_range = np.linspace(spot * 0.7, spot * 1.3, 200)
            payoff = payoff_at_expiry(legs, spot_range)
            fig_pay = go.Figure()
            fig_pay.add_trace(go.Scatter(x=spot_range, y=payoff, mode="lines", name="P&L at expiry",
                                          line=dict(width=3)))
            fig_pay.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_pay.add_vline(x=spot, line_dash="dash", annotation_text="Current Spot")
            fig_pay.update_layout(xaxis_title=f"Underlying Price at Expiry ({ccy})",
                                   yaxis_title=f"P&L ({ccy})", height=400)
            st.plotly_chart(fig_pay, use_container_width=True)

            max_profit = payoff.max()
            max_loss = payoff.min()
            breakevens = spot_range[np.where(np.diff(np.sign(payoff)))[0]]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Max Profit (in range shown)", f"{ccy}{max_profit:.2f}")
            mc2.metric("Max Loss (in range shown)", f"{ccy}{max_loss:.2f}")
            mc3.metric("Breakeven(s)", ", ".join(f"{ccy}{b:.2f}" for b in breakevens) if len(breakevens) else "N/A")

            st.markdown("### Portfolio Greeks (aggregated across all legs)")
            pg = portfolio_greeks(legs, spot, rate, div_yield, vol_for_pricing, tau)
            pgc = st.columns(5)
            for i, k in enumerate(["delta", "gamma", "vega", "theta", "rho"]):
                pgc[i].metric(k.capitalize(), f"{pg[k]:.3f}")
            pgc2 = st.columns(3)
            for i, k in enumerate(["vanna", "volga", "charm"]):
                pgc2[i].metric(k.capitalize(), f"{pg[k]:.4f}")

            st.session_state["current_legs"] = legs
        else:
            st.info("No options chain available for this ticker — strategy builder needs live options data.")

    # ---------------- TAB 5: Risk & Hedging ----------------
    with tabs[4]:
        if have_options:
            st.markdown("### Dynamic Delta-Hedge Simulation (simulated forward path)")
            h1, h2 = st.columns(2)
            with h1:
                rehedge_every = st.slider("Rehedge every N steps", 1, 10, 1)
            with h2:
                cost_bps = st.slider("Transaction cost (bps per trade)", 0, 50, 5)

            sim_path = simulate_gbm_path(spot, rate, div_yield, vol_for_pricing, tau, n_steps=60, seed=7)
            hedge_log, final_pnl = simulate_delta_hedge(sim_path, strike, rate, div_yield, vol_for_pricing,
                                                         tau, option_type, rehedge_every, cost_bps)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=hedge_log["spot"], name="Simulated Spot Path"))
            fig2.update_layout(height=280, yaxis_title="Price", xaxis_title="Step")
            st.plotly_chart(fig2, use_container_width=True)
            st.metric("Simulated Hedge P&L (selling this option & delta-hedging)", f"{ccy}{final_pnl:.2f}")
            hedge_pnl = final_pnl

            st.markdown("### Value at Risk / Conditional VaR")
            legs_for_risk = st.session_state.get("current_legs", [
                {"type": option_type, "strike": strike, "position": -1,
                 "premium": bsm_price(spot, strike, rate, div_yield, vol_for_pricing, tau, option_type)}
            ])
            pg_risk = portfolio_greeks(legs_for_risk, spot, rate, div_yield, vol_for_pricing, tau)
            delta_dollar = pg_risk["delta"] * spot

            confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95)
            var_p, cvar_p = var_cvar_parametric(delta_dollar, vol_for_pricing, confidence)
            var_h, cvar_h = var_cvar_historical(hist["log_return"], delta_dollar, confidence)

            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown("**Parametric (delta-normal)**")
                st.metric(f"1-day VaR ({confidence:.0%})", f"{ccy}{var_p:,.2f}")
                st.metric(f"1-day CVaR ({confidence:.0%})", f"{ccy}{cvar_p:,.2f}")
            with vc2:
                st.markdown("**Historical simulation**")
                st.metric(f"1-day VaR ({confidence:.0%})", f"{ccy}{var_h:,.2f}" if not np.isnan(var_h) else "N/A")
                st.metric(f"1-day CVaR ({confidence:.0%})", f"{ccy}{cvar_h:,.2f}" if not np.isnan(cvar_h) else "N/A")
            st.caption("VaR/CVaR here approximate the CURRENT strategy from the Strategy Builder tab (uses "
                       "its net delta exposure). Build a strategy in Tab 4 first for a multi-leg risk read.")

            st.markdown("### Scenario Stress Test")
            stress_df = stress_test(legs_for_risk, spot, rate, div_yield, vol_for_pricing, tau)
            st.dataframe(stress_df.style.format({"portfolio_value": f"{ccy}{{:.2f}}", "pnl_vs_base": f"{ccy}{{:+.2f}}"}),
                         use_container_width=True)
        else:
            st.info("No options chain available for this ticker — hedging/risk tools need live options data.")

    # ---------------- TAB 6: Backtest & Track Record ----------------
    with tabs[5]:
        st.markdown("### Historical Volatility Signal Backtest (walk-forward)")
        st.caption("Free historical options-chain data isn't available, so this backtests a "
                   "variance-swap-style proxy directly on price history: forecast near-term vol, "
                   "compare it to what actually realized afterward, and simulate selling/buying vol "
                   "accordingly. The signal threshold is calibrated on the first half of history only "
                   "and tested out-of-sample on the second half (walk-forward), to avoid overfitting.")

        bc1, bc2 = st.columns(2)
        with bc1:
            z_thresh = st.slider("Signal z-score threshold", 0.25, 2.0, 0.75, 0.25)
        with bc2:
            train_frac = st.slider("Train fraction (walk-forward split)", 0.3, 0.7, 0.5, 0.1)

        bt_df, bt_stats = historical_vol_signal_backtest(hist, z_threshold=z_thresh, train_frac=train_frac)
        if not bt_df.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(y=bt_df["pnl"].cumsum(), name="Cumulative P&L (proxy units)"))
            fig_bt.update_layout(height=300, xaxis_title="Out-of-sample period", yaxis_title="Cumulative P&L")
            st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown("**Performance Stats (out-of-sample)**")
            stat_cols = st.columns(len(bt_stats))
            for i, (k, v) in enumerate(bt_stats.items()):
                if isinstance(v, float):
                    stat_cols[i].metric(k, f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
                else:
                    stat_cols[i].metric(k, str(v))
            st.caption("These stats are on a volatility-trading PROXY (not real option P&L), and Sharpe/"
                       "Sortino/Calmar are computed on proxy units, not $ returns — treat as directional "
                       "evidence of whether the signal has edge, not as a literal expected return.")
        else:
            st.info("Not enough historical data to run a meaningful backtest for this ticker.")

        st.markdown("---")
        st.markdown("### 📜 Recommendation Track Record")
        st.caption("Every verdict this engine generates gets logged here (locally, on your machine) so you "
                   "can look back and see how the calls would have played out.")
        history_df = get_recommendation_history()
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No recommendations logged yet — they're saved automatically each time you view the "
                   "Final Recommendation tab.")

    # ---------------- TAB 7: Final Recommendation ----------------
    with tabs[6]:
        st.markdown("## 🎯 Rule-Based Signals")
        fwd = theoretical_forward_price(spot, rate, div_yield, tau if tau else 30 / 365)
        f1, f2 = st.columns(2)
        f1.metric("Theoretical Forward/Futures Fair Value", f"{ccy}{fwd:.2f}")
        market_futures = f2.number_input("Market futures price (optional, if you have a quote)", value=0.0)

        rec_col1, rec_col2 = st.columns(2)

        if have_options:
            moneyness = strike / spot
            days_to_expiry = int(tau * 365)
            with rec_col1:
                st.markdown("#### 📊 Options Verdict")
                opt_verdict, opt_notes, opt_score = generate_options_recommendation(
                    iv, g_vol if not np.isnan(g_vol) else e_vol, h_vol, trend, moneyness, days_to_expiry)
                st.markdown(f"**{opt_verdict}**")
                for n in opt_notes:
                    st.markdown(f"- {n}")
                log_recommendation(ticker_input, "options", opt_verdict, opt_score, spot)
        else:
            with rec_col1:
                st.markdown("#### 📊 Options Verdict")
                st.info("No options chain available for this ticker — can't generate an options-specific verdict.")

        with rec_col2:
            st.markdown("#### 📦 Futures Verdict")
            fut_verdict, fut_notes, fut_score = generate_futures_recommendation(
                spot, fwd, market_futures, rate, div_yield, trend, tau if tau else 30 / 365, ccy)
            st.markdown(f"**{fut_verdict}**")
            for n in fut_notes:
                st.markdown(f"- {n}")
            log_recommendation(ticker_input, "futures", fut_verdict, fut_score, spot)

        st.info("⚠️ Both verdicts are transparent, rule-based heuristics for learning purposes — they show "
                "exactly which inputs (vol gap, trend, carry, time decay, moneyness) drove the call. They "
                "are **not** financial advice, don't account for your risk tolerance or portfolio, and "
                "shouldn't be the sole basis for a real trade.")

    # ---------------- FINAL REPORT (compiles results from every tab above) ----------------
    st.markdown("---")
    st.markdown("## 📄 Full Analysis Report")
    st.caption("Compiles everything computed above — market snapshot, volatility estimates, pricing, "
               "Greeks, hedge simulation, backtest stats, and both recommendations — into one downloadable "
               "report. Visit the other tabs first so their results are included; anything not yet computed "
               "shows as N/A.")

    report_text = generate_summary_report(
        ticker_input, ccy, spot, rate, div_yield, source,
        h_vol, e_vol, g_vol, iv, ml_vol, mf_iv,
        have_options, expiry, strike, option_type, tau,
        bsm_p, binom_p, mc_p, greeks1, greeks2,
        heston_p, jump_p, hedge_pnl,
        fwd, market_futures,
        opt_verdict, opt_notes, fut_verdict, fut_notes,
        bt_stats,
    )

    with st.expander("Preview report", expanded=False):
        st.text(report_text)

    st.download_button(
        label="⬇️ Download Full Report (.txt)",
        data=report_text,
        file_name=f"optika_report_{ticker_input.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )

    st.markdown(
        '<div class="optika-footer">◈ Optika — Derivatives Intelligence Engine · '
        'Built with Python, Streamlit, SciPy &amp; scikit-learn · '
        'Data: Yahoo Finance / NSE India / Stooq · Educational tool, not financial advice</div>',
        unsafe_allow_html=True,
    )