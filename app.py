"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ███████╗██████╗  ██████╗ ███████╗
║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       █████╗  ██║  ██║██║  ███╗█████╗
║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══╝  ██║  ██║██║   ██║██╔══╝
║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ███████╗██████╔╝╚██████╔╝███████╗
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
║                                                                          ║
║              QUANT EDGE — Institutional Derivatives Analytics           ║
║              Multi-Source Real-Time · Advanced Pricing · Risk           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import io, json, math, sqlite3, time, re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from scipy.integrate import quad

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = Path("quant_edge.db")

CURRENCY_SYMBOLS = {
    "USD":"$","INR":"₹","GBP":"£","EUR":"€","JPY":"¥",
    "HKD":"HK$","CAD":"C$","AUD":"A$","CNY":"¥","SGD":"S$",
    "CHF":"CHF ","KRW":"₩",
}

# Institutional color palette (WCAG-AA compliant on dark bg)
PALETTE = {
    "primary":   "#6366f1",   # indigo — brand
    "secondary": "#22d3ee",   # cyan
    "success":   "#10b981",   # emerald
    "warning":   "#f59e0b",   # amber
    "danger":    "#ef4444",   # red
    "accent":    "#a78bfa",   # violet
    "muted":     "#64748b",   # slate
    "bg":        "#0b0f1a",
    "surface":   "#151b2b",
    "border":    "#1f2937",
}

PLOTLY_TEMPLATE = "plotly_dark"

POPULAR_TICKERS = [
    ("RELIANCE.NS", "Reliance Industries (RELIANCE.NS) — NSE"),
    ("INFY.NS",     "Infosys Ltd (INFY.NS) — NSE"),
    ("TCS.NS",      "Tata Consultancy Services (TCS.NS) — NSE"),
    ("HDFCBANK.NS", "HDFC Bank (HDFCBANK.NS) — NSE"),
    ("ICICIBANK.NS","ICICI Bank (ICICIBANK.NS) — NSE"),
    ("^NSEI",       "Nifty 50 Index (^NSEI) — NSE"),
    ("^NSEBANK",    "Bank Nifty (^NSEBANK) — NSE"),
    ("AAPL",        "Apple Inc. (AAPL) — NASDAQ"),
    ("MSFT",        "Microsoft Corp. (MSFT) — NASDAQ"),
    ("TSLA",        "Tesla Inc. (TSLA) — NASDAQ"),
    ("NVDA",        "NVIDIA Corp. (NVDA) — NASDAQ"),
    ("SPY",         "SPDR S&P 500 ETF (SPY)"),
    ("QQQ",         "Invesco QQQ (QQQ)"),
    ("GC=F",        "Gold Futures (GC=F)"),
    ("BTC-USD",     "Bitcoin USD (BTC-USD)"),
]

INDIAN_COMPANIES = {
    "reliance":"RELIANCE.NS","tcs":"TCS.NS","infosys":"INFY.NS",
    "infy":"INFY.NS","hdfc":"HDFCBANK.NS","hdfc bank":"HDFCBANK.NS",
    "icici":"ICICIBANK.NS","icici bank":"ICICIBANK.NS",
    "sbi":"SBIN.NS","sbin":"SBIN.NS","state bank":"SBIN.NS",
    "wipro":"WIPRO.NS","itc":"ITC.NS","tata motors":"TATAMOTORS.NS",
    "tata steel":"TATASTEEL.NS","hindustan unilever":"HINDUNILVR.NS",
    "hul":"HINDUNILVR.NS","bharti airtel":"BHARTIARTL.NS",
    "airtel":"BHARTIARTL.NS","asian paints":"ASIANPAINT.NS",
    "maruti":"MARUTI.NS","maruti suzuki":"MARUTI.NS",
    "bajaj finance":"BAJFINANCE.NS","bajaj auto":"BAJAJ-AUTO.NS",
    "adani":"ADANIENT.NS","adani enterprises":"ADANIENT.NS",
    "adani ports":"ADANIPORTS.NS","adani green":"ADANIGREEN.NS",
    "nifty":"^NSEI","nifty50":"^NSEI","nifty 50":"^NSEI",
    "bank nifty":"^NSEBANK","banknifty":"^NSEBANK",
    "sensex":"^BSESN","axis bank":"AXISBANK.NS","axis":"AXISBANK.NS",
    "kotak":"KOTAKBANK.NS","larsen":"LT.NS","l&t":"LT.NS",
    "ongc":"ONGC.NS","coal india":"COALINDIA.NS","ntpc":"NTPC.NS",
    "power grid":"POWERGRID.NS","hcl":"HCLTECH.NS","hcl tech":"HCLTECH.NS",
    "tech mahindra":"TECHM.NS","sun pharma":"SUNPHARMA.NS",
    "dr reddy":"DRREDDY.NS","cipla":"CIPLA.NS","divis":"DIVISLAB.NS",
    "titan":"TITAN.NS","nestle":"NESTLEIND.NS","britannia":"BRITANNIA.NS",
    "zomato":"ZOMATO.NS","paytm":"PAYTM.NS","nykaa":"NYKAA.NS",
    "ioc":"IOC.NS","indian oil":"IOC.NS","bpcl":"BPCL.NS",
}

US_COMPANIES = {
    "apple":"AAPL","microsoft":"MSFT","google":"GOOGL","alphabet":"GOOGL",
    "amazon":"AMZN","tesla":"TSLA","meta":"META","facebook":"META",
    "netflix":"NFLX","nvidia":"NVDA","amd":"AMD","intel":"INTC",
    "s&p 500":"^GSPC","sp500":"^GSPC","nasdaq":"^IXIC","dow":"^DJI",
    "spy":"SPY","qqq":"QQQ","gold":"GC=F","oil":"CL=F",
    "bitcoin":"BTC-USD","ethereum":"ETH-USD",
    "berkshire":"BRK-B","jpmorgan":"JPM","visa":"V","walmart":"WMT",
}

NSE_FO_TICKERS = {
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","AXISBANK",
    "KOTAKBANK","BHARTIARTL","ITC","HINDUNILVR","LT","BAJFINANCE",
    "ASIANPAINT","MARUTI","WIPRO","HCLTECH","TECHM","SUNPHARMA",
    "TATAMOTORS","TATASTEEL","POWERGRID","NTPC","ONGC","COALINDIA",
    "ADANIENT","ADANIPORTS","BAJAJFINSV","BRITANNIA","CIPLA",
    "DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HDFCLIFE","HEROMOTOCO",
    "HINDALCO","INDUSINDBK","JSWSTEEL","M&M","NESTLEIND","SBILIFE",
    "TATACONSUM","TITAN","ULTRACEMCO","UPL","VEDL","BPCL","IOC",
}

# =============================================================================
# DATABASE
# =============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, ticker TEXT, spot REAL, iv REAL,
        hist_vol REAL, ewma_vol REAL, garch_vol REAL,
        rate REAL, div_yield REAL,
        options_verdict TEXT, futures_verdict TEXT,
        data_source TEXT
    )""")
    conn.commit()
    conn.close()

def save_snapshot(ticker, spot, iv, h_vol, e_vol, g_vol,
                   rate, div_yield, opt_v, fut_v, source):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT INTO snapshots
            (ts,ticker,spot,iv,hist_vol,ewma_vol,garch_vol,
             rate,div_yield,options_verdict,futures_verdict,data_source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), ticker, spot, iv,
             h_vol, e_vol, g_vol, rate, div_yield, opt_v, fut_v, source))
        conn.commit()
        conn.close()
    except Exception:
        pass

init_db()

# =============================================================================
# SEARCH
# =============================================================================

def search_ticker_live(query, max_results=12):
    query = query.strip()
    if not query:
        return POPULAR_TICKERS

    results, seen = [], set()
    key, key_upper = query.lower(), query.upper()

    def add(sym, label, score):
        if sym not in seen:
            results.append((sym, label, score))
            seen.add(sym)

    # Exact Indian match — highest priority
    for name, sym in INDIAN_COMPANIES.items():
        if key == name:
            add(sym, f"⭐ {name.title()} ({sym}) — NSE India", 100)

    # Exact US match
    for name, sym in US_COMPANIES.items():
        if key == name:
            add(sym, f"⭐ {name.title()} ({sym})", 95)

    # NSE F&O exact ticker
    if key_upper in NSE_FO_TICKERS:
        add(f"{key_upper}.NS", f"⭐ {key_upper} ({key_upper}.NS) — NSE F&O", 90)

    # Indian substring
    for name, sym in INDIAN_COMPANIES.items():
        if key in name and key != name:
            add(sym, f"🇮🇳 {name.title()} ({sym}) — NSE", 80)

    # NSE F&O prefix
    for fo_sym in NSE_FO_TICKERS:
        if fo_sym.startswith(key_upper) and fo_sym != key_upper:
            add(f"{fo_sym}.NS", f"🇮🇳 {fo_sym} ({fo_sym}.NS) — NSE F&O", 70)

    # US substring
    for name, sym in US_COMPANIES.items():
        if key in name and key != name:
            add(sym, f"🇺🇸 {name.title()} ({sym})", 60)

    has_strong_indian = any(s >= 80 for _, _, s in results)

    # Yahoo search — filtered when strong local match exists
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": max_results,
                    "newsCount": 0, "enableFuzzyQuery": True},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        if resp.status_code == 200:
            for q in resp.json().get("quotes", []):
                sym = q.get("symbol", "")
                name = q.get("shortname") or q.get("longname") or sym
                exch = q.get("exchDisp", "")
                if not sym or sym in seen:
                    continue
                is_indian = sym.endswith(".NS") or sym.endswith(".BO")
                if has_strong_indian and not is_indian:
                    continue
                flag = ("🇮🇳" if is_indian
                         else "🇺🇸" if exch in ["NASDAQ","NYSE","NYSEArca"]
                         else "🌐")
                score = 65 if is_indian else 40
                add(sym, f"{flag} {name} ({sym}) — {exch}", score)
    except Exception:
        pass

    for sym, label in POPULAR_TICKERS:
        if sym.upper().startswith(key_upper) and sym not in seen:
            add(sym, label, 30)

    if not results:
        add(query.upper().replace(" ", ""),
             f"{query.upper()} (as-is)", 10)

    results.sort(key=lambda x: -x[2])
    return [(s, l) for s, l, _ in results[:max_results]]

# =============================================================================
# DATA — SPOT & HISTORY
# =============================================================================

def _rt_spot_yfinance(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        p = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        if p and p > 0:
            return float(p), "Yahoo Finance (live)"
    except Exception:
        pass
    return None, None

def _rt_spot_nse(ticker):
    if not ticker.endswith(".NS"):
        return None, None
    try:
        symbol = ticker.replace(".NS", "")
        s = requests.Session()
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
             "Accept": "application/json",
             "Referer": "https://www.nseindia.com/"}
        s.get("https://www.nseindia.com/", headers=h, timeout=5)
        r = s.get(f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
                    headers=h, timeout=5)
        if r.status_code == 200:
            p = r.json().get("priceInfo", {}).get("lastPrice")
            if p:
                return float(p), "NSE India (live)"
    except Exception:
        pass
    return None, None

def _rt_spot_stooq(ticker):
    """Stooq — free, no key, decent global coverage."""
    try:
        clean = ticker.lower().replace("^","").replace("=f",".f")
        # Convert to Stooq format
        if clean.endswith(".ns"):
            clean = clean.replace(".ns", ".in")
        elif not any(clean.endswith(x) for x in [".in",".uk",".de",".jp",".f"]):
            clean = clean + ".us"
        url = f"https://stooq.com/q/l/?s={clean}&f=sd2t2ohlcv&h&e=csv"
        r = requests.get(url, timeout=5)
        if r.status_code == 200 and len(r.text) > 20:
            lines = r.text.strip().split("\n")
            if len(lines) >= 2:
                cols = lines[1].split(",")
                if len(cols) >= 7 and cols[6] not in ("N/D",""):
                    return float(cols[6]), "Stooq (live)"
    except Exception:
        pass
    return None, None

def get_realtime_spot(ticker, polygon_key="", av_key="", td_key=""):
    if ticker.endswith(".NS"):
        p, s = _rt_spot_nse(ticker)
        if p: return p, s
    p, s = _rt_spot_yfinance(ticker)
    if p: return p, s
    p, s = _rt_spot_stooq(ticker)
    if p: return p, s
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_yf_history(ticker, period="5y"):
    try:
        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if h.empty:
            return None, None, None
        h = h.copy()
        h["log_return"] = np.log(h["Close"] / h["Close"].shift(1))
        return float(h["Close"].iloc[-1]), h, "Yahoo Finance"
    except Exception:
        return None, None, None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_stooq_history(ticker):
    """Stooq CSV history — public, no key, up to 10+ years."""
    try:
        clean = ticker.lower().replace("^","").replace("=f",".f")
        if clean.endswith(".ns"):
            clean = clean.replace(".ns", ".in")
        elif not any(clean.endswith(x) for x in [".in",".uk",".de",".jp",".f"]):
            clean = clean + ".us"
        url = f"https://stooq.com/q/d/l/?s={clean}&i=d"
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "Date" in r.text[:20]:
            df = pd.read_csv(io.StringIO(r.text))
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
            return float(df["Close"].iloc[-1]), df, "Stooq"
    except Exception:
        pass
    return None, None, None

def get_spot_and_history(ticker, av_key="", td_key=""):
    """Try Yahoo → Stooq → Yahoo max period as fallbacks."""
    spot, hist, src = _fetch_yf_history(ticker, period="5y")
    if hist is not None and len(hist) >= 300:
        return spot, hist, src
    # Try Stooq
    spot2, hist2, src2 = _fetch_stooq_history(ticker)
    if hist2 is not None and (hist is None or len(hist2) > len(hist)):
        return spot2, hist2, src2
    # Last resort: Yahoo max
    if hist is None:
        spot3, hist3, src3 = _fetch_yf_history(ticker, period="max")
        if hist3 is not None:
            return spot3, hist3, src3
    return spot, hist, src

@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate():
    try:
        irx = yf.Ticker("^IRX").history(period="5d")
        if not irx.empty:
            r = float(irx["Close"].iloc[-1]) / 100
            if 0 < r < 0.25:
                return r
    except Exception:
        pass
    return 0.045

@st.cache_data(ttl=3600, show_spinner=False)
def get_dividend_yield(ticker, spot):
    try:
        info = yf.Ticker(ticker).info
        r = info.get("trailingAnnualDividendRate")
        if r and spot and r > 0:
            dy = r / spot
            if 0 <= dy < 0.5:
                return dy
        dy = info.get("dividendYield")
        if dy is None:
            return 0.0
        dy = dy if dy < 1 else dy / 100
        return dy if dy <= 0.5 else 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def get_currency_symbol(ticker):
    try:
        code = yf.Ticker(ticker).info.get("currency", "USD")
        return CURRENCY_SYMBOLS.get(code, code + " ")
    except Exception:
        return "$"

# =============================================================================
# OPTIONS
# =============================================================================

def _standardise_chain(df):
    for c in ["bid","ask","lastPrice","openInterest","volume"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "openInterest" else 0
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0)
    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce").fillna(0)
    ba = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid_price"] = np.where(ba, (df["bid"]+df["ask"])/2, df["lastPrice"])
    return df

def _get_nse_session():
    s = requests.Session()
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
         "Accept": "application/json",
         "Accept-Language": "en-US,en;q=0.9",
         "Referer": "https://www.nseindia.com/option-chain"}
    s.headers.update(h)
    try:
        s.get("https://www.nseindia.com/", timeout=6)
        time.sleep(0.3)
        s.get("https://www.nseindia.com/option-chain", timeout=6)
    except Exception:
        pass
    return s

def _options_nse_live(ticker):
    if not (ticker.endswith(".NS") or ticker in ["^NSEI","^NSEBANK"]):
        return [], {}, None, None
    if ticker == "^NSEI":
        symbol, is_idx = "NIFTY", True
    elif ticker == "^NSEBANK":
        symbol, is_idx = "BANKNIFTY", True
    else:
        symbol, is_idx = ticker.replace(".NS", ""), False
    try:
        s = _get_nse_session()
        ep = "indices" if is_idx else "equities"
        url = f"https://www.nseindia.com/api/option-chain-{ep}?symbol={symbol}"
        data = None
        for attempt in range(3):
            try:
                r = s.get(url, timeout=8)
                if r.status_code == 200 and r.text.strip():
                    try:
                        data = r.json()
                        break
                    except Exception:
                        pass
                time.sleep(0.5*(attempt+1))
            except Exception:
                time.sleep(0.5)
        if not data:
            return [], {}, None, None
        records = data.get("records", {})
        underlying = records.get("underlyingValue", 0)
        raw = records.get("data", [])
        if not raw:
            return [], {}, None, None
        rows = []
        for item in raw:
            strike = item.get("strikePrice", 0)
            exp_str = item.get("expiryDate", "")
            if not exp_str or not strike:
                continue
            try:
                exp = datetime.strptime(exp_str, "%d-%b-%Y").strftime("%Y-%m-%d")
            except Exception:
                continue
            for side, ot in [(item.get("CE"), "call"), (item.get("PE"), "put")]:
                if not side:
                    continue
                rows.append({
                    "strike": float(strike), "type": ot, "expiry": exp,
                    "lastPrice": float(side.get("lastPrice", 0) or 0),
                    "bid": float(side.get("bidprice", 0) or 0),
                    "ask": float(side.get("askPrice", 0) or 0),
                    "openInterest": int(side.get("openInterest", 0) or 0),
                    "volume": int(side.get("totalTradedVolume", 0) or 0),
                    "impliedVolatility": float(side.get("impliedVolatility", 0) or 0)/100,
                })
        if not rows:
            return [], {}, None, None
        df = _standardise_chain(pd.DataFrame(rows))
        expiries = sorted(df["expiry"].dropna().unique().tolist())
        chains = {e: df[df["expiry"]==e].copy() for e in expiries}
        return expiries, chains, "NSE India (live)", underlying
    except Exception:
        return [], {}, None, None

def _parse_yahoo_opt(item, opt_type):
    def gv(x, d=0):
        if isinstance(x, dict):
            return x.get("raw", d)
        return x if x is not None else d
    try:
        return {
            "strike": gv(item.get("strike")),
            "lastPrice": gv(item.get("lastPrice")),
            "bid": gv(item.get("bid")),
            "ask": gv(item.get("ask")),
            "openInterest": gv(item.get("openInterest")),
            "volume": gv(item.get("volume")),
            "impliedVolatility": gv(item.get("impliedVolatility")),
            "type": opt_type,
        }
    except Exception:
        return None

def _options_yfinance_robust(ticker):
    chains = {}
    try:
        tk = yf.Ticker(ticker)
        exp_list = list(tk.options)
        if exp_list:
            for e in exp_list[:8]:
                try:
                    oc = tk.option_chain(e)
                    calls = oc.calls.copy(); puts = oc.puts.copy()
                    calls["type"], puts["type"] = "call", "put"
                    df = pd.concat([calls, puts], ignore_index=True)
                    df = _standardise_chain(df)
                    if len(df) > 0:
                        chains[e] = df
                except Exception:
                    continue
            if chains:
                return sorted(chains.keys()), chains, "Yahoo Finance"
    except Exception:
        pass
    try:
        url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
        h = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=h, timeout=8)
        if r.status_code == 200:
            data = r.json()
            result = data.get("optionChain", {}).get("result", [])
            if result:
                item = result[0]
                for ob in item.get("options", []):
                    exp_ts = ob.get("expirationDate", 0)
                    exp = datetime.utcfromtimestamp(exp_ts).strftime("%Y-%m-%d")
                    rows = ([_parse_yahoo_opt(c,"call") for c in ob.get("calls",[])] +
                             [_parse_yahoo_opt(p,"put") for p in ob.get("puts",[])])
                    rows = [r for r in rows if r]
                    if rows:
                        chains[exp] = _standardise_chain(pd.DataFrame(rows))
                if chains:
                    return sorted(chains.keys()), chains, "Yahoo v7 API"
    except Exception:
        pass
    return [], {}, None

def _options_cboe_delayed(ticker):
    try:
        clean = ticker.upper().split(".")[0]
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{clean}.json"
        r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return [], {}, None
        raw = r.json().get("data", [])
        if not raw:
            return [], {}, None
        rows = []
        for item in raw:
            opt = item.get("option", "")
            if not opt:
                continue
            try:
                m = re.search(r"(\d{6})([CP])(\d{8})$", opt.replace(" ",""))
                if not m:
                    continue
                exp_str = m.group(1); pc = m.group(2)
                strike = int(m.group(3)) / 1000
                exp = f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}"
                ot = "call" if pc == "C" else "put"
                bid = float(item.get("bid",0) or 0)
                ask = float(item.get("ask",0) or 0)
                last = float(item.get("last",0) or 0)
                rows.append({
                    "strike": strike, "type": ot, "expiry": exp,
                    "lastPrice": last, "bid": bid, "ask": ask,
                    "openInterest": int(item.get("open_interest",0) or 0),
                    "volume": int(item.get("volume",0) or 0),
                    "impliedVolatility": float(item.get("iv",0) or 0) / 100,
                })
            except Exception:
                continue
        if not rows:
            return [], {}, None
        df = _standardise_chain(pd.DataFrame(rows))
        exps = sorted(df["expiry"].dropna().unique().tolist())
        chains = {e: df[df["expiry"]==e].copy() for e in exps}
        return exps, chains, "CBOE Delayed"
    except Exception:
        return [], {}, None

def _synthetic_chain(spot, rate, div_yield, vol):
    days_list = [7, 14, 30, 60, 90, 180]
    pcts = np.array([0.60,0.70,0.80,0.85,0.90,0.925,0.95,0.975,
                       1.00,1.025,1.05,1.075,1.10,1.15,1.20,1.30,1.40])
    strikes = np.round(spot * pcts, 2)
    today = datetime.now()
    chains, expiries = {}, []
    for days in days_list:
        exp = (today + timedelta(days=days)).strftime("%Y-%m-%d")
        tau = days / 365
        rows = []
        for K in strikes:
            for ot in ["call","put"]:
                p = bsm_price(spot, float(K), rate, div_yield, vol, tau, ot)
                sp = max(p*0.02, 0.01)
                rows.append({
                    "strike": float(K), "type": ot, "expiry": exp,
                    "lastPrice": p, "bid": max(0.0, p-sp/2),
                    "ask": p+sp/2, "mid_price": p,
                    "openInterest": 0, "volume": 0,
                    "impliedVolatility": vol, "synthetic": True,
                })
        chains[exp] = pd.DataFrame(rows)
        expiries.append(exp)
    return expiries, chains

def get_options_data(ticker, polygon_key=""):
    debug = []
    is_indian = ticker.endswith(".NS") or ticker in ["^NSEI","^NSEBANK"]
    is_us = not is_indian and not any(
        ticker.upper().endswith(s) for s in [".NS",".BO",".L",".TO",".AX",".HK"])

    if is_indian:
        debug.append("🇮🇳 Trying NSE India...")
        exp, chains, src, nse_spot = _options_nse_live(ticker)
        if exp:
            debug.append(f"   ✅ {src}: {len(exp)} expiries")
            return exp, chains, src, False, debug, nse_spot
        debug.append("   ❌ NSE returned no data")

    debug.append("🌐 Trying Yahoo Finance...")
    exp, chains, src = _options_yfinance_robust(ticker)
    if exp:
        debug.append(f"   ✅ {src}: {len(exp)} expiries")
        return exp, chains, src, False, debug, None
    debug.append("   ❌ Yahoo returned no data")

    if is_us:
        debug.append("🇺🇸 Trying CBOE Delayed...")
        exp, chains, src = _options_cboe_delayed(ticker.upper().split(".")[0])
        if exp:
            debug.append(f"   ✅ {src}: {len(exp)} expiries")
            return exp, chains, src, False, debug, None
        debug.append("   ❌ CBOE returned no data")

    debug.append("⚠️ Falling back to synthetic BSM chain")
    return None, None, "Synthetic (BSM)", True, debug, None

# =============================================================================
# PRICING
# =============================================================================

def bsm_d1d2(S, K, r, q, v, T):
    d1 = (np.log(S/K) + (r-q+0.5*v**2)*T) / (v*np.sqrt(T))
    return d1, d1 - v*np.sqrt(T)

def bsm_price(S, K, r, q, v, T, opt="call"):
    if T <= 0 or v <= 0:
        return max(0, S-K) if opt=="call" else max(0, K-S)
    d1, d2 = bsm_d1d2(S, K, r, q, v, T)
    if opt == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bsm_greeks(S, K, r, q, v, T, opt="call"):
    z = {k:0.0 for k in ["delta","gamma","vega","theta","rho","vanna","volga","charm"]}
    if T <= 0 or v <= 0:
        return z
    try:
        d1, d2 = bsm_d1d2(S, K, r, q, v, T)
        pdf = norm.pdf(d1)
        dq, dr, st_ = np.exp(-q*T), np.exp(-r*T), np.sqrt(T)
        gamma = dq*pdf / (S*v*st_)
        vega = S*dq*pdf*st_ / 100
        vanna = -dq*pdf*d2 / v
        volga = vega*d1*d2 / v
        charm = dq*pdf*(2*(r-q)*T - d2*v*st_) / (2*T*v*st_)
        if opt == "call":
            delta = dq*norm.cdf(d1)
            theta = (-S*dq*pdf*v/(2*st_) - r*K*dr*norm.cdf(d2) + q*S*dq*norm.cdf(d1))/365
            rho = K*T*dr*norm.cdf(d2) / 100
        else:
            delta = dq*(norm.cdf(d1)-1)
            theta = (-S*dq*pdf*v/(2*st_) + r*K*dr*norm.cdf(-d2) - q*S*dq*norm.cdf(-d1))/365
            rho = -K*T*dr*norm.cdf(-d2) / 100
            charm = -charm
        return {"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,
                "rho":rho,"vanna":vanna,"volga":volga,"charm":charm}
    except Exception:
        return z

def binomial_price(S, K, r, q, v, T, opt="call", style="european", steps=200):
    if T <= 0 or v <= 0:
        return max(0, S-K) if opt=="call" else max(0, K-S)
    dt = T/steps; u = np.exp(v*np.sqrt(dt)); d = 1/u
    p = (np.exp((r-q)*dt)-d)/(u-d); disc = np.exp(-r*dt)
    j = np.arange(steps+1)
    prices = S*(u**(steps-j))*(d**j)
    vals = np.maximum(prices-K, 0) if opt=="call" else np.maximum(K-prices, 0)
    for i in range(steps-1, -1, -1):
        vals = disc*(p*vals[:-1] + (1-p)*vals[1:])
        if style == "american":
            ji = np.arange(i+1)
            pi = S*(u**(i-ji))*(d**ji)
            intr = np.maximum(pi-K, 0) if opt=="call" else np.maximum(K-pi, 0)
            vals = np.maximum(vals, intr)
    return float(vals[0])

def monte_carlo_price(S, K, r, q, v, T, opt="call", n_paths=50_000, seed=42):
    if T <= 0 or v <= 0:
        p = max(0, S-K) if opt=="call" else max(0, K-S)
        return p, 0.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    ST = S*np.exp((r-q-0.5*v**2)*T + v*np.sqrt(T)*z)
    pay = np.maximum(ST-K, 0) if opt=="call" else np.maximum(K-ST, 0)
    dp = np.exp(-r*T)*pay
    return float(dp.mean()), float(dp.std(ddof=1)/np.sqrt(n_paths))

def heston_char(phi, S, K, r, q, T, v0, k, th, sv, rh):
    xi = k - rh*sv*1j*phi
    d = np.sqrt(xi**2 + sv**2*(phi**2+1j*phi))
    g = (xi-d)/(xi+d); edt = np.exp(-d*T)
    C = (r-q)*1j*phi*T + k*th/sv**2 * ((xi-d)*T - 2*np.log((1-g*edt)/(1-g)))
    D = (xi-d)/sv**2 * (1-edt)/(1-g*edt)
    return np.exp(C + D*v0 + 1j*phi*np.log(S*np.exp((r-q)*T)))

def heston_price(S, K, r, q, T, v0, k, th, sv, rh, opt="call"):
    if T <= 0:
        return max(0, S-K) if opt=="call" else max(0, K-S)
    Kl = np.log(K)
    def I1(p):
        cf = heston_char(p-1j, S, K, r, q, T, v0, k, th, sv, rh)
        cf0 = heston_char(-1j, S, K, r, q, T, v0, k, th, sv, rh)
        return np.real(np.exp(-1j*p*Kl)*cf/(1j*p*cf0))
    def I2(p):
        cf = heston_char(p, S, K, r, q, T, v0, k, th, sv, rh)
        return np.real(np.exp(-1j*p*Kl)*cf/(1j*p))
    try:
        P1 = 0.5 + (1/math.pi)*quad(I1, 0, 200, limit=100)[0]
        P2 = 0.5 + (1/math.pi)*quad(I2, 0, 200, limit=100)[0]
        c = S*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2
        return max(0, c) if opt=="call" else max(0, c - S*np.exp(-q*T) + K*np.exp(-r*T))
    except Exception:
        return bsm_price(S, K, r, q, max(np.sqrt(v0),0.01), T, opt)

def calibrate_heston(S, r, q, T, ks, ivs, opt="call"):
    if len(ks) < 3:
        return {"v0":0.04,"kappa":2.0,"theta":0.04,"sigma_v":0.3,"rho":-0.7}
    def obj(p):
        v0,k,th,sv,rh = p
        if v0<=0 or k<=0 or th<=0 or sv<=0 or abs(rh)>=1:
            return 1e6
        errs = []
        for K, miv in zip(ks, ivs):
            hp = heston_price(S, K, r, q, T, v0, k, th, sv, rh, opt)
            biv = implied_vol(hp, S, K, r, q, T, opt)
            if not np.isnan(biv):
                errs.append((biv-miv)**2)
        return np.mean(errs) if errs else 1e6
    try:
        res = minimize(obj, [0.04,2.0,0.04,0.3,-0.7],
                       bounds=[(1e-4,1),(0.1,10),(1e-4,1),(0.01,1),(-0.99,0.99)],
                       method="L-BFGS-B", options={"maxiter":200})
        v0,k,th,sv,rh = res.x
        return {"v0":v0,"kappa":k,"theta":th,"sigma_v":sv,"rho":rh}
    except Exception:
        return {"v0":0.04,"kappa":2.0,"theta":0.04,"sigma_v":0.3,"rho":-0.7}

def merton_jump_price(S, K, r, q, v, T, lam=0.75, mu=0, sig=0.15, opt="call", n=40):
    if T <= 0:
        return max(0, S-K) if opt=="call" else max(0, K-S)
    lp = lam*np.exp(mu + 0.5*sig**2)
    price = 0.0
    for kk in range(n):
        w = (np.exp(-lp*T)*(lp*T)**kk) / math.factorial(kk)
        vk = np.sqrt(v**2 + kk*sig**2/max(T,1e-6))
        rk = r - lam*(np.exp(mu+0.5*sig**2)-1) + kk*(mu+0.5*sig**2)/max(T,1e-6)
        price += w*bsm_price(S, K, rk, q, vk, T, opt)
    return float(price)

def implied_vol(mp, S, K, r, q, T, opt="call"):
    if T <= 0 or pd.isna(mp) or mp <= 0:
        return np.nan
    intr = max(0, S-K) if opt=="call" else max(0, K-S)
    if mp < intr*0.99:
        return np.nan
    def obj(v):
        return bsm_price(S, K, r, q, v, T, opt) - mp
    try:
        return brentq(obj, 1e-4, 5.0, maxiter=200)
    except ValueError:
        return np.nan

def model_free_iv(chain, S, r, T):
    try:
        if chain.empty:
            return np.nan
        pc = "mid_price" if "mid_price" in chain.columns else "lastPrice"
        calls = chain[(chain["type"]=="call") & (chain["strike"]>S)].sort_values("strike")
        puts = chain[(chain["type"]=="put") & (chain["strike"]<S)].sort_values("strike")
        if len(calls) < 2 and len(puts) < 2:
            return np.nan
        ss = 0.0; F = S*np.exp(r*T)
        for df in [calls, puts]:
            if len(df) < 2:
                continue
            ks = df["strike"].values; qs = df[pc].values
            dks = np.diff(ks)
            for i in range(len(dks)):
                ss += 2*dks[i]/(ks[i+1]**2)*np.exp(r*T)*(qs[i]+qs[i+1])/2
        ss -= (F/S - 1)**2
        ss = max(0, ss)/T
        return float(np.sqrt(ss))
    except Exception:
        return np.nan

def build_vol_surface(chain_dict, S, r, q, opt="call"):
    rows = []
    for exp, chain in chain_dict.items():
        try:
            T = max((datetime.strptime(exp,"%Y-%m-%d")-datetime.now()).days, 1)/365
        except Exception:
            continue
        if chain.empty: continue
        pc = "mid_price" if "mid_price" in chain.columns else "lastPrice"
        sub = chain[chain["type"]==opt].copy()
        if sub.empty: continue
        for _, row in sub.iterrows():
            K = float(row["strike"]); mp = row.get(pc, np.nan)
            oi = row.get("openInterest", 0)
            if pd.isna(mp) or mp <= 0: continue
            iv = implied_vol(mp, S, K, r, q, T, opt)
            if np.isnan(iv) or iv<0.01 or iv>3: continue
            rows.append({"expiry":exp,"strike":K,"moneyness":K/S,
                          "iv":iv,"tau":T,"oi":float(oi) if oi else 0.0})
    return pd.DataFrame(rows)

def calculate_pcr(chain):
    if chain.empty:
        return None, None
    ci = chain[chain["type"]=="call"]["openInterest"].sum()
    pi = chain[chain["type"]=="put"]["openInterest"].sum()
    cv = chain[chain["type"]=="call"]["volume"].sum()
    pv = chain[chain["type"]=="put"]["volume"].sum()
    return (pi/ci if ci>0 else None), (pv/cv if cv>0 else None)

# =============================================================================
# VOL ESTIMATION
# =============================================================================

def historical_vol(lr, window=252):
    r = lr.dropna().tail(window)
    return float(r.std()*np.sqrt(252)) if len(r) >= 10 else np.nan

def ewma_vol(lr, lam=0.94):
    r = lr.dropna().values
    if len(r) < 10:
        return np.nan
    var = r[0]**2
    for ret in r[1:]:
        var = lam*var + (1-lam)*ret**2
    return float(np.sqrt(var*252))

def garch_forecast_vol(lr, horizon=5):
    r = lr.dropna()*100
    if len(r) < 100 or not HAS_ARCH:
        return ewma_vol(lr)
    try:
        fit = arch_model(r, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
        fc = fit.forecast(horizon=horizon, reindex=False)
        vf = fc.variance.values[-1].mean()
        return float(np.sqrt(vf)/100*np.sqrt(252))
    except Exception:
        return ewma_vol(lr)

def xgboost_vol_forecast(lr, fd=5):
    if not HAS_XGB or len(lr.dropna()) < 120:
        return np.nan
    try:
        r = lr.dropna().copy()
        df = pd.DataFrame({"r": r})
        for lag in [1,2,3,5]:
            df[f"abs{lag}"] = df["r"].abs().shift(lag)
        for w in [5,10,21]:
            df[f"rv{w}"] = df["r"].rolling(w).std()*np.sqrt(252)
        df["target"] = df["r"].rolling(fd).std().shift(-fd)*np.sqrt(252)
        df = df.dropna()
        if len(df) < 60:
            return np.nan
        s = int(len(df)*0.8)
        X_tr, y_tr = df.iloc[:s,:-1].values, df.iloc[:s,-1].values
        X_te = df.iloc[-1:,:-1].values
        m = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                              learning_rate=0.05, verbosity=0)
        m.fit(X_tr, y_tr)
        return max(float(m.predict(X_te)[0]), 0.01)
    except Exception:
        return np.nan

def realized_vol_series(lr, window=21):
    return lr.rolling(window).std()*np.sqrt(252)

# =============================================================================
# STRATEGIES
# =============================================================================

STRATEGIES = {
    "Long Call":       [("call",+1,1.00)],
    "Long Put":        [("put",+1,1.00)],
    "Short Call":      [("call",-1,1.00)],
    "Short Put":       [("put",-1,1.00)],
    "Long Straddle":   [("call",+1,1.00),("put",+1,1.00)],
    "Short Straddle":  [("call",-1,1.00),("put",-1,1.00)],
    "Long Strangle":   [("call",+1,1.05),("put",+1,0.95)],
    "Bull Call Spread":[("call",+1,1.00),("call",-1,1.05)],
    "Bear Put Spread": [("put",+1,1.00),("put",-1,0.95)],
    "Iron Condor":     [("put",-1,0.90),("put",+1,0.95),
                        ("call",+1,1.05),("call",-1,1.10)],
    "Butterfly":       [("call",+1,0.95),("call",-2,1.00),("call",+1,1.05)],
}

def strategy_payoff(strat, S, r, q, v, T):
    legs = STRATEGIES.get(strat, [])
    sr = np.linspace(S*0.5, S*1.5, 300)
    net_pay = np.zeros(300); net_prem = 0.0; info = []
    for (ot, qty, mm) in legs:
        K = S*mm
        pr = bsm_price(S, K, r, q, v, T, ot)
        net_prem += qty*pr
        pay = np.maximum(sr-K, 0) if ot=="call" else np.maximum(K-sr, 0)
        net_pay += qty*pay
        info.append({"type":ot,"qty":qty,"strike":round(K,2),"premium":round(pr,4)})
    pnl = net_pay + net_prem
    bes = []
    for i in range(len(pnl)-1):
        if pnl[i]*pnl[i+1] < 0:
            be = sr[i] + (0-pnl[i])/(pnl[i+1]-pnl[i])*(sr[i+1]-sr[i])
            bes.append(round(float(be), 2))
    return sr, pnl, net_prem, info, bes, float(pnl.max()), float(pnl.min())

# =============================================================================
# RISK
# =============================================================================

def calculate_var_cvar(lr, conf=0.95, days=1, pv=100_000):
    r = lr.dropna()
    if len(r) < 30:
        return {}
    sr = r*np.sqrt(days)
    mu, sig = sr.mean(), sr.std()
    z = norm.ppf(1-conf)
    pvar = -(mu+z*sig)*pv
    pcvar = -(mu-sig*norm.pdf(z)/(1-conf))*pv
    hvar = -np.percentile(sr, (1-conf)*100)*pv
    mask = sr <= (-hvar/pv)
    hcvar = -sr[mask].mean()*pv if mask.any() else hvar
    return {"param_var":pvar,"param_cvar":pcvar,"hist_var":hvar,"hist_cvar":hcvar}

def stress_test(S, K, r, q, v, T, gr, opt, qty=1):
    base = bsm_price(S, K, r, q, v, T, opt)
    sh = {
        "Spot −20%":{"dS":-0.20*S,"dV":0,"dR":0},
        "Spot −10%":{"dS":-0.10*S,"dV":0,"dR":0},
        "Spot −5%": {"dS":-0.05*S,"dV":0,"dR":0},
        "Spot +5%": {"dS":+0.05*S,"dV":0,"dR":0},
        "Spot +10%":{"dS":+0.10*S,"dV":0,"dR":0},
        "Spot +20%":{"dS":+0.20*S,"dV":0,"dR":0},
        "Vol +25%": {"dS":0,"dV":+0.25*v,"dR":0},
        "Vol +50%": {"dS":0,"dV":+0.50*v,"dR":0},
        "Vol −25%": {"dS":0,"dV":-0.25*v,"dR":0},
        "Rate +100bp":{"dS":0,"dV":0,"dR":+0.01},
        "Rate −100bp":{"dS":0,"dV":0,"dR":-0.01},
        "Crash":    {"dS":-0.20*S,"dV":+0.50*v,"dR":-0.005},
    }
    out = {}
    for n, s in sh.items():
        exact = (bsm_price(S+s["dS"], K, r+s["dR"], q, v+s["dV"], T, opt) - base)*qty
        approx = (gr["delta"]*s["dS"] + 0.5*gr["gamma"]*s["dS"]**2
                   + gr["vega"]*100*(s["dV"]/0.01)*0.01
                   + gr["rho"]*100*(s["dR"]/0.01)*0.01)*qty
        out[n] = {"approx_pnl":approx, "exact_pnl":exact}
    return out

# =============================================================================
# HEDGING & FUTURES
# =============================================================================

def simulate_gbm_path(S, r, q, v, T, n=60, seed=None):
    rng = np.random.default_rng(seed)
    dt = T/n; z = rng.standard_normal(n)
    lr = (r-q-0.5*v**2)*dt + v*np.sqrt(dt)*z
    return S*np.exp(np.concatenate([[0], np.cumsum(lr)]))

def simulate_delta_hedge(path, K, r, q, v, T, opt="call", rehedge=1, cost_bps=5):
    n = len(path); dt = T/(n-1)
    cash = bsm_price(path[0], K, r, q, v, T, opt)
    prev_d, shares, rows = 0.0, 0.0, []
    for i in range(n):
        trem = max(T-i*dt, 1e-6); s = path[i]
        delta = bsm_greeks(s, K, r, q, v, trem, opt)["delta"]
        if i % rehedge == 0 or i == n-1:
            trade = delta - prev_d
            cash -= trade*s + abs(trade)*s*(cost_bps/10_000)
            shares += trade; prev_d = delta
        cash *= np.exp(r*dt)
        rows.append({"step":i,"spot":s,"delta":delta,"cash":cash})
    payoff = max(path[-1]-K, 0) if opt=="call" else max(K-path[-1], 0)
    return pd.DataFrame(rows), cash + shares*path[-1] - payoff

def theoretical_forward_price(S, r, q, T):
    return S*np.exp((r-q)*T)

# =============================================================================
# BACKTEST
# =============================================================================

def walk_forward_backtest(hist, vol_threshold=0.15, n_windows=5, min_days=100):
    r = hist["log_return"].dropna()
    total = len(r)
    if total < min_days:
        return pd.DataFrame(), f"Need ≥{min_days} days, got {total}"
    if total >= 504: lw, nw = 252, min(n_windows, 5)
    elif total >= 252: lw, nw = 126, min(n_windows, 4)
    elif total >= 150: lw, nw = 63, min(n_windows, 3)
    else: lw, nw = 42, min(n_windows, 3)
    step = max((total - lw)//(nw+1), 20)
    results = []
    for w in range(nw):
        te = lw + w*step; ts = te; tend = min(ts+step, total)
        if tend - ts < 5: continue
        tr = r.iloc[:te]; testr = r.iloc[ts:tend]
        ev = ewma_vol(tr); hv = historical_vol(tr, window=min(252, len(tr)))
        sig = ("sell" if not np.isnan(ev) and not np.isnan(hv)
                and ev > hv*(1+vol_threshold) else "hold")
        rv = float(testr.std()*np.sqrt(252)) if len(testr) > 5 else np.nan
        pnl = (ev-rv) if sig == "sell" and not np.isnan(rv) else 0.0
        results.append({"window":w+1,"train_days":te,"ewma_vol":ev,
                          "hist_vol":hv,"signal":sig,"realized_vol":rv,
                          "approx_pnl":pnl})
    df = pd.DataFrame(results)
    return df, f"Adaptive: {total} days → {lw}d train × {nw} windows"

def performance_stats(pnl):
    r = pnl.dropna()
    if len(r) < 2:
        return {}
    mu = r.mean(); sd = r.std(ddof=1) if len(r) > 1 else 0
    sharpe = mu/sd*np.sqrt(252) if sd > 0 else np.nan
    down = r[r<0].std(ddof=1) if len(r[r<0]) > 1 else 0
    sortino = mu/down*np.sqrt(252) if down > 0 else np.nan
    cr = (1+r).cumprod(); peak = cr.cummax()
    max_dd = float(((cr-peak)/peak).min())
    calmar = mu*252/abs(max_dd) if max_dd < 0 else np.nan
    if len(r) >= 3:
        rng = np.random.default_rng(42)
        boots = [rng.choice(r.values, size=len(r), replace=True).mean() for _ in range(2000)]
        pv = float(np.mean(np.array(boots) <= 0))
    else:
        pv = np.nan
    return {"sharpe":sharpe,"sortino":sortino,"calmar":calmar,
            "max_dd":max_dd,"p_value":pv}

# =============================================================================
# RECOMMENDATIONS
# =============================================================================

def rec_options(iv, fv, hv, trend, mn, days, pcr=None):
    notes, score = [], 0
    if not np.isnan(iv) and not np.isnan(fv) and fv > 0:
        gap = (iv-fv)/fv
        if gap > 0.15:
            score -= 2; notes.append(f"IV ({iv:.1%}) is {gap:.0%} above forecast ({fv:.1%}) — options **rich**.")
        elif gap < -0.15:
            score += 2; notes.append(f"IV ({iv:.1%}) is {abs(gap):.0%} below forecast ({fv:.1%}) — options **cheap**.")
        else:
            notes.append(f"IV ({iv:.1%}) in line with forecast ({fv:.1%}).")
    if trend == "up":
        score += 1; notes.append("Uptrend (50d > 200d MA) — bullish bias.")
    elif trend == "down":
        score -= 1; notes.append("Downtrend (50d < 200d MA) — bearish bias.")
    if days < 14:
        score -= 1; notes.append(f"Only {days}d to expiry — theta risk.")
    if pcr is not None:
        if pcr > 1.3:
            score += 1; notes.append(f"PCR={pcr:.2f} — contrarian bullish.")
        elif pcr < 0.7:
            score -= 1; notes.append(f"PCR={pcr:.2f} — contrarian bearish.")
        else:
            notes.append(f"PCR={pcr:.2f} — neutral.")
    if score >= 2: v = "🟢 LEAN: BUY OPTIONS"
    elif score <= -2: v = "🔴 LEAN: SELL PREMIUM"
    else: v = "⚖️ NEUTRAL"
    return v, notes, score

def rec_futures(S, fwd, mkt, r, q, trend, T, ccy):
    notes, score = [], 0
    carry = r - q
    lbl = "contango" if carry > 0 else "backwardation"
    notes.append(f"Net carry {carry:+.2%} → **{lbl}**. Fair fwd {ccy}{fwd:.2f}.")
    arb = False
    if mkt and mkt > 0:
        mis = (mkt-fwd)/fwd
        if abs(mis) > 0.005:
            arb = True
            if mis > 0:
                score += 2; notes.append(f"Market fut {mis:+.2%} above fair → cash-and-carry arb.")
            else:
                score -= 2; notes.append(f"Market fut {mis:+.2%} below fair → reverse arb.")
    if trend == "up":
        score += 1; notes.append("Uptrend → LONG futures.")
    elif trend == "down":
        score -= 1; notes.append("Downtrend → SHORT futures.")
    if arb:
        v = "🔄 ARB: BUY futures" if score > 0 else "🔄 ARB: SELL futures"
    elif score >= 2: v = "🟢 LEAN: LONG FUTURES"
    elif score <= -2: v = "🔴 LEAN: SHORT FUTURES"
    else: v = "⚖️ NEUTRAL"
    return v, notes, score

# =============================================================================
# PROFESSIONAL REPORT
# =============================================================================

def build_professional_report(rd):
    ccy = rd.get("ccy","$"); tkr = rd.get("ticker","N/A")
    S = rd.get("spot",0); r = rd.get("rate",0); q = rd.get("div_yield",0)
    K = rd.get("strike",0); T = rd.get("tau",0); opt = rd.get("option_type","call")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    v = rd.get("volatility",{}); pr = rd.get("pricing",{})
    gk = rd.get("greeks",{}); rk = rd.get("risk",{})
    fd = rd.get("futures",{}); bt = rd.get("backtest",{})
    ss = rd.get("scenarios",{}); pc = rd.get("pcr")

    def fv(x, fmt="{:.2%}"):
        return fmt.format(x) if (x is not None and not (isinstance(x,float) and np.isnan(x))) else "N/A"
    def fp(x):
        return f"{ccy}{x:.4f}" if x else "N/A"

    lines = []
    lines.append("# 📊 QUANT EDGE — INSTITUTIONAL DERIVATIVES ANALYSIS")
    lines.append("")
    lines.append(f"**Instrument:** `{tkr}` | **Report ID:** `QE-{datetime.now().strftime('%Y%m%d-%H%M%S')}`")
    lines.append(f"**Generated:** {now} | **Data Source:** {rd.get('data_source','N/A')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## 📌 EXECUTIVE SUMMARY")
    lines.append("")
    lines.append(f"**Underlying:** {tkr} trading at **{ccy}{S:,.2f}**")
    lines.append(f"**Contract Analysed:** {opt.upper()} @ Strike {ccy}{K:,.2f}, "
                  f"{int(T*365)} days to expiry")
    lines.append("")
    lines.append(f"### Key Findings")
    lines.append(f"- **Implied Volatility:** {fv(v.get('iv'))} vs GARCH Forecast: {fv(v.get('garch'))}")
    lines.append(f"- **Model Price Range:** {fp(pr.get('bsm'))} (BSM) → {fp(pr.get('heston'))} (Heston)")
    lines.append(f"- **Market Mid:** {fp(pr.get('market'))}")
    lines.append(f"- **Delta:** {gk.get('delta',0):.4f} | **Gamma:** {gk.get('gamma',0):.6f} | "
                  f"**Vega:** {gk.get('vega',0):.4f} | **Theta:** {gk.get('theta',0):.4f}/day")
    if pc is not None:
        lines.append(f"- **Put-Call Ratio (OI):** {pc:.2f} — "
                      f"{'Bearish sentiment' if pc>1 else 'Bullish sentiment' if pc<0.7 else 'Neutral'}")
    lines.append("")
    lines.append(f"### Trading Signals")
    lines.append(f"- **Options:** {rd.get('opt_verdict','N/A')}")
    lines.append(f"- **Futures:** {rd.get('fut_verdict','N/A')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Market Environment
    lines.append("## 1️⃣ MARKET ENVIRONMENT")
    lines.append("")
    lines.append("| Parameter | Value | Notes |")
    lines.append("|-----------|-------|-------|")
    lines.append(f"| Spot Price | {ccy}{S:,.2f} | Live/last available |")
    lines.append(f"| Risk-Free Rate | {r:.2%} | 13W T-Bill (^IRX) |")
    lines.append(f"| Dividend Yield | {q:.2%} | Trailing 12M |")
    lines.append(f"| Net Carry (r-q) | {r-q:+.2%} | {'Contango' if r>q else 'Backwardation'} |")
    lines.append(f"| Time to Expiry | {T*365:.0f} days ({T:.4f} yrs) | |")
    lines.append(f"| Moneyness (K/S) | {K/S:.4f} | "
                  f"{'ITM' if (opt=='call' and K<S) or (opt=='put' and K>S) else 'OTM' if abs(K/S-1)>0.02 else 'ATM'} |")
    lines.append("")

    # Section 2: Volatility Analysis
    lines.append("## 2️⃣ VOLATILITY ANALYSIS")
    lines.append("")
    lines.append("### 2.1 Multi-Model Volatility Estimates")
    lines.append("")
    lines.append("| Model | Estimate | Description |")
    lines.append("|-------|----------|-------------|")
    lines.append(f"| Historical (1Y) | {fv(v.get('hist'))} | Realized annualised std dev |")
    lines.append(f"| EWMA (λ=0.94) | {fv(v.get('ewma'))} | Exponentially-weighted moving avg |")
    lines.append(f"| GARCH(1,1) | {fv(v.get('garch'))} | 5-day-ahead conditional forecast |")
    lines.append(f"| XGBoost ML | {fv(v.get('xgb'))} | Gradient boosting on lagged returns |")
    lines.append(f"| Implied Vol (ATM) | {fv(v.get('iv'))} | BSM-inverted from market price |")
    lines.append(f"| Model-Free IV | {fv(v.get('mfiv'))} | CBOE VIX-style, all strikes |")
    lines.append("")

    if v.get("iv") and v.get("garch"):
        vg = (v["iv"] - v["garch"])/v["garch"]*100
        lines.append(f"**Interpretation:** IV is **{vg:+.1f}%** relative to GARCH forecast.")
        if vg > 15:
            lines.append("→ Market pricing significantly higher risk than statistical models suggest. "
                          "Options are **RICH**. Vol-selling strategies favoured.")
        elif vg < -15:
            lines.append("→ Market pricing significantly lower risk than statistical models suggest. "
                          "Options are **CHEAP**. Vol-buying strategies favoured.")
        else:
            lines.append("→ Market pricing broadly consistent with statistical forecasts. No clear vol arbitrage.")
    lines.append("")

    # Section 3: Pricing Model Comparison
    lines.append("## 3️⃣ PRICING MODEL COMPARISON")
    lines.append("")
    lines.append("| Model | Price | Assumption | Advantage |")
    lines.append("|-------|-------|------------|-----------|")
    lines.append(f"| **Black-Scholes-Merton** | {fp(pr.get('bsm'))} | Constant vol, no jumps | Closed-form, fast |")
    lines.append(f"| **Binomial (American)** | {fp(pr.get('binom'))} | Discrete GBM | Handles early exercise |")
    lines.append(f"| **Monte Carlo** | {fp(pr.get('mc'))} ± {pr.get('mc_se',0):.4f} | GBM paths, {pr.get('mc_paths',50000):,} sim | Flexible, path-dependent |")
    lines.append(f"| **Heston Stochastic Vol** | {fp(pr.get('heston'))} | Mean-reverting vol | Captures vol smile |")
    lines.append(f"| **Merton Jump-Diffusion** | {fp(pr.get('merton'))} | GBM + Poisson jumps | Captures tail risk |")
    lines.append(f"| **Market Mid** | {fp(pr.get('market'))} | Bid/ask midpoint | Actual quoted price |")
    lines.append("")

    if pr.get("market") and pr.get("bsm"):
        diff = (pr["market"] - pr["bsm"])/pr["bsm"]*100
        lines.append(f"**Market vs BSM Divergence:** {diff:+.2f}%")
        if abs(diff) > 5:
            lines.append("→ Meaningful divergence between market and theoretical BSM price. "
                          "Consider Heston/Jump models for a more accurate benchmark.")
    lines.append("")

    # Section 4: Greeks
    lines.append("## 4️⃣ RISK SENSITIVITIES (GREEKS)")
    lines.append("")
    lines.append("### 4.1 First-Order Greeks")
    lines.append("")
    lines.append("| Greek | Symbol | Value | Interpretation |")
    lines.append("|-------|--------|-------|----------------|")
    lines.append(f"| Delta | Δ | {gk.get('delta',0):.4f} | {abs(gk.get('delta',0))*100:.1f}% probability proxy |")
    lines.append(f"| Gamma | Γ | {gk.get('gamma',0):.6f} | Rate of change of Δ |")
    lines.append(f"| Vega | ν | {gk.get('vega',0):.4f} | P&L per 1% vol move |")
    lines.append(f"| Theta | Θ | {gk.get('theta',0):.4f}/day | Daily time decay |")
    lines.append(f"| Rho | ρ | {gk.get('rho',0):.4f} | P&L per 1% rate move |")
    lines.append("")
    lines.append("### 4.2 Second-Order Greeks (Advanced)")
    lines.append("")
    lines.append("| Greek | Value | What It Measures |")
    lines.append("|-------|-------|------------------|")
    lines.append(f"| Vanna | {gk.get('vanna',0):.4f} | Change in Δ as vol changes |")
    lines.append(f"| Volga (Vomma) | {gk.get('volga',0):.4f} | Convexity of Vega |")
    lines.append(f"| Charm | {gk.get('charm',0):.4f} | Change in Δ as time passes |")
    lines.append("")

    # Section 5: Risk
    lines.append("## 5️⃣ VALUE-AT-RISK ANALYSIS")
    lines.append("")
    lines.append(f"**Position Value:** {ccy}{rd.get('pos_val',100000):,} | **Confidence:** {rd.get('var_conf',0.95):.0%} | **Horizon:** 1 day")
    lines.append("")
    lines.append("| Metric | Parametric | Historical Simulation | Interpretation |")
    lines.append("|--------|------------|----------------------|----------------|")
    lines.append(f"| VaR | {ccy}{rk.get('param_var',0):,.0f} | {ccy}{rk.get('hist_var',0):,.0f} | Max loss expected |")
    lines.append(f"| CVaR (Expected Shortfall) | {ccy}{rk.get('param_cvar',0):,.0f} | {ccy}{rk.get('hist_cvar',0):,.0f} | Avg loss beyond VaR |")
    lines.append("")

    # Section 6: Stress Test
    if ss:
        lines.append("## 6️⃣ STRESS TEST SCENARIOS")
        lines.append("")
        lines.append("| Scenario | Approx P&L | Exact P&L | Impact |")
        lines.append("|----------|-----------|-----------|--------|")
        for name, val in ss.items():
            impact = "🟢 Profit" if val["exact_pnl"] > 0 else "🔴 Loss"
            lines.append(f"| {name} | {ccy}{val['approx_pnl']:+.2f} | {ccy}{val['exact_pnl']:+.2f} | {impact} |")
        lines.append("")

    # Section 7: Futures
    lines.append("## 7️⃣ FUTURES & FORWARD ANALYSIS")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Spot | {ccy}{S:,.2f} |")
    lines.append(f"| Theoretical Fair Forward | {ccy}{fd.get('theoretical',0):.2f} |")
    lines.append(f"| Cost-of-Carry | {fd.get('carry',0):+.2%} |")
    lines.append(f"| Term Structure | {'Contango' if fd.get('carry',0)>0 else 'Backwardation'} |")
    lines.append("")

    # Section 8: Backtest
    if bt.get("df") is not None and not bt["df"].empty:
        lines.append("## 8️⃣ WALK-FORWARD BACKTEST")
        lines.append("")
        lines.append("**Strategy:** Sell vol when EWMA > Historical × (1 + threshold)")
        lines.append("")
        lines.append("| Window | Signal | Train EWMA | Train Hist | Realized | P&L |")
        lines.append("|--------|--------|-----------|------------|----------|-----|")
        for _, row in bt["df"].iterrows():
            lines.append(f"| W{int(row['window'])} | {row['signal']} | "
                          f"{row.get('ewma_vol',0):.2%} | {row.get('hist_vol',0):.2%} | "
                          f"{row.get('realized_vol',0):.2%} | {row.get('approx_pnl',0):+.4f} |")
        lines.append("")
        ps = bt.get("perf_stats", {})
        if ps:
            lines.append("**Performance Metrics:**")
            lines.append("")
            lines.append(f"- Sharpe Ratio: **{ps.get('sharpe',0):.2f}**")
            lines.append(f"- Sortino Ratio: **{ps.get('sortino',0):.2f}**")
            lines.append(f"- Calmar Ratio: **{ps.get('calmar',0):.2f}**")
            lines.append(f"- Max Drawdown: **{ps.get('max_dd',0):.2%}**")
            pv = ps.get("p_value", np.nan)
            if not np.isnan(pv):
                lines.append(f"- Bootstrap p-value: **{pv:.3f}** "
                              f"({'✅ Statistically significant' if pv<0.05 else '⚠️ Not significant'})")
        lines.append("")

    # Section 9: Recommendations
    lines.append("## 9️⃣ TRADING RECOMMENDATIONS")
    lines.append("")
    lines.append("### Options Strategy")
    lines.append(f"**{rd.get('opt_verdict','N/A')}**")
    lines.append("")
    for n in rd.get("opt_notes", []):
        lines.append(f"- {n}")
    lines.append("")
    lines.append("### Futures Strategy")
    lines.append(f"**{rd.get('fut_verdict','N/A')}**")
    lines.append("")
    for n in rd.get("fut_notes", []):
        lines.append(f"- {n}")
    lines.append("")

    # Disclaimer
    lines.append("---")
    lines.append("")
    lines.append("## ⚠️ DISCLAIMER")
    lines.append("")
    lines.append("This report is generated by **QUANT EDGE**, an educational quantitative finance tool. "
                  "All models, pricing outputs, risk metrics, and recommendations are for **research and "
                  "educational purposes only**. They do not constitute investment advice, financial advice, "
                  "trading advice, or any other sort of advice. Past performance is not indicative of future "
                  "results. Options and futures trading involves substantial risk of loss and is not suitable "
                  "for all investors. Please conduct your own research and consult with a qualified financial "
                  "advisor before making any trading decisions.")
    lines.append("")
    lines.append(f"*QUANT EDGE Report ID: QE-{datetime.now().strftime('%Y%m%d-%H%M%S')} | "
                  f"Generated {now}*")

    return "\n".join(lines)

# =============================================================================
# PLOTLY STYLE HELPERS
# =============================================================================

def apply_pro_style(fig, title="", height=400, show_legend=True):
    """Style a Plotly figure while staying theme-agnostic.

    Colors are intentionally left to Streamlit's native theming
    (see st.plotly_chart(..., theme="streamlit") calls below), which
    automatically adapts to the user's light/dark mode. We only set
    structural properties here (height, margins, transparent backgrounds)
    so the chart blends into the surrounding app regardless of theme.
    """
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ) if show_legend else dict(),
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig

# =============================================================================
# STREAMLIT UI — QUANT EDGE
# =============================================================================

st.set_page_config(
    page_title="QUANT EDGE — Derivatives Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Theme-adaptive CSS — follows the device/browser's light or dark mode
# preference automatically via `prefers-color-scheme`, and also respects
# Streamlit's own theme variables so it stays correct if the user manually
# overrides the theme from Settings.
st.markdown("""
<style>
/* ---- Design tokens (dark default) ---- */
:root {
    --qe-bg-1: #0b0f1a;
    --qe-bg-2: #0f1729;
    --qe-surface-1: #151b2b;
    --qe-surface-2: #1a2138;
    --qe-border: #1f2937;
    --qe-text-heading: #e2e8f0;
    --qe-text-primary: #f1f5f9;
    --qe-text-secondary: #94a3b8;
    --qe-text-muted: #64748b;
    --qe-accent: #a78bfa;
    --qe-accent-2: #67e8f9;
}

/* ---- Light-mode overrides (device/browser preference) ---- */
@media (prefers-color-scheme: light) {
    :root {
        --qe-bg-1: #f8fafc;
        --qe-bg-2: #eef2f9;
        --qe-surface-1: #ffffff;
        --qe-surface-2: #f1f5f9;
        --qe-border: #e2e8f0;
        --qe-text-heading: #0f172a;
        --qe-text-primary: #1e293b;
        --qe-text-secondary: #475569;
        --qe-text-muted: #64748b;
        --qe-accent: #7c3aed;
        --qe-accent-2: #0891b2;
    }
}

/* Global */
.stApp {
    background: linear-gradient(180deg, var(--qe-bg-1) 0%, var(--qe-bg-2) 100%);
}
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1600px;
}

/* Brand */
.qe-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 16px 0; border-bottom: 1px solid var(--qe-border); margin-bottom: 16px;
}
.qe-logo {
    font-size: 1.9rem; font-weight: 900; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #6366f1 0%, #22d3ee 50%, #10b981 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.qe-tagline {
    color: var(--qe-text-muted); font-size: 0.72rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--qe-bg-2) 0%, var(--qe-bg-1) 100%);
    border-right: 1px solid var(--qe-border);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--qe-accent); font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 700; margin-top: 20px; margin-bottom: 8px;
    border-left: 3px solid #6366f1; padding-left: 8px;
}

/* Ensure the sidebar open/close control is always visible, including on
   mobile where the sidebar starts collapsed behind a hamburger icon. */
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--qe-surface-1) 0%, var(--qe-surface-2) 100%);
    padding: 14px 16px; border-radius: 10px;
    border: 1px solid var(--qe-border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.15);
    transition: transform 0.2s, border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px); border-color: #6366f1;
}
[data-testid="stMetricLabel"] {
    color: var(--qe-text-secondary) !important; font-size: 0.7rem !important;
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: var(--qe-text-primary) !important; font-size: 1.35rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; font-weight: 600; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none; color: white; font-weight: 600;
    padding: 10px 24px; border-radius: 8px;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(99,102,241,0.5);
}

/* Section headers */
.qe-section {
    background: linear-gradient(90deg, rgba(99,102,241,0.15) 0%, transparent 100%);
    border-left: 4px solid #6366f1; padding: 12px 16px;
    margin: 16px 0 12px 0; border-radius: 4px;
    font-weight: 700; font-size: 1.1rem; color: var(--qe-text-heading);
}

/* Cards (feature tiles, signal cards, etc.) */
.qe-card {
    background: linear-gradient(135deg, var(--qe-surface-1), var(--qe-surface-2));
    border: 1px solid var(--qe-border);
    border-radius: 12px;
}
.qe-card-title { color: var(--qe-accent); }
.qe-card-title-cyan { color: var(--qe-accent-2); }
.qe-card-body { color: var(--qe-text-secondary); }
.qe-card-heading { color: var(--qe-text-primary); }

/* Badges */
.qe-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 12px; font-size: 0.7rem;
    font-weight: 600; margin: 2px 4px 2px 0;
}
.badge-live { background: rgba(16,185,129,0.15); color: #059669; border: 1px solid #10b981; }
.badge-delayed { background: rgba(245,158,11,0.15); color: #b45309; border: 1px solid #f59e0b; }
.badge-nse { background: rgba(249,115,22,0.15); color: #c2410c; border: 1px solid #f97316; }
.badge-synth { background: rgba(107,114,128,0.15); color: var(--qe-text-secondary); border: 1px solid #6b7280; }
@media (prefers-color-scheme: dark) {
    .badge-live { color: #6ee7b7; }
    .badge-delayed { color: #fcd34d; }
    .badge-nse { color: #fdba74; }
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--qe-border); border-radius: 8px; overflow: hidden;
}

/* Radio (nav) */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(148,163,184,0.08); border: 1px solid var(--qe-border);
    padding: 10px 14px; border-radius: 8px; margin: 3px 0;
    transition: all 0.15s; width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.1); border-color: #6366f1;
}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(90deg, rgba(99,102,241,0.3), rgba(99,102,241,0.05));
    border-color: #6366f1; box-shadow: 0 0 0 1px #6366f1;
}

/* Input */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    background: var(--qe-surface-1) !important;
    border: 1px solid var(--qe-border) !important;
    color: var(--qe-text-primary) !important;
}

/* Info/warning boxes */
.stAlert {
    background: var(--qe-surface-1) !important;
    border: 1px solid var(--qe-border) !important;
    border-left: 4px solid #6366f1 !important;
}

/* Divider */
hr { border-color: var(--qe-border) !important; margin: 20px 0 !important; }

/* Hide Streamlit branding chrome (menu + footer) without hiding the
   header bar itself, since the header holds the sidebar toggle that
   mobile/narrow-viewport users need to open the left navigation panel. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hide the "Manage app" / Deploy button and related toolbar actions
   (Streamlit Cloud viewer chrome) without touching the sidebar toggle. */
.stAppDeployButton, .stDeployButton { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# BRAND HEADER (main area)
# =============================================================================

st.markdown("""
<div class="qe-brand">
  <div>
    <div class="qe-logo">📊 QUANT EDGE</div>
    <div class="qe-tagline">Institutional Derivatives Analytics · Real-Time Multi-Source</div>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — Brand, Navigation, Search, Config
# =============================================================================

with st.sidebar:
    # Brand
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px 0;
                 border-bottom: 1px solid var(--qe-border); margin-bottom: 12px;">
      <div style="font-size: 1.6rem; font-weight: 900; letter-spacing: -0.5px;
                   background: linear-gradient(135deg, #6366f1, #22d3ee, #10b981);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 QUANT EDGE
      </div>
      <div style="color: var(--qe-text-muted); font-size: 0.65rem; text-transform: uppercase;
                   letter-spacing: 1.2px; margin-top: 4px;">
        DERIVATIVES ANALYTICS
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Search
    st.markdown("### 🔍 Instrument")
    raw_query = st.text_input(
        "Search", value="",
        placeholder="Reliance, TCS, AAPL, ^NSEI...",
        key="main_search", label_visibility="collapsed").strip()

    ticker_input = None
    if raw_query:
        candidates = search_ticker_live(raw_query)
        if candidates:
            labels = [c[1] for c in candidates]
            symbols = [c[0] for c in candidates]
            chosen = st.selectbox("Match:", labels, index=0,
                                    key="ticker_select", label_visibility="collapsed")
            ticker_input = symbols[labels.index(chosen)]
    else:
        pop_labels = [p[1] for p in POPULAR_TICKERS]
        pop_symbols = [p[0] for p in POPULAR_TICKERS]
        chosen_pop = st.selectbox("Quick pick:", pop_labels, index=0,
                                    key="popular_select", label_visibility="collapsed")
        ticker_input = pop_symbols[pop_labels.index(chosen_pop)]

    run_btn = st.button("▶  ANALYSE", type="primary", use_container_width=True)

    st.markdown("---")

    # Navigation — the requested left-panel nav
    st.markdown("### 🧭 Navigation")
    section = st.radio(
        "Section", [
            "🎯 Overview",
            "📊 Pricing & Volatility",
            "🏛️ Greeks",
            "🌐 Volatility Surface",
            "🔮 Forecasting",
            "🎨 Strategies",
            "🛡️ Hedging",
            "📦 Futures",
            "⚠️ Risk",
            "🔁 Backtest",
            "📋 Report",
        ],
        label_visibility="collapsed",
        key="nav_section",
    )

    st.markdown("---")

    # Configuration
    st.markdown("### ⚙️ Configuration")
    n_mc = st.select_slider("Monte Carlo paths",
                              [10_000,25_000,50_000,100_000], value=50_000)
    bsteps = st.select_slider("Binomial tree steps",
                                [100,200,300,500], value=200)
    var_conf = st.slider("VaR confidence level", 0.90, 0.99, 0.95, 0.01)
    pos_val = st.number_input("Position notional", 10_000, 10_000_000, 100_000, 10_000)
    rehedge_n = st.slider("Rehedge every N steps", 1, 10, 1)
    hedge_cost = st.slider("Transaction cost (bps)", 0, 50, 5)

    with st.expander("Advanced models"):
        heston_on = st.checkbox("Heston stochastic vol", value=True)
        merton_on = st.checkbox("Merton jump-diffusion", value=True)
        xgb_on = st.checkbox("XGBoost vol forecast", value=True)
        surf_exp = st.slider("Expiries for vol surface", 1, 8, 3)

    with st.expander("🔑 API Keys (optional)"):
        polygon_key = st.text_input("Polygon.io", value="", type="password")
        av_key = st.text_input("Alpha Vantage", value="", type="password")
        td_key = st.text_input("Twelve Data", value="", type="password")

    with st.expander("🔧 Debug"):
        show_debug = st.checkbox("Show data source log", value=False)

    st.markdown("---")
    st.caption("💡 QUANT EDGE v3.0")

# =============================================================================
# MAIN — Run analysis
# =============================================================================

if not (run_btn or raw_query) or not ticker_input:
    # Landing
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
      <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
      <h2 style="color: var(--qe-text-heading); font-weight: 700;">Welcome to QUANT EDGE</h2>
      <p style="color: var(--qe-text-secondary); font-size: 1.05rem; max-width: 600px; margin: 20px auto;">
        Institutional-grade derivatives analytics with real-time multi-source data,
        advanced pricing models, and comprehensive risk analysis.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], [
        ("🌐", "Multi-Source", "NSE India · Yahoo · CBOE · Stooq · Polygon"),
        ("💎", "5 Pricing Models", "BSM · Binomial · Monte Carlo · Heston · Merton"),
        ("📈", "Advanced Vol", "Historical · EWMA · GARCH · XGBoost · Model-Free"),
        ("🛡️", "Full Risk Suite", "Greeks · VaR/CVaR · Stress · Backtest"),
    ]):
        col.markdown(f"""
        <div class="qe-card" style="padding: 24px 20px; text-align: center; height: 160px;">
          <div style="font-size: 2.2rem; margin-bottom: 8px;">{icon}</div>
          <div class="qe-card-title" style="font-weight: 700; font-size: 0.95rem;
                       margin-bottom: 8px;">{title}</div>
          <div class="qe-card-body" style="font-size: 0.78rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Search an instrument** in the left sidebar and click **▶ ANALYSE** to begin.")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────
with st.spinner(f"Loading {ticker_input}..."):
    spot_h, hist, hist_src = get_spot_and_history(ticker_input, av_key, td_key)

if spot_h is None or hist is None:
    st.error(f"❌ No data available for **{ticker_input}**. Try a different instrument.")
    st.stop()

rt_spot, rt_src = get_realtime_spot(ticker_input, polygon_key, av_key, td_key)
spot = rt_spot if rt_spot else spot_h
rt_src = rt_src if rt_spot else f"{hist_src} (last close)"

rate = get_risk_free_rate()
div_yield = get_dividend_yield(ticker_input, spot)
ccy = get_currency_symbol(ticker_input)

with st.spinner("Fetching options chain..."):
    exps_all, chains_all, opt_src, is_synth, debug_log, nse_spot = \
        get_options_data(ticker_input, polygon_key)

if nse_spot and nse_spot > 0:
    spot = float(nse_spot)
    rt_src = "NSE India (option chain)"

if is_synth or exps_all is None:
    hv_pre = historical_vol(hist["log_return"])
    exps_all, chains_all = _synthetic_chain(spot, rate, div_yield, hv_pre or 0.25)
    opt_src = "Synthetic (BSM)"
    is_synth = True

# ── Top data strip ─────────────────────────────────────────────────────────
top_c1, top_c2, top_c3, top_c4, top_c5, top_c6 = st.columns([1.5, 1, 1, 1, 1, 1.5])
top_c1.metric("Spot", f"{ccy}{spot:,.2f}")
top_c2.metric("Risk-Free", f"{rate:.2%}")
top_c3.metric("Div Yield", f"{div_yield:.2%}")
top_c4.metric("Expiries", len(exps_all))
top_c5.metric("History", f"{len(hist)}d")
top_c6.metric("Updated", datetime.now().strftime("%H:%M:%S"))

# Source badges
rt_badge = "badge-nse" if "NSE" in rt_src else ("badge-live" if rt_spot else "badge-delayed")
opt_badge = ("badge-nse" if "NSE" in opt_src
             else "badge-synth" if is_synth else "badge-live")
st.markdown(f"""
<div style="margin: 8px 0 12px 0;">
  <span class="qe-badge {rt_badge}">📡 Price · {rt_src}</span>
  <span class="qe-badge {opt_badge}">📋 Options · {opt_src}</span>
  <span class="qe-badge badge-live">💱 {ccy}{ticker_input}</span>
</div>
""", unsafe_allow_html=True)

if show_debug and debug_log:
    with st.expander("🔧 Data Source Log", expanded=False):
        for line in debug_log:
            st.text(line)

if is_synth:
    st.warning("⚠️ **Synthetic chain in use.** For real options data try: "
                "RELIANCE.NS, TCS.NS, ^NSEI (NSE); AAPL, MSFT, SPY, TSLA (US).")

# ── Contract selectors ─────────────────────────────────────────────────────
sc1, sc2, sc3 = st.columns(3)
with sc1:
    expiry = st.selectbox("📅 Expiry", exps_all, index=min(2, len(exps_all)-1))
chain = chains_all.get(expiry, pd.DataFrame())
strikes = sorted(chain["strike"].unique().tolist()) if not chain.empty else []
def_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i]-spot)) if strikes else 0
with sc2:
    strike = float(st.selectbox("💵 Strike", strikes, index=def_idx)
                    if strikes else st.number_input("Strike", value=float(spot)))
with sc3:
    option_type = st.selectbox("📝 Type", ["call", "put"])

tau = max((datetime.strptime(expiry,"%Y-%m-%d")-datetime.now()).days, 1) / 365
pc_col = "mid_price" if (not chain.empty and "mid_price" in chain.columns) else "lastPrice"
if not chain.empty:
    rs = chain[(chain["strike"]==strike) & (chain["type"]==option_type)]
    market_price = (float(rs[pc_col].iloc[0])
                     if (not rs.empty and pc_col in rs.columns
                         and not pd.isna(rs[pc_col].iloc[0])) else np.nan)
else:
    market_price = np.nan

# ── Compute all metrics ─────────────────────────────────────────────────────
with st.spinner("Computing analytics..."):
    h_vol = historical_vol(hist["log_return"])
    e_vol = ewma_vol(hist["log_return"])
    g_vol = garch_forecast_vol(hist["log_return"])
    x_vol = xgboost_vol_forecast(hist["log_return"]) if xgb_on else np.nan
    iv = implied_vol(market_price, spot, strike, rate, div_yield, tau, option_type)
    vol_fp = iv if not np.isnan(iv) else (g_vol if not np.isnan(g_vol) else (h_vol or 0.25))
    mfiv = model_free_iv(chain, spot, rate, tau) if not chain.empty else np.nan
    pcr_oi, pcr_vol = calculate_pcr(chain) if not chain.empty else (None, None)

    bsm_p = bsm_price(spot, strike, rate, div_yield, vol_fp, tau, option_type)
    binom_p = binomial_price(spot, strike, rate, div_yield, vol_fp, tau,
                               option_type, "american", bsteps)
    mc_p, mc_err = monte_carlo_price(spot, strike, rate, div_yield,
                                       vol_fp, tau, option_type, n_mc)

    heston_p = np.nan; h_params = {}
    if heston_on and not chain.empty:
        sd = []
        for _, rr in chain[chain["type"]==option_type].iterrows():
            mp2 = rr.get(pc_col, np.nan)
            if pd.isna(mp2) or mp2 <= 0: continue
            iv2 = implied_vol(mp2, spot, float(rr["strike"]), rate, div_yield, tau, option_type)
            if not np.isnan(iv2) and 0.01 < iv2 < 3:
                sd.append((float(rr["strike"]), iv2))
        h_params = (calibrate_heston(spot, rate, div_yield, tau,
                                       [x[0] for x in sd], [x[1] for x in sd], option_type)
                     if len(sd) >= 3
                     else {"v0":vol_fp**2,"kappa":2.0,"theta":vol_fp**2,
                             "sigma_v":0.3,"rho":-0.5})
        heston_p = heston_price(spot, strike, rate, div_yield, tau,
                                 h_params["v0"], h_params["kappa"], h_params["theta"],
                                 h_params["sigma_v"], h_params["rho"], option_type)
    merton_p = (merton_jump_price(spot, strike, rate, div_yield, vol_fp, tau,
                                    opt=option_type) if merton_on else np.nan)

    greeks = bsm_greeks(spot, strike, rate, div_yield, vol_fp, tau, option_type)
    hist["ma50"] = hist["Close"].rolling(50).mean()
    hist["ma200"] = hist["Close"].rolling(200).mean()
    lma = hist.dropna(subset=["ma50","ma200"])
    trend = ("up" if not lma.empty and lma.iloc[-1]["ma50"] > lma.iloc[-1]["ma200"]
              else ("down" if not lma.empty else "flat"))
    var_dict = calculate_var_cvar(hist["log_return"], var_conf, 1, pos_val)
    fwd = theoretical_forward_price(spot, rate, div_yield, tau)

st.markdown("---")

# =============================================================================
# SECTION RENDERING
# =============================================================================

if section == "🎯 Overview":
    st.markdown('<div class="qe-section">🎯 Executive Overview</div>', unsafe_allow_html=True)

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Implied Vol", f"{iv:.1%}" if not np.isnan(iv) else "N/A")
    k2.metric("BSM Price", f"{ccy}{bsm_p:.4f}")
    k3.metric("Delta", f"{greeks['delta']:.4f}")
    k4.metric("Theta/day", f"{greeks['theta']:.4f}")

    # Price chart with MA overlay
    st.markdown("### Price Trajectory")
    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close",
                                  line=dict(color=PALETTE["primary"], width=2),
                                  fill="tozeroy", fillcolor="rgba(99,102,241,0.05)"))
    fig_px.add_trace(go.Scatter(x=hist.index, y=hist["ma50"], name="MA 50",
                                  line=dict(color=PALETTE["warning"], width=1.5, dash="dot")))
    fig_px.add_trace(go.Scatter(x=hist.index, y=hist["ma200"], name="MA 200",
                                  line=dict(color=PALETTE["danger"], width=1.5, dash="dash")))
    apply_pro_style(fig_px, f"{ticker_input} — Price History with Moving Averages", 420)
    st.plotly_chart(fig_px, use_container_width=True, theme="streamlit")

    # Signals
    ov, on_notes, _ = rec_options(iv, g_vol if not np.isnan(g_vol) else e_vol,
                                     h_vol, trend, strike/spot, int(tau*365), pcr_oi)
    fv_sig, fn_notes, _ = rec_futures(spot, fwd, 0.0, rate, div_yield, trend, tau, ccy)

    st.markdown("### Trading Signals")
    sig_c1, sig_c2 = st.columns(2)
    with sig_c1:
        st.markdown(f"""
        <div class="qe-card" style="border-left: 4px solid #6366f1; padding: 20px;">
          <div class="qe-card-title" style="font-size: 0.75rem; font-weight: 700;
                       text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
            📊 OPTIONS
          </div>
          <div class="qe-card-heading" style="font-size: 1.2rem; font-weight: 700; margin-bottom: 12px;">
            {ov}
          </div>
          <div class="qe-card-body" style="font-size: 0.85rem; line-height: 1.5;">
            {"<br>• ".join([""] + on_notes)}
          </div>
        </div>
        """, unsafe_allow_html=True)
    with sig_c2:
        st.markdown(f"""
        <div class="qe-card" style="border-left: 4px solid #22d3ee; padding: 20px;">
          <div class="qe-card-title-cyan" style="font-size: 0.75rem; font-weight: 700;
                       text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
            📦 FUTURES
          </div>
          <div class="qe-card-heading" style="font-size: 1.2rem; font-weight: 700; margin-bottom: 12px;">
            {fv_sig}
          </div>
          <div class="qe-card-body" style="font-size: 0.85rem; line-height: 1.5;">
            {"<br>• ".join([""] + fn_notes)}
          </div>
        </div>
        """, unsafe_allow_html=True)

elif section == "📊 Pricing & Volatility":
    st.markdown('<div class="qe-section">📊 Pricing & Volatility Analysis</div>',
                  unsafe_allow_html=True)

    # Vol metrics
    st.markdown("### Volatility Estimates")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Historical", f"{h_vol:.1%}" if not np.isnan(h_vol) else "N/A")
    v2.metric("EWMA", f"{e_vol:.1%}" if not np.isnan(e_vol) else "N/A")
    v3.metric("GARCH", f"{g_vol:.1%}" if not np.isnan(g_vol) else "N/A")
    v4.metric("XGBoost", f"{x_vol:.1%}" if not np.isnan(x_vol) else "N/A")
    v5.metric("Implied", f"{iv:.1%}" if not np.isnan(iv) else "N/A")
    v6.metric("Model-Free IV", f"{mfiv:.1%}" if not np.isnan(mfiv) else "N/A")

    if pcr_oi is not None:
        p1, p2 = st.columns(2)
        p1.metric("Put-Call Ratio (OI)", f"{pcr_oi:.2f}",
                   "Bearish" if pcr_oi > 1 else "Bullish")
        if pcr_vol is not None:
            p2.metric("Put-Call Ratio (Vol)", f"{pcr_vol:.2f}",
                       "Bearish" if pcr_vol > 1 else "Bullish")

    # Vol time series
    rv21 = realized_vol_series(hist["log_return"], 21)
    rv63 = realized_vol_series(hist["log_return"], 63)
    lam_e = 0.94
    var_e = hist["log_return"].dropna().iloc[0]**2
    evals = []
    for ret in hist["log_return"].dropna():
        var_e = lam_e*var_e + (1-lam_e)*ret**2
        evals.append(np.sqrt(var_e*252))
    ewma_s = pd.Series(evals, index=hist["log_return"].dropna().index)

    fig_v = go.Figure()
    fig_v.add_trace(go.Scatter(x=rv21.index, y=rv21, name="21d Realized",
                                 line=dict(color=PALETTE["primary"], width=2)))
    fig_v.add_trace(go.Scatter(x=rv63.index, y=rv63, name="63d Realized",
                                 line=dict(color=PALETTE["secondary"], width=2)))
    fig_v.add_trace(go.Scatter(x=ewma_s.index, y=ewma_s, name="EWMA",
                                 line=dict(color=PALETTE["warning"], width=1.5, dash="dot")))
    if not np.isnan(iv):
        fig_v.add_hline(y=float(iv), line_dash="dash", line_color=PALETTE["danger"],
                          annotation_text=f"Current IV: {iv:.1%}",
                          annotation_font=dict(color=PALETTE["danger"]))
    fig_v.update_yaxes(tickformat=".0%")
    apply_pro_style(fig_v, "Realized vs Implied Volatility", 420)
    st.plotly_chart(fig_v, use_container_width=True, theme="streamlit")

    st.markdown("### Option Pricing — Multi-Model Comparison")
    if is_synth:
        st.info("💡 Market price is BSM-theoretic (synthetic chain).")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("BSM", f"{ccy}{bsm_p:.4f}")
    m2.metric("Binomial", f"{ccy}{binom_p:.4f}")
    m3.metric("Monte Carlo", f"{ccy}{mc_p:.4f}", f"±{mc_err:.4f}")
    m4.metric("Heston", f"{ccy}{heston_p:.4f}" if not np.isnan(heston_p) else "N/A")
    m5.metric("Merton", f"{ccy}{merton_p:.4f}" if not np.isnan(merton_p) else "N/A")
    m6.metric("Market", f"{ccy}{market_price:.4f}" if not np.isnan(market_price) else "N/A")

    mnames = ["BSM","Binomial","Monte Carlo","Heston","Merton JD","Market"]
    mprices = [bsm_p, binom_p, mc_p,
                heston_p if not np.isnan(heston_p) else 0,
                merton_p if not np.isnan(merton_p) else 0,
                market_price if not np.isnan(market_price) else 0]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
               PALETTE["warning"], PALETTE["danger"], PALETTE["muted"]]

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        x=mnames, y=mprices, marker=dict(color=colors, line=dict(color="#0b0f1a", width=1)),
        text=[f"{ccy}{v:.4f}" for v in mprices], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Price: %{text}<extra></extra>",
    ))
    if not np.isnan(market_price):
        fig_b.add_hline(y=float(market_price), line_dash="dash",
                         annotation_text="Market Mid", annotation_position="right")
    apply_pro_style(fig_b, "Pricing Model Comparison", 400, show_legend=False)
    fig_b.update_yaxes(title_text=f"Price ({ccy})")
    st.plotly_chart(fig_b, use_container_width=True, theme="streamlit")

    # IV Smile
    if not chain.empty:
        st.markdown("### Implied Volatility Smile")
        smile = []
        for _, rr in chain[chain["type"]==option_type].iterrows():
            mp2 = rr.get(pc_col, np.nan)
            if pd.isna(mp2) or mp2 <= 0: continue
            k2 = float(rr["strike"])
            iv2 = implied_vol(mp2, spot, k2, rate, div_yield, tau, option_type)
            if not np.isnan(iv2) and 0.01 < iv2 < 3:
                smile.append({"strike":k2, "iv":iv2, "moneyness":k2/spot,
                                "oi":float(rr.get("openInterest",0) or 0)})
        if smile:
            sdf = pd.DataFrame(smile).sort_values("strike")
            fig_sm = make_subplots(specs=[[{"secondary_y":True}]])
            fig_sm.add_trace(go.Scatter(
                x=sdf["moneyness"], y=sdf["iv"], mode="lines+markers", name="IV",
                line=dict(color=PALETTE["primary"], width=2.5),
                marker=dict(size=8, color=PALETTE["primary"])
            ), secondary_y=False)
            fig_sm.add_trace(go.Bar(
                x=sdf["moneyness"], y=sdf["oi"], name="Open Interest",
                marker=dict(color=PALETTE["secondary"], opacity=0.3)
            ), secondary_y=True)
            fig_sm.add_vline(x=1.0, line_dash="dash", line_color=PALETTE["accent"],
                              annotation_text="ATM")
            fig_sm.update_yaxes(title_text="Implied Vol", tickformat=".0%", secondary_y=False)
            fig_sm.update_yaxes(title_text="Open Interest", secondary_y=True)
            fig_sm.update_xaxes(title_text="Moneyness (K/S)")
            apply_pro_style(fig_sm, "IV Smile with Open Interest Overlay", 420)
            st.plotly_chart(fig_sm, use_container_width=True, theme="streamlit")

elif section == "🏛️ Greeks":
    st.markdown('<div class="qe-section">🏛️ Risk Sensitivities (Greeks)</div>',
                  unsafe_allow_html=True)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Δ Delta", f"{greeks['delta']:.4f}")
    g2.metric("Γ Gamma", f"{greeks['gamma']:.6f}")
    g3.metric("ν Vega", f"{greeks['vega']:.4f}")
    g4.metric("Θ Theta/d", f"{greeks['theta']:.4f}")
    g5.metric("ρ Rho", f"{greeks['rho']:.4f}")

    st.markdown("### Second-Order Greeks")
    s1, s2, s3 = st.columns(3)
    s1.metric("Vanna", f"{greeks['vanna']:.4f}")
    s2.metric("Volga", f"{greeks['volga']:.4f}")
    s3.metric("Charm", f"{greeks['charm']:.4f}")

    # Greeks vs Spot
    sr_range = np.linspace(spot*0.7, spot*1.3, 100)
    deltas = [bsm_greeks(s2, strike, rate, div_yield, vol_fp, tau, option_type)["delta"] for s2 in sr_range]
    gammas = [bsm_greeks(s2, strike, rate, div_yield, vol_fp, tau, option_type)["gamma"] for s2 in sr_range]
    vannas = [bsm_greeks(s2, strike, rate, div_yield, vol_fp, tau, option_type)["vanna"] for s2 in sr_range]
    charms = [bsm_greeks(s2, strike, rate, div_yield, vol_fp, tau, option_type)["charm"] for s2 in sr_range]

    fig_gk = make_subplots(rows=2, cols=2,
                            subplot_titles=("Delta", "Gamma", "Vanna", "Charm"),
                            vertical_spacing=0.14, horizontal_spacing=0.10)
    for arr, r_, c_, col in [(deltas,1,1,PALETTE["primary"]),
                                (gammas,1,2,PALETTE["secondary"]),
                                (vannas,2,1,PALETTE["warning"]),
                                (charms,2,2,PALETTE["success"])]:
        fig_gk.add_trace(go.Scatter(x=sr_range, y=arr, line=dict(color=col, width=2.5),
                                     fill="tozeroy", fillcolor=col.replace(")",",0.1)").replace("rgb","rgba"),
                                     showlegend=False), row=r_, col=c_)
        fig_gk.add_vline(x=float(spot), line_dash="dash",
                          opacity=0.5, row=r_, col=c_)
    apply_pro_style(fig_gk, "Greeks vs Spot Price", 550, show_legend=False)
    st.plotly_chart(fig_gk, use_container_width=True, theme="streamlit")

elif section == "🌐 Volatility Surface":
    st.markdown('<div class="qe-section">🌐 Volatility Surface</div>', unsafe_allow_html=True)

    sel_exp = exps_all[:min(surf_exp, len(exps_all))]
    with st.spinner("Building surface..."):
        surf_df = build_vol_surface(
            {e:chains_all[e] for e in sel_exp if e in chains_all},
            spot, rate, div_yield, option_type)

    if surf_df.empty:
        st.warning("Not enough data to build vol surface.")
    else:
        pivot = surf_df.pivot_table(values="iv", index="expiry",
                                     columns="strike", aggfunc="mean").sort_index()
        col_vals = np.array([float(c) for c in pivot.columns])
        closest = float(col_vals[np.argmin(np.abs(col_vals-strike))])

        fig_sf = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[float(k) for k in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[[0, "#1e1b4b"],[0.25,"#312e81"],[0.5,"#6366f1"],
                         [0.75,"#22d3ee"],[1,"#10b981"]],
            colorbar=dict(title="IV", tickformat=".0%", thickness=15, len=0.75),
            hovertemplate="Strike: %{x}<br>Expiry: %{y}<br>IV: %{z:.2%}<extra></extra>",
        ))
        fig_sf.add_vline(x=closest, line_dash="dash",
                          annotation_text=f"K={strike:.0f}")
        fig_sf.update_xaxes(title="Strike")
        fig_sf.update_yaxes(title="Expiry")
        apply_pro_style(fig_sf, "Implied Volatility Surface", 480, show_legend=False)
        st.plotly_chart(fig_sf, use_container_width=True, theme="streamlit")

        # Term structure
        st.markdown("### ATM Volatility Term Structure")
        term_rows = []
        for e2 in sel_exp:
            try:
                t2 = max((datetime.strptime(e2,"%Y-%m-%d")-datetime.now()).days,1)/365
            except Exception:
                continue
            c2 = chains_all.get(e2, pd.DataFrame())
            if c2.empty: continue
            sub = c2[c2["type"]==option_type]
            if sub.empty: continue
            atm_k = float(sub.iloc[(sub["strike"]-spot).abs().argsort()[:1]]["strike"].values[0])
            arow = sub[sub["strike"]==atm_k]
            if arow.empty: continue
            mp2 = arow[pc_col if pc_col in arow.columns else "lastPrice"].iloc[0]
            iv2 = implied_vol(mp2, spot, atm_k, rate, div_yield, t2, option_type)
            if not np.isnan(iv2) and 0.01 < iv2 < 3:
                term_rows.append({"days": float(t2*365), "atm_iv": iv2, "expiry": e2})
        if term_rows:
            tdf = pd.DataFrame(term_rows).sort_values("days")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=tdf["days"], y=tdf["atm_iv"], mode="lines+markers",
                line=dict(color=PALETTE["primary"], width=3),
                marker=dict(size=12, color=PALETTE["primary"]),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                hovertemplate="Days: %{x:.0f}<br>ATM IV: %{y:.2%}<extra></extra>",
            ))
            if not np.isnan(h_vol):
                fig_ts.add_hline(y=float(h_vol), line_dash="dash",
                                  line_color=PALETTE["warning"],
                                  annotation_text=f"Hist Vol {h_vol:.1%}")
            fig_ts.update_yaxes(title="ATM Implied Vol", tickformat=".0%")
            fig_ts.update_xaxes(title="Days to Expiry")
            apply_pro_style(fig_ts, "ATM IV Term Structure", 380, show_legend=False)
            st.plotly_chart(fig_ts, use_container_width=True, theme="streamlit")

elif section == "🔮 Forecasting":
    st.markdown('<div class="qe-section">🔮 Volatility Forecasting Models</div>',
                  unsafe_allow_html=True)

    # Model comparison
    forecasts = {
        "Historical (252d)": h_vol,
        "EWMA (λ=0.94)": e_vol,
        "GARCH(1,1)": g_vol,
        "XGBoost ML": x_vol,
        "Model-Free IV": mfiv,
        "Market IV": iv,
    }
    fc_valid = {k:v for k,v in forecasts.items() if v is not None and not np.isnan(v)}

    if fc_valid:
        st.markdown("### Model Forecast Comparison")
        fig_fc = go.Figure(go.Bar(
            x=list(fc_valid.keys()), y=list(fc_valid.values()),
            marker=dict(color=[PALETTE["primary"], PALETTE["secondary"],
                                 PALETTE["success"], PALETTE["warning"],
                                 PALETTE["accent"], PALETTE["danger"]][:len(fc_valid)]),
            text=[f"{v:.1%}" for v in fc_valid.values()], textposition="outside",
        ))
        fig_fc.update_yaxes(tickformat=".0%", title="Annualised Vol")
        apply_pro_style(fig_fc, "Volatility Forecast — Cross-Model Comparison", 380, show_legend=False)
        st.plotly_chart(fig_fc, use_container_width=True, theme="streamlit")

    # GARCH conditional vol
    if HAS_ARCH and len(hist["log_return"].dropna()) > 100:
        st.markdown("### GARCH(1,1) Conditional Volatility Time Series")
        try:
            rs = hist["log_return"].dropna()*100
            gf = arch_model(rs, vol="Garch", p=1, q=1).fit(disp="off")
            cv = gf.conditional_volatility/100*np.sqrt(252)
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=cv.index, y=cv, name="GARCH Cond. Vol",
                line=dict(color=PALETTE["success"], width=2),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
            ))
            fig_g.update_yaxes(tickformat=".0%")
            apply_pro_style(fig_g, "GARCH(1,1) Historical Conditional Volatility", 380, show_legend=False)
            st.plotly_chart(fig_g, use_container_width=True, theme="streamlit")
            st.caption(f"**Model parameters:** ω={gf.params.get('omega',0):.2e}, "
                        f"α={gf.params.get('alpha[1]',0):.4f}, "
                        f"β={gf.params.get('beta[1]',0):.4f}, "
                        f"Persistence (α+β)={gf.params.get('alpha[1]',0)+gf.params.get('beta[1]',0):.4f}")
        except Exception as e:
            st.info(f"GARCH chart unavailable: {e}")

    # Return distribution
    st.markdown("### Return Distribution Analysis")
    ret_c = hist["log_return"].dropna()
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(
        x=ret_c, nbinsx=80, histnorm="probability density",
        marker=dict(color=PALETTE["primary"], opacity=0.6),
        name="Observed"
    ))
    xr = np.linspace(ret_c.min(), ret_c.max(), 200)
    fig_r.add_trace(go.Scatter(
        x=xr, y=norm.pdf(xr, ret_c.mean(), ret_c.std()),
        name="Normal Fit", line=dict(color=PALETTE["warning"], width=2.5),
    ))
    fig_r.update_xaxes(title="Log Return")
    fig_r.update_yaxes(title="Density")
    apply_pro_style(fig_r, "Return Distribution vs Normal", 380)
    st.plotly_chart(fig_r, use_container_width=True, theme="streamlit")

    # Skew/kurtosis
    skew = ret_c.skew(); kurt = ret_c.kurtosis()
    sk1, sk2, sk3 = st.columns(3)
    sk1.metric("Skewness", f"{skew:.4f}", "Negative tail" if skew < -0.1 else "Positive tail" if skew > 0.1 else "Symmetric")
    sk2.metric("Excess Kurtosis", f"{kurt:.4f}", "Heavy tails" if kurt > 1 else "Normal-ish")
    sk3.metric("Sample Size", f"{len(ret_c):,} days")

elif section == "🎨 Strategies":
    st.markdown('<div class="qe-section">🎨 Multi-Leg Strategy Builder</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns([2,1,1])
    with sc1:
        sname = st.selectbox("Strategy", list(STRATEGIES.keys()))
    with sc2:
        svol = st.slider("Vol (%)", 5, 150, int(vol_fp*100)) / 100
    with sc3:
        stau = st.slider("Days", 7, 365, max(7, int(tau*365)))

    sr2, pnl2, nprem, linfo, bes, maxp, maxl = strategy_payoff(
        sname, spot, rate, div_yield, svol, stau/365)

    ldf = pd.DataFrame(linfo)
    st.dataframe(ldf, use_container_width=True, hide_index=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Premium", f"{ccy}{nprem:.4f}", "Received" if nprem > 0 else "Paid")
    m2.metric("Breakevens", ", ".join([f"{ccy}{b:.2f}" for b in bes]) or "None")
    m3.metric("Max Profit", f"{ccy}{maxp:.2f}" if maxp < 1e5 else "Unlimited")
    m4.metric("Max Loss", f"{ccy}{abs(maxl):.2f}" if maxl > -1e5 else "Unlimited")

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=sr2, y=np.maximum(pnl2, 0), fill="tozeroy",
                                 fillcolor=f"rgba(16,185,129,0.2)",
                                 line=dict(width=0), name="Profit zone", showlegend=False))
    fig_p.add_trace(go.Scatter(x=sr2, y=np.minimum(pnl2, 0), fill="tozeroy",
                                 fillcolor=f"rgba(239,68,68,0.2)",
                                 line=dict(width=0), name="Loss zone", showlegend=False))
    fig_p.add_trace(go.Scatter(x=sr2, y=pnl2, mode="lines", name="P&L",
                                 line=dict(color=PALETTE["primary"], width=3)))
    fig_p.add_hline(y=0.0, line_dash="dash", opacity=0.5)
    fig_p.add_vline(x=float(spot), line_color=PALETTE["warning"], line_dash="dot",
                     annotation_text="Spot")
    for be in bes:
        fig_p.add_vline(x=float(be), line_color=PALETTE["success"], line_dash="dash",
                         annotation_text=f"BE {ccy}{be:.1f}")
    fig_p.update_xaxes(title=f"Spot at Expiry ({ccy})")
    fig_p.update_yaxes(title=f"P&L ({ccy})")
    apply_pro_style(fig_p, f"{sname} — Payoff at Expiry", 480)
    st.plotly_chart(fig_p, use_container_width=True, theme="streamlit")

elif section == "🛡️ Hedging":
    st.markdown('<div class="qe-section">🛡️ Delta-Hedge Simulation</div>', unsafe_allow_html=True)

    path = simulate_gbm_path(spot, rate, div_yield, vol_fp, tau, 60, seed=7)
    hlog, fpnl = simulate_delta_hedge(path, strike, rate, div_yield, vol_fp,
                                        tau, option_type, rehedge_n, hedge_cost)
    hc1, hc2, hc3 = st.columns(3)
    hc1.metric("Hedge P&L", f"{ccy}{fpnl:.4f}", "Profit" if fpnl > 0 else "Loss")
    hc2.metric("Premium Received", f"{ccy}{bsm_p:.4f}")
    hc3.metric("Rehedge Every", f"{rehedge_n} step(s)")

    fig_h = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Simulated Spot Path", "Delta Over Time"),
                            vertical_spacing=0.10)
    fig_h.add_trace(go.Scatter(y=hlog["spot"], line=dict(color=PALETTE["primary"], width=2),
                                 fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                                 showlegend=False), row=1, col=1)
    fig_h.add_hline(y=float(strike), line_dash="dash", line_color=PALETTE["warning"],
                     annotation_text="Strike", row=1, col=1)
    fig_h.add_trace(go.Scatter(y=hlog["delta"], line=dict(color=PALETTE["success"], width=2),
                                 fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
                                 showlegend=False), row=2, col=1)
    apply_pro_style(fig_h, "Delta-Hedge Path Simulation", 480, show_legend=False)
    st.plotly_chart(fig_h, use_container_width=True, theme="streamlit")

    # Multi-path distribution
    st.markdown("### 100-Path Hedge P&L Distribution")
    pnls = []
    for s_ in range(100):
        p_ = simulate_gbm_path(spot, rate, div_yield, vol_fp, tau, 60, seed=s_)
        _, pn = simulate_delta_hedge(p_, strike, rate, div_yield, vol_fp,
                                       tau, option_type, rehedge_n, hedge_cost)
        pnls.append(pn)
    parr = np.array(pnls)
    fig_pd = go.Figure()
    fig_pd.add_trace(go.Histogram(x=parr, nbinsx=30,
                                    marker=dict(color=PALETTE["primary"], opacity=0.7)))
    fig_pd.add_vline(x=0.0, line_dash="dash")
    fig_pd.add_vline(x=float(parr.mean()), line_color=PALETTE["warning"], line_dash="dot",
                      annotation_text=f"Mean {ccy}{parr.mean():.2f}")
    fig_pd.update_xaxes(title=f"P&L ({ccy})")
    fig_pd.update_yaxes(title="Frequency")
    apply_pro_style(fig_pd, "Hedge P&L Distribution (100 GBM paths)", 380, show_legend=False)
    st.plotly_chart(fig_pd, use_container_width=True, theme="streamlit")

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Mean", f"{ccy}{parr.mean():.4f}")
    dc2.metric("Std Dev", f"{ccy}{parr.std():.4f}")
    dc3.metric("Best", f"{ccy}{parr.max():.4f}")
    dc4.metric("Worst", f"{ccy}{parr.min():.4f}")

elif section == "📦 Futures":
    st.markdown('<div class="qe-section">📦 Futures & Forward Analysis</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Spot", f"{ccy}{spot:,.2f}")
    fc2.metric("Fair Forward", f"{ccy}{fwd:.2f}")
    fc3.metric("Carry", f"{rate-div_yield:+.2%}",
                "Contango" if rate > div_yield else "Backwardation")

    mkt_fut = st.number_input("Market futures price (optional)", value=0.0,
                                step=0.5, format="%.2f")

    tenors = np.array([1,7,14,30,60,90,120,180,252])/252
    fwds = [float(theoretical_forward_price(spot, rate, div_yield, float(t))) for t in tenors]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=(tenors*252).astype(int).tolist(), y=fwds,
                                  mode="lines+markers",
                                  line=dict(color=PALETTE["primary"], width=3),
                                  marker=dict(size=10, color=PALETTE["primary"]),
                                  fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"))
    fig_fc.add_hline(y=float(spot), line_dash="dot",
                      annotation_text=f"Spot {ccy}{spot:.2f}")
    if mkt_fut > 0:
        fig_fc.add_hline(y=float(mkt_fut), line_dash="dash", line_color=PALETTE["danger"],
                          annotation_text=f"Market {ccy}{mkt_fut:.2f}")
    fig_fc.update_xaxes(title="Days to Expiry")
    fig_fc.update_yaxes(title=f"Forward Price ({ccy})")
    apply_pro_style(fig_fc, "Cost-of-Carry Forward Curve", 380, show_legend=False)
    st.plotly_chart(fig_fc, use_container_width=True, theme="streamlit")

    ov, on_notes, _ = rec_options(iv, g_vol if not np.isnan(g_vol) else e_vol,
                                     h_vol, trend, strike/spot, int(tau*365), pcr_oi)
    fv_sig, fn_notes, _ = rec_futures(spot, fwd, mkt_fut, rate, div_yield, trend, tau, ccy)

    st.markdown("### Trading Signals")
    fv1, fv2 = st.columns(2)
    with fv1:
        st.markdown(f"**📊 Options:** {ov}")
        for n in on_notes:
            st.markdown(f"- {n}")
    with fv2:
        st.markdown(f"**📦 Futures:** {fv_sig}")
        for n in fn_notes:
            st.markdown(f"- {n}")

elif section == "⚠️ Risk":
    st.markdown('<div class="qe-section">⚠️ Value-at-Risk & Stress Testing</div>',
                  unsafe_allow_html=True)

    if var_dict:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f"Param VaR", f"{ccy}{var_dict['param_var']:,.0f}")
        r2.metric(f"Param CVaR", f"{ccy}{var_dict['param_cvar']:,.0f}")
        r3.metric(f"Hist VaR", f"{ccy}{var_dict['hist_var']:,.0f}")
        r4.metric(f"Hist CVaR", f"{ccy}{var_dict['hist_cvar']:,.0f}")

    st.markdown("### Stress Test Scenarios")
    sc_dict = stress_test(spot, strike, rate, div_yield, vol_fp, tau, greeks, option_type)
    srows = [{"Scenario":k, "Approx P&L":round(v["approx_pnl"],4),
                "Exact P&L":round(v["exact_pnl"],4)} for k,v in sc_dict.items()]
    sdf = pd.DataFrame(srows)

    fig_st = go.Figure(go.Bar(
        y=sdf["Scenario"], x=sdf["Exact P&L"], orientation="h",
        marker=dict(color=[PALETTE["success"] if v > 0 else PALETTE["danger"]
                            for v in sdf["Exact P&L"]]),
        text=[f"{ccy}{v:+.2f}" for v in sdf["Exact P&L"]], textposition="outside",
    ))
    fig_st.add_vline(x=0.0, line_dash="dash", opacity=0.5)
    fig_st.update_xaxes(title=f"P&L ({ccy})")
    apply_pro_style(fig_st, "Option P&L Under Stress Scenarios", 480, show_legend=False)
    st.plotly_chart(fig_st, use_container_width=True, theme="streamlit")

    st.dataframe(sdf, use_container_width=True, hide_index=True)

elif section == "🔁 Backtest":
    st.markdown('<div class="qe-section">🔁 Walk-Forward Backtest</div>', unsafe_allow_html=True)

    bt_thr = st.slider("Vol threshold (%)", 5, 30, 15) / 100
    with st.spinner("Running adaptive backtest..."):
        btdf, note = walk_forward_backtest(hist, vol_threshold=bt_thr)

    st.info(f"📊 {note}")

    if btdf.empty:
        st.warning("Insufficient data. Try a ticker with longer history.")
    else:
        st.dataframe(btdf.style.format({
            "ewma_vol":"{:.2%}","hist_vol":"{:.2%}",
            "realized_vol":"{:.2%}","approx_pnl":"{:.4f}"
        }), use_container_width=True, hide_index=True)

        ps = performance_stats(btdf["approx_pnl"])
        if ps:
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Sharpe", f"{ps.get('sharpe',0):.2f}"
                       if not np.isnan(ps.get('sharpe',np.nan)) else "N/A")
            b2.metric("Sortino", f"{ps.get('sortino',0):.2f}"
                       if not np.isnan(ps.get('sortino',np.nan)) else "N/A")
            b3.metric("Calmar", f"{ps.get('calmar',0):.2f}"
                       if not np.isnan(ps.get('calmar',np.nan)) else "N/A")
            b4.metric("Max DD", f"{ps.get('max_dd',0):.2%}")
            pv = ps.get("p_value", np.nan)
            b5.metric("Bootstrap p", f"{pv:.3f}" if not np.isnan(pv) else "N/A",
                       "✅ Sig." if not np.isnan(pv) and pv < 0.05 else "❌ Not sig.")

        fig_bt = go.Figure(go.Bar(
            x=[f"W{int(r)}" for r in btdf["window"]],
            y=btdf["approx_pnl"],
            marker=dict(color=[PALETTE["success"] if v > 0 else PALETTE["danger"]
                                 for v in btdf["approx_pnl"]]),
            text=[f"{v:+.4f}" for v in btdf["approx_pnl"]], textposition="outside"
        ))
        fig_bt.add_hline(y=0.0, line_dash="dash", opacity=0.5)
        apply_pro_style(fig_bt, "Walk-Forward P&L by Window", 380, show_legend=False)
        st.plotly_chart(fig_bt, use_container_width=True, theme="streamlit")

        btdf["cum_pnl"] = btdf["approx_pnl"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=btdf["window"], y=btdf["cum_pnl"], mode="lines+markers",
            line=dict(color=PALETTE["primary"], width=3),
            marker=dict(size=12, color=PALETTE["primary"]),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.15)",
        ))
        fig_cum.add_hline(y=0.0, line_dash="dash", opacity=0.5)
        apply_pro_style(fig_cum, "Cumulative Backtest P&L", 320, show_legend=False)
        st.plotly_chart(fig_cum, use_container_width=True, theme="streamlit")

elif section == "📋 Report":
    st.markdown('<div class="qe-section">📋 Comprehensive Analysis Report</div>',
                  unsafe_allow_html=True)

    ov2, on_notes2, _ = rec_options(iv, g_vol if not np.isnan(g_vol) else e_vol,
                                        h_vol, trend, strike/spot if spot else 1,
                                        int(tau*365), pcr_oi)
    fv2, fn_notes2, _ = rec_futures(spot, fwd, 0.0, rate, div_yield, trend, tau, ccy)

    btdf2, _ = walk_forward_backtest(hist, vol_threshold=0.15)
    ps2 = performance_stats(btdf2["approx_pnl"]) if not btdf2.empty else {}
    sc_dict = stress_test(spot, strike, rate, div_yield, vol_fp, tau, greeks, option_type)

    rd = {
        "ticker": ticker_input, "ccy": ccy, "spot": spot,
        "rate": rate, "div_yield": div_yield,
        "strike": strike, "tau": tau, "option_type": option_type,
        "data_source": f"{rt_src} | Options: {opt_src}",
        "pos_val": pos_val, "var_conf": var_conf,
        "volatility": {"hist":h_vol,"ewma":e_vol,"garch":g_vol,
                         "xgb":x_vol,"iv":iv,"mfiv":mfiv},
        "pricing": {"bsm":bsm_p,"binom":binom_p,"mc":mc_p,"mc_se":mc_err,
                      "mc_paths":n_mc,
                      "heston":heston_p if not np.isnan(heston_p) else 0,
                      "merton":merton_p if not np.isnan(merton_p) else 0,
                      "market":market_price if not np.isnan(market_price) else 0},
        "greeks": greeks, "risk": var_dict,
        "futures": {"theoretical":fwd,"carry":rate-div_yield},
        "backtest": {"df":btdf2,"perf_stats":ps2},
        "scenarios": sc_dict,
        "pcr": pcr_oi,
        "opt_verdict": ov2, "fut_verdict": fv2,
        "opt_notes": on_notes2, "fut_notes": fn_notes2,
    }
    rpt = build_professional_report(rd)

    st.download_button("⬇️ Download Full Report (Markdown)",
                        data=rpt.encode("utf-8"),
                        file_name=f"QUANT_EDGE_{ticker_input}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        mime="text/markdown", type="primary")

    with st.expander("📄 View Report Preview", expanded=True):
        st.markdown(rpt)

    save_snapshot(ticker_input, spot, iv if not np.isnan(iv) else None,
                   h_vol, e_vol, g_vol, rate, div_yield, ov2, fv2, rt_src)

    st.info("⚠️ Educational tool only. Not financial advice.")