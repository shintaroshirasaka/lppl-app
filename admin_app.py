import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st
import os
import requests
import time
import hmac
import hashlib
import base64
import io

# =======================================================
# DESIGN SYSTEM: Tech-Luxury Theme
# =======================================================
THEME = {
    "bg_main": "#0E1117",       # Streamlit default dark
    "bg_chart": "#121212",      # Deep matte black for charts
    "gold": "#C5A059",          # Champagne Gold (Key Accent)
    "platinum": "#E0E0E0",      # Platinum (Main Text/Line)
    "grid": "#333333",          # Subtle Grid
    "red_glow": "#FF453A",      # Sophisticated Red
    "blue_glow": "#0A84FF",     # Sophisticated Blue
    "green_glow": "#30D158",    # Sophisticated Green
    "font_serif": "DejaVu Serif", 
    "font_sans": "DejaVu Sans", 
}

# Set Global Matplotlib Styles for Luxury Feel
plt.rcParams['font.family'] = THEME['font_sans']
plt.rcParams['axes.facecolor'] = THEME['bg_chart']
plt.rcParams['figure.facecolor'] = THEME['bg_main']
plt.rcParams['text.color'] = THEME['platinum']
plt.rcParams['axes.labelcolor'] = '#888888'
plt.rcParams['xtick.color'] = '#888888'
plt.rcParams['ytick.color'] = '#888888'
plt.rcParams['grid.color'] = THEME['grid']
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.edgecolor'] = '#333333'

# =======================================================
# AUTH GATE: Require signed short-lived token (?t=...)
# =======================================================
def _b64url_decode(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("utf-8"))

def verify_token(token: str, secret: str) -> tuple[bool, str]:
    try:
        part_payload, part_sig = token.split(".", 1)
        payload = _b64url_decode(part_payload).decode("utf-8")
        sig = _b64url_decode(part_sig).decode("utf-8")
        email, exp_str = payload.split("|", 1)
        exp = int(exp_str)
        if time.time() > exp:
            return (False, "")
        expected = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return (False, "")
        return (True, email)
    except Exception:
        return (False, "")

# ---- REQUIRE TOKEN (no warnings: use st.query_params only) ----
OS_TOKEN_SECRET = os.environ.get("OS_TOKEN_SECRET_ADMIN", "").strip()
token = st.query_params.get("t", "")

if not OS_TOKEN_SECRET or not token:
    st.stop()

ok, authed_email = verify_token(token, OS_TOKEN_SECRET)
if not ok:
    st.stop()

# ---- OPTIONAL ADMIN EMAIL ALLOWLIST ----
ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
)
if ADMIN_EMAILS:
    if authed_email.strip().lower() not in ADMIN_EMAILS:
        st.stop()


# =======================================================
# Cache settings
# =======================================================
PRICE_TTL_SECONDS = 15 * 60         # yfinance cache
FIT_TTL_SECONDS = 24 * 60 * 60      # fit cache
SCRAPE_TTL_SECONDS = 60 * 60        # scraping cache (1 hour)


# =======================================================
# FINAL SCORE SETTINGS (calendar-day distance)
# =======================================================
UP_FUTURE_NEAR_DAYS = 30
UP_FUTURE_FAR_DAYS  = 180
UP_PAST_NEAR_DAYS   = 7
UP_PAST_FAR_DAYS    = 120
DOWN_FUTURE_NEAR_DAYS = 7
DOWN_FUTURE_FAR_DAYS  = 60
DOWN_PAST_NEAR_DAYS   = 7
DOWN_PAST_FAR_DAYS    = 60


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _lin_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x0 == x1:
        return y0
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def _clamp_p0_into_bounds(p0, lb, ub, eps=1e-6):
    p0 = np.asarray(p0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


# =======================================================
# Mid-term Quant Table Settings
# =======================================================
MID_TH = {
    "A_R_ok": 0.70,     "A_R_ng": 0.40,
    "B_R2_ok": 0.50,    "B_R2_ng": 0.20,
    "C_beta_ok_low": 0.80, "C_beta_ok_high": 1.50,
    "C_beta_ng_low": 0.50, "C_beta_ng_high": 2.00,
    "E_cmgr_ok": 0.0075, "E_cmgr_ng": 0.0000,
    "F_sharpe_ok": 1.00, "F_sharpe_ng": 0.50,
    "G_mdd_ok": -0.15,   "G_mdd_ng": -0.30,
    "F2_alpha_ok": 0.05, "F2_alpha_ng": 0.00,
    "F3_alpha_ok": 0.05, "F3_alpha_ng": 0.00,
    "H_R2_ok": 0.25,     "H_R2_ng": 0.10,
    "I_beta_ok_low": 0.60, "I_beta_ok_high": 1.40,
    "I_beta_ng_low": 0.30, "I_beta_ng_high": 2.00,
    "J_vol_ok": 0.25,    "J_vol_ng": 0.40,
}

def _mid_beta_r2(x: pd.Series, y: pd.Series):
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, r2

def _mid_fmt_value(metric: str, v: float) -> str:
    if metric.startswith("E: CMGR"):
        return f"{v*100:.2f}%"
    if metric in ["D: Max Drawdown", "F2: Alpha (Relative Outperformance vs Benchmark)"]:
        return f"{v*100:.2f}%"
    if metric.startswith("F: Sharpe"):
        return f"{v:.2f}"
    if metric.startswith("F3: Alpha"):
        return f"{v*100:.2f}%"
    if metric.startswith("D: Deviation Vol"):
        return f"{v:.2f} (pp)"
    if metric.startswith("J: Rolling Vol"):
        return f"{v*100:.2f}%"
    return f"{v:.4f}"

def _mid_judge(metric: str, value: float, dev: pd.DataFrame, bench: str, window: int) -> str:
    if metric == "A: Deviation R":
        if value >= MID_TH["A_R_ok"]: return "OK"
        if value < MID_TH["A_R_ng"]:  return "NG"
        return "Caution"
    if metric == "B: Deviation R²":
        if value >= MID_TH["B_R2_ok"]: return "OK"
        if value < MID_TH["B_R2_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("C: Deviation Beta"):
        if MID_TH["C_beta_ok_low"] <= value <= MID_TH["C_beta_ok_high"]: return "OK"
        if value < MID_TH["C_beta_ng_low"] or value > MID_TH["C_beta_ng_high"]: return "NG"
        return "Caution"
    if metric.startswith("D: Deviation Vol"):
        bench_latest = dev[bench].rolling(window).std(ddof=1).dropna().iloc[-1]
        ratio = value / bench_latest if bench_latest != 0 else np.nan
        if ratio <= 1.5: return "OK"
        if ratio > 2.5:  return "NG"
        return "Caution"
    if metric == "D: Max Drawdown":
        if value >= MID_TH["G_mdd_ok"]: return "OK"
        if value < MID_TH["G_mdd_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("E: CMGR"):
        if value >= MID_TH["E_cmgr_ok"]: return "OK"
        if value < MID_TH["E_cmgr_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("F: Sharpe"):
        if value >= MID_TH["F_sharpe_ok"]: return "OK"
        if value < MID_TH["F_sharpe_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("F2: Alpha"):
        if value >= MID_TH["F2_alpha_ok"]: return "OK"
        if value < MID_TH["F2_alpha_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("F3: Alpha"):
        if value >= MID_TH["F3_alpha_ok"]: return "OK"
        if value < MID_TH["F3_alpha_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("H: Rolling R²"):
        if value >= MID_TH["H_R2_ok"]: return "OK"
        if value < MID_TH["H_R2_ng"]:  return "NG"
        return "Caution"
    if metric.startswith("I: Rolling Beta"):
        if MID_TH["I_beta_ok_low"] <= value <= MID_TH["I_beta_ok_high"]: return "OK"
        if value < MID_TH["I_beta_ng_low"] or value > MID_TH["I_beta_ng_high"]: return "NG"
        return "Caution"
    if metric.startswith("J: Rolling Vol"):
        if value <= MID_TH["J_vol_ok"]: return "OK"
        if value > MID_TH["J_vol_ng"]:  return "NG"
        return "Caution"
    return "Caution"

def _mid_threshold_text(metric: str) -> str:
    return "-" # Simplified for English view


# =======================================================
# Price fetch helpers
# =======================================================
def _pick_price_field(df: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            raise ValueError("NO_PRICE_FIELD")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("NO_PRICE_FIELD")
    return s

@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_price_series_cached(ticker: str, start_date: date, end_date: date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")
    s = _pick_price_field(df, ticker).dropna()
    if s.empty or len(s) < 30:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")
    vals = s.to_numpy(dtype="float64")
    if not np.all(np.isfinite(vals)):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")
    if np.any(vals <= 0):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")
    return s

@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_prices_pair_cached(ticker: str, bench: str, start_date: date, end_date: date) -> pd.DataFrame:
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        [ticker, bench],
        start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError("NO_DATA_PAIR")
    s_t = _pick_price_field(df, ticker).dropna()
    s_b = _pick_price_field(df, bench).dropna()
    close = pd.concat([s_t.rename(ticker), s_b.rename(bench)], axis=1).dropna()
    if close.empty or len(close) < 30:
        raise ValueError("NO_DATA_PAIR")
    vals = close.to_numpy(dtype="float64")
    if not np.all(np.isfinite(vals)):
        raise ValueError("NO_DATA_PAIR")
    if np.any(vals <= 0):
        raise ValueError("NO_DATA_PAIR")
    return close


def build_midterm_quant_table(ticker: str, bench: str, start_date: date, end_date: date, window: int = 20) -> pd.DataFrame:
    prices = fetch_prices_pair_cached(ticker, bench, start_date, end_date)
    if len(prices) < window + 10:
        raise ValueError("Not enough data points for mid-term table.")
    base = prices.iloc[0]
    index100 = prices / base * 100
    dev = index100 - 100
    ret = np.log(prices / prices.shift(1)).dropna()
    A_R = dev[ticker].corr(dev[bench])
    B_R2 = A_R ** 2
    C_beta, _ = _mid_beta_r2(dev[bench], dev[ticker])
    D_vol_latest = dev[ticker].rolling(window).std(ddof=1).dropna().iloc[-1]
    start_dt = prices.index[0].date()
    end_dt = prices.index[-1].date()
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    months = max(months, 1)
    E_cmgr = (prices[ticker].iloc[-1] / prices[ticker].iloc[0]) ** (1 / months) - 1
    F_sharpe = ret[ticker].mean() / ret[ticker].std(ddof=1) * np.sqrt(252)
    p = prices[ticker]
    dd = p / p.cummax() - 1
    max_dd = dd.min()
    max_dd_date = dd.idxmin().date()
    ticker_total = prices[ticker].iloc[-1] / prices[ticker].iloc[0]
    bench_total = prices[bench].iloc[-1] / prices[bench].iloc[0]
    F2_alpha_rel = (ticker_total / bench_total) - 1
    X = ret[bench]
    Y = ret[ticker]
    beta_daily = np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1)
    alpha_daily = Y.mean() - beta_daily * X.mean()
    F3_alpha_annual = alpha_daily * 252
    roll_r = ret[ticker].rolling(window).corr(ret[bench])
    H_R2_latest = (roll_r ** 2).dropna().iloc[-1]
    I_beta_latest = (ret[ticker].rolling(window).cov(ret[bench]) / ret[bench].rolling(window).var(ddof=1)).dropna().iloc[-1]
    J_vol_annual_latest = (ret[ticker].rolling(window).std(ddof=1) * np.sqrt(252)).dropna().iloc[-1]

    # ENGLISH ROWS
    rows = [
        ("Market Fit", "A: Deviation R", A_R, _mid_threshold_text("A: Deviation R"), "Trend Similarity"),
        ("Market Fit", "B: Deviation R2", B_R2, _mid_threshold_text("B: Deviation R²"), "Trend Explanation"),
        ("Market Fit", f"C: Deviation Beta (vs {bench})", C_beta, _mid_threshold_text(f"C: Deviation Beta"), "Deviation Sensitivity"),
        ("Market Fit", f"D: Deviation Vol (rolling {window})", D_vol_latest, _mid_threshold_text(f"D: Deviation Vol"), "Deviation Std"),
        ("Market Fit", "D: Max Drawdown", max_dd, _mid_threshold_text("D: Max Drawdown"), f"Max Drawdown (Peak: {max_dd_date})"),
        ("Efficiency", f"E: CMGR (monthly, ~{months} m)", E_cmgr, _mid_threshold_text(f"E: CMGR"), "Monthly CMGR"),
        ("Efficiency", "F: Sharpe (annualized)", F_sharpe, _mid_threshold_text("F: Sharpe"), "Risk Adjusted Return"),
        ("Efficiency", "F2: Alpha (Relative)", F2_alpha_rel, _mid_threshold_text("F2: Alpha"), "Total Excess Return"),
        ("Efficiency", "F3: Alpha (Annualized)", F3_alpha_annual, _mid_threshold_text("F3: Alpha"), "Market Neutral Alpha"),
        ("Causality", f"H: Rolling R2 (daily, {window})", H_R2_latest, _mid_threshold_text(f"H: Rolling R²"), "Recent Relation"),
        ("Causality", f"I: Rolling Beta (daily, {window})", I_beta_latest, _mid_threshold_text(f"I: Rolling Beta"), "Recent Beta"),
        ("Causality", f"J: Rolling Vol (annual, {window})", J_vol_annual_latest, _mid_threshold_text(f"J: Rolling Vol"), "Recent Volatility"),
    ]

    df = pd.DataFrame(rows, columns=["Block", "Metric", "Value", "Threshold", "Note"])
    df["Judgement"] = [_mid_judge(m, v, dev=dev, bench=bench, window=window) for m, v in zip(df["Metric"], df["Value"])]
    df_display = df.copy()
    df_display["Value"] = [_mid_fmt_value(m, v) for m, v in zip(df["Metric"], df["Value"])]
    return df_display


# -------------------------------------------------------
# Bubble decision flow
# -------------------------------------------------------
def bubble_judgement(r2: float, m: float, omega: float) -> tuple[str, dict]:
    info = {
        "r2_ok": bool(np.isfinite(r2) and r2 >= 0.65),
        "m_ok": bool(np.isfinite(m) and (0.25 <= m <= 0.70)),
        "omega_ok": bool(np.isfinite(omega) and (6.0 <= omega <= 13.0)),
        "omega_high": bool(np.isfinite(omega) and omega >= 18.0),
    }
    # ENGLISH VERDICTS
    if not info["r2_ok"]: return "Pending (Weak Shape)", info
    if not info["m_ok"]: return "Normal Uptrend", info
    if info["omega_ok"]: return "Bubble-like", info
    if info["omega_high"]: return "Pseudo-Bubble (Fast)", info
    return "Pseudo-Bubble", info

def render_bubble_flow(r2: float, m: float, omega: float):
    verdict, info = bubble_judgement(r2, m, omega)
    def yn(v: bool) -> str: return "YES" if v else "NO"
    lines = []
    lines.append(f"1. R2 >= 0.65 ?   (R2={r2:.2f})")
    lines.append(f" |  {yn(info['r2_ok'])} -> " + ("Next" if info["r2_ok"] else "Pending"))
    if info["r2_ok"]:
        lines.append(" |")
        lines.append(f" |  2. m in [0.25, 0.70] ?   (m={m:.2f})")
        lines.append(f" |     {yn(info['m_ok'])} -> " + ("Next" if info["m_ok"] else "Normal Uptrend"))
        if info["m_ok"]:
            lines.append(" |     |")
            lines.append(f" |     |  3. omega in [6, 13] ?   (w={omega:.2f})")
            if info["omega_ok"]:
                lines.append(" |     |     YES -> Bubble-like")
            else:
                lines.append(" |     |     NO -> Pseudo-Bubble")
    lines.append("")
    lines.append(f"[Verdict] {verdict}")
    st.markdown("### Bubble Check Flow (Admin)")
    st.code("\n".join(lines), language="text")

def admin_interpretation_text(bubble_res: dict, end_date: date) -> tuple[str, list[str]]:
    pdict = bubble_res.get("param_dict", {})
    r2 = float(bubble_res.get("r2", float("nan")))
    m = float(pdict.get("m", float("nan")))
    omega = float(pdict.get("omega", float("nan")))
    tc_date = pd.Timestamp(bubble_res.get("tc_date")).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    days_to_tc = int((tc_date - end_norm).days)
    bullets: list[str] = []
    
    WARNING_DAYS = 14
    
    if days_to_tc < 0:
        bullets.append(f"Passed t_c by {abs(days_to_tc)} days.")
    elif days_to_tc <= WARNING_DAYS:
        bullets.append(f"Days to t_c: {days_to_tc} (Warning Zone < 14).")
    else:
        bullets.append(f"Days to t_c: {days_to_tc} (Safe Zone).")
        
    verdict, _ = bubble_judgement(r2, m, omega)
    bullets.append(f"Verdict: {verdict}")
    
    return verdict, bullets


# -------------------------------------------------------
# LPPL Model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)

def fit_lppl_bubble(price_series: pd.Series):
    """Uptrend fit (robust bounds) - FIXED DATE CALCULATION"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)
    N = len(t)
    
    # Initial guess
    A_init = float(np.mean(log_price))
    p0 = [A_init, -1.0, 0.1, 0.5, N + 20, 8.0, 0.0]
    
    # Bounds
    A_low = float(np.min(log_price) - 2.0)
    A_high = float(np.max(log_price) + 2.0)
    lower_bounds = [A_low, -20, -20, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N + 250, 25.0, np.pi]
    
    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)
    
    params, _ = curve_fit(lppl, t, log_price, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
    
    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)
    
    ss_res = float(np.sum((log_price - log_fit) ** 2))
    ss_tot = float(np.sum((log_price - log_price.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((log_price - log_fit) ** 2)))
    
    # --- FIX: Correct date conversion ---
    tc_days = float(params[4])
    last_idx = N - 1
    future_units = tc_days - last_idx
    last_date = price_series.index[-1]
    tc_date = last_date + timedelta(days=future_units)
    
    A, B, C, m, tc, omega, phi = [float(x) for x in params]
    abs_c_over_b = float(abs(C / B)) if abs(B) > 1e-12 else float("inf")
    log_period = float(2.0 * np.pi / omega) if omega != 0 else float("inf")
    
    bounds_info = {"A_low": A_low, "A_high": A_high, "B_low": -20.0, "B_high": 20.0, "C_low": -20.0, "C_high": 20.0,
                   "m_low": 0.01, "m_high": 0.99, "tc_low": float(N + 1), "tc_high": float(N + 250),
                   "omega_low": 2.0, "omega_high": 25.0, "phi_low": float(-np.pi), "phi_high": float(np.pi)}
                    
    return {
        "params": np.asarray(params, dtype=float), 
        "param_dict": {"A": A, "B": B, "C": C, "m": m, "tc": tc, "omega": omega, "phi": phi, "abs_C_over_B": abs_c_over_b, "log_period_2pi_over_omega": log_period}, 
        "price_fit": price_fit, 
        "log_fit": log_fit, 
        "r2": float(r2), 
        "rmse": rmse, 
        "tc_date": tc_date, 
        "tc_days": tc_days, 
        "bounds_info": bounds_info, 
        "N": int(N)
    }

def fit_lppl_negative_bubble(price_series: pd.Series, peak_date, min_points: int = 10, min_drop_ratio: float = 0.03):
    """Downtrend fit (negative bubble) - FIXED DATE CALCULATION"""
    down_series = price_series[price_series.index >= peak_date].copy()
    if len(down_series) < min_points: return {"ok": False}
    
    peak_price = float(price_series.loc[peak_date])
    last_price = float(down_series.iloc[-1])
    drop_ratio = (peak_price - last_price) / peak_price
    if drop_ratio < min_drop_ratio: return {"ok": False}
    
    price_down = down_series.values.astype(float)
    t_down = np.arange(len(price_down), dtype=float)
    log_down = np.log(price_down)
    neg_log_down = -log_down
    N_down = len(t_down)
    
    A_init = float(np.mean(neg_log_down))
    p0 = [A_init, -1.0, 0.1, 0.5, N_down + 15, 8.0, 0.0]
    
    A_low = float(np.min(neg_log_down) - 2.0)
    A_high = float(np.max(neg_log_down) + 2.0)
    lower_bounds = [A_low, -20, -20, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N_down + 200, 25.0, np.pi]
    
    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)
    
    try:
        params_down, _ = curve_fit(lppl, t_down, neg_log_down, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
    except Exception:
        return {"ok": False}
        
    neg_log_fit = lppl(t_down, *params_down)
    log_fit = -neg_log_fit
    price_fit_down = np.exp(log_fit)
    
    ss_res = np.sum((neg_log_down - neg_log_fit) ** 2)
    ss_tot = np.sum((neg_log_down - neg_log_down.mean()) ** 2)
    r2_down = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # --- FIX: Correct date conversion for negative bubble ---
    tc_days = float(params_down[4])
    last_idx = N_down - 1
    future_units = tc_days - last_idx
    last_date = down_series.index[-1]
    tc_bottom_date = last_date + timedelta(days=future_units)
    
    return {
        "ok": True, 
        "down_series": down_series, 
        "price_fit_down": price_fit_down, 
        "r2": float(r2_down), 
        "tc_date": tc_bottom_date, 
        "tc_days": tc_days, 
        "params": np.asarray(params_down, dtype=float)
    }


# -------------------------------------------------------
# Cache helpers
# -------------------------------------------------------
def series_cache_key(s: pd.Series) -> str:
    idx = s.index.astype("int64").to_numpy()
    vals = s.to_numpy(dtype="float64")
    h = hashlib.sha256()
    h.update(idx.tobytes())
    h.update(vals.tobytes())
    return h.hexdigest()

@st.cache_data(ttl=FIT_TTL_SECONDS, show_spinner=False)
def fit_lppl_bubble_cached(price_key: str, price_values: np.ndarray, idx_int: np.ndarray):
    idx = pd.to_datetime(idx_int)
    s = pd.Series(price_values, index=idx)
    return fit_lppl_bubble(s)

@st.cache_data(ttl=FIT_TTL_SECONDS, show_spinner=False)
def fit_lppl_negative_bubble_cached(price_key: str, price_values: np.ndarray, idx_int: np.ndarray, peak_date_int: int, min_points: int, min_drop_ratio: float):
    idx = pd.to_datetime(idx_int)
    s = pd.Series(price_values, index=idx)
    peak_date = pd.to_datetime(peak_date_int)
    return fit_lppl_negative_bubble(s, peak_date=peak_date, min_points=min_points, min_drop_ratio=min_drop_ratio)


# =======================================================
# (FIXED) ROBUST SCRAPING FUNCTION: Multi-Source + IRBank + Yahoo
# =======================================================
@st.cache_data(ttl=SCRAPE_TTL_SECONDS, show_spinner=False)
def fetch_jp_margin_data_robust(ticker: str) -> pd.DataFrame:
    """
    Multiple Sources for JP Margin data.
    """
    code = ticker.replace(".T", "").strip()
    if not code.isdigit():
        return pd.DataFrame()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Referer": "https://www.google.com/"
    }

    # Data cleaning helper
    def clean_num(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('株', '').replace('円', '').replace('倍', '').replace('%', '').replace('-', '0').strip()
            if not x: return 0
            try: return float(x)
            except: return 0
        return x

    def parse_date_col(df, col_name="Date"):
        dates = pd.to_datetime(df[col_name], errors='coerce')
        if dates.isna().any():
            today = date.today()
            current_year = today.year
            def try_parse(x):
                if pd.isna(x) or str(x).strip() == "": return pd.NaT
                x_clean = str(x).split(' ')[0]
                x_clean = x_clean.replace("年", "/").replace("月", "/").replace("日", "")
                try: return pd.to_datetime(x_clean)
                except: pass
                try:
                    dt = pd.to_datetime(f"{current_year}/{x_clean}")
                    if dt.date() > today + timedelta(days=2):
                        dt = dt.replace(year=current_year - 1)
                    return dt
                except: return pd.NaT
            dates = df[col_name].apply(try_parse)
        return dates

    # ----- 1. IRBank Scraper (Priority) -----
    def _scrape_irbank():
        urls = [f"https://irbank.net/{code}/fee", f"https://irbank.net/{code}/karauri"]
        for url in urls:
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code != 200: continue
                try: dfs = pd.read_html(io.BytesIO(res.content))
                except: continue
                target = pd.DataFrame()
                for df in dfs:
                    col_str = " ".join([str(c) for c in df.columns])
                    if "日付" in col_str and ("逆日歩" in col_str or "融資" in col_str or "貸株" in col_str or "残" in col_str):
                        target = df
                        if isinstance(target.columns, pd.MultiIndex):
                            target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                        else:
                            target.columns = [str(c).strip() for c in target.columns]
                        break
                if target.empty: continue
                rename_map = {}
                for c in target.columns:
                    if "日付" in c: rename_map[c] = "Date"
                    elif "逆日歩" in c: rename_map[c] = "Hibush"
                    elif "融資" in c and "残" in c: rename_map[c] = "MarginBuy"
                    elif "貸株" in c and "残" in c: rename_map[c] = "MarginSell"
                    elif "売残" in c and "信用" not in c: rename_map[c] = "MarginSell"
                    elif "買残" in c and "信用" not in c: rename_map[c] = "MarginBuy"
                target = target.rename(columns=rename_map)
                if "Date" not in target.columns: continue
                target["Date"] = parse_date_col(target, "Date")
                for col in ["MarginBuy", "MarginSell", "Hibush"]:
                    if col in target.columns: target[col] = target[col].apply(clean_num)
                df_clean = target.dropna(subset=["Date"]).sort_values("Date")
                if not df_clean.empty and ("MarginBuy" in df_clean.columns or "MarginSell" in df_clean.columns):
                    return df_clean
            except: continue
        return pd.DataFrame()

    # ----- 2. Minkabu Scraper -----
    def _scrape_minkabu():
        url = f"https://minkabu.jp/stock/{code}/margin"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and ("融資" in col_str or "貸株" in col_str or "残" in col_str):
                    target = df
                    if isinstance(target.columns, pd.MultiIndex):
                        target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                    else:
                        target.columns = [str(c).strip() for c in target.columns]
                    if "逆日歩" in col_str or "品貸料" in col_str: break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "逆日歩" in c or "品貸料" in c: rename_map[c] = "Hibush"
                elif ("融資" in c and "残" in c) or ("買残" in c) or ("買い" in c and "残" in c):
                    if "増減" not in c and "回転" not in c: rename_map[c] = "MarginBuy"
                elif ("貸株" in c and "残" in c) or ("売残" in c) or ("売り" in c and "残" in c):
                    if "増減" not in c and "回転" not in c: rename_map[c] = "MarginSell"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell", "Hibush"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 3. Karauri.net Scraper -----
    def _scrape_karauri():
        url = f"https://karauri.net/{code}/"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and ("売残" in col_str or "貸株" in col_str):
                    target = df
                    if isinstance(target.columns, pd.MultiIndex):
                        target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                    else:
                        target.columns = [str(c).strip() for c in target.columns]
                    if "逆日歩" in col_str: break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "逆日歩" in c: rename_map[c] = "Hibush"
                elif ("融資" in c and "残" in c) or ("買残" in c):
                    if "増減" not in c: rename_map[c] = "MarginBuy"
                elif ("貸株" in c and "残" in c) or ("売残" in c):
                    if "増減" not in c: rename_map[c] = "MarginSell"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell", "Hibush"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 4. Yahoo Finance JP Scraper (Backup) -----
    def _scrape_yahoo():
        url = f"https://finance.yahoo.co.jp/quote/{code}.T/margin"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and "売り残" in col_str and "買い残" in col_str:
                    target = df
                    target.columns = [str(c).strip() for c in target.columns]
                    break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "売り残" in c and "増減" not in c: rename_map[c] = "MarginSell"
                elif "買い残" in c and "増減" not in c: rename_map[c] = "MarginBuy"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            target["Hibush"] = 0
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 5. Kabutan Scraper (Final Fallback) -----
    def _scrape_kabutan():
        url = f"https://kabutan.jp/stock/finance?code={code}&mode=m"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and "売残" in col_str and "買残" in col_str:
                    target = df
                    target.columns = [str(c).strip() for c in target.columns]
                    break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "売残" in c and "増減" not in c: rename_map[c] = "MarginSell"
                elif "買残" in c and "増減" not in c: rename_map[c] = "MarginBuy"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            target["Hibush"] = 0
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # --- EXECUTION FLOW ---
    df = _scrape_irbank()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
    
    df = _scrape_minkabu()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
    
    df = _scrape_karauri()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df

    df = _scrape_yahoo()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
        
    df = _scrape_kabutan()
    return df


# =======================================================
# (ADDED BACK) Compute Signal and Score Function
# =======================================================
def compute_signal_and_score(tc_up_date, end_date, down_tc_date) -> tuple[str, int]:
    now = pd.Timestamp(end_date).normalize()
    tc_up = pd.Timestamp(tc_up_date).normalize()
    
    # Priority 1: Downtrend Risk
    if down_tc_date is not None:
        down_tc = pd.Timestamp(down_tc_date).normalize()
        delta = (down_tc - now).days
        if delta > 0:
            s = _lin_map(delta, DOWN_FUTURE_NEAR_DAYS, DOWN_FUTURE_FAR_DAYS, 90, 80)
            return ("HIGH", int(round(_clamp(s, 80, 90))))
        past = abs(delta)
        s = _lin_map(past, DOWN_PAST_NEAR_DAYS, DOWN_PAST_FAR_DAYS, 90, 100)
        return ("HIGH", int(round(_clamp(s, 90, 100))))
    
    # Priority 2: Uptrend Caution
    gap = (tc_up - now).days
    WARNING_BUFFER = 14

    # SAFE
    if gap > WARNING_BUFFER:
        s = _lin_map(gap, UP_FUTURE_NEAR_DAYS, UP_FUTURE_FAR_DAYS, 59, 0)
        return ("SAFE", int(round(_clamp(s, 0, 59))))

    # CAUTION
    past_warning = WARNING_BUFFER - gap
    s = _lin_map(past_warning, 0, UP_PAST_FAR_DAYS, 60, 79)
    return ("CAUTION", int(round(_clamp(s, 60, 79))))


# =======================================================
# Tech-Luxury Rendering Functions (NEW)
# =======================================================
def draw_luxury_watermark(ax):
    """Draws a subtle, high-end watermark."""
    text_obj = ax.text(0.98, 0.02, "OUT-STANDER", transform=ax.transAxes,
            fontsize=14, color=THEME['gold'], fontweight='normal',
            fontname=THEME['font_serif'], alpha=0.4, ha='right', va='bottom', zorder=1)

def annotate_line_end(ax, x_date, y_val, text, color, offset=(5, 0)):
    """Places text directly at the end of a line (No Legend Box)."""
    ax.annotate(
        text, 
        xy=(x_date, y_val), 
        xytext=offset, 
        textcoords="offset points",
        color=color, 
        fontsize=9, 
        fontweight='bold', 
        fontfamily=THEME['font_sans'],
        va='center',
        bbox=dict(boxstyle="square,pad=0.1", fc=THEME['bg_chart'], ec="none", alpha=0.7)
    )

def render_tech_luxury_chart(price_series, bubble_res, neg_res, ticker, score, signal_label, gain):
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # 1. Main Price Line (Platinum, thin, crisp)
    ax.plot(price_series.index, price_series.values, color=THEME['platinum'], 
            linewidth=1.2, alpha=0.9, zorder=5)
    # Direct Label for Ticker
    annotate_line_end(ax, price_series.index[-1], price_series.values[-1], 
                      ticker, THEME['platinum'])

    # 2. Up Model (Champagne Gold, Smooth)
    ax.plot(price_series.index, bubble_res["price_fit"], color=THEME['gold'], 
            linewidth=1.8, alpha=1.0, zorder=6)
    annotate_line_end(ax, price_series.index[-1], bubble_res["price_fit"][-1], 
                      "Model", THEME['gold'], offset=(5, 10))

    # 3. Events (Vertical Lines with Top Labels)
    tc_date = bubble_res["tc_date"]
    tc_str = pd.Timestamp(tc_date).strftime('%Y-%m-%d')
    ax.axvline(tc_date, color=THEME['red_glow'], linestyle=":", linewidth=1, alpha=0.8)
    ax.text(tc_date, ax.get_ylim()[1], f"Turn\n{tc_str}", color=THEME['red_glow'], 
            ha='center', va='bottom', fontsize=8, backgroundcolor=THEME['bg_chart'])

    peak_date = price_series.idxmax()
    peak_str = pd.Timestamp(peak_date).strftime('%Y-%m-%d')
    ax.axvline(peak_date, color='#555555', linestyle=":", linewidth=1, alpha=0.6)
    ax.text(peak_date, ax.get_ylim()[1], f"Peak\n{peak_str}", color='#888888', 
            ha='center', va='bottom', fontsize=8, backgroundcolor=THEME['bg_chart'])

    # 4. Down Model (if exists)
    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color=THEME['blue_glow'], linewidth=1.2, alpha=0.8)
        ax.plot(down.index, neg_res["price_fit_down"], color=THEME['green_glow'], 
                linestyle="--", linewidth=1.5, alpha=0.8)
        
        tc_down = neg_res["tc_date"]
        tc_d_str = pd.Timestamp(tc_down).strftime('%Y-%m-%d')
        ax.axvline(tc_down, color=THEME['green_glow'], linestyle=":", linewidth=1)
        ax.text(tc_down, ax.get_ylim()[0], f"Bottom\n{tc_d_str}", color=THEME['green_glow'], 
                ha='center', va='top', fontsize=8)

    # Watermark
    draw_luxury_watermark(ax)
    
    return fig


# =======================================================
# Render Graph Pack (UPDATED for Dark Theme)
# =======================================================
def render_graph_pack_from_prices(prices, ticker, bench, window=20, trading_days=252):
    base = prices.iloc[0]
    index100 = prices / base * 100.0
    dev = index100 - 100.0
    ret = np.log(prices / prices.shift(1)).dropna()

    # 1. Cumulative
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(index100.index, index100[ticker], label=f"{ticker}", color=THEME['gold'])
    ax.plot(index100.index, index100[bench],  label=f"{bench}",  color=THEME['blue_glow'])
    ax.set_title("Cumulative Performance (Index = 100)", color="white")
    ax.legend(facecolor=THEME['bg_chart'], labelcolor="white")
    st.pyplot(fig)

    # 2. Scatter Dev
    X = dev[bench].dropna(); Y = dev[ticker].dropna()
    common = X.index.intersection(Y.index); X = X.loc[common]; Y = Y.loc[common]
    slope_dev, intercept_dev = np.polyfit(X.values, Y.values, 1)
    x_sorted = X.sort_values(); y_line = slope_dev * x_sorted + intercept_dev

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X, Y, alpha=0.6, color=THEME['red_glow'])
    ax.plot(x_sorted, y_line, color=THEME['blue_glow'])
    ax.set_title(f"Price Deviation Scatter ({ticker} vs {bench})", color="white")
    st.pyplot(fig)
    st.write(f"Deviation regression: slope={slope_dev:.6f}, intercept={intercept_dev:.6f}")

    # 3. Rolling Vol Dev
    vol_dev_t = dev[ticker].rolling(int(window)).std(ddof=1)
    vol_dev_b = dev[bench].rolling(int(window)).std(ddof=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(vol_dev_t.index, vol_dev_t, label=f"{ticker}", color=THEME['gold'])
    ax.plot(vol_dev_b.index, vol_dev_b, label=f"{bench}", color=THEME['blue_glow'])
    ax.set_title(f"Rolling Volatility of Price Deviation (Window = {int(window)})", color="white")
    ax.legend(facecolor=THEME['bg_chart'], labelcolor="white")
    st.pyplot(fig)

    # 4. Drawdown
    p = prices[ticker]; running_max = p.cummax(); dd = (p / running_max) - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dd.index, dd * 100.0, color=THEME['platinum'])
    ax.set_title(f"Drawdown (%) - {ticker}", color="white")
    st.pyplot(fig)

    # 5. Scatter Returns
    Xr = ret[bench].dropna(); Yr = ret[ticker].dropna()
    common_r = Xr.index.intersection(Yr.index); Xr = Xr.loc[common_r]; Yr = Yr.loc[common_r]
    slope_ret, intercept_ret = np.polyfit(Xr.values, Yr.values, 1)
    xr_sorted = Xr.sort_values(); yr_line = slope_ret * xr_sorted + intercept_ret
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Xr, Yr, alpha=0.6, color=THEME['red_glow'])
    ax.plot(xr_sorted, yr_line, color=THEME['blue_glow'])
    ax.set_title(f"Daily Log Returns Scatter ({ticker} vs {bench})", color="white")
    st.pyplot(fig)
    
    # 6. Rolling Vol Returns
    roll_vol = ret[ticker].rolling(int(window)).std(ddof=1) * np.sqrt(float(trading_days))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll_vol.index, roll_vol * 100.0, color=THEME['gold'])
    ax.set_title(f"Rolling Volatility (Annualized, Window = {int(window)})", color="white")
    st.pyplot(fig)

    # 7. JP MARGIN
    if ticker.endswith(".T"):
        st.markdown("---")
        st.subheader(f"Margin Balance & Reverse Repo Fee (JP) - {ticker}")
        
        with st.spinner("Fetching JP Margin Data..."):
            margin_df = fetch_jp_margin_data_robust(ticker)
            
        if not margin_df.empty:
            fig, ax1 = plt.subplots(figsize=(11, 6))
            dates = margin_df["Date"]
            
            if "MarginBuy" in margin_df.columns and "MarginSell" in margin_df.columns:
                ax1.fill_between(dates, margin_df["MarginBuy"], color=THEME['blue_glow'], alpha=0.2, label="Margin Buy (Long)")
                ax1.plot(dates, margin_df["MarginBuy"], color=THEME['blue_glow'], linewidth=1.5)
                
                ax1.fill_between(dates, margin_df["MarginSell"], color=THEME['red_glow'], alpha=0.2, label="Margin Sell (Short)")
                ax1.plot(dates, margin_df["MarginSell"], color=THEME['red_glow'], linewidth=1.5)
                
                ax1.set_ylabel("Balance (Shares)", color="white")
                ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Hibush
            has_hibu = "Hibush" in margin_df.columns and margin_df["Hibush"].sum() > 0
            if has_hibu:
                ax2 = ax1.twinx()
                colors = [THEME['gold'] if v > 0 else "#333333" for v in margin_df["Hibush"]]
                ax2.bar(dates, margin_df["Hibush"], color=colors, alpha=0.6, width=0.6, label="Rev Repo Fee")
                ax2.set_ylabel("Fee (JPY)", color=THEME['gold'])
                ax2.tick_params(axis='y', colors=THEME['gold'])
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor=THEME['bg_chart'], labelcolor="white")
            else:
                ax1.legend(loc="upper left", facecolor=THEME['bg_chart'], labelcolor="white")

            st.pyplot(fig)
            st.caption("Source: IRBank / Minkabu / Karauri / Yahoo / Kabutan (Auto-fallback).")
            with st.expander("Show Details"):
                st.dataframe(margin_df.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.warning("No JP Margin Data found.")


# -------------------------------------------------------
# Streamlit app (ADMIN)
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander Admin", layout="wide")
    
    # CSS: Tech-Luxury Instrument Panel
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Roboto+Mono:wght@300;400&display=swap');
        
        .stApp {{ background-color: {THEME['bg_main']} !important; }}
        div[data-testid="stMarkdownContainer"] p, label {{ color: #ffffff !important; }}
        
        /* Header Container */
        .instrument-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }}
        
        /* Ticker Styling */
        .ticker-title {{
            font-family: 'Cinzel', serif; /* Elegant Serif */
            font-size: 32px;
            color: #FFFFFF;
            letter-spacing: 2px;
        }}
        .ticker-sub {{
            font-family: 'Roboto Mono', monospace;
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }}

        /* Score Instrument Styling */
        .score-panel {{
            text-align: right;
            font-family: 'Roboto Mono', monospace;
        }}
        .score-label {{
            font-size: 10px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .score-value {{
            font-size: 32px;
            color: {THEME['gold']};
            font-weight: 300;
        }}
        .status-dot {{
            height: 10px;
            width: 10px;
            background-color: #333;
            border-radius: 50%;
            display: inline-block;
            margin-left: 8px;
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
        }}
        .status-dot.safe {{ background-color: {THEME['blue_glow']}; box-shadow: 0 0 8px {THEME['blue_glow']}; }}
        .status-dot.caution {{ background-color: #FFD700; box-shadow: 0 0 8px #FFD700; }}
        .status-dot.high {{ background-color: {THEME['red_glow']}; box-shadow: 0 0 8px {THEME['red_glow']}; }}
        
        /* Remove Streamlit Elements */
        div[data-testid="stHeader"] {{ display: none; }}
        footer {{ display: none; }}
        </style>
        """, unsafe_allow_html=True)

    def show_error_black(msg: str):
        st.markdown(f"""<div style="background-color: #141518; border: 1px solid #2a2c30; padding: 14px; color: white;">{msg}</div>""", unsafe_allow_html=True)

    # Input Form (Minimalist)
    with st.form("input_form"):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1: ticker = st.text_input("Ticker", value="", placeholder="e.g. NVDA")
        with col2: start_date = st.date_input("Start", date.today() - timedelta(days=220))
        with col3: end_date = st.date_input("End", date.today())
        with col4: submitted = st.form_submit_button("ANALYZE")

    if not submitted:
        st.info("Awaiting Input...")
        st.stop()
    
    if not ticker.strip():
        show_error_black("Invalid Ticker")
        st.stop()

    # Calculation Block
    try:
        price_series = fetch_price_series_cached(ticker.strip(), start_date, end_date)
    except:
        show_error_black("Ticker Not Found or No Data")
        st.stop()

    if len(price_series) < 30:
        show_error_black("Data too short")
        st.stop()

    key = series_cache_key(price_series)
    idx_int = price_series.index.astype("int64").to_numpy()
    vals = price_series.to_numpy(dtype="float64")

    try:
        bubble_res = fit_lppl_bubble_cached(key, vals, idx_int)
    except:
        show_error_black("LPPL Fit Failed")
        st.stop()

    peak_date = price_series.idxmax(); peak_price = float(price_series.max())
    gain = peak_price / float(price_series.iloc[0])
    
    neg_res = fit_lppl_negative_bubble_cached(key, vals, idx_int, int(pd.Timestamp(peak_date).value), 10, 0.03)
    
    tc_up_date = pd.Timestamp(bubble_res["tc_date"])
    down_tc_date = pd.Timestamp(neg_res["tc_date"]) if neg_res.get("ok") else None
    signal_label, score = compute_signal_and_score(tc_up_date, pd.Timestamp(end_date), down_tc_date)

    # Determine Status Dot Class
    status_class = "safe" if signal_label == "SAFE" else "high" if signal_label == "HIGH" else "caution"

    # -------------------------------------------------------
    # 1. RENDER HEADER (Instrument Panel)
    # -------------------------------------------------------
    st.markdown(f"""
        <div class="instrument-header">
            <div>
                <div class="ticker-title">{ticker.upper()}</div>
                <div class="ticker-sub">MARKET ANALYSIS SYSTEM</div>
            </div>
            <div class="score-panel">
                <div class="score-label">Risk Score</div>
                <div>
                    <span class="score-value">{score}</span>
                    <span class="status-dot {status_class}"></span>
                </div>
                <div class="score-label" style="margin-top:4px;">Signal: {signal_label}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # 2. RENDER CHART (Clean, Direct Labeling)
    # -------------------------------------------------------
    fig = render_tech_luxury_chart(price_series, bubble_res, neg_res, ticker, score, signal_label, gain)
    st.pyplot(fig)

    # Admin Debug Section (Collapsed by default to keep UI clean)
    with st.expander("Admin Diagnostics"):
        pdict = bubble_res.get("param_dict", {})
        r2 = float(bubble_res.get("r2", np.nan))
        m = float(pdict.get("m", np.nan))
        omega = float(pdict.get("omega", np.nan))
        render_bubble_flow(r2, m, omega)
        admin_interpretation_text(bubble_res, end_date)

    # Quant table expander
    with st.expander("Mid-term Quant Table", expanded=False):
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1: bench = st.text_input("Benchmark", value="ACWI")
        with col_b2: mid_window = st.number_input("Window", min_value=5, max_value=120, value=20, step=1)
        try:
            df_mid = build_midterm_quant_table(ticker=ticker.strip(), bench=bench.strip(), start_date=start_date, end_date=end_date, window=int(mid_window))
            st.dataframe(df_mid[["Block", "Metric", "Value", "Threshold", "Judgement", "Note"]], use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.markdown("---")
    with st.expander("Extra Graphs", expanded=False):
        try:
            prices_pair = fetch_prices_pair_cached(ticker=ticker.strip(), bench=bench.strip(), start_date=start_date, end_date=end_date)
            render_graph_pack_from_prices(prices=prices_pair, ticker=ticker.strip(), bench=bench.strip(), window=int(mid_window))
        except Exception:
            pass

if __name__ == "__main__":
    main()
