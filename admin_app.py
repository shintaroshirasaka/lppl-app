import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
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
import platform

# =======================================================
# FONT SETUP (Luxury / Serif Style)
# =======================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

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

OS_TOKEN_SECRET = os.environ.get("OS_TOKEN_SECRET_ADMIN", "").strip()
token = st.query_params.get("t", "")

if not OS_TOKEN_SECRET or not token:
    st.stop()

ok, authed_email = verify_token(token, OS_TOKEN_SECRET)
if not ok:
    st.stop()

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
PRICE_TTL_SECONDS = 15 * 60
FIT_TTL_SECONDS = 24 * 60 * 60
SCRAPE_TTL_SECONDS = 60 * 60


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

def _clean_num_global(x):
    """Shared numeric cleaning helper for all scrapers."""
    if isinstance(x, str):
        x = x.replace(',', '').replace('株', '').replace('円', '').replace('倍', '').replace('%', '').replace('-', '0').strip()
        if not x:
            return 0
        try:
            return float(x)
        except Exception:
            return 0
    return x


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
        return "WATCH"
    if metric == "B: Deviation R²":
        if value >= MID_TH["B_R2_ok"]: return "OK"
        if value < MID_TH["B_R2_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("C: Deviation β"):
        if MID_TH["C_beta_ok_low"] <= value <= MID_TH["C_beta_ok_high"]: return "OK"
        if value < MID_TH["C_beta_ng_low"] or value > MID_TH["C_beta_ng_high"]: return "NG"
        return "WATCH"
    if metric.startswith("D: Deviation Vol"):
        bench_latest = dev[bench].rolling(window).std(ddof=1).dropna().iloc[-1]
        ratio = value / bench_latest if bench_latest != 0 else np.nan
        if ratio <= 1.5: return "OK"
        if ratio > 2.5:  return "NG"
        return "WATCH"
    if metric == "D: Max Drawdown":
        if value >= MID_TH["G_mdd_ok"]: return "OK"
        if value < MID_TH["G_mdd_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("E: CMGR"):
        if value >= MID_TH["E_cmgr_ok"]: return "OK"
        if value < MID_TH["E_cmgr_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("F: Sharpe"):
        if value >= MID_TH["F_sharpe_ok"]: return "OK"
        if value < MID_TH["F_sharpe_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("F2: Alpha"):
        if value >= MID_TH["F2_alpha_ok"]: return "OK"
        if value < MID_TH["F2_alpha_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("F3: Alpha"):
        if value >= MID_TH["F3_alpha_ok"]: return "OK"
        if value < MID_TH["F3_alpha_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("H: Rolling R²"):
        if value >= MID_TH["H_R2_ok"]: return "OK"
        if value < MID_TH["H_R2_ng"]:  return "NG"
        return "WATCH"
    if metric.startswith("I: Rolling β"):
        if MID_TH["I_beta_ok_low"] <= value <= MID_TH["I_beta_ok_high"]: return "OK"
        if value < MID_TH["I_beta_ng_low"] or value > MID_TH["I_beta_ng_high"]: return "NG"
        return "WATCH"
    if metric.startswith("J: Rolling Vol"):
        if value <= MID_TH["J_vol_ok"]: return "OK"
        if value > MID_TH["J_vol_ng"]:  return "NG"
        return "WATCH"
    return "WATCH"

def _mid_threshold_text(metric: str) -> str:
    if metric == "A: Deviation R":
        return f"OK≥{MID_TH['A_R_ok']:.2f} / NG<{MID_TH['A_R_ng']:.2f}"
    if metric == "B: Deviation R²":
        return f"OK≥{MID_TH['B_R2_ok']:.2f} / NG<{MID_TH['B_R2_ng']:.2f}"
    if metric.startswith("C: Deviation β"):
        return f"OK:{MID_TH['C_beta_ok_low']:.2f}–{MID_TH['C_beta_ok_high']:.2f} / NG<{MID_TH['C_beta_ng_low']:.2f} or >{MID_TH['C_beta_ng_high']:.2f}"
    if metric.startswith("D: Deviation Vol"):
        return "OK:(Stock/Idx)≤1.5 / NG>2.5"
    if metric == "D: Max Drawdown":
        return f"OK≥{MID_TH['G_mdd_ok']*100:.0f}% / NG<{MID_TH['G_mdd_ng']*100:.0f}%"
    if metric.startswith("E: CMGR"):
        return f"OK≥{MID_TH['E_cmgr_ok']*100:.2f}%/mo / NG<{MID_TH['E_cmgr_ng']*100:.2f}%/mo"
    if metric.startswith("F: Sharpe"):
        return f"OK≥{MID_TH['F_sharpe_ok']:.2f} / NG<{MID_TH['F_sharpe_ng']:.2f}"
    if metric.startswith("F2: Alpha"):
        return f"OK≥{MID_TH['F2_alpha_ok']*100:.0f}% / NG<{MID_TH['F2_alpha_ng']*100:.0f}%"
    if metric.startswith("F3: Alpha"):
        return f"OK≥{MID_TH['F3_alpha_ok']*100:.0f}% / NG<{MID_TH['F3_alpha_ng']*100:.0f}%"
    if metric.startswith("H: Rolling R²"):
        return f"OK≥{MID_TH['H_R2_ok']:.2f} / NG<{MID_TH['H_R2_ng']:.2f}"
    if metric.startswith("I: Rolling β"):
        return f"OK:{MID_TH['I_beta_ok_low']:.2f}–{MID_TH['I_beta_ok_high']:.2f} / NG<{MID_TH['I_beta_ng_low']:.2f} or >{MID_TH['I_beta_ng_high']:.2f}"
    if metric.startswith("J: Rolling Vol"):
        return f"OK≤{MID_TH['J_vol_ok']*100:.0f}% / NG>{MID_TH['J_vol_ng']*100:.0f}%"
    return "-"


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

    rows = [
        ("1. Market Fit", "A: Deviation R", A_R, _mid_threshold_text("A: Deviation R"), "Correlation of deviations"),
        ("1. Market Fit", "B: Deviation R²", B_R2, _mid_threshold_text("B: Deviation R²"), "Explanatory power of deviation"),
        ("1. Market Fit", f"C: Deviation β (vs {bench})", C_beta, _mid_threshold_text(f"C: Deviation β (vs {bench})"), "Sensitivity in deviation space"),
        ("1. Market Fit", f"D: Deviation Vol (rolling {window})", D_vol_latest, _mid_threshold_text(f"D: Deviation Vol (rolling {window})"), "Std Dev of deviation"),
        ("1. Market Fit", "D: Max Drawdown", max_dd, _mid_threshold_text("D: Max Drawdown"), f"Max Drawdown (Date: {max_dd_date})"),
        ("2. Efficiency", f"E: CMGR (monthly, ~{months} months)", E_cmgr, _mid_threshold_text(f"E: CMGR (monthly, ~{months} months)"), "Compounded Monthly Growth Rate"),
        ("2. Efficiency", "F: Sharpe (annualized, log returns, rf=0)", F_sharpe, _mid_threshold_text("F: Sharpe (annualized, log returns, rf=0)"), "Risk-adjusted returns"),
        ("2. Efficiency", "F2: Alpha (Relative Outperformance vs Benchmark)", F2_alpha_rel, _mid_threshold_text("F2: Alpha (Relative Outperformance vs Benchmark)"), "Total return alpha over period"),
        ("2. Efficiency", "F3: Alpha (Regression, daily -> annualized)", F3_alpha_annual, _mid_threshold_text("F3: Alpha (Regression, daily -> annualized)"), "CAPM Alpha"),
        ("3. Causality", f"H: Rolling R² (daily, {window})", H_R2_latest, _mid_threshold_text(f"H: Rolling R² (daily, {window})"), "Recent correlation strength"),
        ("3. Causality", f"I: Rolling β (daily, {window})", I_beta_latest, _mid_threshold_text(f"I: Rolling β (daily, {window})"), "Recent Beta"),
        ("3. Causality", f"J: Rolling Vol (annualized, {window})", J_vol_annual_latest, _mid_threshold_text(f"J: Rolling Vol (annualized, {window})"), "Recent volatility"),
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
    if not info["r2_ok"]: return "Inconclusive (Weak LPPL shape)", info
    if not info["m_ok"]: return "Normal Uptrend", info
    if info["omega_ok"]: return "Bubble-like Rise", info
    if info["omega_high"]: return "Bubble-like (Warning: Fast Freq)", info
    return "Bubble-like (Warning)", info

def render_bubble_flow(r2: float, m: float, omega: float):
    verdict, info = bubble_judgement(r2, m, omega)
    def yn(v: bool) -> str: return "YES" if v else "NO"
    lines = []
    lines.append(f"1. R² ≥ 0.65 ?   (R²={r2:.2f})")
    lines.append(f" ├ {yn(info['r2_ok'])} → " + ("Next" if info["r2_ok"] else "Inconclusive (Weak LPPL shape)"))
    if info["r2_ok"]:
        lines.append(" │")
        lines.append(f" │   2. m ∈ [0.25, 0.70] ?   (m={m:.2f})")
        lines.append(f" │    ├ {yn(info['m_ok'])} → " + ("Next" if info["m_ok"] else "Normal Uptrend"))
        if info["m_ok"]:
            lines.append(" │    │")
            lines.append(f" │    │   3. ω ∈ [6, 13] ?   (ω={omega:.2f})")
            if info["omega_ok"]:
                lines.append(" │    │    ├ YES → Bubble-like Rise")
            else:
                if np.isfinite(omega) and omega >= 18:
                    lines.append(" │    │    ├ NO (≥18) → Bubble-like (Warning: Fast Freq)")
                else:
                    lines.append(" │    │    ├ NO → Bubble-like (Warning)")
            lines.append(" │    │")
    lines.append("")
    lines.append(f"[Result] {verdict}")
    st.markdown("### Bubble Decision Flow (Admin)")
    st.code("\n".join(lines), language="text")

def admin_interpretation_text(bubble_res: dict, end_date: date) -> tuple[str, list[str]]:
    pdict = bubble_res.get("param_dict", {})
    r2 = float(bubble_res.get("r2", float("nan")))
    rmse = float(bubble_res.get("rmse", float("nan")))
    m = float(pdict.get("m", float("nan")))
    omega = float(pdict.get("omega", float("nan")))
    c_over_b = float(pdict.get("abs_C_over_B", float("nan")))
    tc_date = pd.Timestamp(bubble_res.get("tc_date")).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    days_to_tc = int((tc_date - end_norm).days)
    bullets: list[str] = []
    WARNING_DAYS = 14
    if days_to_tc < 0:
        bullets.append(f"t_c has already passed {abs(days_to_tc)} days ago (Potential structural peak).")
        bullets.append("New trend following is risky. Consider partial profit taking or hedging.")
    elif days_to_tc <= WARNING_DAYS:
        bullets.append(f"Days to t_c: {days_to_tc} (DANGER ZONE: Within 14 days).")
        bullets.append("Early warning period. Be cautious with new longs (reduce size). Prepare to exit.")
    else:
        bullets.append(f"Days to t_c: {days_to_tc} (Safe Zone).")
    verdict, _ = bubble_judgement(r2, m, omega)
    bullets.append(f"Bubble Verdict (R²->m->ω): {verdict}")
    if np.isfinite(r2) and r2 < 0.65:
        bullets.append(f"R²={r2:.2f}: Weak shape -> Inconclusive.")
    if np.isfinite(m) and m >= 0.85:
        bullets.append(f"m={m:.2f}: Near upper limit -> Boundary solution suspected.")
    if np.isfinite(omega) and omega >= 18:
        bullets.append(f"ω={omega:.2f}: Frequency too high -> Noise fitting suspected.")
    if np.isfinite(rmse):
        bullets.append(f"RMSE(log)={rmse:.3f}: Fit stability indicator.")
    if np.isfinite(c_over_b) and c_over_b >= 2.0:
        bullets.append(f"|C/B|={c_over_b:.2f}: Oscillation too strong -> Overfitting suspected.")
    if verdict == "Bubble-like Rise":
        summary = "Bubble-like rise (Valid m band + High R² + Typical ω). Restrict chasing, prepare for exit/hedge."
    elif verdict.startswith("Bubble-like"):
        summary = "Bubble-like (Shape resembles bubble but has warning signs). Be cautious with chasing, advance risk management."
    elif verdict == "Normal Uptrend":
        summary = "Difficult to classify as a typical bubble. However, if tc is near, adjust stance accordingly."
    else:
        summary = "Inconclusive due to weak LPPL shape. Change window or use other indicators."
    return summary, bullets


# -------------------------------------------------------
# LPPL Model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)

def fit_lppl_bubble(price_series: pd.Series):
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)
    N = len(t)
    A_init = float(np.mean(log_price))
    p0 = [A_init, -1.0, 0.1, 0.5, N + 20, 8.0, 0.0]
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
    code = ticker.replace(".T", "").strip()
    if not code.isdigit():
        return pd.DataFrame()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Referer": "https://www.google.com/"
    }

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
# NEW: FINRA Short Volume Scraper (US Stocks)
# =======================================================
@st.cache_data(ttl=SCRAPE_TTL_SECONDS, show_spinner=False)
def fetch_us_short_volume(ticker: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches daily FINRA Reg SHO Short Volume data for US-listed stocks.
    Primary source: FINRA public API (no auth required)
      POST https://api.finra.org/data/group/otcmarket/name/regShoDaily
    Returns: DataFrame[Date, ShortVolume, TotalVolume, NonShortVolume, ShortRatio]
    Data is aggregated across all TRFs (NQTRF, NYTRF, NCTRF).
    """
    symbol = ticker.strip().upper()
    non_us_suffixes = [".T", ".HK", ".L", ".DE", ".PA", ".AS", ".MI", ".SW", ".TO", ".AX"]
    if any(symbol.endswith(sfx) for sfx in non_us_suffixes):
        return pd.DataFrame()

    def _try_finra_api():
        url = "https://api.finra.org/data/group/otcmarket/name/regShoDaily"
        api_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "limit": 5000,
            "compareFilters": [
                {
                    "compareType": "equal",
                    "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                    "fieldValue": symbol
                }
            ]
        }
        try:
            res = requests.post(url, headers=api_headers, json=payload, timeout=15)
            if res.status_code != 200:
                return pd.DataFrame()
            data = res.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                return pd.DataFrame()
            raw = pd.DataFrame(data)
            required_cols = ["tradeReportDate", "shortParQuantity", "totalParQuantity"]
            for col in required_cols:
                if col not in raw.columns:
                    return pd.DataFrame()
            raw["tradeReportDate"] = pd.to_datetime(raw["tradeReportDate"], errors="coerce")
            raw["shortParQuantity"] = pd.to_numeric(raw["shortParQuantity"], errors="coerce").fillna(0)
            raw["totalParQuantity"] = pd.to_numeric(raw["totalParQuantity"], errors="coerce").fillna(0)
            if "shortExemptParQuantity" in raw.columns:
                raw["shortExemptParQuantity"] = pd.to_numeric(raw["shortExemptParQuantity"], errors="coerce").fillna(0)
            daily = raw.groupby("tradeReportDate").agg(
                ShortVolume=("shortParQuantity", "sum"),
                TotalVolume=("totalParQuantity", "sum")
            ).reset_index()
            daily = daily.rename(columns={"tradeReportDate": "Date"})
            daily = daily.dropna(subset=["Date"]).sort_values("Date")
            daily["TotalVolume"] = daily["TotalVolume"].replace(0, np.nan)
            daily["ShortRatio"] = (daily["ShortVolume"] / daily["TotalVolume"] * 100).round(2)
            daily["NonShortVolume"] = (daily["TotalVolume"] - daily["ShortVolume"]).clip(lower=0)
            daily = daily.dropna(subset=["ShortVolume"])
            if len(daily) > 0:
                return daily.tail(lookback_days).reset_index(drop=True)
        except Exception:
            pass
        return pd.DataFrame()

    df = _try_finra_api()
    if not df.empty and "ShortVolume" in df.columns:
        return df
    return pd.DataFrame()


# =======================================================
# NEW: JP Short Selling Ratio Scraper (karauri.net daily data)
# =======================================================
@st.cache_data(ttl=SCRAPE_TTL_SECONDS, show_spinner=False)
def fetch_jp_short_selling_ratio(ticker: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Scrapes daily short selling ratio data for JP stocks.
    Source: karauri.net (空売り比率ページ)
    Returns: DataFrame[Date, ShortVolume, TotalVolume, NonShortVolume, ShortRatio]
    """
    code = ticker.replace(".T", "").strip()
    if not code.isdigit():
        return pd.DataFrame()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Referer": "https://www.google.com/"
    }

    def _try_karauri_ratio():
        url = f"https://karauri.net/{code}/"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                return pd.DataFrame()
            try:
                dfs = pd.read_html(io.BytesIO(res.content))
            except Exception:
                return pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                has_ratio = "空売り" in col_str and ("比率" in col_str or "%" in col_str)
                has_volume = "出来高" in col_str or "数量" in col_str
                has_date = "日付" in col_str or "日" in col_str
                if has_date and (has_ratio or has_volume):
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(map(str, c)).strip() for c in df.columns]
                    else:
                        df.columns = [str(c).strip() for c in df.columns]
                    rename_map = {}
                    for c in df.columns:
                        if "日付" in c or c == "日":
                            rename_map[c] = "Date"
                        elif "空売り" in c and ("比率" in c or "%" in c):
                            rename_map[c] = "ShortRatio"
                        elif "空売り" in c and ("数量" in c or "出来高" in c or "残高" in c):
                            rename_map[c] = "ShortVolume"
                        elif "出来高" in c and "空売り" not in c:
                            rename_map[c] = "TotalVolume"
                        elif "売残" in c:
                            rename_map[c] = "ShortVolume"
                        elif "返済" in c:
                            pass
                    df = df.rename(columns=rename_map)
                    if "Date" not in df.columns:
                        continue
                    today = date.today()
                    current_year = today.year
                    def try_parse_jp(x):
                        if pd.isna(x) or str(x).strip() == "":
                            return pd.NaT
                        x_clean = str(x).split(' ')[0]
                        x_clean = x_clean.replace("年", "/").replace("月", "/").replace("日", "")
                        try:
                            return pd.to_datetime(x_clean)
                        except Exception:
                            pass
                        try:
                            dt = pd.to_datetime(f"{current_year}/{x_clean}")
                            if dt.date() > today + timedelta(days=2):
                                dt = dt.replace(year=current_year - 1)
                            return dt
                        except Exception:
                            return pd.NaT
                    df["Date"] = df["Date"].apply(try_parse_jp)
                    for col in ["ShortVolume", "TotalVolume", "ShortRatio"]:
                        if col in df.columns:
                            df[col] = df[col].apply(_clean_num_global)
                    df = df.dropna(subset=["Date"]).sort_values("Date")
                    if "TotalVolume" in df.columns and "ShortVolume" in df.columns:
                        df["NonShortVolume"] = (df["TotalVolume"] - df["ShortVolume"]).clip(lower=0)
                        if "ShortRatio" not in df.columns:
                            df["ShortRatio"] = np.where(
                                df["TotalVolume"] > 0,
                                (df["ShortVolume"] / df["TotalVolume"] * 100).round(2),
                                0
                            )
                    elif "ShortRatio" in df.columns and "ShortVolume" not in df.columns:
                        df["ShortVolume"] = 0
                        df["TotalVolume"] = 0
                        df["NonShortVolume"] = 0
                    if len(df) > 0:
                        return df.tail(lookback_days).reset_index(drop=True)
        except Exception:
            pass
        return pd.DataFrame()

    def _try_jpx_short_ratio():
        url = f"https://karauri.net/{code}/ratio/"
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code != 200:
                return pd.DataFrame()
            try:
                dfs = pd.read_html(io.BytesIO(res.content))
            except Exception:
                return pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if ("日付" in col_str or "日" in col_str) and ("比率" in col_str or "%" in col_str):
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(map(str, c)).strip() for c in df.columns]
                    else:
                        df.columns = [str(c).strip() for c in df.columns]
                    rename_map = {}
                    for c in df.columns:
                        if "日付" in c or c == "日":
                            rename_map[c] = "Date"
                        elif "比率" in c or "%" in c:
                            rename_map[c] = "ShortRatio"
                        elif "空売り" in c and ("数量" in c or "出来高" in c):
                            rename_map[c] = "ShortVolume"
                        elif "出来高" in c:
                            rename_map[c] = "TotalVolume"
                    df = df.rename(columns=rename_map)
                    if "Date" not in df.columns:
                        continue
                    today_dt = date.today()
                    current_yr = today_dt.year
                    def try_parse_jp2(x):
                        if pd.isna(x) or str(x).strip() == "":
                            return pd.NaT
                        x_clean = str(x).split(' ')[0].replace("年", "/").replace("月", "/").replace("日", "")
                        try:
                            return pd.to_datetime(x_clean)
                        except Exception:
                            pass
                        try:
                            dt = pd.to_datetime(f"{current_yr}/{x_clean}")
                            if dt.date() > today_dt + timedelta(days=2):
                                dt = dt.replace(year=current_yr - 1)
                            return dt
                        except Exception:
                            return pd.NaT
                    df["Date"] = df["Date"].apply(try_parse_jp2)
                    for col in ["ShortVolume", "TotalVolume", "ShortRatio"]:
                        if col in df.columns:
                            df[col] = df[col].apply(_clean_num_global)
                    df = df.dropna(subset=["Date"]).sort_values("Date")
                    if "TotalVolume" in df.columns and "ShortVolume" in df.columns:
                        df["NonShortVolume"] = (df["TotalVolume"] - df["ShortVolume"]).clip(lower=0)
                    if len(df) > 0:
                        return df.tail(lookback_days).reset_index(drop=True)
        except Exception:
            pass
        return pd.DataFrame()

    df = _try_karauri_ratio()
    if not df.empty and "ShortRatio" in df.columns:
        return df
    df = _try_jpx_short_ratio()
    if not df.empty and "ShortRatio" in df.columns:
        return df
    return pd.DataFrame()


# =======================================================
# NEW: Short Volume Chart Renderer (HNWI Dark Theme)
# =======================================================
def render_short_volume_chart(df: pd.DataFrame, ticker: str, is_jp: bool = False):
    """
    Renders FINRA-style short volume chart:
    - Stacked bars: ShortVolume (red/gold) + NonShortVolume (grey)
    - Line: ShortRatio (%) on right axis with data labels
    - Dashed line at 50% baseline
    - HNWI dark theme
    """
    HNWI_BG = "#050505"
    HNWI_AX_BG = "#0b0c0e"
    TEXT_COLOR = "#F0F0F0"
    TICK_COLOR = "#888888"
    SHORT_BAR_COLOR = "#C5504F"
    NONSHORT_BAR_COLOR = "#555555"
    RATIO_LINE_COLOR = "#5B9BD5"
    BASELINE_COLOR = "#C5A059"

    dates = df["Date"]
    has_volumes = "TotalVolume" in df.columns and df["TotalVolume"].sum() > 0
    has_ratio = "ShortRatio" in df.columns and df["ShortRatio"].sum() > 0

    if not has_ratio and not has_volumes:
        st.warning("Short volume data has no usable values.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(HNWI_BG)
    ax1.set_facecolor(HNWI_AX_BG)

    bar_width = 0.7
    x_pos = np.arange(len(dates))

    if has_volumes:
        short_vals = df["ShortVolume"].fillna(0).values
        nonshort_vals = df.get("NonShortVolume", pd.Series(dtype=float)).fillna(0).values
        if len(nonshort_vals) == 0:
            nonshort_vals = np.zeros(len(short_vals))

        ax1.bar(x_pos, short_vals, width=bar_width, color=SHORT_BAR_COLOR, alpha=0.85, label="Short Volume", zorder=3)
        ax1.bar(x_pos, nonshort_vals, width=bar_width, bottom=short_vals, color=NONSHORT_BAR_COLOR, alpha=0.5, label="Non-Short Volume", zorder=2)

        ax1.set_ylabel("Volume (shares)", color=TEXT_COLOR, fontname='serif')
        max_vol = max(short_vals.max() + nonshort_vals.max(), 1)
        if max_vol >= 1_000_000:
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x/1_000_000:.1f}M"))
        else:
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax1.set_xticks(x_pos)
    date_labels = [d.strftime('%m-%d') for d in dates]
    ax1.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=7)
    ax1.tick_params(colors=TICK_COLOR)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#333333')
    ax1.spines['bottom'].set_color('#333333')
    ax1.grid(color="#333333", linestyle=":", linewidth=0.5, alpha=0.3, axis='y')

    if has_ratio:
        ax2 = ax1.twinx()
        ax2.set_facecolor(HNWI_AX_BG)
        ratio_vals = df["ShortRatio"].fillna(0).values

        ax2.plot(x_pos, ratio_vals, color=RATIO_LINE_COLOR, linewidth=2.0, marker='o', markersize=4, zorder=10, label="Short Vol Ratio (%)")

        for i, (xp, rv) in enumerate(zip(x_pos, ratio_vals)):
            if rv > 0:
                ax2.annotate(f"{rv:.1f}%", (xp, rv), textcoords="offset points", xytext=(0, 10),
                             ha='center', va='bottom', fontsize=7, color=RATIO_LINE_COLOR, fontweight='bold', zorder=11)

        ax2.axhline(y=50.0, color=BASELINE_COLOR, linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
        ax2.text(len(x_pos) - 0.5, 50.5, "50%", color=BASELINE_COLOR, fontsize=9, fontweight='bold', va='bottom', ha='right', fontname='serif')

        y_min = max(min(ratio_vals) - 10, 0)
        y_max = min(max(ratio_vals) + 15, 100)
        ax2.set_ylim(y_min, y_max)
        ax2.set_ylabel("Short Volume Ratio (%)", color=RATIO_LINE_COLOR, fontname='serif')
        ax2.tick_params(axis='y', colors=RATIO_LINE_COLOR)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('#333333')
        ax2.spines['bottom'].set_color('#333333')

    title_prefix = "JPX" if is_jp else "FINRA"
    title_text = f"{title_prefix} Short Volume & Ratio: {ticker}"
    ax1.set_title(title_text, color=TEXT_COLOR, fontweight='normal', fontname='serif', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_ratio:
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2
    else:
        all_lines = lines1
        all_labels = labels1

    ax1.legend(all_lines, all_labels, loc="upper left", facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, frameon=False, fontsize=8)

    ax1.text(0.95, 0.03, "OUT-STANDER", transform=ax1.transAxes,
             fontsize=24, color='#3d3320', fontweight='bold',
             fontname='serif', ha='right', va='bottom', zorder=0, alpha=0.9)

    fig.tight_layout()
    st.pyplot(fig)


# =======================================================
# FINAL SCORE (UPDATED WITH 14-DAY WARNING RULE)
# =======================================================
def compute_signal_and_score(tc_up_date, end_date, down_tc_date) -> tuple[str, int]:
    now = pd.Timestamp(end_date).normalize()
    tc_up = pd.Timestamp(tc_up_date).normalize()
    if down_tc_date is not None:
        down_tc = pd.Timestamp(down_tc_date).normalize()
        delta = (down_tc - now).days
        if delta > 0:
            s = _lin_map(delta, DOWN_FUTURE_NEAR_DAYS, DOWN_FUTURE_FAR_DAYS, 90, 80)
            return ("HIGH", int(round(_clamp(s, 80, 90))))
        past = abs(delta)
        s = _lin_map(past, DOWN_PAST_NEAR_DAYS, DOWN_PAST_FAR_DAYS, 90, 100)
        return ("HIGH", int(round(_clamp(s, 90, 100))))
    gap = (tc_up - now).days
    WARNING_BUFFER = 14
    if gap > WARNING_BUFFER:
        s = _lin_map(gap, UP_FUTURE_NEAR_DAYS, UP_FUTURE_FAR_DAYS, 59, 0)
        return ("SAFE", int(round(_clamp(s, 0, 59))))
    past_warning = WARNING_BUFFER - gap
    s = _lin_map(past_warning, 0, UP_PAST_FAR_DAYS, 60, 79)
    return ("CAUTION", int(round(_clamp(s, 60, 79))))


# =======================================================
# Render Graph Pack
# =======================================================
def draw_score_overlay(ax, score: int, label: str):
    if score < 60:
        score_color = "#3CB371"
    elif score < 80:
        score_color = "#ffc53d"
    else:
        score_color = "#ff4d4f"
    x_pos = 0.02
    y_pos = 0.86
    ax.text(x_pos, y_pos - 0.08, str(score), transform=ax.transAxes,
            fontsize=36, color=score_color, fontweight='bold', ha='left', va='bottom', fontname='serif', zorder=20)


def draw_logo_overlay(ax):
    ax.text(0.95, 0.03, "OUT-STANDER", transform=ax.transAxes,
            fontsize=24, color='#3d3320', fontweight='bold',
            fontname='serif', ha='right', va='bottom', zorder=0, alpha=0.9)


def render_graph_pack_from_prices(prices, ticker, bench, window=20, trading_days=252):
    base = prices.iloc[0]
    index100 = prices / base * 100.0
    dev = index100 - 100.0
    ret = np.log(prices / prices.shift(1)).dropna()

    HNWI_BG = "#050505"
    HNWI_AX_BG = "#0b0c0e"
    TEXT_COLOR = "#F0F0F0"
    TICK_COLOR = "#888888"
    HNWI_TICKER_COLOR = "#C5A059"
    HNWI_BENCH_COLOR = "#A0A0A0"
    grid_kwargs = {"color": "#333333", "linestyle": ":", "linewidth": 0.5, "alpha": 0.5}

    def style_hnwi_ax(ax, title=None):
        ax.set_facecolor(HNWI_AX_BG)
        if title:
            ax.set_title(title, color=TEXT_COLOR, fontweight='normal', fontname='serif')
        ax.tick_params(colors=TICK_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.grid(**grid_kwargs)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        draw_logo_overlay(ax)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Cumulative Return (Indexed)")
    ax.plot(index100.index, index100[ticker], label=f"{ticker}", color=HNWI_TICKER_COLOR, linewidth=1.5)
    ax.plot(index100.index, index100[bench],  label=f"{bench}",  color=HNWI_BENCH_COLOR, linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (Base=100)")
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, frameon=False)
    st.pyplot(fig)

    X = dev[bench].dropna(); Y = dev[ticker].dropna()
    common = X.index.intersection(Y.index); X = X.loc[common]; Y = Y.loc[common]
    slope_dev, intercept_dev = np.polyfit(X.values, Y.values, 1)
    x_sorted = X.sort_values(); y_line = slope_dev * x_sorted + intercept_dev

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Deviation Scatter & Trend")
    ax.scatter(X, Y, alpha=0.6, color=HNWI_TICKER_COLOR, edgecolor='none', s=40)
    ax.plot(x_sorted, y_line, color=HNWI_TICKER_COLOR, linewidth=2.0, alpha=0.8)
    ax.set_xlabel(f"{bench} Deviation (pp)")
    ax.set_ylabel(f"{ticker} Deviation (pp)")
    st.pyplot(fig)

    vol_dev_t = dev[ticker].rolling(int(window)).std(ddof=1)
    vol_dev_b = dev[bench].rolling(int(window)).std(ddof=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Deviation Volatility (Rolling)")
    ax.plot(vol_dev_t.index, vol_dev_t, label=f"{ticker}", color=HNWI_TICKER_COLOR, linewidth=1.5)
    ax.plot(vol_dev_b.index, vol_dev_b, label=f"{bench}", color=HNWI_BENCH_COLOR, linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Deviation (pp)")
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, frameon=False)
    st.pyplot(fig)

    p = prices[ticker]; running_max = p.cummax(); dd = (p / running_max) - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Drawdown Profile")
    ax.fill_between(dd.index, dd * 100.0, 0, color=HNWI_TICKER_COLOR, alpha=0.3)
    ax.plot(dd.index, dd * 100.0, color=HNWI_TICKER_COLOR, linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_ylim(top=0.5)
    st.pyplot(fig)

    Xr = ret[bench].dropna(); Yr = ret[ticker].dropna()
    common_r = Xr.index.intersection(Yr.index); Xr = Xr.loc[common_r]; Yr = Yr.loc[common_r]
    slope_ret, intercept_ret = np.polyfit(Xr.values, Yr.values, 1)
    xr_sorted = Xr.sort_values(); yr_line = slope_ret * xr_sorted + intercept_ret

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Daily Returns Scatter")
    ax.scatter(Xr, Yr, alpha=0.6, color=HNWI_TICKER_COLOR, edgecolor='none', s=40)
    ax.plot(xr_sorted, yr_line, color=HNWI_TICKER_COLOR, linewidth=2.0, alpha=0.8)
    ax.set_xlabel(f"{bench} Daily Log Return")
    ax.set_ylabel(f"{ticker} Daily Log Return")
    st.pyplot(fig)

    roll_vol = ret[ticker].rolling(int(window)).std(ddof=1) * np.sqrt(float(trading_days))
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title="Annualized Volatility (Rolling)")
    ax.plot(roll_vol.index, roll_vol * 100.0, color=HNWI_TICKER_COLOR, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Vol (%)")
    st.pyplot(fig)

    if ticker.endswith(".T"):
        st.markdown("---")
        st.subheader(f"■ Margin Balance & Reverse Repo Fee (JP Data: {ticker})")
        with st.spinner("Fetching JP Margin Data (IRBank/Minkabu/Karauri/Yahoo/Kabutan)..."):
            margin_df = fetch_jp_margin_data_robust(ticker)
        if not margin_df.empty:
            fig, ax1 = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor(HNWI_BG)
            style_hnwi_ax(ax1, title=None)
            dates = margin_df["Date"]
            if "MarginBuy" in margin_df.columns and "MarginSell" in margin_df.columns:
                ax1.fill_between(dates, margin_df["MarginBuy"], color=HNWI_TICKER_COLOR, alpha=0.3, label="Margin Buy (Longs)")
                ax1.plot(dates, margin_df["MarginBuy"], color=HNWI_TICKER_COLOR, linewidth=1.5)
                ax1.fill_between(dates, margin_df["MarginSell"], color=HNWI_BENCH_COLOR, alpha=0.3, label="Margin Sell (Shorts)")
                ax1.plot(dates, margin_df["MarginSell"], color=HNWI_BENCH_COLOR, linewidth=1.5)
                ax1.set_ylabel("Margin Balance (Shares)")
                ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            ax1.set_xlabel("Date")
            has_hibu = "Hibush" in margin_df.columns and margin_df["Hibush"].sum() > 0
            if has_hibu:
                ax2 = ax1.twinx()
                ax2.set_facecolor(HNWI_AX_BG)
                ax2.tick_params(colors=TICK_COLOR)
                ax2.yaxis.label.set_color(TEXT_COLOR)
                ax2.spines['top'].set_visible(False)
                ax2.spines['left'].set_visible(False)
                ax2.spines['right'].set_color('#333333')
                ax2.spines['bottom'].set_color('#333333')
                colors = ["#FFD700" if v > 0 else "#333333" for v in margin_df["Hibush"]]
                ax2.bar(dates, margin_df["Hibush"], color=colors, alpha=0.6, width=0.6, label="Rev. Repo Fee (Hibush)")
                ax2.set_ylabel("Fee (JPY)", color="#FFD700")
                ax2.tick_params(axis='y', colors="#FFD700")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, frameon=False)
            else:
                ax1.legend(loc="upper left", facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, frameon=False)
            st.pyplot(fig)
            st.caption("※Source: IRBank / Minkabu / Karauri / Yahoo / Kabutan (Auto-switch). Showing recent data only.")
            with st.expander("View Margin Data Details"):
                st.dataframe(margin_df.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.warning("Could not fetch Japanese margin data (All sources failed).")


# -------------------------------------------------------
# Streamlit app (ADMIN)
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander Admin", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
        .stApp { background-color: #050505 !important; color: #ffffff !important; }
        div[data-testid="stMarkdownContainer"] p, label { color: #ffffff !important; font-family: 'Times New Roman', serif; }
        input.st-ai, input.st-ah, div[data-baseweb="input"] { background-color: #111111 !important; color: #ffffff !important; border-color: #333 !important; }
        input { color: #ffffff !important; }
        input::placeholder { color: rgba(255,255,255,0.45) !important; }
        div[data-baseweb="input"] svg { fill: #ffffff !important; }
        div[data-testid="stFormSubmitButton"] button { background-color: #1a1a1a !important; color: #d4af37 !important; border: 1px solid #333 !important; font-family: 'Times New Roman', serif; }
        div[data-testid="stFormSubmitButton"] button:hover { background-color: #d4af37 !important; border-color: #d4af37 !important; color: #000000 !important; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
        .custom-error { background-color: #141518; border: 1px solid #2a2c30; border-radius: 0px; padding: 14px 18px; color: #ffffff; font-size: 0.95rem; margin-top: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def show_error_black(msg: str):
        st.markdown(f"""<div class="custom-error">{msg}</div>""", unsafe_allow_html=True)

    with st.form("input_form"):
        ticker = st.text_input("Ticker", value="", placeholder="Ex: NVDA / 0700.HK / 7203.T")
        today = date.today()
        default_start = today - timedelta(days=220)
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Start", default_start)
        with col2: end_date = st.date_input("End", today)
        submitted = st.form_submit_button("Run Analysis")

    if not submitted: st.stop()
    if not ticker.strip():
        show_error_black("Invalid Ticker or No Data.")
        st.stop()

    try:
        price_series = fetch_price_series_cached(ticker.strip(), start_date, end_date)
    except Exception:
        show_error_black("Invalid Ticker or No Data.")
        st.stop()

    if len(price_series) < 30:
        show_error_black("Invalid Ticker or No Data (Not enough history).")
        st.stop()

    key = series_cache_key(price_series)
    idx_int = price_series.index.astype("int64").to_numpy()
    vals = price_series.to_numpy(dtype="float64")

    try:
        bubble_res = fit_lppl_bubble_cached(key, vals, idx_int)
    except Exception:
        show_error_black("LPPL Fit Failed (Uptrend).")
        st.stop()

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price_val = float(price_series.iloc[0])
    gain = peak_price / start_price_val
    gain_pct = (gain - 1.0) * 100.0

    peak_date_int = int(pd.Timestamp(peak_date).value)
    neg_res = fit_lppl_negative_bubble_cached(key, vals, idx_int, peak_date_int=peak_date_int, min_points=10, min_drop_ratio=0.03)

    tc_up_date = pd.Timestamp(bubble_res["tc_date"])
    end_ts = pd.Timestamp(end_date)
    down_tc_date = pd.Timestamp(neg_res["tc_date"]) if neg_res.get("ok") else None

    signal_label, score = compute_signal_and_score(tc_up_date, end_ts, down_tc_date)

    # -----------------------------------------------------------
    # MAIN CHART RENDER
    # -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))
    BG_COLOR = "#050505"
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.plot(price_series.index, price_series.values, color="#F0F0F0", linewidth=0.8, alpha=0.9, zorder=5)
    GOLD_COLOR = "#C5A059"
    ax.plot(price_series.index, bubble_res["price_fit"], color=GOLD_COLOR, linewidth=2.0, alpha=1.0, zorder=6)
    ax.axvline(bubble_res["tc_date"], color="#ff4d4f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(peak_date, color="white", linestyle=":", linewidth=0.5, alpha=0.4)
    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="cyan", linewidth=0.8, alpha=0.7)
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="#008b8b", linewidth=1.5, alpha=0.8)
        ax.axvline(neg_res["tc_date"], color="#00ff00", linestyle="--", linewidth=1.2, alpha=0.8)
    last_date = price_series.index[-1]
    total_days = (last_date - price_series.index[0]).days
    margin_days = int(total_days * 0.15)
    margin_limit_date = last_date + timedelta(days=margin_days)
    ax.set_xlim(right=margin_limit_date)
    last_price = price_series.values[-1]
    last_model_val = bubble_res["price_fit"][-1]
    text_date_offset = last_date + timedelta(days=2)
    ax.text(text_date_offset, last_price, f" ← {ticker.strip()}", color="#F0F0F0",
            fontsize=10, fontweight='bold', fontname='serif', va='center', zorder=10)
    ax.text(text_date_offset, last_model_val, f" ← Model", color=GOLD_COLOR,
            fontsize=10, fontweight='bold', fontname='serif', va='center', zorder=10)
    peak_val = price_series.max()
    peak_dt = price_series.idxmax()
    ax.text(peak_dt, peak_val * 1.05, f"Peak\n{peak_dt.strftime('%Y-%m-%d')}",
            color="#888888", fontsize=7, ha='center', fontname='sans-serif')
    ax.text(0.02, 0.92, ticker.strip(), transform=ax.transAxes,
            fontsize=28, color="#F0F0F0", fontweight='normal', fontname='serif')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.grid(color="#333333", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', colors='#888888', labelsize=8)
    ax.tick_params(axis='y', colors='#888888', labelsize=8)
    draw_score_overlay(ax, score, signal_label)
    draw_logo_overlay(ax)
    st.pyplot(fig)

    # -----------------------------------------------------------
    # NEW: SHORT VOLUME ANALYSIS SECTION
    # -----------------------------------------------------------
    ticker_clean = ticker.strip()
    is_jp = ticker_clean.endswith(".T")
    is_non_us = any(ticker_clean.endswith(sfx) for sfx in [".T", ".HK", ".L", ".DE", ".PA", ".AS", ".MI", ".SW"])
    is_us = not is_non_us

    st.markdown("---")
    st.markdown("### ■ Short Volume / 空売り比率 Analysis")

    if is_us:
        with st.spinner(f"Fetching FINRA Short Volume for {ticker_clean}..."):
            short_df = fetch_us_short_volume(ticker_clean)
        if not short_df.empty:
            render_short_volume_chart(short_df, ticker_clean, is_jp=False)
            st.caption("※Source: FINRA Reg SHO Daily Short Sale Volume API (public, no auth). Data aggregated across all TRFs (NQTRF/NYTRF/NCTRF).")
            with st.expander("View Short Volume Raw Data"):
                display_cols = [c for c in ["Date", "ShortVolume", "TotalVolume", "NonShortVolume", "ShortRatio"] if c in short_df.columns]
                st.dataframe(short_df[display_cols].sort_values("Date", ascending=False), use_container_width=True)
            avg_ratio = short_df["ShortRatio"].mean() if "ShortRatio" in short_df.columns else 0
            max_ratio = short_df["ShortRatio"].max() if "ShortRatio" in short_df.columns else 0
            max_ratio_date = short_df.loc[short_df["ShortRatio"].idxmax(), "Date"] if "ShortRatio" in short_df.columns and len(short_df) > 0 else "N/A"
            min_ratio = short_df["ShortRatio"].min() if "ShortRatio" in short_df.columns else 0
            min_ratio_date = short_df.loc[short_df["ShortRatio"].idxmin(), "Date"] if "ShortRatio" in short_df.columns and len(short_df) > 0 else "N/A"
            st.markdown("**Quick Stats (FINRA Short Volume)**")
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Avg Ratio", f"{avg_ratio:.1f}%")
            with stat_cols[1]:
                st.metric("Max Ratio", f"{max_ratio:.1f}%", delta=f"{max_ratio_date.strftime('%m/%d') if hasattr(max_ratio_date, 'strftime') else max_ratio_date}")
            with stat_cols[2]:
                st.metric("Min Ratio", f"{min_ratio:.1f}%", delta=f"{min_ratio_date.strftime('%m/%d') if hasattr(min_ratio_date, 'strftime') else min_ratio_date}")
            with stat_cols[3]:
                baseline_label = "Bullish (<40%)" if avg_ratio < 40 else ("Bearish (>50%)" if avg_ratio > 50 else "Neutral (40-50%)")
                st.metric("Sentiment", baseline_label)
        else:
            st.warning("Could not fetch FINRA short volume data. The FINRA API may be temporarily unavailable.")
            st.info("Supported: US-listed stocks (NASDAQ/NYSE/AMEX). Data from FINRA Reg SHO API (api.finra.org).")

    elif is_jp:
        with st.spinner(f"Fetching JP Short Selling Data for {ticker_clean}..."):
            short_df = fetch_jp_short_selling_ratio(ticker_clean)
        if not short_df.empty and "ShortRatio" in short_df.columns:
            render_short_volume_chart(short_df, ticker_clean, is_jp=True)
            st.caption("※Source: karauri.net (JPX空売り報告). Scraping-based — may break if site structure changes.")
            with st.expander("View Short Selling Raw Data"):
                display_cols = [c for c in ["Date", "ShortVolume", "TotalVolume", "NonShortVolume", "ShortRatio"] if c in short_df.columns]
                st.dataframe(short_df[display_cols].sort_values("Date", ascending=False), use_container_width=True)
            avg_ratio = short_df["ShortRatio"].mean()
            st.markdown(f"**Avg Short Ratio (period): {avg_ratio:.1f}%**")
        else:
            st.info("JP short selling ratio data not available from karauri.net for this ticker. Margin balance data is available in the Graphs section below.")
    else:
        st.info(f"Short volume analysis is currently supported for US stocks (FINRA) and JP stocks (.T). Ticker '{ticker_clean}' is not in a supported market.")

    # -----------------------------------------------------------
    # EXISTING SECTIONS BELOW
    # -----------------------------------------------------------
    if signal_label == "HIGH":
        risk_label = "High"; risk_color = "#ff4d4f"
    elif signal_label == "CAUTION":
        risk_label = "Caution"; risk_color = "#ffc53d"
    else:
        risk_label = "Safe"; risk_color = "#3CB371"

    st.markdown("---")
    pdict = bubble_res.get("param_dict", {})
    r2 = float(bubble_res.get("r2", np.nan))
    m = float(pdict.get("m", np.nan))
    omega = float(pdict.get("omega", np.nan))
    render_bubble_flow(r2, m, omega)
    verdict, _ = bubble_judgement(r2, m, omega)
    st.markdown("### Admin Indicators (Minimal Explanation for Bubble Check)")
    tc_up_norm = pd.Timestamp(bubble_res["tc_date"]).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    days_to_tc = int((tc_up_norm - end_norm).days)
    rmse = float(bubble_res.get("rmse", np.nan))
    c_over_b = float(pdict.get("abs_C_over_B", np.nan))
    log_period = float(pdict.get("log_period_2pi_over_omega", np.nan))
    N = int(bubble_res.get("N", 0))
    admin_rows = [
        ["1. R² (log space)", r2, "Bubble condition: R²≥0.65."],
        ["2. m (Most Important)", m, "Bubble-like: m∈[0.25,0.70] (Typical 0.3-0.6)."],
        ["3. ω", omega, "Bubble-like: ω∈[6,13]. ω≥18 is too fast -> Bubble-like (Warning)."],
        ["Verdict (Bubble)", verdict, "Final result from flow above."],
        ["t_c (Date Approx)", str(tc_up_norm.date()), "Near/Past tc = Potential end stage."],
        ["Days to t_c", days_to_tc, "End stage proximity (Shorter/Negative = Higher risk)."],
        ["RMSE (log space)", rmse, "Fit stability (Lower is better)."],
        ["|C/B|", c_over_b, "≥2.0 increases suspicion of overfitting."],
        ["2π/ω", log_period, "Period representation."],
        ["Fit window N", N, "Period N."]
    ]
    st.dataframe(pd.DataFrame(admin_rows, columns=["Indicator", "Value", "Note"]), use_container_width=True)
    st.markdown("#### Raw Params (A, B, C, m, tc, ω, φ)")
    raw_params = bubble_res.get("params", np.array([], dtype=float)).astype(float)
    if raw_params.size == 7:
        raw_df = pd.DataFrame([raw_params], columns=["A", "B", "C", "m", "tc", "omega", "phi"])
    else:
        raw_df = pd.DataFrame([raw_params])
    st.dataframe(raw_df, use_container_width=True)
    st.markdown("#### Fit Quality Flags (Sanity Check)")
    binfo = bubble_res.get("bounds_info", {})
    if raw_params.size == 7:
        A_, B_, C_, m_, tc_, omega_, phi_ = [float(x) for x in raw_params]
    else:
        A_, B_, C_, m_, tc_, omega_, phi_ = [np.nan]*7
    def near_bound(x, lo, hi, tol=0.02):
        if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi == lo: return False
        r = (x - lo) / (hi - lo)
        return (r < tol) or (r > 1 - tol)
    flags = []
    if np.isfinite(m_) and (m_ < 0.2 or m_ > 0.8): flags.append("m is outside typical bubble band.")
    if near_bound(m_, binfo.get("m_low", np.nan), binfo.get("m_high", np.nan)): flags.append("m is near boundary (Suspected boundary solution).")
    if near_bound(tc_, binfo.get("tc_low", np.nan), binfo.get("tc_high", np.nan)): flags.append("tc is near boundary.")
    if near_bound(omega_, binfo.get("omega_low", np.nan), binfo.get("omega_high", np.nan)): flags.append("ω is near boundary.")
    if np.isfinite(c_over_b) and c_over_b > 2.0: flags.append("|C/B| is large (Excessive oscillation).")
    if flags: st.write(pd.DataFrame({"Flags": flags}))
    else: st.write("No clear red flags in simple heuristics.")

    st.markdown("### Interpretation for Investment (Admin Only)")
    summary, bullets = admin_interpretation_text(bubble_res, end_date)
    with st.expander("Investment Interpretation Note (Admin Only)", expanded=True):
        st.markdown(f"**Summary**: {summary}")
        st.markdown("**Points**")
        st.markdown("\n".join([f"- {b}" for b in bullets]))
        st.markdown("---")
        st.markdown("**Usage (Practical)**")
        st.markdown("- Bubble check uses **R²(Fit) x m(Shape) x ω(Naturalness)**.\n- t_c indicates 'Is it late stage?', not 'Is it a bubble?'.")

    st.markdown("---")
    with st.expander("Mid-term Quant Table (A-J + Threshold + Judgement)", expanded=False):
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1: bench = st.text_input("Benchmark (Index)", value="ACWI", help="Ex: ACWI / ^GSPC / ^N225")
        with col_b2: mid_window = st.number_input("WINDOW (Table)", min_value=5, max_value=120, value=20, step=1)
        try:
            df_mid = build_midterm_quant_table(ticker=ticker.strip(), bench=bench.strip(), start_date=start_date, end_date=end_date, window=int(mid_window))
            st.dataframe(df_mid[["Block", "Metric", "Value", "Threshold", "Judgement", "Note"]], use_container_width=True, hide_index=True)
        except Exception as e:
            show_error_black(f"Failed to build table: {e}")

    st.markdown("---")
    with st.expander("Graphs (Index/Deviation/Regression/Vol/Drawdown)", expanded=True):
        gcol1, gcol2, gcol3 = st.columns([2, 1, 1])
        with gcol1: graph_bench = st.text_input("Benchmark (For Graph)", value="ACWI")
        with gcol2: graph_window = st.number_input("WINDOW (For Graph)", min_value=5, max_value=120, value=20, step=1)
        with gcol3: trading_days = st.number_input("Trading days/year", min_value=200, max_value=365, value=252, step=1)
        try:
            prices_pair = fetch_prices_pair_cached(ticker=ticker.strip(), bench=graph_bench.strip(), start_date=start_date, end_date=end_date)
            st.caption(f"Data: {prices_pair.index.min().date()} to {prices_pair.index.max().date()} | rows={len(prices_pair)}")
            render_graph_pack_from_prices(prices=prices_pair, ticker=ticker.strip(), bench=graph_bench.strip(), window=int(graph_window), trading_days=int(trading_days))
        except Exception as e:
            show_error_black(f"Failed to render graphs: {e}")

if __name__ == "__main__":
    main()
