import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st
import os
import time
import hmac
import hashlib
import base64

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
# AUTH GATE: Admin-style auth (OS_TOKEN_SECRET_ADMIN + ADMIN_EMAILS)
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


# -------------------------------------------------------
# LPPL-like model
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
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 20
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    A_low = float(np.min(log_price) - 2.0)
    A_high = float(np.max(log_price) + 2.0)

    lower_bounds = [A_low, -20, -20, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N + 250, 25.0, np.pi]

    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)

    params, _ = curve_fit(
        lppl,
        t,
        log_price,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)

    ss_res = np.sum((log_price - log_fit) ** 2)
    ss_tot = np.sum((log_price - log_price.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    tc_val = float(params[4])
    last_idx = N - 1
    future_units = tc_val - last_idx

    last_date = price_series.index[-1]
    tc_date = last_date + timedelta(days=future_units)

    return {
        "params": params,
        "price_fit": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_val,
    }


def fit_lppl_negative_bubble(
    price_series: pd.Series,
    peak_date,
    min_points: int = 10,
    min_drop_ratio: float = 0.03,
):
    down_series = price_series[price_series.index >= peak_date].copy()

    if len(down_series) < min_points:
        return {"ok": False}

    peak_price = float(price_series.loc[peak_date])
    last_price = float(down_series.iloc[-1])
    drop_ratio = (peak_price - last_price) / peak_price
    if drop_ratio < min_drop_ratio:
        return {"ok": False}

    price_down = down_series.values.astype(float)
    t_down = np.arange(len(price_down), dtype=float)

    log_down = np.log(price_down)
    neg_log_down = -log_down

    N_down = len(t_down)

    A_init = float(np.mean(neg_log_down))
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N_down + 15
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    A_low = float(np.min(neg_log_down) - 2.0)
    A_high = float(np.max(neg_log_down) + 2.0)

    lower_bounds = [A_low, -20, -20, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N_down + 200, 25.0, np.pi]

    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)

    try:
        params_down, _ = curve_fit(
            lppl,
            t_down,
            neg_log_down,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
    except Exception:
        return {"ok": False}

    neg_log_fit = lppl(t_down, *params_down)
    log_fit = -neg_log_fit
    price_fit_down = np.exp(log_fit)

    ss_res = np.sum((neg_log_down - neg_log_fit) ** 2)
    ss_tot = np.sum((neg_log_down - neg_log_down.mean()) ** 2)
    r2_down = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    tc_val = float(params_down[4])
    last_idx = N_down - 1
    future_units = tc_val - last_idx

    last_date = down_series.index[-1]
    tc_bottom_date = last_date + timedelta(days=future_units)

    return {
        "ok": True,
        "down_series": down_series,
        "price_fit_down": price_fit_down,
        "r2": r2_down,
        "tc_date": tc_bottom_date,
        "tc_days": tc_val,
        "params": params_down,
    }


# -------------------------------------------------------
# Price fetch (cached) + strict validation
# -------------------------------------------------------
@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_price_series_cached(ticker: str, start_date: date, end_date: date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            raise ValueError("INVALID_TICKER_OR_NO_DATA")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("INVALID_TICKER_OR_NO_DATA")

    s = s.dropna()

    if s.empty or len(s) < 30:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    vals = s.to_numpy(dtype="float64")
    if not np.all(np.isfinite(vals)):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    if np.any(vals <= 0):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    return s


# -------------------------------------------------------
# fit cache helpers
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
def fit_lppl_negative_bubble_cached(
    price_key: str,
    price_values: np.ndarray,
    idx_int: np.ndarray,
    peak_date_int: int,
    min_points: int,
    min_drop_ratio: float,
):
    idx = pd.to_datetime(idx_int)
    s = pd.Series(price_values, index=idx)
    peak_date = pd.to_datetime(peak_date_int)
    return fit_lppl_negative_bubble(
        s,
        peak_date=peak_date,
        min_points=min_points,
        min_drop_ratio=min_drop_ratio,
    )


# =======================================================
# FINAL: Phase + time-distance scoring (date-based)
# =======================================================
def compute_signal_and_score(tc_up_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             down_tc_date: pd.Timestamp | None) -> tuple[str, int]:
    now = pd.Timestamp(end_date).normalize()
    tc_up = pd.Timestamp(tc_up_date).normalize()

    if down_tc_date is not None:
        down_tc = pd.Timestamp(down_tc_date).normalize()
        delta = (down_tc - now).days

        if delta > 0:
            s = _lin_map(
                x=delta,
                x0=DOWN_FUTURE_NEAR_DAYS,
                x1=DOWN_FUTURE_FAR_DAYS,
                y0=90,
                y1=80,
            )
            return ("HIGH", int(round(_clamp(s, 80, 90))))

        past = abs(delta)
        s = _lin_map(
            x=past,
            x0=DOWN_PAST_NEAR_DAYS,
            x1=DOWN_PAST_FAR_DAYS,
            y0=90,
            y1=100,
        )
        return ("HIGH", int(round(_clamp(s, 90, 100))))

    gap = (tc_up - now).days
    WARNING_BUFFER = 14

    if gap > WARNING_BUFFER:
        s = _lin_map(
            x=gap,
            x0=UP_FUTURE_NEAR_DAYS,
            x1=UP_FUTURE_FAR_DAYS,
            y0=59,
            y1=0,
        )
        return ("SAFE", int(round(_clamp(s, 0, 59))))

    past_warning = WARNING_BUFFER - gap

    s = _lin_map(
        x=past_warning,
        x0=0,
        x1=UP_PAST_FAR_DAYS,
        y0=60,
        y1=79,
    )
    return ("CAUTION", int(round(_clamp(s, 60, 79))))


# =======================================================
# Chart overlay helpers (from admin)
# =======================================================
def draw_score_overlay(ax, score: int, label: str):
    if score < 60:
        score_color = "#3CB371"
    elif score < 80:
        score_color = "#ffc53d"
    else:
        score_color = "#ff4d4f"
    ax.text(
        0.02, 0.78, str(score),
        transform=ax.transAxes, fontsize=36, color=score_color,
        fontweight='bold', ha='left', va='bottom', fontname='serif', zorder=20,
    )


def draw_logo_overlay(ax):
    ax.text(
        0.95, 0.03, "OUT-STANDER",
        transform=ax.transAxes, fontsize=24, color='#3d3320',
        fontweight='bold', fontname='serif', ha='right', va='bottom',
        zorder=0, alpha=0.9,
    )


# -------------------------------------------------------
# Streamlit app
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
        .stApp {
            background-color: #050505 !important;
            color: #ffffff !important;
        }
        div[data-testid="stMarkdownContainer"] p, label {
            color: #ffffff !important;
            font-family: 'Times New Roman', serif;
        }
        input.st-ai, input.st-ah, div[data-baseweb="input"] {
            background-color: #111111 !important;
            color: #ffffff !important;
            border-color: #333 !important;
        }
        input { color: #ffffff !important; }
        input::placeholder { color: rgba(255,255,255,0.45) !important; }
        div[data-baseweb="input"] svg { fill: #ffffff !important; }
        div[data-testid="stFormSubmitButton"] button {
            background-color: #1a1a1a !important;
            color: #d4af37 !important;
            border: 1px solid #333 !important;
            font-family: 'Times New Roman', serif;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #d4af37 !important;
            border-color: #d4af37 !important;
            color: #000000 !important;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
        .custom-error {
            background-color: #141518;
            border: 1px solid #2a2c30;
            border-radius: 0px;
            padding: 14px 18px;
            color: #ffffff;
            font-size: 0.95rem;
            margin-top: 12px;
        }
        .brand-header {
            font-family: 'Playfair Display', 'Times New Roman', serif;
            font-size: 1.4rem;
            font-weight: 400;
            color: #C5A059;
            letter-spacing: 0.15em;
            padding: 18px 0 8px 0;
            border-bottom: 1px solid #1a1a1a;
            margin-bottom: 16px;
        }
        .info-card {
            background-color: #0b0c0e;
            border: 1px solid #1a1c1f;
            border-radius: 0px;
            padding: 16px 20px;
            margin-top: 8px;
        }
        .info-card .card-label {
            font-size: 0.75rem;
            color: #666;
            font-family: 'Times New Roman', serif;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .info-card .card-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f0f0f0;
            font-family: 'Times New Roman', serif;
            line-height: 1.2;
        }
        .info-card .card-sub {
            font-size: 0.8rem;
            color: #555;
            font-family: 'Times New Roman', serif;
            margin-top: 2px;
        }
        .score-legend {
            background-color: #0b0c0e;
            border: 1px solid #1a1c1f;
            border-radius: 0px;
            padding: 10px 20px;
            margin-top: 8px;
            display: flex;
            gap: 28px;
            align-items: center;
        }
        .score-legend .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'Times New Roman', serif;
            font-size: 0.82rem;
        }
        .score-legend .legend-range {
            font-weight: 700;
        }
        .score-legend .legend-label {
            font-weight: 400;
            color: #777;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def show_error_black(msg: str):
        st.markdown(
            f"""<div class="custom-error">{msg}</div>""",
            unsafe_allow_html=True,
        )

    # ---- Minimal brand header (replaces banner) ----
    st.markdown('<div class="brand-header">OUT-STANDER</div>', unsafe_allow_html=True)

    # ---- Input form ----
    with st.form("input_form"):
        ticker = st.text_input(
            "Ticker",
            value="",
            placeholder="e.g., NVDA / 0700.HK / 7203.T"
        )

        today = date.today()
        default_start = today - timedelta(days=220)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", default_start)
        with col2:
            end_date = st.date_input("End", today)

        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    if not ticker.strip():
        show_error_black("Invalid ticker symbol or no price data.")
        st.stop()

    try:
        price_series = fetch_price_series_cached(ticker.strip(), start_date, end_date)
    except ValueError:
        show_error_black("Invalid ticker symbol or no price data.")
        st.stop()
    except Exception:
        show_error_black("Invalid ticker symbol or no price data.")
        st.stop()

    if len(price_series) < 30:
        show_error_black("Invalid ticker symbol or no price data.")
        st.stop()

    # ---- fit caches ----
    key = series_cache_key(price_series)
    idx_int = price_series.index.astype("int64").to_numpy()
    vals = price_series.to_numpy(dtype="float64")

    try:
        bubble_res = fit_lppl_bubble_cached(key, vals, idx_int)
    except Exception:
        show_error_black("Invalid ticker symbol or no price data.")
        st.stop()

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price_val = float(price_series.iloc[0])
    gain = peak_price / start_price_val
    gain_pct = (gain - 1.0) * 100.0

    peak_date_int = int(pd.Timestamp(peak_date).value)
    neg_res = fit_lppl_negative_bubble_cached(
        key, vals, idx_int,
        peak_date_int=peak_date_int,
        min_points=10,
        min_drop_ratio=0.03,
    )

    tc_up_date = pd.Timestamp(bubble_res["tc_date"])
    end_ts = pd.Timestamp(end_date)
    down_tc_date = pd.Timestamp(neg_res["tc_date"]) if neg_res.get("ok") else None

    signal_label, score = compute_signal_and_score(tc_up_date, end_ts, down_tc_date)

    # =======================================================
    # CHART (Admin-style HNWI design)
    # =======================================================
    BG_COLOR = "#050505"
    GOLD_COLOR = "#C5A059"

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Price line
    ax.plot(price_series.index, price_series.values,
            color="#F0F0F0", linewidth=0.8, alpha=0.9, zorder=5)

    # Up model (gold)
    ax.plot(price_series.index, bubble_res["price_fit"],
            color=GOLD_COLOR, linewidth=2.0, alpha=1.0, zorder=6)

    # Turn (up) vertical line
    ax.axvline(bubble_res["tc_date"], color="#ff4d4f",
               linestyle="--", linewidth=1.2, alpha=0.8)

    # Peak vertical line
    ax.axvline(peak_date, color="white", linestyle=":", linewidth=0.5, alpha=0.4)

    # Negative bubble (if detected)
    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values,
                color="cyan", linewidth=0.8, alpha=0.7)
        ax.plot(down.index, neg_res["price_fit_down"],
                "--", color="#008b8b", linewidth=1.5, alpha=0.8)
        ax.axvline(neg_res["tc_date"], color="#00ff00",
                   linestyle="--", linewidth=1.2, alpha=0.8)

    # Right margin (15% for label readability)
    last_date = price_series.index[-1]
    total_days = (last_date - price_series.index[0]).days
    margin_days = int(total_days * 0.15)
    margin_limit_date = last_date + timedelta(days=margin_days)
    ax.set_xlim(right=margin_limit_date)

    # Right-side labels (instead of legend box)
    last_price = price_series.values[-1]
    last_model_val = bubble_res["price_fit"][-1]
    text_date_offset = last_date + timedelta(days=2)
    ax.text(text_date_offset, last_price, f" \u2190 {ticker.strip()}",
            color="#F0F0F0", fontsize=10, fontweight='bold',
            fontname='serif', va='center', zorder=10)
    ax.text(text_date_offset, last_model_val, " \u2190 Model",
            color=GOLD_COLOR, fontsize=10, fontweight='bold',
            fontname='serif', va='center', zorder=10)

    # Peak annotation (inside chart, avoid top overflow)
    peak_val = price_series.max()
    peak_dt = price_series.idxmax()
    y_min_data = price_series.min()
    y_max_data = peak_val
    y_range = y_max_data - y_min_data
    ax.set_ylim(y_min_data - y_range * 0.03, y_max_data + y_range * 0.12)
    ax.text(peak_dt, peak_val + y_range * 0.04,
            f"Peak\n{peak_dt.strftime('%Y-%m-%d')}",
            color="#888888", fontsize=7, ha='center', fontname='sans-serif')

    # Ticker name (top left)
    ax.text(0.02, 0.92, ticker.strip(),
            transform=ax.transAxes, fontsize=28, color="#F0F0F0",
            fontweight='normal', fontname='serif')

    # Axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.grid(color="#333333", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', colors='#888888', labelsize=8)
    ax.tick_params(axis='y', colors='#888888', labelsize=8)

    # Score overlay + Logo watermark
    draw_score_overlay(ax, score, signal_label)
    draw_logo_overlay(ax)

    st.pyplot(fig)
    plt.close(fig)

    # =======================================================
    # CARD GROUP (Below chart)
    # =======================================================

    # ---- Row 1: Score Legend (horizontal) ----
    score_legend_html = """
    <div class="score-legend">
        <div class="legend-item">
            <span class="legend-range" style="color: #ff4d4f;">80 – 100</span>
            <span class="legend-label">Risk</span>
        </div>
        <div class="legend-item">
            <span class="legend-range" style="color: #ffc53d;">60 – 79</span>
            <span class="legend-label">Caution</span>
        </div>
        <div class="legend-item">
            <span class="legend-range" style="color: #3CB371;">0 – 59</span>
            <span class="legend-label">Stable</span>
        </div>
    </div>
    """
    st.markdown(score_legend_html, unsafe_allow_html=True)

    # ---- Row 2: Info cards (5 columns) ----
    if score < 60:
        display_label = "Stable"
        label_color = "#3CB371"
    elif score < 80:
        display_label = "Caution"
        label_color = "#ffc53d"
    else:
        display_label = "Risk"
        label_color = "#ff4d4f"

    tc_up_str = pd.Timestamp(bubble_res["tc_date"]).strftime("%Y-%m-%d")
    peak_date_str = pd.Timestamp(peak_date).strftime("%Y-%m-%d")
    days_to_turn_up = (pd.Timestamp(bubble_res["tc_date"]).normalize() - pd.Timestamp(end_date).normalize()).days

    if neg_res.get("ok"):
        tc_down_str = pd.Timestamp(neg_res["tc_date"]).strftime("%Y-%m-%d")
        days_to_turn_down = (pd.Timestamp(neg_res["tc_date"]).normalize() - pd.Timestamp(end_date).normalize()).days
    else:
        tc_down_str = "N/A"
        days_to_turn_down = None

    def _days_display(d):
        if d is None:
            return ""
        if d > 0:
            return f"{d}d ahead"
        elif d == 0:
            return "Today"
        else:
            return f"{abs(d)}d ago"

    col_score, col_peak, col_gain, col_turn_up, col_turn_down = st.columns(5)

    with col_score:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Score</div>
                <div class="card-value" style="color: {label_color};">{score}</div>
                <div class="card-sub" style="color: {label_color};">{display_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_peak:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Peak</div>
                <div class="card-value">{peak_date_str}</div>
                <div class="card-sub">{peak_price:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_gain:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Gain (Start → Peak)</div>
                <div class="card-value">{gain:.2f}x</div>
                <div class="card-sub" style="color: #3CB371;">{gain_pct:+.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_turn_up:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Turn (Up)</div>
                <div class="card-value" style="color: #ff4d4f;">{tc_up_str}</div>
                <div class="card-sub">{_days_display(days_to_turn_up)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_turn_down:
        down_color = "#00ff00" if tc_down_str != "N/A" else "#555"
        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">Turn (Down)</div>
                <div class="card-value" style="color: {down_color};">{tc_down_str}</div>
                <div class="card-sub">{_days_display(days_to_turn_down)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
