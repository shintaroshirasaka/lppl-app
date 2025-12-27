import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st
import os

# =======================================================
# AUTH GATE: Require signed short-lived token (?t=...)
#   - Render env var: OS_TOKEN_SECRET
#   - Query param: ?t=<token>
# Token format:
#   base64url("email|exp").base64url(hex(hmac_sha256("email|exp", secret)))
# =======================================================
import time
import hmac
import hashlib
import base64


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
OS_TOKEN_SECRET = os.environ.get("OS_TOKEN_SECRET", "").strip()
token = st.query_params.get("t", "")

if not OS_TOKEN_SECRET or not token:
    st.error("Login required.")
    st.stop()

ok, authed_email = verify_token(token, OS_TOKEN_SECRET)
if not ok:
    st.error("Login required.")
    st.stop()


# =======================================================
# Cache settings (ここだけ調整すればOK)
# =======================================================
PRICE_TTL_SECONDS = 15 * 60         # yfinanceのキャッシュ（例: 15分）
FIT_TTL_SECONDS = 24 * 60 * 60      # fit結果のキャッシュ（例: 24時間）

# =======================================================
# NEW SCORE SETTINGS (あなた提案の「時間距離ベース」スコア)
# =======================================================
# どこまでを「近い/遠い」とみなすか（営業日ベースの目安）
UP_FUTURE_NEAR_DAYS = 30     # t_c が30営業日以内なら、緑の上限付近
UP_FUTURE_FAR_DAYS  = 180    # t_c が180営業日以上先なら、緑の下限付近

UP_PAST_NEAR_DAYS   = 7      # t_c が直近過去（7営業日以内）→黄色の下限付近
UP_PAST_FAR_DAYS    = 120    # t_c が120営業日以上過去 →黄色の上限付近

DOWN_TC_NEAR_DAYS   = 30     # down_tc が30日以内に迫る → 赤強め
DOWN_TC_FAR_DAYS    = 120    # down_tc が120日以上先 → 赤弱め（それでも赤）

DOWN_PAST_NEAR_DAYS = 7      # down_tc が過去に入って間もない → 赤中
DOWN_PAST_FAR_DAYS  = 60     # down_tc が過去に長く入っている → 100に近づく


# -------------------------------------------------------
# LPPL-like model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    """上昇局面へのフィット"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)

    N = len(t)

    # 初期値
    A_init = np.mean(log_price)
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 20
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    # 境界
    lower_bounds = [-10, -10, -10, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [10, 10, 10, 0.99, N + 250, 25.0, np.pi]

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
    r2 = 1 - ss_res / ss_tot

    first_date = price_series.index[0]
    tc_days = float(params[4])
    tc_date = first_date + timedelta(days=tc_days)

    return {
        "params": params,
        "price_fit": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days,
    }


def fit_lppl_negative_bubble(
    price_series: pd.Series,
    peak_date,
    min_points: int = 10,
    min_drop_ratio: float = 0.03,
):
    """下落局面へのフィット（負バブル）"""
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
    neg_log_down = -log_down  # 下落をバブルとして見る

    N_down = len(t_down)
    A_init = np.mean(neg_log_down)
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N_down + 15
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    lower_bounds = [-10, -10, -10, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [10, 10, 10, 0.99, N_down + 200, 25.0, np.pi]

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
    r2_down = 1 - ss_res / ss_tot

    first_down_date = down_series.index[0]
    tc_days = float(params_down[4])
    tc_bottom_date = first_down_date + timedelta(days=tc_days)

    return {
        "ok": True,
        "down_series": down_series,
        "price_fit_down": price_fit_down,
        "r2": r2_down,
        "tc_date": tc_bottom_date,
        "tc_days": tc_days,
        "params": params_down,
    }


# -------------------------------------------------------
# 価格データ取得（キャッシュ版）
# -------------------------------------------------------
@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_price_series_cached(ticker: str, start_date: date, end_date: date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError("価格データが取得できません。")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            raise ValueError("終値カラムが見つかりません。")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("終値カラムが見つかりません。")

    return s.dropna()


# -------------------------------------------------------
# fit結果キャッシュ（Seriesを安定キー化）
# -------------------------------------------------------
def series_cache_key(s: pd.Series) -> str:
    idx = s.index.astype("int64").to_numpy()  # datetime -> ns int
    vals = s.to_numpy(dtype="float64")
    h = hashlib.sha256()
    h.update(idx.tobytes())
    h.update(vals.tobytes())
    return h.hexdigest()


@st.cache_data(ttl=FIT_TTL_SECONDS, show_spinner=False)
def fit_lppl_bubble_cached(price_key: str, price_values: np.ndarray, idx_int: np.ndarray):
    idx = pd.to_datetime(idx_int)  # ns int -> datetime
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
# NEW: Signal & Score (あなた提案の定義)
#   - green: tc is future
#   - yellow: tc is past but no down_tc
#   - red: down_tc exists
# score: within each regime, increases with "urgency"
# =======================================================
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lin_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    # x0->y0, x1->y1 (linear), clamp outside
    if x0 == x1:
        return y0
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def compute_signal_and_score(
    last_index: float,
    tc_index: float,
    down_tc_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> tuple[str, int]:
    """
    Returns: (signal_label, score_int)
      signal_label in {"SAFE","CAUTION","HIGH"}
      score_int in 0..100
    """

    # 1) RED regime if down_tc exists
    if down_tc_date is not None:
        delta_days = (down_tc_date.normalize() - end_date.normalize()).days

        # down_tc in the future -> early red (80..95)
        if delta_days > 0:
            # closer -> higher
            s = _lin_map(
                x=delta_days,
                x0=DOWN_TC_FAR_DAYS,  # far future -> weaker red
                x1=DOWN_TC_NEAR_DAYS, # near future -> stronger red
                y0=80,
                y1=95
            )
            s = _clamp(s, 80, 95)
            return ("HIGH", int(round(s)))

        # down_tc already in the past -> deeper red (90..100)
        past_days = abs(delta_days)
        s = _lin_map(
            x=past_days,
            x0=DOWN_PAST_NEAR_DAYS,
            x1=DOWN_PAST_FAR_DAYS,
            y0=90,
            y1=100
        )
        s = _clamp(s, 90, 100)
        return ("HIGH", int(round(s)))

    # 2) No down_tc: decide by tc_index vs now (last_index)
    gap = tc_index - last_index

    # GREEN regime: tc in future
    if gap > 0:
        # far -> low, near -> high (0..59)
        s = _lin_map(
            x=gap,
            x0=UP_FUTURE_FAR_DAYS,
            x1=UP_FUTURE_NEAR_DAYS,
            y0=0,
            y1=59
        )
        s = _clamp(s, 0, 59)
        return ("SAFE", int(round(s)))

    # YELLOW regime: tc in past
    past = abs(gap)

    s = _lin_map(
        x=past,
        x0=UP_PAST_NEAR_DAYS,
        x1=UP_PAST_FAR_DAYS,
        y0=60,
        y1=79
    )
    s = _clamp(s, 60, 79)
    return ("CAUTION", int(round(s)))


# -------------------------------------------------------
# Streamlit アプリ本体
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0c0e !important;
            color: #ffffff !important;
        }
        div[data-testid="stMarkdownContainer"] p, label {
            color: #ffffff !important;
        }
        input.st-ai, input.st-ah, div[data-baseweb="input"] {
            background-color: #1a1c1f !important;
            color: #ffffff !important;
            border-color: #444 !important;
        }
        input { color: #ffffff !important; }
        input::placeholder { color: rgba(255,255,255,0.45) !important; }
        div[data-baseweb="input"] svg { fill: #ffffff !important; }
        div[data-testid="stFormSubmitButton"] button {
            background-color: #222428 !important;
            color: #ffffff !important;
            border: 1px solid #555 !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #444 !important;
            border-color: #888 !important;
            color: #ffffff !important;
        }
        div[data-testid="stFormSubmitButton"] button:active { color: #ffffff !important; }
        div[data-testid="stFormSubmitButton"] p { color: #ffffff !important; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists("banner.png"):
        st.image("banner.png", use_container_width=True)
    else:
        st.title("Out-stander")

    with st.form("input_form"):
        ticker = st.text_input(
            "Ticker",
            value="",
            placeholder="例: NVDA / 0700.HK / 7203.T"
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
        st.error("Tickerを入力してください（例: NVDA / 0700.HK / 7203.T）")
        st.stop()

    try:
        price_series = fetch_price_series_cached(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.stop()

    if len(price_series) < 30:
        st.error("データが少なすぎます。期間を伸ばしてください。")
        st.stop()

    # --- fit キャッシュ用キー ---
    key = series_cache_key(price_series)
    idx_int = price_series.index.astype("int64").to_numpy()
    vals = price_series.to_numpy(dtype="float64")

    # --- curve_fit（上昇） ---
    bubble_res = fit_lppl_bubble_cached(key, vals, idx_int)

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price_val = float(price_series.iloc[0])
    gain = peak_price / start_price_val
    gain_pct = (gain - 1.0) * 100.0

    # --- curve_fit（下落） ---
    peak_date_int = int(pd.Timestamp(peak_date).value)
    neg_res = fit_lppl_negative_bubble_cached(
        key,
        vals,
        idx_int,
        peak_date_int=peak_date_int,
        min_points=10,
        min_drop_ratio=0.03,
    )

    # ===================================================
    # NEW SCORE: distance-based regime score
    # ===================================================
    last_index = float(len(price_series) - 1)
    tc_index = float(bubble_res["tc_days"])

    down_tc_date = None
    if neg_res.get("ok"):
        down_tc_date = pd.Timestamp(neg_res["tc_date"])

    signal_label, score = compute_signal_and_score(
        last_index=last_index,
        tc_index=tc_index,
        down_tc_date=down_tc_date,
        end_date=pd.Timestamp(end_date),
    )

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(price_series.index, price_series.values, color="gray", label=ticker)
    ax.plot(price_series.index, bubble_res["price_fit"], color="orange", label="Up model")

    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--",
               label=f"Turn (up) {pd.Timestamp(bubble_res['tc_date']).date()}")
    ax.axvline(peak_date, color="white", linestyle=":",
               label=f"Peak {pd.Timestamp(peak_date).date()}")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="cyan", label="Down")
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="green", label="Down model")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--",
                   label=f"Turn (down) {pd.Timestamp(neg_res['tc_date']).date()}")

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")

    st.pyplot(fig)

    # ===================================================
    # UI label (signal derived from regime)
    # ===================================================
    if signal_label == "HIGH":
        risk_label = "High"
        risk_color = "#ff4d4f"
    elif signal_label == "CAUTION":
        risk_label = "Caution"
        risk_color = "#ffc53d"
    else:
        risk_label = "Safe"
        risk_color = "#52c41a"

    col_score, col_gain = st.columns(2)

    with col_score:
        score_card_html = f"""
        <div style="
            background-color: #141518;
            border: 1px solid #2a2c30;
            border-radius: 12px;
            padding: 18px 20px 16px 20px;
            margin-top: 8px;
        ">
            <div style="font-size: 0.85rem; color: #a0a2a8; margin-bottom: 6px;">
                Score
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px;">
                <div style="font-size: 40px; font-weight: 700; color: #f5f5f5;">
                    {score}
                </div>
                <div style="
                    padding: 2px 10px;
                    border-radius: 999px;
                    background-color: {risk_color}33;
                    color: {risk_color};
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    {risk_label}
                </div>
            </div>
        </div>
        """
        st.markdown(score_card_html, unsafe_allow_html=True)

    with col_gain:
        gain_card_html = f"""
        <div style="
            background-color: #141518;
            border: 1px solid #2a2c30;
            border-radius: 12px;
            padding: 18px 20px 16px 20px;
            margin-top: 8px;
        ">
            <div style="font-size: 0.85rem; color: #a0a2a8; margin-bottom: 6px;">
                Gain
            </div>
            <div style="font-size: 36px; font-weight: 700; color: #f5f5f5; line-height: 1.1;">
                {gain:.2f}x
            </div>
            <div style="font-size: 0.9rem; color: #a0a2a8; margin-top: 4px;">
                Start → Peak
            </div>
            <div style="
                margin-top: 6px;
                display: inline-block;
                padding: 2px 10px;
                border-radius: 999px;
                background-color: #102915;
                color: #52c41a;
                font-size: 0.85rem;
                font-weight: 500;
            ">
                {gain_pct:+.1f}%
            </div>
        </div>
        """
        st.markdown(gain_card_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
