# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st

# -------------------------------------------------------
# LPPL-like internal model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B*(dt**m) + C*(dt**m)*np.cos(omega*np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    logp = np.log(price)

    N = len(t)
    p0 = [np.mean(logp), -1.0, 0.1, 0.5, N+20, 8.0, 0.0]
    lower = [-10, -10, -10, 0.01, N+1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N+250, 25.0, np.pi]

    params, _ = curve_fit(lppl, t, logp, p0=p0, bounds=(lower, upper), maxfev=20000)
    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)

    # Internal R²
    ss_res = np.sum((logp - log_fit)**2)
    ss_tot = np.sum((logp - logp.mean())**2)
    r2 = 1 - ss_res/ss_tot

    tc_days = float(params[4])
    tc_date = price_series.index[0] + timedelta(days=tc_days)

    return {
        "params": params,
        "price_fit": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days,
    }


def fit_lppl_negative_bubble(price_series, peak_date, min_points=10, min_drop_ratio=0.03):
    down = price_series[price_series.index >= peak_date].copy()
    if len(down) < min_points:
        return {"ok": False}

    peak_price = float(price_series.loc[peak_date])
    last_price = float(down.iloc[-1])
    if (peak_price - last_price)/peak_price < min_drop_ratio:
        return {"ok": False}

    price = down.values.astype(float)
    t = np.arange(len(price), dtype=float)
    neg = -np.log(price)

    N = len(t)
    p0 = [np.mean(neg), -1.0, 0.1, 0.5, N+15, 8.0, 0.0]
    lower = [-10, -10, -10, 0.01, N+1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N+200, 25.0, np.pi]

    try:
        params, _ = curve_fit(lppl, t, neg, p0=p0, bounds=(lower, upper), maxfev=20000)
    except Exception:
        return {"ok": False}

    neg_fit = lppl(t, *params)
    price_fit = np.exp(-neg_fit)

    ss_res = np.sum((neg - neg_fit)**2)
    ss_tot = np.sum((neg - neg.mean())**2)
    r2 = 1 - ss_res/ss_tot

    tc_days = float(params[4])
    tc_date = down.index[0] + timedelta(days=tc_days)

    return {
        "ok": True,
        "down_series": down,
        "price_fit_down": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days,
        "params": params,
    }


# -------------------------------------------------------
# Bubble Score
# -------------------------------------------------------
def bubble_score(r2_up, m, tc_index, last_index):
    r_score = max(0.0, min(1.0, (r2_up - 0.5)/0.5))
    m_score = max(0.0, 1.0 - 2*abs(m - 0.5))

    gap = tc_index - last_index
    if gap <= 0:
        tc_score = 1.0
    elif gap <= 30:
        tc_score = 1.0
    elif gap >= 120:
        tc_score = 0.0
    else:
        tc_score = 1.0 - (gap - 30)/(120 - 30)

    s = 0.4*r_score + 0.3*m_score + 0.3*tc_score
    return int(round(100*max(0.0, min(1.0, s))))


# -------------------------------------------------------
# Data
# -------------------------------------------------------
def fetch_price_series(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("No price data.")
    if isinstance(df.columns, pd.MultiIndex):
        s = df[("Adj Close", ticker)] if ("Adj Close", ticker) in df else df[("Close", ticker)]
    else:
        s = df["Adj Close"] if "Adj Close" in df else df["Close"]
    return s.dropna()


# -------------------------------------------------------
# Streamlit App (Minimal UI)
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander", layout="wide")

    st.title("Out-stander")  # simple, minimal

    # -------- Input --------
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", "AMD")
        with col2:
            today = date.today()
            start_date = st.date_input("Start", today - timedelta(days=220))
        end_date = st.date_input("End", today)

        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    price_series = fetch_price_series(ticker, start_date, end_date)
    if len(price_series) < 30:
        st.error("Insufficient data.")
        st.stop()

    bubble_res = fit_lppl_bubble(price_series)

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price = float(price_series.iloc[0])
    rise_ratio = peak_price/start_price
    rise_percent = (rise_ratio - 1)*100

    params_up = bubble_res["params"]
    r2_up = bubble_res["r2"]
    m_up = params_up[3]
    tc_index = float(bubble_res["tc_days"])
    last_index = len(price_series) - 1
    score = bubble_score(r2_up, m_up, tc_index, last_index)

    try:
        neg_res = fit_lppl_negative_bubble(price_series, peak_date)
    except Exception:
        neg_res = {"ok": False}

    # -------- Graph --------
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(price_series.index, price_series.values, color="lightgray", label=f"{ticker}")
    ax.plot(price_series.index, bubble_res["price_fit"], color="orange", label="Uptrend model")
    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--", label="Turning (up)")
    ax.axvline(peak_date, color="black", linestyle=":", label="Peak")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="blue", label="Downtrend")
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="green", label="Down model")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--", label="Turning (down)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -------- Bubble Score --------
    st.subheader("Score")
    st.markdown(f"<h1 style='font-size:48px'>{score}</h1>", unsafe_allow_html=True)

    if score >= 80:
        st.markdown("<h2>🔴 High Risk</h2>", unsafe_allow_html=True)
    elif score >= 60:
        st.markdown("<h2>🟡 Caution</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>🟢 Safe</h2>", unsafe_allow_html=True)

    # -------- Gain --------
    st.subheader("Gain")
    st.metric("Start → Peak", f"{rise_ratio:.2f}x", f"{rise_percent:+.1f}%")

    # -------- No Summary Table (fully removed) --------


if __name__ == "__main__":
    main()
