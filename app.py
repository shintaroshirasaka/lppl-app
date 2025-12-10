# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st


# -------------------------------------------------------
# LPPL-like Model
# -------------------------------------------------------

def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = np.maximum(tc - t, 1e-6)
    return A + B*(dt**m) + C*(dt**m)*np.cos(omega*np.log(dt) + phi)


def fit_lppl_bubble(series):
    price = series.values.astype(float)
    t = np.arange(len(price))
    logp = np.log(price)

    N = len(t)
    p0 = [np.mean(logp), -1.0, 0.1, 0.5, N+20, 8.0, 0]
    lower = [-10, -10, -10, 0.01, N+1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N+250, 25.0, np.pi]

    params, _ = curve_fit(lppl, t, logp, p0=p0, bounds=(lower,upper), maxfev=20000)
    log_fit = lppl(t,*params)
    price_fit = np.exp(log_fit)

    ss_res = np.sum((logp-log_fit)**2)
    ss_tot = np.sum((logp-np.mean(logp))**2)
    r2 = 1 - ss_res/ss_tot

    tc_days = params[4]
    tc_date = series.index[0] + timedelta(days=tc_days)

    return {
        "params": params,
        "price_fit": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days
    }


def fit_lppl_negative_bubble(series, peak_date, min_points=10, min_drop_ratio=0.03):

    down = series[series.index >= peak_date]
    if len(down) < min_points:
        return {"ok": False}

    peak_price = float(series.loc[peak_date])
    last_price = float(down.iloc[-1])
    if (peak_price - last_price)/peak_price < min_drop_ratio:
        return {"ok": False}

    price = down.values
    t = np.arange(len(price))
    neg = -np.log(price)

    N = len(t)
    p0 = [np.mean(neg), -1.0, 0.1, 0.5, N+15, 8.0, 0]
    lower = [-10,-10,-10,0.01,N+1,2.0,-np.pi]
    upper = [10,10,10,0.99,N+200,25.0,np.pi]

    try:
        params,_ = curve_fit(lppl,t,neg,p0=p0,bounds=(lower,upper),maxfev=20000)
    except:
        return {"ok": False}

    neg_fit = lppl(t,*params)
    price_fit = np.exp(-neg_fit)

    ss_res = np.sum((neg-neg_fit)**2)
    ss_tot = np.sum((neg-np.mean(neg))**2)
    r2 = 1 - ss_res/ss_tot

    tc_days = params[4]
    tc_date = down.index[0] + timedelta(days=tc_days)

    return {
        "ok":True,
        "down_series":down,
        "price_fit_down":price_fit,
        "r2":r2,
        "tc_date":tc_date,
        "tc_days":tc_days
    }


# -------------------------------------------------------
# Bubble Score
# -------------------------------------------------------

def bubble_score(r2, m, tc_index, last_index):

    r_score = max(0,min(1,(r2-0.5)/0.5))
    m_score = max(0,1-2*abs(m-0.5))

    gap = tc_index-last_index
    if gap <= 0:
        tc_score = 1
    elif gap <= 30:
        tc_score = 1
    elif gap >= 120:
        tc_score = 0
    else:
        tc_score = 1-(gap-30)/90

    score = 0.4*r_score + 0.3*m_score + 0.3*tc_score
    return int(round(100*max(0,min(1,score))))


# -------------------------------------------------------
# Fetch Price
# -------------------------------------------------------

def fetch_price_series(ticker,start_date,end_date):
    df = yf.download(ticker,
                     start=start_date.strftime("%Y-%m-%d"),
                     end=(end_date+timedelta(days=1)).strftime("%Y-%m-%d"),
                     auto_adjust=False)
    if df.empty:
        raise ValueError("No data")
    return df["Adj Close"].dropna()


# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------

def main():

    st.set_page_config(page_title="Out-stander", layout="wide")

    # =============================
    # DARK THEME CSS（白枠完全除去）
    # =============================
    st.markdown("""
    <style>

    /* Main background */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0c0e !important;
    }

    /* Header transparent */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    /* Label */
    label {
        color: #d0d0d0 !important;
    }

    /* Input base styling */
    input, textarea {
        background-color: #1c1d1f !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #444 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Remove focus/hover outline */
    input:focus, input:hover, input:active {
        outline: none !important;
        box-shadow: none !important;
        border-color: #666 !important;
    }

    /* Streamlit wrapper border 強制黒化 */
    div[data-baseweb="input"] {
        border: 1px solid #444 !important;
        background-color: #1c1d1f !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"]:hover {
        border: 1px solid #666 !important;
    }
    div[data-baseweb="input"]:focus-within {
        border: 1px solid #777 !important;
        box-shadow: none !important;
    }

    /* Run button */
    .stButton > button {
        background-color: #222428 !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #555 !important;
    }
    .stButton > button:hover {
        background-color: #333 !important;
        border-color: #777 !important;
    }
    .stButton > button:disabled {
        background-color: #1c1d1f !important;
        color: #777 !important;
        border-color: #444 !important;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f2f2f2 !important;
    }

    /* Plot */
    .stPyplotCanvas {
        background-color: #0b0c0e !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("Out-stander")

    # -----------------------------
    # Input Form
    # -----------------------------
    with st.form("form"):

        ticker = st.text_input("Ticker", "TSM")

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

    # -----------------------------
    # Data
    # -----------------------------
    series = fetch_price_series(ticker, start_date, end_date)

    if len(series) < 30:
        st.error("Insufficient data.")
        st.stop()

    # -----------------------------
    # Models
    # -----------------------------
    up = fit_lppl_bubble(series)

    peak_date = series.idxmax()
    peak_price = float(series.max())
    start_price_v = float(series.iloc[0])
    gain = peak_price/start_price_v
    gain_pct = (gain-1)*100

    params = up["params"]
    r2 = up["r2"]
    m = params[3]
    tc_index = up["tc_days"]
    last_index = len(series)-1
    score = bubble_score(r2,m,tc_index,last_index)

    neg = fit_lppl_negative_bubble(series, peak_date)

    # -----------------------------
    # Plot
    # -----------------------------
    fig,ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(series.index, series.values, color="gray", label=ticker)
    ax.plot(series.index, up["price_fit"], color="orange", label="Up model")

    ax.axvline(up["tc_date"], color="red", linestyle="--",
               label=f"Turn (up) {up['tc_date'].date()}")
    ax.axvline(peak_date, color="white", linestyle=":",
               label=f"Peak {peak_date.date()}")

    if neg.get("ok"):
        down = neg["down_series"]
        ax.plot(down.index, down.values, color="cyan", label="Down")
        ax.plot(down.index, neg["price_fit_down"], "--",
                color="green", label="Down model")
        ax.axvline(neg["tc_date"], color="green", linestyle="--",
                   label=f"Turn (down) {neg['tc_date'].date()}")

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")

    st.pyplot(fig)

    # -----------------------------
    # UI Output
    # -----------------------------
    st.subheader("Score")
    st.markdown(f"<h1>{score}</h1>", unsafe_allow_html=True)

    if score >= 80:
        st.write("🔴 High Risk")
    elif score >= 60:
        st.write("🟡 Caution")
    else:
        st.write("🟢 Safe")

    st.subheader("Gain")
    st.metric("Start → Peak", f"{gain:.2f}x", f"{gain_pct:+.1f}%")



if __name__ == "__main__":
    main()
