# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st

# -------------------------------------------------------
# 内部モデル（LPPL 形）
# -------------------------------------------------------


def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt**m) + C * (dt**m) * np.cos(omega * np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    """上昇局面へのフィット"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)

    N = len(t)
    A_init = np.mean(log_price)
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 20
    omega_init = 8.0
    phi_init = 0.0

    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]
    lower = [-10, -10, -10, 0.01, N + 1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N + 250, 25.0, np.pi]

    params, _ = curve_fit(
        lppl,
        t,
        log_price,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000,
    )

    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)

    # R² は内部でのみ利用
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
    """下落局面のフィット（負バブル）"""

    down = price_series[price_series.index >= peak_date].copy()
    if len(down) < min_points:
        return {"ok": False}

    peak_price = float(price_series.loc[peak_date])
    last_price = float(down.iloc[-1])
    drop_ratio = (peak_price - last_price) / peak_price

    if drop_ratio < min_drop_ratio:
        return {"ok": False}

    price = down.values.astype(float)
    t = np.arange(len(price), dtype=float)
    logp = np.log(price)
    neg = -logp

    N = len(t)
    A_init = np.mean(neg)
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 15
    omega_init = 8.0
    phi_init = 0.0

    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]
    lower = [-10, -10, -10, 0.01, N + 1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N + 200, 25.0, np.pi]

    try:
        params, _ = curve_fit(
            lppl,
            t,
            neg,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception:
        return {"ok": False}

    neg_fit = lppl(t, *params)
    price_fit = np.exp(-neg_fit)

    ss_res = np.sum((neg - neg_fit) ** 2)
    ss_tot = np.sum((neg - neg.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    first_date = down.index[0]
    tc_days = float(params[4])
    tc_date = first_date + timedelta(days=tc_days)

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
# Bubble Score（0〜100）
# -------------------------------------------------------


def bubble_score(r2_up, m, tc_index, last_index):
    r_score = max(0.0, min(1.0, (r2_up - 0.5) / 0.5))
    m_score = max(0.0, 1.0 - 2 * abs(m - 0.5))

    gap = tc_index - last_index
    if gap <= 0:
        tc_score = 1.0
    elif gap <= 30:
        tc_score = 1.0
    elif gap >= 120:
        tc_score = 0.0
    else:
        tc_score = 1.0 - (gap - 30) / (120 - 30)

    score_raw = 0.4 * r_score + 0.3 * m_score + 0.3 * tc_score
    return int(round(100 * max(0.0, min(1.0, score_raw))))


# -------------------------------------------------------
# データ取得
# -------------------------------------------------------


def fetch_price_series(ticker, start_date, end_date):
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError("価格データが取得できませんでした。")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        else:
            s = df[("Close", ticker)]
    else:
        s = df["Adj Close"] if "Adj Close" in df else df["Close"]

    return s.dropna()


# -------------------------------------------------------
# Streamlit アプリ本体
# -------------------------------------------------------


def main():
    st.set_page_config(page_title="Out-stander", layout="wide")

    st.title("Out-stander")

    # ---------------- 入力フォーム ----------------
    with st.form("input_form"):
        # 1行目：Ticker（フル幅）
        ticker = st.text_input("Ticker", "AMD")

        # 2行目：Start / End（2カラム）
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

    # ---------------- データ取得 ----------------
    price_series = fetch_price_series(ticker, start_date, end_date)
    if len(price_series) < 30:
        st.error("Insufficient data.")
        st.stop()

    # ---------------- 上昇構造解析 ----------------
    bubble_res = fit_lppl_bubble(price_series)

    # 最高値・上昇倍率
    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price = float(price_series.iloc[0])
    rise_ratio = peak_price / start_price
    rise_percent = (rise_ratio - 1.0) * 100.0

    # Bubble Score
    params_up = bubble_res["params"]
    r2_up = bubble_res["r2"]  # 内部のみ利用
    m_up = params_up[3]
    tc_index = float(bubble_res["tc_days"])
    last_index = float(len(price_series) - 1)
    score = bubble_score(r2_up, m_up, tc_index, last_index)

    # 下落構造解析
    try:
        neg_res = fit_lppl_negative_bubble(price_series, peak_date)
    except Exception:
        neg_res = {"ok": False}

    # ---------------- グラフ ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price_series.index, price_series.values,
            color="lightgray", label=f"{ticker}")
    ax.plot(price_series.index, bubble_res["price_fit"],
            color="orange", label="Model (up)")

    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--",
               label=f"Turning (up) {bubble_res['tc_date'].date()}")
    ax.axvline(peak_date, color="black", linestyle=":",
               label=f"Peak {peak_date.date()}")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="blue", label="Down")
        ax.plot(down.index, neg_res["price_fit_down"],
                "--", color="green", label="Model (down)")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--",
                   label=f"Turning (down) {neg_res['tc_date'].date()}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    st.pyplot(fig)

    # ---------------- バブル度スコア ----------------
    st.subheader("バブル度スコア")
    st.markdown(f"<h1 style='font-size:48px'>{score}</h1>",
                unsafe_allow_html=True)

    if score >= 80:
        st.markdown("<h2>🔴 High Risk</h2>", unsafe_allow_html=True)
    elif score >= 60:
        st.markdown("<h2>🟡 Caution</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2>🟢 Safe</h2>", unsafe_allow_html=True)

    # ---------------- 上昇倍率 ----------------
    st.subheader("上昇倍率")
    st.metric("Start → Peak", f"{rise_ratio:.2f}x", f"{rise_percent:+.1f}%")

    # 構造的転換点サマリーは非表示（表そのものを出さない）


if __name__ == "__main__":
    main()

