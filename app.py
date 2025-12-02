# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta

import streamlit as st

# -------------------------------------------------------
# モデル
# -------------------------------------------------------

def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B*(dt**m) + C*(dt**m)*np.cos(omega*np.log(dt)+phi)


def fit_lppl_bubble(price_series):
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)

    N = len(t)
    p0 = [np.mean(log_price), -1.0, 0.1, 0.5, N+20, 8.0, 0.0]
    lower = [-10, -10, -10, 0.01, N+1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N+250, 25.0, np.pi]

    params, _ = curve_fit(lppl, t, log_price, p0=p0,
                          bounds=(lower, upper), maxfev=20000)

    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)

    ss_res = np.sum((log_price - log_fit)**2)
    ss_tot = np.sum((log_price - np.mean(log_price))**2)
    r2 = 1 - ss_res/ss_tot

    first_day = price_series.index[0]
    tc_days = float(params[4])
    tc_date = first_day + timedelta(days=tc_days)

    return {
        "params": params,
        "price_fit": price_fit,
        "r2": r2,
        "tc_days": tc_days,
        "tc_date": tc_date
    }


def fit_lppl_negative_bubble(price_series, peak_date,
                             min_points=10, min_drop_ratio=0.03):

    down = price_series[price_series.index >= peak_date]
    if len(down) < min_points:
        return {"ok": False}

    peak_price = float(price_series.loc[peak_date])
    last = float(down.iloc[-1])
    if (peak_price - last)/peak_price < min_drop_ratio:
        return {"ok": False}

    price = down.values.astype(float)
    t = np.arange(len(price))
    neg = -np.log(price)

    N = len(t)
    p0 = [np.mean(neg), -1.0, 0.1, 0.5, N+15, 8.0, 0.0]
    lower = [-10, -10, -10, 0.01, N+1, 2.0, -np.pi]
    upper = [10, 10, 10, 0.99, N+200, 25.0, np.pi]

    try:
        params, _ = curve_fit(lppl, t, neg, p0=p0,
                              bounds=(lower, upper), maxfev=20000)
    except Exception:
        return {"ok": False}

    neg_fit = lppl(t, *params)
    price_fit = np.exp(-neg_fit)

    ss_res = np.sum((neg - neg_fit)**2)
    ss_tot = np.sum((neg - np.mean(neg))**2)
    r2 = 1 - ss_res/ss_tot

    first = down.index[0]
    tc_days = float(params[4])
    tc_date = first + timedelta(days=tc_days)

    return {
        "ok": True,
        "down_series": down,
        "price_fit_down": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days
    }


# -------------------------------------------------------
# Bubble Score
# -------------------------------------------------------

def bubble_score(r2, m, tc_index, last_index):

    r_score = max(0, min(1, (r2 - 0.5) / 0.5))
    m_score = max(0, 1 - 2*abs(m - 0.5))

    gap = tc_index - last_index
    if gap <= 0:
        tc_score = 1.0
    elif gap <= 30:
        tc_score = 1.0
    elif gap >= 120:
        tc_score = 0.0
    else:
        tc_score = 1 - (gap - 30)/(120 - 30)

    score = 0.4*r_score + 0.3*m_score + 0.3*tc_score
    score = int(round(100*max(0, min(1, score))))

    return score


# -------------------------------------------------------
# データ取得
# -------------------------------------------------------

def fetch_price_series(ticker, start, end):
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False
    )
    if df.empty:
        raise ValueError("データ取得失敗")

    if isinstance(df.columns, pd.MultiIndex):
        s = df[("Adj Close", ticker)] if ("Adj Close", ticker) in df else df[("Close", ticker)]
    else:
        s = df["Adj Close"] if "Adj Close" in df else df["Close"]

    return s.dropna()


# -------------------------------------------------------
# Streamlit アプリ
# -------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="アウトスタンダー")

    st.title("アウトスタンダー（株価解析アプリ）")
    st.caption("※投資助言ではなく、数理モデルによるリサーチツールです。")

    # ------------- 入力フォーム -------------
    with st.form("form"):
        st.write("### 入力パラメータ")

        ticker = st.text_input("ティッカー", "AMD")

        today = date.today()
        default_start = today - timedelta(days=220)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", default_start)
        with col2:
            end_date = st.date_input("終了日", today)

        submitted = st.form_submit_button("解析を実行")

    if not submitted:
        st.stop()

    # ------------- データ取得 -------------
    price = fetch_price_series(ticker, start_date, end_date)

    if len(price) < 30:
        st.error("データが不足しています。期間を伸ばしてください。")
        st.stop()

    # 上昇バブル解析
    bubble = fit_lppl_bubble(price)

    # 最高値
    peak_date = price.idxmax()
    peak_price = float(price.max())
    start_price = float(price.iloc[0])

    rise_ratio = peak_price / start_price
    rise_percent = (rise_ratio - 1)*100

    # Bubble Score
    params = bubble["params"]
    r2 = bubble["r2"]
    m = params[3]
    tc_index = bubble["tc_days"]
    last_index = len(price)-1

    score = bubble_score(r2, m, tc_index, last_index)

    # 下落バブル
    neg = fit_lppl_negative_bubble(price, peak_date)

    # ----------------------------------------------------------
    # ① 統合グラフ（最初に表示）
    # ----------------------------------------------------------
    st.write("### 統合グラフ")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price.index, price, color="lightgray", label=f"{ticker} price")
    ax.plot(price.index, bubble["price_fit"], color="orange", label="Model (uptrend)")
    ax.axvline(bubble["tc_date"], color="red", linestyle="--", label=f"Internal collapse {bubble['tc_date'].date()}")
    ax.axvline(peak_date, color="black", linestyle=":", label=f"Price peak {peak_date.date()}")

    if neg["ok"]:
        down = neg["down_series"]
        ax.plot(down.index, down.values, color="blue", label="downtrend")
        ax.plot(down.index, neg["price_fit_down"], "--", color="green", label="Model (down)")
        ax.axvline(neg["tc_date"], color="green", linestyle="--", label=f"Bottom {neg['tc_date'].date()}")

    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # ----------------------------------------------------------
    # ② バブル度スコア
    # ----------------------------------------------------------
    st.write("### バブル度スコア")
    st.caption("Bubble Score (0–100)")

    if score >= 80:
        icon = "🔴"; title = "危険"
    elif score >= 60:
        icon = "🟡"; title = "注意"
    else:
        icon = "🟢"; title = "安全"

    st.markdown(
        f"""
        <div style="margin-top:0px;">
            <div style="font-size:42px; font-weight:bold; line-height:1;">
                {score}
            </div>
            <div style="font-size:36px; font-weight:bold; line-height:1.1;">
                {icon} {title}
            </div>
        </div>
        <div style="margin-bottom:25px;"></div>
        """,
        unsafe_allow_html=True
    )

    # ----------------------------------------------------------
    # ③ 上昇倍率（参考）
    # ----------------------------------------------------------
    st.write("### 上昇倍率（参考）")
    st.metric("開始日 → 最高値", f"{rise_ratio:.2f}倍", f"{rise_percent:+.1f}%")

    # ----------------------------------------------------------
    # ④ 候補日サマリー
    # ----------------------------------------------------------
    st.write("### 候補日サマリー")

    rows = [
        ["内部崩壊候補日（上昇）", bubble["tc_date"].date(), round(r2, 4)],
        ["最高値の日付", peak_date.date(), None],
        ["バブル度スコア", f"{score}/100", None],
        ["開始日→最高値の上昇倍率", f"{rise_ratio:.2f}倍", None],
    ]

    if neg["ok"]:
        rows.append(["内部底候補日（下落）", neg["tc_date"].date(), round(neg["r2"], 4)])
    else:
        rows.append(["内部底候補日（下落）", "該当なし", None])

    st.table(pd.DataFrame(rows, columns=["イベント", "数値", "R²"]))


if __name__ == "__main__":
    main()
