# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta

import streamlit as st

# -------------------------------------------------------
# 数理モデル本体（内部では LPPL 形だが外には出さない）
# -------------------------------------------------------


def lppl(t, A, B, C, m, tc, omega, phi):
    """log-price 用の数理モデル"""
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)  # log(0) 回避
    return A + B * (dt**m) + C * (dt**m) * np.cos(omega * np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    """上昇局面へのモデルフィット"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)

    N = len(t)

    # 初期値
    A_init = np.mean(log_price)
    B_init = -1
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 20
    omega_init = 8.0
    phi_init = 0.0

    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]
    lower_bounds = [-10, -10, -10, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [10, 10, 10, 0.99, N + 250, 25.0, np.pi]

    params, cov = curve_fit(
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
        "cov": cov,
        "price_fit": price_fit,
        "r2": r2,
        "tc_date": tc_date,
        "tc_days": tc_days,
    }


def fit_lppl_negative_bubble(price_series, peak_date, min_points=10, min_drop_ratio=0.03):
    """下落局面の負バブル解析（成功しない場合は ok=False）"""

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
    A_init = np.mean(neg_log_down)
    B_init = -1
    C_init = 0.1
    m_init = 0.5
    tc_init = N_down + 15
    omega_init = 8.0
    phi_init = 0.0

    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]
    lower_bounds = [-10, -10, -10, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [10, 10, 10, 0.99, N_down + 200, 25.0, np.pi]

    try:
        params_down, cov_down = curve_fit(
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
    price_fit_down = np.exp(-neg_log_fit)

    ss_res = np.sum((neg_log_down - neg_log_fit) ** 2)
    ss_tot = np.sum((neg_log_down - neg_log_down.mean()) ** 2)
    r2_down = 1 - ss_res / ss_tot

    first_down_date = down_series.index[0]
    tc_days = float(params_down[4])
    tc_date = first_down_date + timedelta(days=tc_days)

    return {
        "ok": True,
        "down_series": down_series,
        "price_fit_down": price_fit_down,
        "r2": r2_down,
        "tc_date": tc_date,
        "tc_days": tc_days,
        "params": params_down,
    }


# -------------------------------------------------------
# Bubble Score
# -------------------------------------------------------


def bubble_score(r2_up, m, tc_index, last_index):
    """Bubble Score（0〜100）"""
    r_score = max(0, min(1, (r2_up - 0.5) / 0.5))
    m_score = max(0, 1 - 2 * abs(m - 0.5))

    gap = tc_index - last_index
    if gap <= 0:
        tc_score = 1.0
    elif gap <= 30:
        tc_score = 1.0
    elif gap >= 120:
        tc_score = 0.0
    else:
        tc_score = 1.0 - (gap - 30) / (120 - 30)

    score_0_1 = 0.4 * r_score + 0.3 * m_score + 0.3 * tc_score
    score = int(round(100 * max(0, min(1, score_0_1))))

    return score, {
        "r_component": r_score,
        "m_component": m_score,
        "tc_component": tc_score,
    }


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
    st.set_page_config(page_title="アウトスタンダー（株価解析アプリ）", layout="wide")

    st.title("アウトスタンダー（株価解析アプリ）")
    st.caption("※投資助言ではなく、数理モデルによるリサーチツールです。")

    # --- 入力フォーム ---
    with st.form("input_form"):
        st.write("### 入力パラメータ")

        ticker = st.text_input("ティッカー（例: AMD, PLTR, 9988.HK）", "AMD")

        today = date.today()
        default_end = today
        default_start = today - timedelta(days=220)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", default_start)
        with col2:
            end_date = st.date_input("終了日", default_end)

        submitted = st.form_submit_button("解析を実行")

    if not submitted:
        st.stop()

    # --- データ取得 ---
    price_series = fetch_price_series(ticker, start_date, end_date)

    if len(price_series) < 30:
        st.error("データが不足しています。期間を伸ばしてください。")
        st.stop()

    # --- 上昇バブル解析 ---
    bubble_res = fit_lppl_bubble(price_series)

    # 開始日〜最高値の倍率
    start_price = float(price_series.iloc[0])
    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())

    rise_ratio = peak_price / start_price
    rise_percent = (rise_ratio - 1) * 100

    # --- Bubble Score ---
    params_up = bubble_res["params"]
    r2_up = bubble_res["r2"]
    m_up = float(params_up[3])
    tc_index = bubble_res["tc_days"]
    last_index = len(price_series) - 1

    score, score_detail = bubble_score(r2_up, m_up, tc_index, last_index)

    # --- Score 表示（1回だけ）---
    st.write("### バブル度スコア（現在の期間設定）")
    st.caption("Bubble Score (0–100)")

    if score >= 80:
        icon = "🔴"
        title = "危険"
    elif score >= 60:
        icon = "🟡"
        title = "注意"
    else:
        icon = "🟢"
        title = "安全"

    st.markdown(
        f"""
        <div style="margin-top:10px;">
            <div style="font-size:42px; font-weight:bold;">{score}</div>
            <div style="font-size:36px; font-weight:bold;">{icon} {title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 上昇倍率（参考情報） ---
    st.write("### 上昇倍率（参考）")
    st.metric("開始日 → 最高値", f"{rise_ratio:.2f}倍", f"{rise_percent:+.1f}%")

    # --- スコア内訳 ---
    with st.expander("バブル度スコアの内訳"):
        st.write(
            f"- R² 成分: {score_detail['r_component']:.2f}\n"
            f"- 形状パラメータ m 成分: {score_detail['m_component']:.2f}\n"
            f"- t_c の近さ成分: {score_detail['tc_component']:.2f}"
        )

    # --- 下落局面（negative bubble） ---
    try:
        neg_res = fit_lppl_negative_bubble(price_series, peak_date)
    except Exception:
        neg_res = {"ok": False}

    # --- サマリー表 ---
    st.write("### 候補日サマリー")

    rows = [
        ["内部崩壊候補日（上昇局面）", bubble_res["tc_date"].date(), round(r2_up, 4)],
        ["価格の最高値日", peak_date.date(), None],
        ["バブル度スコア（現在）", f"{score}/100", None],
        ["開始日→最高値の上昇倍率", f"{rise_ratio:.2f}倍", None],
    ]

    if neg_res.get("ok"):
        rows.append(
            ["内部底候補日（下落局面）", neg_res["tc_date"].date(), round(neg_res["r2"], 4)]
        )
    else:
        rows.append(
            ["内部底候補日（下落局面）", "該当なし", None]
        )

    st.table(pd.DataFrame(rows, columns=["イベント", "日付 / スコア / 倍率", "R²"]))

    # --- グラフ ---
    st.write("### 統合グラフ")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price_series.index, price_series.values, color="lightgray", label=f"{ticker} price (actual)")
    ax.plot(price_series.index, bubble_res["price_fit"], color="orange", label="Model (uptrend)")
    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--", label=f"Internal collapse ({bubble_res['tc_date'].date()})")
    ax.axvline(peak_date, color="black", linestyle=":", label=f"Price peak ({peak_date.date()})")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="blue", label=f"{ticker} downtrend")
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="green", label="Model (downtrend)")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--", label=f"Bottom candidate ({neg_res['tc_date'].date()})")

    ax.set_title(f"{ticker} — Bubble → Collapse → Negative Bubble")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)

    st.pyplot(fig)


if __name__ == "__main__":
    main()
