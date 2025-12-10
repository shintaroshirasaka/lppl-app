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
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt**m) + C * (dt**m) * np.cos(omega * np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    """上昇局面へのモデルフィット"""
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

    # R² は内部でのみ使用（表示には出さない）
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
    """下落局面の負バブル解析（成功しない場合は ok=False）"""

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

    # R²（下落用）も内部でのみ利用可能
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
    """バブル度スコア（0〜100）"""
    # R² 成分
    r_score = max(0.0, min(1.0, (r2_up - 0.5) / 0.5))
    # m 成分
    m_score = max(0.0, 1.0 - 2 * abs(m - 0.5))

    # t_c の近さ成分
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
    score = int(round(100 * max(0.0, min(1.0, score_raw))))
    return score


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

    # ---------------- 入力フォーム ----------------
    with st.form("input_form"):
        st.write("### 入力パラメータ")

        ticker = st.text_input("ティッカー（例: AMD, PLTR, TSM, 9988.HK）", "AMD")

        today = date.today()
        default_start = today - timedelta(days=220)
        default_end = today

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", default_start)
        with col2:
            end_date = st.date_input("終了日", default_end)

        submitted = st.form_submit_button("解析を実行")

    if not submitted:
        st.stop()

    # ---------------- データ取得 ----------------
    price_series = fetch_price_series(ticker, start_date, end_date)

    if len(price_series) < 30:
        st.error("データが不足しています。期間を伸ばしてください。")
        st.stop()

    # ---------------- 上昇バブル解析 ----------------
    bubble_res = fit_lppl_bubble(price_series)

    # 最高値＆上昇倍率
    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price = float(price_series.iloc[0])

    rise_ratio = peak_price / start_price
    rise_percent = (rise_ratio - 1.0) * 100.0

    # ---------------- Bubble Score ----------------
    params_up = bubble_res["params"]
    r2_up = bubble_res["r2"]          # 内部でのみ利用
    m_up = params_up[3]
    tc_index = float(bubble_res["tc_days"])
    last_index = float(len(price_series) - 1)

    score = bubble_score(r2_up, m_up, tc_index, last_index)

    # ---------------- 負バブル解析（下落） ----------------
    try:
        neg_res = fit_lppl_negative_bubble(price_series, peak_date)
    except Exception:
        neg_res = {"ok": False}

    # ------------------------------------------------
    # ① 統合グラフ
    # ------------------------------------------------
    st.write("### 統合グラフ")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        price_series.index,
        price_series.values,
        color="lightgray",
        label=f"{ticker} price",
    )
    ax.plot(
        price_series.index,
        bubble_res["price_fit"],
        color="orange",
        label="Model (uptrend)",
    )
    ax.axvline(
        bubble_res["tc_date"],
        color="red",
        linestyle="--",
        label=f"Internal collapse {bubble_res['tc_date'].date()}",
    )
    ax.axvline(
        peak_date,
        color="black",
        linestyle=":",
        label=f"Price peak {peak_date.date()}",
    )

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(
            down.index,
            down.values,
            color="blue",
            label=f"{ticker} downtrend",
        )
        ax.plot(
            down.index,
            neg_res["price_fit_down"],
            "--",
            color="green",
            label="Model (downtrend)",
        )
        ax.axvline(
            neg_res["tc_date"],
            color="green",
            linestyle="--",
            label=f"Bottom {neg_res['tc_date'].date()}",
        )

    ax.set_title(f"{ticker} — Bubble → Collapse → Negative Bubble")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)

    st.pyplot(fig)

    # ------------------------------------------------
    # ② バブル度スコア
    # ------------------------------------------------
    st.write("### バブル度スコア")
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
        <div style="margin-top:0px;">
            <div style="font-size:42px; font-weight:bold; line-height:1;">
                {score}
            </div>
            <div style="font-size:36px; font-weight:bold; line-height:1.1;">
                {icon} {title}
            </div>
        </div>
        <div style="margin-bottom:30px;"></div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------
    # ③ 上昇倍率（参考）
    # ------------------------------------------------
    st.write("### 上昇倍率（参考）")
    st.metric("開始日 → 最高値", f"{rise_ratio:.2f}倍", f"{rise_percent:+.1f}%")

    # ------------------------------------------------
    # ④ 候補日サマリー（R² は非表示）
    # ------------------------------------------------
    st.write("### 候補日サマリー")

    rows = [
        ["内部崩壊候補日（上昇）", bubble_res["tc_date"].date()],
        ["最高値の日付", peak_date.date()],
        ["バブル度スコア", f"{score} / 100"],
        ["開始日→最高値の上昇倍率", f"{rise_ratio:.2f}倍"],
    ]

    if neg_res.get("ok"):
        rows.append(
            ["内部底候補日（下落）", neg_res["tc_date"].date()]
        )
    else:
        rows.append(["内部底候補日（下落）", "該当なし"])

    summary_df = pd.DataFrame(rows, columns=["イベント", "数値 / 日付"])
    st.table(summary_df)


if __name__ == "__main__":
    main()
