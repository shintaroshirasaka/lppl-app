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


def fit_lppl_bubble(price_series):
    """
    上昇局面に対して数理モデルをフィットする。
    price_series: pandas Series (index=DatetimeIndex, values=price)
    """
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


def fit_lppl_negative_bubble(price_series, peak_date, min_points=10):
    """
    価格ピーク以降の下落局面に対して、負のバブル（ネガティブバブル）をフィットする。
    - 条件を満たさない場合やフィットに失敗した場合は ok=False で返す。
    """
    down_series = price_series[price_series.index >= peak_date].copy()

    if len(down_series) < min_points:
        return {"ok": False, "reason": "points_short"}

    price_down = down_series.values.astype(float)
    t_down = np.arange(len(price_down), dtype=float)

    log_down = np.log(price_down)
    neg_log_down = -log_down  # 下落をバブルとして扱う

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
        params_down, cov_down = curve_fit(
            lppl,
            t_down,
            neg_log_down,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000,
        )
    except Exception as e:
        return {"ok": False, "reason": "curve_fit_failed", "error": str(e)}

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
# データ取得（yfinance）
# -------------------------------------------------------


def fetch_price_series(ticker: str, start_date: date, end_date: date) -> pd.Series:
    """
    yfinance から株価データを取得し、終値 Series を返す。
    MultiIndex / 単一 Index 両方に対応。
    """
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError("価格データが取得できませんでした。ティッカーや期間を確認してください。")

    # MultiIndex の場合
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)].dropna()
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].dropna()
        else:
            raise ValueError("終値カラム（Adj Close / Close）が見つかりません。")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            s = df["Close"].dropna()
        else:
            raise ValueError("終値カラム（Adj Close / Close）が見つかりません。")

    return s


# -------------------------------------------------------
# Streamlit アプリ本体
# -------------------------------------------------------


def main():
    st.set_page_config(
        page_title="アウトスタンダー（株価解析アプリ）",
        layout="wide",
    )

    # タイトル & 説明
    st.title("アウトスタンダー（株価解析アプリ）")
    st.write(
        "【バブル → 内部崩壊 → 最高値 → ネガティブバブル → 内部底候補】"
        "のような株価の構造変化を、独自の数理モデルで推定します。"
    )
    st.caption("※投資助言ではなく、数理モデルによるリサーチ用ツールです。")

    # -----------------------------
    # 入力フォーム
    # -----------------------------
    with st.form("input_form"):
        st.write("### 入力パラメータ")

        # ティッカーは1列で表示
        ticker = st.text_input(
            "ティッカー / 証券コード（例: AMD, PLTR, AVGO, 7203.T, 9988.HK）",
            value="AMD",
        )

        today = date.today()
        default_end = today
        default_start = today - timedelta(days=250)

        # 開始日・終了日は2カラムで横並び
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("開始日", value=default_start)
        with col2:
            end_date = st.date_input("終了日", value=default_end)

        submitted = st.form_submit_button("解析を実行")

    if not submitted:
        st.stop()

    # -----------------------------
    # データ取得
    # -----------------------------
    try:
        price_series = fetch_price_series(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"価格データ取得でエラーが発生しました: {e}")
        st.stop()

    if len(price_series) < 30:
        st.error("データ点数が少なすぎます。もう少し長い期間を指定してください。")
        st.stop()

    # -----------------------------
    # 上昇局面のモデルフィット
    # -----------------------------
    try:
        bubble_res = fit_lppl_bubble(price_series)
    except Exception as e:
        st.error(f"上昇局面の解析でエラーが発生しました: {e}")
        st.stop()

    # 最高値日
    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())

    # 下落局面のモデルフィット（失敗しても致命的エラーにはしない）
    try:
        neg_res = fit_lppl_negative_bubble(price_series, peak_date)
    except Exception as e:
        neg_res = {"ok": False, "reason": "exception", "error": str(e)}

    # --------------------------------------------------
    # サマリー表
    # --------------------------------------------------
    st.write("### 候補日サマリー")

    rows = [
        [
            "内部崩壊候補日（上昇局面）",
            bubble_res["tc_date"].date(),
            round(bubble_res["r2"], 4),
        ],
        ["価格の最高値日", peak_date.date(), None],
    ]

    if neg_res is not None and neg_res.get("ok"):
        rows.append(
            [
                "内部底候補日（下落局面）",
                neg_res["tc_date"].date(),
                round(neg_res["r2"], 4),
            ]
        )
    else:
        rows.append(["内部底候補日（下落局面）", "該当なし", None])

    summary_df = pd.DataFrame(rows, columns=["イベント", "日付", "R² (参考)"])
    st.table(summary_df)

    # --------------------------------------------------
    # 統合グラフ
    # --------------------------------------------------
    st.write("### 統合グラフ")

    fig, ax = plt.subplots(figsize=(10, 5))

    # 全期間の実際の価格（グレー）
    ax.plot(
        price_series.index,
        price_series.values,
        color="lightgray",
        label=f"{ticker} price (actual)",
    )

    # 上昇局面のモデルフィット（オレンジ）
    ax.plot(
        price_series.index,
        bubble_res["price_fit"],
        color="orange",
        label="Model (uptrend)",
    )

    # 内部崩壊候補日（赤）
    ax.axvline(
        bubble_res["tc_date"],
        color="red",
        linestyle="--",
        label=f"Internal collapse ({bubble_res['tc_date'].date()})",
    )

    # 最高値（黒）
    ax.axvline(
        peak_date,
        color="black",
        linestyle=":",
        label=f"Price peak ({peak_date.date()})",
    )

    # 下落局面（あれば）
    if neg_res is not None and neg_res.get("ok"):
        down_series = neg_res["down_series"]
        ax.plot(
            down_series.index,
            down_series.values,
            color="blue",
            label=f"{ticker} downtrend",
        )
        ax.plot(
            down_series.index,
            neg_res["price_fit_down"],
            "--",
            color="green",
            label="Model (downtrend)",
        )
        ax.axvline(
            neg_res["tc_date"],
            color="green",
            linestyle="--",
            label=f"Bottom candidate ({neg_res['tc_date'].date()})",
        )

    ax.set_title(f"{ticker} — Bubble → Collapse → Negative Bubble")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc="best")

    st.pyplot(fig)

    # --------------------------------------------------
    # モデルパラメータ詳細
    # --------------------------------------------------
    with st.expander("モデルパラメータ詳細（上昇局面）"):
        param_names = ["A", "B", "C", "m", "tc", "omega", "phi"]
        up_params_df = pd.DataFrame(
            {
                "parameter": param_names,
                "estimate": bubble_res["params"],
            }
        )
        st.dataframe(up_params_df)

    if neg_res is not None and neg_res.get("ok"):
        with st.expander("モデルパラメータ詳細（下落局面）"):
            down_params_df = pd.DataFrame(
                {
                    "parameter": param_names,
                    "estimate": neg_res["params"],
                }
            )
            st.dataframe(down_params_df)


if __name__ == "__main__":
    main()
