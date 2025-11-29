# ここにさっき渡した app.py のコード全文を貼る
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

# -------------------------
# LPPL モデル
# -------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    dt = tc - t
    dt = np.maximum(dt, 1e-6)  # log のため 0 回避
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)


# -------------------------
# データ取得（MultiIndex 対応）
# -------------------------
def fetch_price_series(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if df.empty:
        raise ValueError("価格データが取得できませんでした。ティッカーや期間を確認してください。")

    # MultiIndex or 単一 Index 対応
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)].dropna()
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].dropna()
        else:
            raise ValueError("Adj Close / Close カラムが見つかりません (MultiIndex)。")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            s = df["Close"].dropna()
        else:
            raise ValueError("Adj Close / Close カラムが見つかりません。")

    return s


# -------------------------
# 上昇バブル LPPL フィット
# -------------------------
def fit_lppl_bubble(price_series):
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


# -------------------------
# ネガティブバブル（底候補）LPPL フィット
# -------------------------
def fit_lppl_negative_bubble(price_series, peak_date):
    down_series = price_series[price_series.index >= peak_date].copy()
    if len(down_series) < 10:
        raise ValueError("ピーク以降のデータが少なすぎます。")

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

    params_down, cov_down = curve_fit(
        lppl,
        t_down,
        neg_log_down,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )

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
        "down_series": down_series,
        "price_fit_down": price_fit_down,
        "r2": r2_down,
        "tc_date": tc_bottom_date,
        "tc_days": tc_days,
    }


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("LPPL バブル解析アプリ")
    st.write("**【バブル → 内部崩壊 → 最高値 → ネガティブバブル → 内部底候補】** を推定します。")
    st.write("※投資助言ではなく、数理モデルによるリサーチ用ツールです。")

    # --- 入力 ---
    ticker = st.text_input("ティッカー / 証券コード（例: AMD, PLTR, AVGO, 7203.T）", value="AMD")

    today = date.today()
    default_end = today
    default_start = today - timedelta(days=250)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日", value=default_start)
    with col2:
        end_date = st.date_input("終了日", value=default_end)

    if st.button("解析を実行"):
        try:
            # 1. データ取得
            price_series = fetch_price_series(ticker, start_date, end_date)

            # 2. 上昇バブル LPPL
            bubble_res = fit_lppl_bubble(price_series)

            # 3. 最高値日
            peak_date = price_series.idxmax()
            peak_price = price_series.max()

            # 4. ネガティブバブル LPPL
            neg_res = fit_lppl_negative_bubble(price_series, peak_date)

            # --- 結果テーブル ---
            summary = pd.DataFrame(
                [
                    ["内部崩壊候補日 (上昇 t_c)", bubble_res["tc_date"].date(), bubble_res["r2"]],
                    ["価格の最高値日", peak_date.date(), np.nan],
                    ["内部底候補日 (下落 t_c)", neg_res["tc_date"].date(), neg_res["r2"]],
                ],
                columns=["イベント", "日付", "R² (参考)"],
            )

            st.subheader("候補日サマリー")
            st.table(summary)

            # --- グラフ描画 ---
            fig, ax = plt.subplots(figsize=(12, 5))

            # 全期間の実際の価格
            ax.plot(price_series.index, price_series.values,
                    color="lightgray", label=f"{ticker} price (actual)")

            # 上昇バブル LPPL フィット
            ax.plot(price_series.index, bubble_res["price_fit"],
                    color="orange", label="LPPL bubble fit (uptrend)")

            # 下落局面の実際の価格
            down_series = neg_res["down_series"]
            ax.plot(down_series.index, down_series.values,
                    color="blue", label=f"{ticker} downtrend (actual)")

            # 重要日縦線
            ax.axvline(bubble_res["tc_date"], color="red", linestyle="--",
                       label=f"Internal collapse t_c↑ ({bubble_res['tc_date'].date()})")

            ax.axvline(peak_date, color="black", linestyle=":",
                       label=f"Price peak ({peak_date.date()})")

            ax.axvline(neg_res["tc_date"], color="green", linestyle="--",
                       label=f"Negative-bubble t_c↓ ({neg_res['tc_date'].date()})")

            ax.set_title(f"{ticker} — Bubble → Collapse → Negative Bubble")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True)
            plt.xticks(rotation=30)
            ax.legend(loc="best")

            st.subheader("統合グラフ")
            st.pyplot(fig)

            # 詳細な数値も見たい場合
            with st.expander("LPPL パラメータ詳細"):
                param_names = ["A", "B", "C", "m", "tc", "omega", "phi"]
                up_params_df = pd.DataFrame(
                    {
                        "parameter": param_names,
                        "estimate": bubble_res["params"],
                    }
                )
                down_params_df = pd.DataFrame(
                    {
                        "parameter": param_names,
                        "estimate": neg_res.get("params_down", [np.nan] * 7),
                    }
                )
                st.write("上昇バブル LPPL 推定値")
                st.dataframe(up_params_df)

            st.success("解析が完了しました。")

        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
