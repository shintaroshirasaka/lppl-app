# ここにさっき渡した app.py のコード全文を貼る
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st
import os  # ファイル存在確認用にインポート追加

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
# 価格データ取得
# -------------------------------------------------------

def fetch_price_series(ticker: str, start_date: date, end_date: date):
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
# Streamlit アプリ本体
# -------------------------------------------------------

def main():
    st.set_page_config(page_title="Out-stander", layout="wide")

    # ---- 簡易ダークテーマ（背景だけ黒） ----
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0b0c0e;
            color: #f2f2f2;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        /* バナーと上部の余白調整 */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 5rem;
        }
        .stTextInput > div > div > input,
        .stDateInput > div > div > input {
            background-color: #1a1c1f;
            color: #ffffff;
            border: 1px solid #444;
        }
        .stButton button {
            background-color: #222428;
            color: #ffffff;
            border-radius: 6px;
            border: 1px solid #555;
        }
        .stButton button:hover {
            background-color: #333333;
            border-color: #777777;
        }
        h1, h2, h3, h4 {
            color: #f2f2f2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- Banner / Title Section -----
    # 画像ファイル 'banner.png' があれば表示し、なければテキストタイトルを表示
    if os.path.exists("banner.png"):
        st.image("banner.png", use_container_width=True) # 古いバージョンの場合 use_column_width=True
    else:
        # 画像がない場合のフォールバック
        st.title("Out-stander")

    # ----- Input Form -----
    with st.form("input_form"):
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

    # ----- Fetch Data -----
    try:
        price_series = fetch_price_series(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.stop()

    if len(price_series) < 30:
        st.error("データが少なすぎます。期間を伸ばしてください。")
        st.stop()

    # ----- 上昇モデル -----
    bubble_res = fit_lppl_bubble(price_series)

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price_val = float(price_series.iloc[0])
    gain = peak_price / start_price_val
    gain_pct = (gain - 1.0) * 100.0

    params_up = bubble_res["params"]
    r2_up = bubble_res["r2"]
    m_up = params_up[3]
    tc_index = float(bubble_res["tc_days"])
    last_index = float(len(price_series) - 1)
    score = bubble_score(r2_up, m_up, tc_index, last_index)

    # ----- 下落モデル -----
    neg_res = fit_lppl_negative_bubble(price_series, peak_date)

    # ----- グラフ -----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(price_series.index, price_series.values,
            color="gray", label=ticker)
    ax.plot(price_series.index, bubble_res["price_fit"],
            color="orange", label="Up model")

    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--",
               label=f"Turn (up) {bubble_res['tc_date'].date()}")
    ax.axvline(peak_date, color="white", linestyle=":",
               label=f"Peak {peak_date.date()}")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values,
                color="cyan", label="Down")
        ax.plot(down.index, neg_res["price_fit_down"],
                "--", color="green", label="Down model")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--",
                   label=f"Turn (down) {neg_res['tc_date'].date()}")

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")

    st.pyplot(fig)

    # --------------------------------------------------
    # Score & Gain Cards
    # --------------------------------------------------
    # Score → Risk label & color
    if score >= 80:
        risk_label = "High"
        risk_color = "#ff4d4f"
    elif score >= 60:
        risk_label = "Caution"
        risk_color = "#ffc53d"
    else:
        risk_label = "Safe"
        risk_color = "#52c41a"

    col_score, col_gain = st.columns(2)

    # ----- Score Card -----
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

    # ----- Gain Card -----
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
