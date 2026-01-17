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
#   - Render env var: OS_TOKEN_SECRET_ADMIN
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
OS_TOKEN_SECRET = os.environ.get("OS_TOKEN_SECRET_ADMIN", "").strip()
token = st.query_params.get("t", "")

if not OS_TOKEN_SECRET or not token:
    st.stop()

ok, authed_email = verify_token(token, OS_TOKEN_SECRET)
if not ok:
    st.stop()

# ---- OPTIONAL ADMIN EMAIL ALLOWLIST ----
ADMIN_EMAILS = set(
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
)
if ADMIN_EMAILS:
    if authed_email.strip().lower() not in ADMIN_EMAILS:
        st.stop()


# =======================================================
# Cache settings
# =======================================================
PRICE_TTL_SECONDS = 15 * 60         # yfinance cache
FIT_TTL_SECONDS = 24 * 60 * 60      # fit cache


# =======================================================
# FINAL SCORE SETTINGS (calendar-day distance)
# =======================================================
UP_FUTURE_NEAR_DAYS = 30
UP_FUTURE_FAR_DAYS  = 180

UP_PAST_NEAR_DAYS   = 7
UP_PAST_FAR_DAYS    = 120

DOWN_FUTURE_NEAR_DAYS = 7
DOWN_FUTURE_FAR_DAYS  = 60

DOWN_PAST_NEAR_DAYS   = 7
DOWN_PAST_FAR_DAYS    = 60


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lin_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x0 == x1:
        return y0
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _clamp_p0_into_bounds(p0, lb, ub, eps=1e-6):
    """初期値が bounds の外に出て curve_fit が落ちるのを防ぐ"""
    p0 = np.asarray(p0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


# =======================================================
# (ADD) Mid-term Quant Table (A~J + Threshold + Judgement)
# =======================================================
MID_TH = {
    # ① 相場適合性
    "A_R_ok": 0.70,     "A_R_ng": 0.40,
    "B_R2_ok": 0.50,    "B_R2_ng": 0.20,
    "C_beta_ok_low": 0.80, "C_beta_ok_high": 1.50,
    "C_beta_ng_low": 0.50, "C_beta_ng_high": 2.00,

    # ② 成果効率
    "E_cmgr_ok": 0.0075, "E_cmgr_ng": 0.0000,
    "F_sharpe_ok": 1.00, "F_sharpe_ng": 0.50,

    # Max Drawdown thresholds
    "G_mdd_ok": -0.15,   "G_mdd_ng": -0.30,

    # F2/F3 alpha thresholds
    "F2_alpha_ok": 0.05, "F2_alpha_ng": 0.00,
    "F3_alpha_ok": 0.05, "F3_alpha_ng": 0.00,

    # ③ 因果監視
    "H_R2_ok": 0.25,     "H_R2_ng": 0.10,
    "I_beta_ok_low": 0.60, "I_beta_ok_high": 1.40,
    "I_beta_ng_low": 0.30, "I_beta_ng_high": 2.00,
    "J_vol_ok": 0.25,    "J_vol_ng": 0.40,
}


def _mid_beta_r2(x: pd.Series, y: pd.Series):
    beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return beta, r2


def _mid_fmt_value(metric: str, v: float) -> str:
    if metric.startswith("E: CMGR"):
        return f"{v*100:.2f}%"
    if metric in ["D: Max Drawdown", "F2: Alpha (Relative Outperformance vs Benchmark)"]:
        return f"{v*100:.2f}%"
    if metric.startswith("F: Sharpe"):
        return f"{v:.2f}"
    if metric.startswith("F3: Alpha"):
        return f"{v*100:.2f}%"
    if metric.startswith("D: Deviation Vol"):
        return f"{v:.2f} (pp)"
    if metric.startswith("J: Rolling Vol"):
        return f"{v*100:.2f}%"
    return f"{v:.4f}"


def _mid_judge(metric: str, value: float, dev: pd.DataFrame, bench: str, window: int) -> str:
    if metric == "A: Deviation R":
        if value >= MID_TH["A_R_ok"]: return "OK"
        if value < MID_TH["A_R_ng"]:  return "NG"
        return "注意"

    if metric == "B: Deviation R²":
        if value >= MID_TH["B_R2_ok"]: return "OK"
        if value < MID_TH["B_R2_ng"]:  return "NG"
        return "注意"

    if metric.startswith("C: Deviation β"):
        if MID_TH["C_beta_ok_low"] <= value <= MID_TH["C_beta_ok_high"]: return "OK"
        if value < MID_TH["C_beta_ng_low"] or value > MID_TH["C_beta_ng_high"]: return "NG"
        return "注意"

    if metric.startswith("D: Deviation Vol"):
        bench_latest = dev[bench].rolling(window).std(ddof=1).dropna().iloc[-1]
        ratio = value / bench_latest if bench_latest != 0 else np.nan
        if ratio <= 1.5: return "OK"
        if ratio > 2.5:  return "NG"
        return "注意"

    if metric == "D: Max Drawdown":
        if value >= MID_TH["G_mdd_ok"]: return "OK"
        if value < MID_TH["G_mdd_ng"]:  return "NG"
        return "注意"

    if metric.startswith("E: CMGR"):
        if value >= MID_TH["E_cmgr_ok"]: return "OK"
        if value < MID_TH["E_cmgr_ng"]:  return "NG"
        return "注意"

    if metric.startswith("F: Sharpe"):
        if value >= MID_TH["F_sharpe_ok"]: return "OK"
        if value < MID_TH["F_sharpe_ng"]:  return "NG"
        return "注意"

    if metric.startswith("F2: Alpha"):
        if value >= MID_TH["F2_alpha_ok"]: return "OK"
        if value < MID_TH["F2_alpha_ng"]:  return "NG"
        return "注意"

    if metric.startswith("F3: Alpha"):
        if value >= MID_TH["F3_alpha_ok"]: return "OK"
        if value < MID_TH["F3_alpha_ng"]:  return "NG"
        return "注意"

    if metric.startswith("H: Rolling R²"):
        if value >= MID_TH["H_R2_ok"]: return "OK"
        if value < MID_TH["H_R2_ng"]:  return "NG"
        return "注意"

    if metric.startswith("I: Rolling β"):
        if MID_TH["I_beta_ok_low"] <= value <= MID_TH["I_beta_ok_high"]: return "OK"
        if value < MID_TH["I_beta_ng_low"] or value > MID_TH["I_beta_ng_high"]: return "NG"
        return "注意"

    if metric.startswith("J: Rolling Vol"):
        if value <= MID_TH["J_vol_ok"]: return "OK"
        if value > MID_TH["J_vol_ng"]:  return "NG"
        return "注意"

    return "注意"


def _mid_threshold_text(metric: str) -> str:
    if metric == "A: Deviation R":
        return f"OK≥{MID_TH['A_R_ok']:.2f} / NG<{MID_TH['A_R_ng']:.2f}"
    if metric == "B: Deviation R²":
        return f"OK≥{MID_TH['B_R2_ok']:.2f} / NG<{MID_TH['B_R2_ng']:.2f}"
    if metric.startswith("C: Deviation β"):
        return f"OK:{MID_TH['C_beta_ok_low']:.2f}–{MID_TH['C_beta_ok_high']:.2f} / NG<{MID_TH['C_beta_ng_low']:.2f} or >{MID_TH['C_beta_ng_high']:.2f}"
    if metric.startswith("D: Deviation Vol"):
        return "OK:（株÷指数）≤1.5 / NG>2.5"
    if metric == "D: Max Drawdown":
        return f"OK≥{MID_TH['G_mdd_ok']*100:.0f}% / NG<{MID_TH['G_mdd_ng']*100:.0f}%"
    if metric.startswith("E: CMGR"):
        return f"OK≥{MID_TH['E_cmgr_ok']*100:.2f}%/月 / NG<{MID_TH['E_cmgr_ng']*100:.2f}%/月"
    if metric.startswith("F: Sharpe"):
        return f"OK≥{MID_TH['F_sharpe_ok']:.2f} / NG<{MID_TH['F_sharpe_ng']:.2f}"
    if metric.startswith("F2: Alpha"):
        return f"OK≥{MID_TH['F2_alpha_ok']*100:.0f}% / NG<{MID_TH['F2_alpha_ng']*100:.0f}%"
    if metric.startswith("F3: Alpha"):
        return f"OK≥{MID_TH['F3_alpha_ok']*100:.0f}% / NG<{MID_TH['F3_alpha_ng']*100:.0f}%"
    if metric.startswith("H: Rolling R²"):
        return f"OK≥{MID_TH['H_R2_ok']:.2f} / NG<{MID_TH['H_R2_ng']:.2f}"
    if metric.startswith("I: Rolling β"):
        return f"OK:{MID_TH['I_beta_ok_low']:.2f}–{MID_TH['I_beta_ok_high']:.2f} / NG<{MID_TH['I_beta_ng_low']:.2f} or >{MID_TH['I_beta_ng_high']:.2f}"
    if metric.startswith("J: Rolling Vol"):
        return f"OK≤{MID_TH['J_vol_ok']*100:.0f}% / NG>{MID_TH['J_vol_ng']*100:.0f}%"
    return "-"


# =======================================================
# (IMPORTANT) Price fetch helpers (Adj Close unified)
#   - LPPL / 判定表 / グラフ すべて Adj Close 優先で統一
# =======================================================
def _pick_price_field(df: pd.DataFrame, ticker: str) -> pd.Series:
    """
    yfinance download columns may be MultiIndex: (field, ticker)
    Prefer 'Adj Close' then 'Close'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            s = df[("Adj Close", ticker)]
        elif ("Close", ticker) in df.columns:
            s = df[("Close", ticker)]
        else:
            raise ValueError("NO_PRICE_FIELD")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise ValueError("NO_PRICE_FIELD")
    return s


@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_price_series_cached(ticker: str, start_date: date, end_date: date) -> pd.Series:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    s = _pick_price_field(df, ticker).dropna()

    if s.empty or len(s) < 30:
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    vals = s.to_numpy(dtype="float64")
    if not np.all(np.isfinite(vals)):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")
    if np.any(vals <= 0):
        raise ValueError("INVALID_TICKER_OR_NO_DATA")

    return s


@st.cache_data(ttl=PRICE_TTL_SECONDS, show_spinner=False)
def fetch_prices_pair_cached(ticker: str, bench: str, start_date: date, end_date: date) -> pd.DataFrame:
    end_exclusive = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        [ticker, bench],
        start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError("NO_DATA_PAIR")

    s_t = _pick_price_field(df, ticker).dropna()
    s_b = _pick_price_field(df, bench).dropna()

    close = pd.concat([s_t.rename(ticker), s_b.rename(bench)], axis=1).dropna()
    if close.empty or len(close) < 30:
        raise ValueError("NO_DATA_PAIR")

    vals = close.to_numpy(dtype="float64")
    if not np.all(np.isfinite(vals)):
        raise ValueError("NO_DATA_PAIR")
    if np.any(vals <= 0):
        raise ValueError("NO_DATA_PAIR")

    return close


def build_midterm_quant_table(
    ticker: str,
    bench: str,
    start_date: date,
    end_date: date,
    window: int = 20,
) -> pd.DataFrame:
    prices = fetch_prices_pair_cached(ticker, bench, start_date, end_date)
    if len(prices) < window + 10:
        raise ValueError("Not enough data points for mid-term table.")

    base = prices.iloc[0]
    index100 = prices / base * 100
    dev = index100 - 100
    ret = np.log(prices / prices.shift(1)).dropna()

    # ① 相場適合性
    A_R = dev[ticker].corr(dev[bench])
    B_R2 = A_R ** 2
    C_beta, _ = _mid_beta_r2(dev[bench], dev[ticker])
    D_vol_latest = dev[ticker].rolling(window).std(ddof=1).dropna().iloc[-1]

    # ② 成果効率
    start_dt = prices.index[0].date()
    end_dt = prices.index[-1].date()
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    months = max(months, 1)

    E_cmgr = (prices[ticker].iloc[-1] / prices[ticker].iloc[0]) ** (1 / months) - 1
    F_sharpe = ret[ticker].mean() / ret[ticker].std(ddof=1) * np.sqrt(252)

    p = prices[ticker]
    dd = p / p.cummax() - 1
    max_dd = dd.min()
    max_dd_date = dd.idxmin().date()

    ticker_total = prices[ticker].iloc[-1] / prices[ticker].iloc[0]
    bench_total = prices[bench].iloc[-1] / prices[bench].iloc[0]
    F2_alpha_rel = (ticker_total / bench_total) - 1

    X = ret[bench]
    Y = ret[ticker]
    beta_daily = np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1)
    alpha_daily = Y.mean() - beta_daily * X.mean()
    F3_alpha_annual = alpha_daily * 252

    # ③ 因果監視（最新値）
    roll_r = ret[ticker].rolling(window).corr(ret[bench])
    H_R2_latest = (roll_r ** 2).dropna().iloc[-1]
    I_beta_latest = (ret[ticker].rolling(window).cov(ret[bench]) / ret[bench].rolling(window).var(ddof=1)).dropna().iloc[-1]
    J_vol_annual_latest = (ret[ticker].rolling(window).std(ddof=1) * np.sqrt(252)).dropna().iloc[-1]

    rows = [
        ("① 相場適合性", "A: Deviation R", A_R, _mid_threshold_text("A: Deviation R"),
         "指数との中長期トレンドの類似度（高いほど同じ動き）"),
        ("① 相場適合性", "B: Deviation R²", B_R2, _mid_threshold_text("B: Deviation R²"),
         "トレンド変動の説明力（指数で説明できる割合）"),
        ("① 相場適合性", f"C: Deviation β (vs {bench})", C_beta, _mid_threshold_text(f"C: Deviation β (vs {bench})"),
         "乖離空間での感応度（指数に対してどれだけ動くか）"),
        ("① 相場適合性", f"D: Deviation Vol (rolling {window})", D_vol_latest, _mid_threshold_text(f"D: Deviation Vol (rolling {window})"),
         "乖離の標準偏差（乖離のブレ幅）"),
        ("① 相場適合性", "D: Max Drawdown", max_dd, _mid_threshold_text("D: Max Drawdown"),
         f"最大下落率（最悪日: {max_dd_date}）"),

        ("② 成果効率", f"E: CMGR (monthly, ~{months} months)", E_cmgr, _mid_threshold_text(f"E: CMGR (monthly, ~{months} months)"),
         "月次複利成長率"),
        ("② 成果効率", "F: Sharpe (annualized, log returns, rf=0)", F_sharpe, _mid_threshold_text("F: Sharpe (annualized, log returns, rf=0)"),
         "リスク調整後リターン（大きいほど効率的）"),
        ("② 成果効率", "F2: Alpha (Relative Outperformance vs Benchmark)", F2_alpha_rel, _mid_threshold_text("F2: Alpha (Relative Outperformance vs Benchmark)"),
         "期間全体の指数に対する超過収益"),
        ("② 成果効率", "F3: Alpha (Regression, daily -> annualized)", F3_alpha_annual, _mid_threshold_text("F3: Alpha (Regression, daily -> annualized)"),
         "市場要因を差し引いたα（回帰）"),

        ("③ 因果監視", f"H: Rolling R² (daily, {window})", H_R2_latest, _mid_threshold_text(f"H: Rolling R² (daily, {window})"),
         "直近で指数との関係が維持されているか"),
        ("③ 因果監視", f"I: Rolling β (daily, {window})", I_beta_latest, _mid_threshold_text(f"I: Rolling β (daily, {window})"),
         "直近β（過剰反応/鈍化の監視）"),
        ("③ 因果監視", f"J: Rolling Vol (annualized, {window})", J_vol_annual_latest, _mid_threshold_text(f"J: Rolling Vol (annualized, {window})"),
         "直近の年率ボラ（リスク局面の確認）"),
    ]

    df = pd.DataFrame(rows, columns=["Block", "Metric", "Value", "Threshold", "Note"])
    df["Judgement"] = [
        _mid_judge(m, v, dev=dev, bench=bench, window=window)
        for m, v in zip(df["Metric"], df["Value"])
    ]

    df_display = df.copy()
    df_display["Value"] = [_mid_fmt_value(m, v) for m, v in zip(df["Metric"], df["Value"])]
    return df_display


# -------------------------------------------------------
# (NEW) Bubble decision flow (R² -> m -> ω)
# -------------------------------------------------------
def bubble_judgement(r2: float, m: float, omega: float) -> tuple[str, dict]:
    """
    ① R² >= 0.65 ?
      NO -> 判定保留（LPPL形状が弱い）
      YES ->
        ② m in [0.25, 0.70] ?
          NO -> 通常の上昇寄り
          YES ->
            ③ ω in [6, 13] ?
              NO（>=18） -> バブル“風”（注意）
              YES -> バブル的な上昇
    """
    info = {
        "r2_ok": bool(np.isfinite(r2) and r2 >= 0.65),
        "m_ok": bool(np.isfinite(m) and (0.25 <= m <= 0.70)),
        "omega_ok": bool(np.isfinite(omega) and (6.0 <= omega <= 13.0)),
        "omega_high": bool(np.isfinite(omega) and omega >= 18.0),
    }

    if not info["r2_ok"]:
        return "判定保留（LPPL形状が弱い）", info
    if not info["m_ok"]:
        return "通常の上昇寄り", info
    if info["omega_ok"]:
        return "バブル的な上昇", info
    if info["omega_high"]:
        return "バブル“風”（注意：短周期すぎ）", info
    return "バブル“風”（注意）", info


def render_bubble_flow(r2: float, m: float, omega: float):
    verdict, info = bubble_judgement(r2, m, omega)

    def yn(v: bool) -> str:
        return "YES" if v else "NO"

    lines = []
    lines.append(f"① R² ≥ 0.65 ?   （R²={r2:.2f}）")
    lines.append(f" ├ {yn(info['r2_ok'])} → " + ("次へ" if info["r2_ok"] else "判定保留（LPPL形状が弱い）"))

    if info["r2_ok"]:
        lines.append(" │")
        lines.append(f" │   ② m ∈ [0.25, 0.70] ?   （m={m:.2f}）")
        lines.append(f" │    ├ {yn(info['m_ok'])} → " + ("次へ" if info["m_ok"] else "通常の上昇寄り"))

        if info["m_ok"]:
            lines.append(" │    │")
            lines.append(f" │    │   ③ ω ∈ [6, 13] ?   （ω={omega:.2f}）")
            if info["omega_ok"]:
                lines.append(" │    │    ├ YES → バブル的な上昇")
            else:
                if np.isfinite(omega) and omega >= 18:
                    lines.append(" │    │    ├ NO（≥18）→ バブル“風”（注意）")
                else:
                    lines.append(" │    │    ├ NO → バブル“風”（注意）")
            lines.append(" │    │")

    lines.append("")
    lines.append(f"【判定】{verdict}")

    st.markdown("### バブル判定フロー（管理者用）")
    st.code("\n".join(lines), language="text")


# -------------------------------------------------------
# (FIX) admin_interpretation_text  ※NameError対策：必ず定義
# -------------------------------------------------------
def admin_interpretation_text(bubble_res: dict, end_date: date) -> tuple[str, list[str]]:
    """
    管理者向け：投資判断に使うための簡易解釈
    返り値: (要約, 箇条書き)
    """
    pdict = bubble_res.get("param_dict", {})
    r2 = float(bubble_res.get("r2", float("nan")))
    rmse = float(bubble_res.get("rmse", float("nan")))
    m = float(pdict.get("m", float("nan")))
    omega = float(pdict.get("omega", float("nan")))
    c_over_b = float(pdict.get("abs_C_over_B", float("nan")))

    tc_date = pd.Timestamp(bubble_res.get("tc_date")).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    days_to_tc = int((tc_date - end_norm).days)

    bullets: list[str] = []

    # 終盤度（tc）
    if days_to_tc < 0:
        bullets.append(f"t_c は既に {abs(days_to_tc)} 日前に通過（構造的ピーク通過の可能性）。")
        bullets.append("新規の追随買いは控えめに。保有なら部分利確・ヘッジを検討。")
    elif days_to_tc <= 30:
        bullets.append(f"t_c まで残り {days_to_tc} 日（危険ゾーンが近い）。")
        bullets.append("新規ロングは慎重（サイズ縮小・分割）。利確/ヘッジ準備。")
    else:
        bullets.append(f"t_c まで残り {days_to_tc} 日（危険ゾーンは監視段階）。")

    # バブル判定（R²→m→ω）
    verdict, _ = bubble_judgement(r2, m, omega)
    bullets.append(f"バブル判定（R²→m→ω）：{verdict}")

    # 補助（信頼度/注意喚起）
    if np.isfinite(r2) and r2 < 0.65:
        bullets.append(f"R²={r2:.2f}：形状が弱め→判定保留寄り。")
    if np.isfinite(m) and m >= 0.85:
        bullets.append(f"m={m:.2f}：上限寄り→境界解の疑い（tc一点は信じすぎない）。")
    if np.isfinite(omega) and omega >= 18:
        bullets.append(f"ω={omega:.2f}：短周期すぎ→ノイズ追随の疑い（バブル“風”注意）。")

    if np.isfinite(rmse):
        bullets.append(f"RMSE(log)={rmse:.3f}：フィット安定度の目安（小さいほど安定）。")
    if np.isfinite(c_over_b) and c_over_b >= 2.0:
        bullets.append(f"|C/B|={c_over_b:.2f}：振動項が強すぎ→“それっぽいフィット”の疑い。")

    # 要約（短く）
    if verdict == "バブル的な上昇":
        summary = "バブル的上昇（m帯域＋R²十分＋ω典型）。追随買いは抑え、利確/ヘッジを織り込む局面。"
    elif verdict.startswith("バブル“風”"):
        summary = "バブル“風”（形はそれっぽいが注意要素あり）。追随買いは慎重、リスク管理を前倒し。"
    elif verdict == "通常の上昇寄り":
        summary = "典型バブルとは言いにくい（mが帯域外）。ただしtcが近いなら姿勢調整は有効。"
    else:
        summary = "LPPL形状が弱く判定保留。期間変更・他指標との併用推奨。"

    return summary, bullets


# -------------------------------------------------------
# LPPL-like model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)


def fit_lppl_bubble(price_series: pd.Series):
    """上昇（バブル）側フィット：log(price) をターゲットに LPPL を当てる"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)

    N = len(t)

    # Initial guess
    A_init = float(np.mean(log_price))
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N + 20
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    # Dynamic bounds for A
    A_low = float(np.min(log_price) - 2.0)
    A_high = float(np.max(log_price) + 2.0)

    lower_bounds = [A_low, -20, -20, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N + 250, 25.0, np.pi]

    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)

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

    ss_res = float(np.sum((log_price - log_fit) ** 2))
    ss_tot = float(np.sum((log_price - log_price.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rmse = float(np.sqrt(np.mean((log_price - log_fit) ** 2)))

    first_date = price_series.index[0]
    tc_days = float(params[4])  # t=0..N-1 の「データ点単位」
    tc_date = first_date + timedelta(days=tc_days)  # カレンダー日近似

    A, B, C, m, tc, omega, phi = [float(x) for x in params]
    abs_c_over_b = float(abs(C / B)) if abs(B) > 1e-12 else float("inf")
    log_period = float(2.0 * np.pi / omega) if omega != 0 else float("inf")

    bounds_info = {
        "A_low": A_low, "A_high": A_high,
        "B_low": -20.0, "B_high": 20.0,
        "C_low": -20.0, "C_high": 20.0,
        "m_low": 0.01, "m_high": 0.99,
        "tc_low": float(N + 1), "tc_high": float(N + 250),
        "omega_low": 2.0, "omega_high": 25.0,
        "phi_low": float(-np.pi), "phi_high": float(np.pi),
    }

    return {
        "params": np.asarray(params, dtype=float),
        "param_dict": {
            "A": A, "B": B, "C": C, "m": m, "tc": tc, "omega": omega, "phi": phi,
            "abs_C_over_B": abs_c_over_b,
            "log_period_2pi_over_omega": log_period,
        },
        "price_fit": price_fit,
        "log_fit": log_fit,
        "r2": float(r2),
        "rmse": rmse,
        "tc_date": tc_date,
        "tc_days": tc_days,
        "bounds_info": bounds_info,
        "N": int(N),
    }


def fit_lppl_negative_bubble(
    price_series: pd.Series,
    peak_date,
    min_points: int = 10,
    min_drop_ratio: float = 0.03,
):
    """下落（ネガティブ・バブル）側フィット：-log(price) に LPPL を当てる（反転）"""
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

    # Initial guess
    A_init = float(np.mean(neg_log_down))
    B_init = -1.0
    C_init = 0.1
    m_init = 0.5
    tc_init = N_down + 15
    omega_init = 8.0
    phi_init = 0.0
    p0 = [A_init, B_init, C_init, m_init, tc_init, omega_init, phi_init]

    # Dynamic bounds for A
    A_low = float(np.min(neg_log_down) - 2.0)
    A_high = float(np.max(neg_log_down) + 2.0)

    lower_bounds = [A_low, -20, -20, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N_down + 200, 25.0, np.pi]

    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)

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
    r2_down = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    first_down_date = down_series.index[0]
    tc_days = float(params_down[4])
    tc_bottom_date = first_down_date + timedelta(days=tc_days)

    return {
        "ok": True,
        "down_series": down_series,
        "price_fit_down": price_fit_down,
        "r2": float(r2_down),
        "tc_date": tc_bottom_date,
        "tc_days": tc_days,
        "params": np.asarray(params_down, dtype=float),
    }


# -------------------------------------------------------
# fit cache helpers
# -------------------------------------------------------
def series_cache_key(s: pd.Series) -> str:
    idx = s.index.astype("int64").to_numpy()
    vals = s.to_numpy(dtype="float64")
    h = hashlib.sha256()
    h.update(idx.tobytes())
    h.update(vals.tobytes())
    return h.hexdigest()


@st.cache_data(ttl=FIT_TTL_SECONDS, show_spinner=False)
def fit_lppl_bubble_cached(price_key: str, price_values: np.ndarray, idx_int: np.ndarray):
    idx = pd.to_datetime(idx_int)
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
# FINAL: Phase + time-distance scoring (date-based)
# =======================================================
def compute_signal_and_score(tc_up_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             down_tc_date: pd.Timestamp | None) -> tuple[str, int]:
    """
    SAFE   : tc_up > now, score 0..59（近いほど高い）
    CAUTION: tc_up < now かつ down_tc なし, score 60..79（古いほど高い）
    HIGH   : down_tc あり, score 80..100（進行度が高いほど高い）
    """
    now = pd.Timestamp(end_date).normalize()
    tc_up = pd.Timestamp(tc_up_date).normalize()

    # HIGH
    if down_tc_date is not None:
        down_tc = pd.Timestamp(down_tc_date).normalize()
        delta = (down_tc - now).days  # future:+, past:-

        if delta > 0:
            s = _lin_map(
                x=delta,
                x0=DOWN_FUTURE_NEAR_DAYS,
                x1=DOWN_FUTURE_FAR_DAYS,
                y0=90,
                y1=80,
            )
            return ("HIGH", int(round(_clamp(s, 80, 90))))

        past = abs(delta)
        s = _lin_map(
            x=past,
            x0=DOWN_PAST_NEAR_DAYS,
            x1=DOWN_PAST_FAR_DAYS,
            y0=90,
            y1=100,
        )
        return ("HIGH", int(round(_clamp(s, 90, 100))))

    # SAFE / CAUTION
    gap = (tc_up - now).days

    if gap > 0:
        s = _lin_map(
            x=gap,
            x0=UP_FUTURE_NEAR_DAYS,
            x1=UP_FUTURE_FAR_DAYS,
            y0=59,
            y1=0,
        )
        return ("SAFE", int(round(_clamp(s, 0, 59))))

    past = abs(gap)
    s = _lin_map(
        x=past,
        x0=UP_PAST_NEAR_DAYS,
        x1=UP_PAST_FAR_DAYS,
        y0=60,
        y1=79,
    )
    return ("CAUTION", int(round(_clamp(s, 60, 79))))


# =======================================================
# (ADD) Graph pack renderer (uses Adj Close unified prices)
# =======================================================
def render_graph_pack_from_prices(
    prices: pd.DataFrame,
    ticker: str,
    bench: str,
    window: int = 20,
    trading_days: int = 252,
):
    # Derived
    base = prices.iloc[0]
    index100 = prices / base * 100.0
    dev = index100 - 100.0
    ret = np.log(prices / prices.shift(1)).dropna()

    # (A) Cumulative index
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.plot(index100.index, index100[ticker], label=f"{ticker} (Index)", color="red")
    ax.plot(index100.index, index100[bench],  label=f"{bench} (Index)",  color="blue")
    ax.set_title("Cumulative Performance (Index = 100)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Index", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")
    st.pyplot(fig)

    # (B) Deviation scatter + regression
    X = dev[bench].dropna()
    Y = dev[ticker].dropna()
    common = X.index.intersection(Y.index)
    X = X.loc[common]
    Y = Y.loc[common]

    slope_dev, intercept_dev = np.polyfit(X.values, Y.values, 1)
    x_sorted = X.sort_values()
    y_line = slope_dev * x_sorted + intercept_dev

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.scatter(X, Y, alpha=0.6, color="red")
    ax.plot(x_sorted, y_line, color="blue")
    ax.set_title(f"Price Deviation Scatter ({ticker} vs {bench}) + Regression", color="white")
    ax.set_xlabel(f"{bench} Deviation (pp)", color="white")
    ax.set_ylabel(f"{ticker} Deviation (pp)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    st.pyplot(fig)
    st.write(f"Deviation regression: slope={slope_dev:.6f}, intercept={intercept_dev:.6f}")

    # (C) Rolling vol of deviation
    vol_dev_t = dev[ticker].rolling(int(window)).std(ddof=1)
    vol_dev_b = dev[bench].rolling(int(window)).std(ddof=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.plot(vol_dev_t.index, vol_dev_t, label=f"{ticker} Deviation Vol", color="red")
    ax.plot(vol_dev_b.index, vol_dev_b, label=f"{bench} Deviation Vol", color="blue")
    ax.set_title(f"Rolling Volatility of Price Deviation (Window = {int(window)})", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Std of Deviation (pp)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")
    st.pyplot(fig)

    # (D) Drawdown
    p = prices[ticker]
    running_max = p.cummax()
    dd = (p / running_max) - 1.0

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.plot(dd.index, dd * 100.0, color="white")
    ax.set_title(f"Drawdown (%) - {ticker}", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Drawdown (%)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    st.pyplot(fig)
    st.write(f"Max Drawdown (%): {dd.min() * 100:.2f}")

    # (E) Daily log return scatter + regression
    Xr = ret[bench].dropna()
    Yr = ret[ticker].dropna()
    common_r = Xr.index.intersection(Yr.index)
    Xr = Xr.loc[common_r]
    Yr = Yr.loc[common_r]

    slope_ret, intercept_ret = np.polyfit(Xr.values, Yr.values, 1)
    xr_sorted = Xr.sort_values()
    yr_line = slope_ret * xr_sorted + intercept_ret

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.scatter(Xr, Yr, alpha=0.6, color="red")
    ax.plot(xr_sorted, yr_line, color="blue")
    ax.set_title(f"Daily Log Returns Scatter ({ticker} vs {bench}) + Regression", color="white")
    ax.set_xlabel(f"{bench} Daily Log Return", color="white")
    ax.set_ylabel(f"{ticker} Daily Log Return", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    st.pyplot(fig)
    st.write(f"Return regression (approx β, α): slope(beta)={slope_ret:.6f}, intercept(alpha/day)={intercept_ret:.6f}")

    # (F) Rolling annualized vol
    roll_vol = ret[ticker].rolling(int(window)).std(ddof=1) * np.sqrt(float(trading_days))

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    ax.plot(roll_vol.index, roll_vol * 100.0, color="red")
    ax.set_title(f"Rolling Volatility of Daily Returns (Annualized, Window = {int(window)}) - {ticker}", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Annualized Vol (%)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    st.pyplot(fig)

    if roll_vol.dropna().empty:
        st.warning("ローリング年率Volが計算できません（windowが大きい/データ不足の可能性）。")
    else:
        st.write(f"Annualized Vol mean (%): {roll_vol.mean() * 100:.2f}")
        st.write(f"Annualized Vol last (%): {roll_vol.dropna().iloc[-1] * 100:.2f}")


# -------------------------------------------------------
# Streamlit app (ADMIN)
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander Admin", layout="wide")

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
        [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }

        .custom-error {
            background-color: #141518;
            border: 1px solid #2a2c30;
            border-radius: 12px;
            padding: 14px 18px;
            color: #ffffff;
            font-size: 0.95rem;
            margin-top: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def show_error_black(msg: str):
        st.markdown(
            f"""
            <div class="custom-error">
                {msg}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Out-stander（管理者用）")
    st.caption(f"認証ユーザー: {authed_email}")

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
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

    # =======================================================
    # LPPL (Adj Close unified)
    # =======================================================
    try:
        price_series = fetch_price_series_cached(ticker.strip(), start_date, end_date)
    except Exception:
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

    if len(price_series) < 30:
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

    # ---- fit caches ----
    key = series_cache_key(price_series)
    idx_int = price_series.index.astype("int64").to_numpy()
    vals = price_series.to_numpy(dtype="float64")

    try:
        bubble_res = fit_lppl_bubble_cached(key, vals, idx_int)
    except Exception:
        show_error_black("フィットに失敗しました（上昇LPPL）。")
        st.stop()

    peak_date = price_series.idxmax()
    peak_price = float(price_series.max())
    start_price_val = float(price_series.iloc[0])
    gain = peak_price / start_price_val
    gain_pct = (gain - 1.0) * 100.0

    peak_date_int = int(pd.Timestamp(peak_date).value)
    neg_res = fit_lppl_negative_bubble_cached(
        key, vals, idx_int,
        peak_date_int=peak_date_int,
        min_points=10,
        min_drop_ratio=0.03,
    )

    tc_up_date = pd.Timestamp(bubble_res["tc_date"])
    end_ts = pd.Timestamp(end_date)
    down_tc_date = pd.Timestamp(neg_res["tc_date"]) if neg_res.get("ok") else None

    signal_label, score = compute_signal_and_score(tc_up_date, end_ts, down_tc_date)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(price_series.index, price_series.values, color="gray", label=ticker.strip())
    ax.plot(price_series.index, bubble_res["price_fit"], color="orange", label="上昇モデル")

    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--",
               label=f"t_c（上昇） {pd.Timestamp(bubble_res['tc_date']).date()}")
    ax.axvline(peak_date, color="white", linestyle=":",
               label=f"ピーク {pd.Timestamp(peak_date).date()}")

    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="cyan", label="下落（ピーク以降）")
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="green", label="下落モデル")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--",
                   label=f"t_c（下落） {pd.Timestamp(neg_res['tc_date']).date()}")

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (Adj Close preferred)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")

    st.pyplot(fig)

    # ---- UI signal ----
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
                スコア
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
                上昇倍率（開始→ピーク）
            </div>
            <div style="font-size: 36px; font-weight: 700; color: #f5f5f5; line-height: 1.1;">
                {gain:.2f}x
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

    # =======================================================
    # Bubble flow + simplified metric explanations
    # =======================================================
    st.markdown("---")
    pdict = bubble_res.get("param_dict", {})
    r2 = float(bubble_res.get("r2", np.nan))
    m = float(pdict.get("m", np.nan))
    omega = float(pdict.get("omega", np.nan))

    render_bubble_flow(r2, m, omega)

    verdict, _ = bubble_judgement(r2, m, omega)

    st.markdown("### 管理者指標（バブル判定のための最小説明）")

    tc_up_norm = pd.Timestamp(bubble_res["tc_date"]).normalize()
    end_norm = pd.Timestamp(end_date).normalize()
    days_to_tc = int((tc_up_norm - end_norm).days)

    rmse = float(bubble_res.get("rmse", np.nan))
    c_over_b = float(pdict.get("abs_C_over_B", np.nan))
    log_period = float(pdict.get("log_period_2pi_over_omega", np.nan))
    N = int(bubble_res.get("N", 0))

    admin_rows = [
        ["① R²（対数空間）", r2,
         "何のため？：LPPL形状が十分に出ているか（m/ωでバブル判定して良いか）。バブル判定の条件：R²≥0.65。"],
        ["② m（最重要）", m,
         "何のため？：上昇が“バブル型の加速形状”か。バブル的：m∈[0.25,0.70]（典型0.3〜0.6）。"],
        ["③ ω", omega,
         "何のため？：振動が“過熱の型として自然”か。バブル的：ω∈[6,13]。ω≥18は短周期すぎ→バブル“風”（注意）。"],
        ["結論（バブル判定）", verdict,
         "上の判定フローの最終結論。投資行動（追随買い抑制・利確/ヘッジ検討）に使う。"],
        ["t_c（日付・近似）", str(tc_up_norm.date()),
         "何のため？：バブル“かどうか”ではなく“終盤かどうか”。tcが近い/通過＝終盤の可能性（※1点=1日換算の近似）。"],
        ["t_cまでの残日数", days_to_tc,
         "何のため？：終盤度（短い/通過ほど終盤リスク↑）。バブル判定そのものは R²・m・ω。"],
        ["RMSE（対数空間）", rmse,
         "何のため？：フィットの安定度（mの信頼補助）。大きいとバブル判定の確度が落ちる。"],
        ["|C/B|", c_over_b,
         "何のため？：上下動の荒さ（副作用）。≥2.0は“それっぽいフィット”の疑いが増える。"],
        ["2π/ω", log_period,
         "何のため？：周期の別表現（基本はωの条件で十分）。"],
        ["フィット期間 N", N,
         "何のため？：短すぎ/長すぎは判定ブレ要因。"],
    ]
    admin_df = pd.DataFrame(admin_rows, columns=["指標", "値", "解説"])
    st.dataframe(admin_df, use_container_width=True)

    st.markdown("#### 生パラメータ（A, B, C, m, tc, ω, φ）")
    raw_params = bubble_res.get("params", np.array([], dtype=float)).astype(float)
    if raw_params.size == 7:
        raw_df = pd.DataFrame([raw_params], columns=["A", "B", "C", "m", "tc", "omega", "phi"])
    else:
        raw_df = pd.DataFrame([raw_params])
    st.dataframe(raw_df, use_container_width=True)

    # =======================================================
    # Fit quality flags（簡易サニティチェック）
    # =======================================================
    st.markdown("#### フィット品質フラグ（簡易サニティチェック）")
    binfo = bubble_res.get("bounds_info", {})
    if raw_params.size == 7:
        A_, B_, C_, m_, tc_, omega_, phi_ = [float(x) for x in raw_params]
    else:
        A_, B_, C_, m_, tc_, omega_, phi_ = [np.nan]*7

    def near_bound(x, lo, hi, tol=0.02):
        if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return False
        r = (x - lo) / (hi - lo)
        return (r < tol) or (r > 1 - tol)

    flags = []
    if np.isfinite(m_) and (m_ < 0.2 or m_ > 0.8):
        flags.append("m が一般的な“バブル的帯域”（概ね0.3〜0.6）から外れています → バブル判定は慎重に。")
    if near_bound(m_, binfo.get("m_low", np.nan), binfo.get("m_high", np.nan)):
        flags.append("m が境界（0.01/0.99付近）に張り付いています → 境界解の疑い（判定の信頼度低下）。")
    if near_bound(tc_, binfo.get("tc_low", np.nan), binfo.get("tc_high", np.nan)):
        flags.append("tc が境界付近です → 制約に押されている可能性（終盤度の解釈は幅で）。")
    if near_bound(omega_, binfo.get("omega_low", np.nan), binfo.get("omega_high", np.nan)):
        flags.append("ω が境界付近です → ωの解釈は慎重に。")
    if np.isfinite(c_over_b) and c_over_b > 2.0:
        flags.append("|C/B| が大きい → 振動項が支配的で“それっぽい線”になりやすい（要注意）。")

    if flags:
        st.write(pd.DataFrame({"フラグ": flags}))
    else:
        st.write("単純なヒューリスティックでは明確な赤信号は見当たりません。")

    # =======================================================
    # 投資判断メモ（管理者のみ）
    # =======================================================
    st.markdown("### 投資判断向けの解釈（管理者のみ）")
    summary, bullets = admin_interpretation_text(bubble_res, end_date)

    with st.expander("投資判断向けの解釈メモ（管理者のみ）", expanded=True):
        st.markdown(f"**要約**：{summary}")
        st.markdown("**ポイント**")
        st.markdown("\n".join([f"- {b}" for b in bullets]))

        st.markdown("---")
        st.markdown("**使い方（実務）**")
        st.markdown(
            "- バブル判定は **R²（信用）×m（形）×ω（自然さ）** の3点で判断。\n"
            "- tc は『バブルかどうか』ではなく『終盤かどうか』。\n"
            "- 結果が「バブル的/バブル風」なら、新規追随買いを控え、部分利確・ヘッジ・サイズ調整を検討。"
        )

    # =======================================================
    # Mid-term Quant 判定表（LPPLの下に表示）
    # =======================================================
    st.markdown("---")
    with st.expander("Mid-term Quant 判定表（A〜J + Threshold + Judgement）", expanded=False):
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            bench = st.text_input("Benchmark（指数）", value="^GSPC", help="例: ^GSPC / ^IXIC / ^N225")
        with col_b2:
            mid_window = st.number_input("WINDOW（判定表）", min_value=5, max_value=120, value=20, step=1)

        try:
            df_mid = build_midterm_quant_table(
                ticker=ticker.strip(),
                bench=bench.strip(),
                start_date=start_date,
                end_date=end_date,
                window=int(mid_window),
            )
            st.dataframe(
                df_mid[["Block", "Metric", "Value", "Threshold", "Judgement", "Note"]],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            show_error_black(f"判定表の生成に失敗しました: {e}")

    # =======================================================
    # (ADD) グラフ（Adj Close unified）
    # =======================================================
    st.markdown("---")
    with st.expander("グラフ（Index/Deviation/Regression/Vol/Drawdown）", expanded=False):
        gcol1, gcol2, gcol3 = st.columns([2, 1, 1])
        with gcol1:
            graph_bench = st.text_input("Benchmark（グラフ用）", value="^GSPC")
        with gcol2:
            graph_window = st.number_input("WINDOW（グラフ用）", min_value=5, max_value=120, value=20, step=1)
        with gcol3:
            trading_days = st.number_input("Trading days/year", min_value=200, max_value=365, value=252, step=1)

        try:
            prices_pair = fetch_prices_pair_cached(
                ticker=ticker.strip(),
                bench=graph_bench.strip(),
                start_date=start_date,
                end_date=end_date,
            )
            st.caption(f"Data: {prices_pair.index.min().date()} to {prices_pair.index.max().date()} | rows={len(prices_pair)}")
            render_graph_pack_from_prices(
                prices=prices_pair,
                ticker=ticker.strip(),
                bench=graph_bench.strip(),
                window=int(graph_window),
                trading_days=int(trading_days),
            )
        except Exception as e:
            show_error_black(f"グラフ表示に失敗しました: {e}")


if __name__ == "__main__":
    main()

