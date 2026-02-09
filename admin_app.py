import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects # 縁取り用
from scipy.optimize import curve_fit
from datetime import date, timedelta
import streamlit as st
import os
import requests
import time
import hmac
import hashlib
import base64
import io
import platform

# =======================================================
# FONT SETUP (Auto-detect Japanese Font)
# =======================================================
def configure_japanese_font():
    """
    OS環境に合わせて日本語フォントを自動設定する。
    Matplotlibのデフォルトでは日本語が表示できない（□□□になる）ため。
    """
    system = platform.system()
    font_path = None
    
    # OSごとの代表的な日本語フォントパス候補
    if system == "Windows":
        font_candidates = [
            "C:\\Windows\\Fonts\\meiryo.ttc",
            "C:\\Windows\\Fonts\\msgothic.ttc",
            "C:\\Windows\\Fonts\\yugothr.ttc"
        ]
    elif system == "Darwin": # Mac
        font_candidates = [
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Osaka.ttf",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
        ]
    else: # Linux (Streamlit Cloud, Docker etc)
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/ipafont-gothic/ipag.ttf"
        ]
        
    for path in font_candidates:
        if os.path.exists(path):
            font_path = path
            break
            
    if font_path:
        # フォントプロパティを生成してMatplotlibに設定
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        return font_prop
    else:
        # 見つからない場合はデフォルト（文字化けする可能性あり）
        return None

# フォント設定を実行（グローバル設定）
JP_FONT = configure_japanese_font()


# =======================================================
# AUTH GATE: Require signed short-lived token (?t=...)
# =======================================================
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
SCRAPE_TTL_SECONDS = 60 * 60        # scraping cache (1 hour)


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
    p0 = np.asarray(p0, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


# =======================================================
# Mid-term Quant Table Settings
# =======================================================
MID_TH = {
    "A_R_ok": 0.70,     "A_R_ng": 0.40,
    "B_R2_ok": 0.50,    "B_R2_ng": 0.20,
    "C_beta_ok_low": 0.80, "C_beta_ok_high": 1.50,
    "C_beta_ng_low": 0.50, "C_beta_ng_high": 2.00,
    "E_cmgr_ok": 0.0075, "E_cmgr_ng": 0.0000,
    "F_sharpe_ok": 1.00, "F_sharpe_ng": 0.50,
    "G_mdd_ok": -0.15,   "G_mdd_ng": -0.30,
    "F2_alpha_ok": 0.05, "F2_alpha_ng": 0.00,
    "F3_alpha_ok": 0.05, "F3_alpha_ng": 0.00,
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
# Price fetch helpers
# =======================================================
def _pick_price_field(df: pd.DataFrame, ticker: str) -> pd.Series:
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


def build_midterm_quant_table(ticker: str, bench: str, start_date: date, end_date: date, window: int = 20) -> pd.DataFrame:
    prices = fetch_prices_pair_cached(ticker, bench, start_date, end_date)
    if len(prices) < window + 10:
        raise ValueError("Not enough data points for mid-term table.")
    base = prices.iloc[0]
    index100 = prices / base * 100
    dev = index100 - 100
    ret = np.log(prices / prices.shift(1)).dropna()
    A_R = dev[ticker].corr(dev[bench])
    B_R2 = A_R ** 2
    C_beta, _ = _mid_beta_r2(dev[bench], dev[ticker])
    D_vol_latest = dev[ticker].rolling(window).std(ddof=1).dropna().iloc[-1]
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
    roll_r = ret[ticker].rolling(window).corr(ret[bench])
    H_R2_latest = (roll_r ** 2).dropna().iloc[-1]
    I_beta_latest = (ret[ticker].rolling(window).cov(ret[bench]) / ret[bench].rolling(window).var(ddof=1)).dropna().iloc[-1]
    J_vol_annual_latest = (ret[ticker].rolling(window).std(ddof=1) * np.sqrt(252)).dropna().iloc[-1]

    rows = [
        ("① 相場適合性", "A: Deviation R", A_R, _mid_threshold_text("A: Deviation R"), "指数との中長期トレンドの類似度"),
        ("① 相場適合性", "B: Deviation R²", B_R2, _mid_threshold_text("B: Deviation R²"), "トレンド変動の説明力"),
        ("① 相場適合性", f"C: Deviation β (vs {bench})", C_beta, _mid_threshold_text(f"C: Deviation β (vs {bench})"), "乖離空間での感応度"),
        ("① 相場適合性", f"D: Deviation Vol (rolling {window})", D_vol_latest, _mid_threshold_text(f"D: Deviation Vol (rolling {window})"), "乖離の標準偏差"),
        ("① 相場適合性", "D: Max Drawdown", max_dd, _mid_threshold_text("D: Max Drawdown"), f"最大下落率（最悪日: {max_dd_date}）"),
        ("② 成果効率", f"E: CMGR (monthly, ~{months} months)", E_cmgr, _mid_threshold_text(f"E: CMGR (monthly, ~{months} months)"), "月次複利成長率"),
        ("② 成果効率", "F: Sharpe (annualized, log returns, rf=0)", F_sharpe, _mid_threshold_text("F: Sharpe (annualized, log returns, rf=0)"), "リスク調整後リターン"),
        ("② 成果効率", "F2: Alpha (Relative Outperformance vs Benchmark)", F2_alpha_rel, _mid_threshold_text("F2: Alpha (Relative Outperformance vs Benchmark)"), "期間全体の指数に対する超過収益"),
        ("② 成果効率", "F3: Alpha (Regression, daily -> annualized)", F3_alpha_annual, _mid_threshold_text("F3: Alpha (Regression, daily -> annualized)"), "市場要因を差し引いたα"),
        ("③ 因果監視", f"H: Rolling R² (daily, {window})", H_R2_latest, _mid_threshold_text(f"H: Rolling R² (daily, {window})"), "直近で指数との関係が維持されているか"),
        ("③ 因果監視", f"I: Rolling β (daily, {window})", I_beta_latest, _mid_threshold_text(f"I: Rolling β (daily, {window})"), "直近β"),
        ("③ 因果監視", f"J: Rolling Vol (annualized, {window})", J_vol_annual_latest, _mid_threshold_text(f"J: Rolling Vol (annualized, {window})"), "直近の年率ボラ"),
    ]

    df = pd.DataFrame(rows, columns=["Block", "Metric", "Value", "Threshold", "Note"])
    df["Judgement"] = [_mid_judge(m, v, dev=dev, bench=bench, window=window) for m, v in zip(df["Metric"], df["Value"])]
    df_display = df.copy()
    df_display["Value"] = [_mid_fmt_value(m, v) for m, v in zip(df["Metric"], df["Value"])]
    return df_display


# -------------------------------------------------------
# Bubble decision flow
# -------------------------------------------------------
def bubble_judgement(r2: float, m: float, omega: float) -> tuple[str, dict]:
    info = {
        "r2_ok": bool(np.isfinite(r2) and r2 >= 0.65),
        "m_ok": bool(np.isfinite(m) and (0.25 <= m <= 0.70)),
        "omega_ok": bool(np.isfinite(omega) and (6.0 <= omega <= 13.0)),
        "omega_high": bool(np.isfinite(omega) and omega >= 18.0),
    }
    if not info["r2_ok"]: return "判定保留（LPPL形状が弱い）", info
    if not info["m_ok"]: return "通常の上昇寄り", info
    if info["omega_ok"]: return "バブル的な上昇", info
    if info["omega_high"]: return "バブル“風”（注意：短周期すぎ）", info
    return "バブル“風”（注意）", info

def render_bubble_flow(r2: float, m: float, omega: float):
    verdict, info = bubble_judgement(r2, m, omega)
    def yn(v: bool) -> str: return "YES" if v else "NO"
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

def admin_interpretation_text(bubble_res: dict, end_date: date) -> tuple[str, list[str]]:
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
    
    # 14日早期警戒ルールをテキストにも反映
    WARNING_DAYS = 14
    
    if days_to_tc < 0:
        bullets.append(f"t_c は既に {abs(days_to_tc)} 日前に通過（構造的ピーク通過の可能性）。")
        bullets.append("新規の追随買いは控えめに。保有なら部分利確・ヘッジを検討。")
    elif days_to_tc <= WARNING_DAYS:
        bullets.append(f"t_c まで残り {days_to_tc} 日（危険ゾーン：14日以内）。")
        bullets.append("早期警戒期間に入っています。新規ロングは慎重（サイズ縮小・分割）。利確/ヘッジ準備。")
    else:
        bullets.append(f"t_c まで残り {days_to_tc} 日（安全圏）。")
        
    verdict, _ = bubble_judgement(r2, m, omega)
    bullets.append(f"バブル判定（R²→m→ω）：{verdict}")
    if np.isfinite(r2) and r2 < 0.65:
        bullets.append(f"R²={r2:.2f}：形状が弱め→判定保留寄り。")
    if np.isfinite(m) and m >= 0.85:
        bullets.append(f"m={m:.2f}：上限寄り→境界解の疑い。")
    if np.isfinite(omega) and omega >= 18:
        bullets.append(f"ω={omega:.2f}：短周期すぎ→ノイズ追随の疑い。")
    if np.isfinite(rmse):
        bullets.append(f"RMSE(log)={rmse:.3f}：フィット安定度の目安。")
    if np.isfinite(c_over_b) and c_over_b >= 2.0:
        bullets.append(f"|C/B|={c_over_b:.2f}：振動項が強すぎ→“それっぽいフィット”の疑い。")
    if verdict == "バブル的な上昇":
        summary = "バブル的上昇（m帯域＋R²十分＋ω典型）。追随買いは抑え、利確/ヘッジを織り込む局面。"
    elif verdict.startswith("バブル“風”"):
        summary = "バブル“風”（形はそれっぽいが注意要素あり）。追随買いは慎重、リスク管理を前倒し。"
    elif verdict == "通常の上昇寄り":
        summary = "典型バブルとは言いにくい。ただしtcが近いなら姿勢調整は有効。"
    else:
        summary = "LPPL形状が弱く判定保留。期間変更・他指標との併用推奨。"
    return summary, bullets


# -------------------------------------------------------
# LPPL Model
# -------------------------------------------------------
def lppl(t, A, B, C, m, tc, omega, phi):
    t = np.asarray(t, dtype=float)
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)

def fit_lppl_bubble(price_series: pd.Series):
    """Uptrend fit (robust bounds) - FIXED DATE CALCULATION"""
    price = price_series.values.astype(float)
    t = np.arange(len(price), dtype=float)
    log_price = np.log(price)
    N = len(t)
    
    # Initial guess
    A_init = float(np.mean(log_price))
    p0 = [A_init, -1.0, 0.1, 0.5, N + 20, 8.0, 0.0]
    
    # Bounds
    A_low = float(np.min(log_price) - 2.0)
    A_high = float(np.max(log_price) + 2.0)
    lower_bounds = [A_low, -20, -20, 0.01, N + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N + 250, 25.0, np.pi]
    
    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)
    
    params, _ = curve_fit(lppl, t, log_price, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
    
    log_fit = lppl(t, *params)
    price_fit = np.exp(log_fit)
    
    ss_res = float(np.sum((log_price - log_fit) ** 2))
    ss_tot = float(np.sum((log_price - log_price.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((log_price - log_fit) ** 2)))
    
    # --- FIX: Correct date conversion ---
    tc_days = float(params[4])
    last_idx = N - 1
    future_units = tc_days - last_idx
    last_date = price_series.index[-1]
    tc_date = last_date + timedelta(days=future_units)
    
    A, B, C, m, tc, omega, phi = [float(x) for x in params]
    abs_c_over_b = float(abs(C / B)) if abs(B) > 1e-12 else float("inf")
    log_period = float(2.0 * np.pi / omega) if omega != 0 else float("inf")
    
    bounds_info = {"A_low": A_low, "A_high": A_high, "B_low": -20.0, "B_high": 20.0, "C_low": -20.0, "C_high": 20.0,
                   "m_low": 0.01, "m_high": 0.99, "tc_low": float(N + 1), "tc_high": float(N + 250),
                   "omega_low": 2.0, "omega_high": 25.0, "phi_low": float(-np.pi), "phi_high": float(np.pi)}
                   
    return {
        "params": np.asarray(params, dtype=float), 
        "param_dict": {"A": A, "B": B, "C": C, "m": m, "tc": tc, "omega": omega, "phi": phi, "abs_C_over_B": abs_c_over_b, "log_period_2pi_over_omega": log_period}, 
        "price_fit": price_fit, 
        "log_fit": log_fit, 
        "r2": float(r2), 
        "rmse": rmse, 
        "tc_date": tc_date, 
        "tc_days": tc_days, 
        "bounds_info": bounds_info, 
        "N": int(N)
    }

def fit_lppl_negative_bubble(price_series: pd.Series, peak_date, min_points: int = 10, min_drop_ratio: float = 0.03):
    """Downtrend fit (negative bubble) - FIXED DATE CALCULATION"""
    down_series = price_series[price_series.index >= peak_date].copy()
    if len(down_series) < min_points: return {"ok": False}
    
    peak_price = float(price_series.loc[peak_date])
    last_price = float(down_series.iloc[-1])
    drop_ratio = (peak_price - last_price) / peak_price
    if drop_ratio < min_drop_ratio: return {"ok": False}
    
    price_down = down_series.values.astype(float)
    t_down = np.arange(len(price_down), dtype=float)
    log_down = np.log(price_down)
    neg_log_down = -log_down
    N_down = len(t_down)
    
    A_init = float(np.mean(neg_log_down))
    p0 = [A_init, -1.0, 0.1, 0.5, N_down + 15, 8.0, 0.0]
    
    A_low = float(np.min(neg_log_down) - 2.0)
    A_high = float(np.max(neg_log_down) + 2.0)
    lower_bounds = [A_low, -20, -20, 0.01, N_down + 1, 2.0, -np.pi]
    upper_bounds = [A_high, 20, 20, 0.99, N_down + 200, 25.0, np.pi]
    
    p0 = _clamp_p0_into_bounds(p0, lower_bounds, upper_bounds)
    
    try:
        params_down, _ = curve_fit(lppl, t_down, neg_log_down, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=20000)
    except Exception:
        return {"ok": False}
        
    neg_log_fit = lppl(t_down, *params_down)
    log_fit = -neg_log_fit
    price_fit_down = np.exp(log_fit)
    
    ss_res = np.sum((neg_log_down - neg_log_fit) ** 2)
    ss_tot = np.sum((neg_log_down - neg_log_down.mean()) ** 2)
    r2_down = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # --- FIX: Correct date conversion for negative bubble ---
    tc_days = float(params_down[4])
    last_idx = N_down - 1
    future_units = tc_days - last_idx
    last_date = down_series.index[-1]
    tc_bottom_date = last_date + timedelta(days=future_units)
    
    return {
        "ok": True, 
        "down_series": down_series, 
        "price_fit_down": price_fit_down, 
        "r2": float(r2_down), 
        "tc_date": tc_bottom_date, 
        "tc_days": tc_days, 
        "params": np.asarray(params_down, dtype=float)
    }


# -------------------------------------------------------
# Cache helpers
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
def fit_lppl_negative_bubble_cached(price_key: str, price_values: np.ndarray, idx_int: np.ndarray, peak_date_int: int, min_points: int, min_drop_ratio: float):
    idx = pd.to_datetime(idx_int)
    s = pd.Series(price_values, index=idx)
    peak_date = pd.to_datetime(peak_date_int)
    return fit_lppl_negative_bubble(s, peak_date=peak_date, min_points=min_points, min_drop_ratio=min_drop_ratio)


# =======================================================
# (FIXED) ROBUST SCRAPING FUNCTION: Multi-Source + IRBank + Yahoo
# =======================================================
@st.cache_data(ttl=SCRAPE_TTL_SECONDS, show_spinner=False)
def fetch_jp_margin_data_robust(ticker: str) -> pd.DataFrame:
    """
    複数ソース（IRBank -> Minkabu -> Karauri -> Yahoo -> Kabutan）を順に試し、
    信用残（MarginBuy/MarginSell）と逆日歩（Hibush）を取得する。
    """
    code = ticker.replace(".T", "").strip()
    if not code.isdigit():
        return pd.DataFrame()

    # アクセス偽装用のヘッダー
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
        "Referer": "https://www.google.com/"
    }

    # ----- 共通: 数値クリーニング関数 -----
    def clean_num(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('株', '').replace('円', '').replace('倍', '').replace('%', '').replace('-', '0').strip()
            if not x: return 0
            try: return float(x)
            except: return 0
        return x

    # ----- 共通: 日付クリーニング関数 -----
    def parse_date_col(df, col_name="Date"):
        dates = pd.to_datetime(df[col_name], errors='coerce')
        if dates.isna().any():
            today = date.today()
            current_year = today.year
            def try_parse(x):
                if pd.isna(x) or str(x).strip() == "": return pd.NaT
                x_clean = str(x).split(' ')[0]
                x_clean = x_clean.replace("年", "/").replace("月", "/").replace("日", "")
                try: return pd.to_datetime(x_clean)
                except: pass
                try:
                    dt = pd.to_datetime(f"{current_year}/{x_clean}")
                    if dt.date() > today + timedelta(days=2):
                        dt = dt.replace(year=current_year - 1)
                    return dt
                except: return pd.NaT
            dates = df[col_name].apply(try_parse)
        return dates

    # ----- 1. IRBank Scraper (Priority) -----
    def _scrape_irbank():
        urls = [f"https://irbank.net/{code}/fee", f"https://irbank.net/{code}/karauri"]
        for url in urls:
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code != 200: continue
                try: dfs = pd.read_html(io.BytesIO(res.content))
                except: continue
                target = pd.DataFrame()
                for df in dfs:
                    col_str = " ".join([str(c) for c in df.columns])
                    if "日付" in col_str and ("逆日歩" in col_str or "融資" in col_str or "貸株" in col_str or "残" in col_str):
                        target = df
                        if isinstance(target.columns, pd.MultiIndex):
                            target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                        else:
                            target.columns = [str(c).strip() for c in target.columns]
                        break
                if target.empty: continue
                rename_map = {}
                for c in target.columns:
                    if "日付" in c: rename_map[c] = "Date"
                    elif "逆日歩" in c: rename_map[c] = "Hibush"
                    elif "融資" in c and "残" in c: rename_map[c] = "MarginBuy"
                    elif "貸株" in c and "残" in c: rename_map[c] = "MarginSell"
                    elif "売残" in c and "信用" not in c: rename_map[c] = "MarginSell"
                    elif "買残" in c and "信用" not in c: rename_map[c] = "MarginBuy"
                target = target.rename(columns=rename_map)
                if "Date" not in target.columns: continue
                target["Date"] = parse_date_col(target, "Date")
                for col in ["MarginBuy", "MarginSell", "Hibush"]:
                    if col in target.columns: target[col] = target[col].apply(clean_num)
                df_clean = target.dropna(subset=["Date"]).sort_values("Date")
                if not df_clean.empty and ("MarginBuy" in df_clean.columns or "MarginSell" in df_clean.columns):
                    return df_clean
            except: continue
        return pd.DataFrame()

    # ----- 2. Minkabu Scraper -----
    def _scrape_minkabu():
        url = f"https://minkabu.jp/stock/{code}/margin"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and ("融資" in col_str or "貸株" in col_str or "残" in col_str):
                    target = df
                    if isinstance(target.columns, pd.MultiIndex):
                        target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                    else:
                        target.columns = [str(c).strip() for c in target.columns]
                    if "逆日歩" in col_str or "品貸料" in col_str: break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "逆日歩" in c or "品貸料" in c: rename_map[c] = "Hibush"
                elif ("融資" in c and "残" in c) or ("買残" in c) or ("買い" in c and "残" in c):
                    if "増減" not in c and "回転" not in c: rename_map[c] = "MarginBuy"
                elif ("貸株" in c and "残" in c) or ("売残" in c) or ("売り" in c and "残" in c):
                    if "増減" not in c and "回転" not in c: rename_map[c] = "MarginSell"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell", "Hibush"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 3. Karauri.net Scraper -----
    def _scrape_karauri():
        url = f"https://karauri.net/{code}/"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and ("売残" in col_str or "貸株" in col_str):
                    target = df
                    if isinstance(target.columns, pd.MultiIndex):
                        target.columns = ['_'.join(map(str, c)).strip() for c in target.columns]
                    else:
                        target.columns = [str(c).strip() for c in target.columns]
                    if "逆日歩" in col_str: break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "逆日歩" in c: rename_map[c] = "Hibush"
                elif ("融資" in c and "残" in c) or ("買残" in c):
                    if "増減" not in c: rename_map[c] = "MarginBuy"
                elif ("貸株" in c and "残" in c) or ("売残" in c):
                    if "増減" not in c: rename_map[c] = "MarginSell"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell", "Hibush"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 4. Yahoo Finance JP Scraper (Backup) -----
    def _scrape_yahoo():
        url = f"https://finance.yahoo.co.jp/quote/{code}.T/margin"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and "売り残" in col_str and "買い残" in col_str:
                    target = df
                    target.columns = [str(c).strip() for c in target.columns]
                    break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "売り残" in c and "増減" not in c: rename_map[c] = "MarginSell"
                elif "買い残" in c and "増減" not in c: rename_map[c] = "MarginBuy"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            target["Hibush"] = 0
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # ----- 5. Kabutan Scraper (Final Fallback) -----
    def _scrape_kabutan():
        url = f"https://kabutan.jp/stock/finance?code={code}&mode=m"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code != 200: return pd.DataFrame()
            try: dfs = pd.read_html(io.BytesIO(res.content))
            except: return pd.DataFrame()
            target = pd.DataFrame()
            for df in dfs:
                col_str = " ".join([str(c) for c in df.columns])
                if "日付" in col_str and "売残" in col_str and "買残" in col_str:
                    target = df
                    target.columns = [str(c).strip() for c in target.columns]
                    break
            if target.empty: return pd.DataFrame()
            rename_map = {}
            for c in target.columns:
                if "日付" in c: rename_map[c] = "Date"
                elif "売残" in c and "増減" not in c: rename_map[c] = "MarginSell"
                elif "買残" in c and "増減" not in c: rename_map[c] = "MarginBuy"
            target = target.rename(columns=rename_map)
            if "Date" not in target.columns: return pd.DataFrame()
            target["Date"] = parse_date_col(target, "Date")
            for col in ["MarginBuy", "MarginSell"]:
                if col in target.columns: target[col] = target[col].apply(clean_num)
            target["Hibush"] = 0
            return target.dropna(subset=["Date"]).sort_values("Date")
        except: return pd.DataFrame()

    # --- EXECUTION FLOW ---
    df = _scrape_irbank()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
    
    df = _scrape_minkabu()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
    
    df = _scrape_karauri()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df

    df = _scrape_yahoo()
    if not df.empty and ("MarginBuy" in df.columns or "MarginSell" in df.columns): return df
        
    df = _scrape_kabutan()
    return df


# =======================================================
# FINAL SCORE (UPDATED WITH 14-DAY WARNING RULE)
# =======================================================
def compute_signal_and_score(tc_up_date, end_date, down_tc_date) -> tuple[str, int]:
    now = pd.Timestamp(end_date).normalize()
    tc_up = pd.Timestamp(tc_up_date).normalize()
    
    # 優先度1: 下落トレンド（High Risk）
    if down_tc_date is not None:
        down_tc = pd.Timestamp(down_tc_date).normalize()
        delta = (down_tc - now).days
        if delta > 0:
            s = _lin_map(delta, DOWN_FUTURE_NEAR_DAYS, DOWN_FUTURE_FAR_DAYS, 90, 80)
            return ("HIGH", int(round(_clamp(s, 80, 90))))
        past = abs(delta)
        s = _lin_map(past, DOWN_PAST_NEAR_DAYS, DOWN_PAST_FAR_DAYS, 90, 100)
        return ("HIGH", int(round(_clamp(s, 90, 100))))
    
    # 優先度2: 上昇トレンド（Safe/Caution判定）
    # 【ロジック変更】tcまで「残り14日」を切ったらCAUTION（黄色）にする
    gap = (tc_up - now).days
    WARNING_BUFFER = 14  # 早期警戒ライン（14日）

    # SAFE: 14日より未来
    if gap > WARNING_BUFFER:
        s = _lin_map(gap, UP_FUTURE_NEAR_DAYS, UP_FUTURE_FAR_DAYS, 59, 0)
        return ("SAFE", int(round(_clamp(s, 0, 59))))

    # CAUTION: 14日以内、当日、または過去
    past_warning = WARNING_BUFFER - gap
    
    s = _lin_map(past_warning, 0, UP_PAST_FAR_DAYS, 60, 79)
    return ("CAUTION", int(round(_clamp(s, 60, 79))))


# =======================================================
# Render Graph Pack
# =======================================================
def draw_score_overlay(ax, score: int, label: str):
    """
    MatplotlibのAxes上にScoreとSignalを描画する
    """
    # 信号色定義
    if label == "HIGH":
        badge_bg = "#ff4d4f"; badge_fg = "white"
    elif label == "CAUTION":
        badge_bg = "#ffc53d"; badge_fg = "black"
    else: # SAFE
        badge_bg = "#52c41a"; badge_fg = "white"

    # 1. "Score" Label (薄いグレー)
    ax.text(0.04, 0.92, "Score", transform=ax.transAxes,
            fontsize=12, color='#aaaaaa', fontweight='normal', zorder=20)
            
    # 2. Score Value (白、大きく)
    ax.text(0.04, 0.83, str(score), transform=ax.transAxes,
            fontsize=32, color='white', fontweight='bold', zorder=20)
            
    # 3. Signal Badge (Scoreの横に配置)
    ax.text(0.18, 0.85, f" {label} ", transform=ax.transAxes,
            fontsize=10, color=badge_fg, fontweight='bold',
            bbox=dict(facecolor=badge_bg, edgecolor='none', boxstyle='round,pad=0.4', alpha=0.95),
            zorder=20, verticalalignment='bottom')

    # 4. 背景パネル (読みやすくするための半透明の黒背景)
    rect = patches.FancyBboxPatch(
        (0.02, 0.81), width=0.28, height=0.16,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        facecolor="#000000", alpha=0.6,
        edgecolor="#333333", linewidth=1,
        zorder=15
    )
    ax.add_patch(rect)


def draw_logo_overlay(ax):
    """
    ロゴ（画像）がないため、テキスト描画でロゴを再現する。
    左下に 'OUT-STANDER' をSerifフォントで配置。
    ★視認性向上のために「縁取り（Outline）」を追加。
    """
    # ロゴ風テキスト (Serif Font, Gold color)
    text_obj = ax.text(0.02, 0.03, "OUT-STANDER", transform=ax.transAxes,
            fontsize=16, color='#e5c07b', fontweight='bold',
            fontname='serif', zorder=20, alpha=0.9)
    
    # 縁取り（黒いアウトライン）を追加して、線と重なっても読めるようにする
    text_obj.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='black'),
        path_effects.Normal()
    ])


def render_graph_pack_from_prices(prices, ticker, bench, window=20, trading_days=252):
    base = prices.iloc[0]
    index100 = prices / base * 100.0
    dev = index100 - 100.0
    ret = np.log(prices / prices.shift(1)).dropna()

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
    
    # 日本語フォント設定があれば適用
    if JP_FONT:
        ax.legend(facecolor="#0b0c0e", labelcolor="white", prop=JP_FONT)
        ax.set_title("Cumulative Performance (Index = 100)", color="white", fontproperties=JP_FONT)
    else:
        ax.legend(facecolor="#0b0c0e", labelcolor="white")
        
    st.pyplot(fig)

    X = dev[bench].dropna(); Y = dev[ticker].dropna()
    common = X.index.intersection(Y.index); X = X.loc[common]; Y = Y.loc[common]
    slope_dev, intercept_dev = np.polyfit(X.values, Y.values, 1)
    x_sorted = X.sort_values(); y_line = slope_dev * x_sorted + intercept_dev

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
    
    if JP_FONT:
        ax.legend(facecolor="#0b0c0e", labelcolor="white", prop=JP_FONT)
    else:
        ax.legend(facecolor="#0b0c0e", labelcolor="white")
    st.pyplot(fig)

    p = prices[ticker]; running_max = p.cummax(); dd = (p / running_max) - 1.0
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

    Xr = ret[bench].dropna(); Yr = ret[ticker].dropna()
    common_r = Xr.index.intersection(Yr.index); Xr = Xr.loc[common_r]; Yr = Yr.loc[common_r]
    slope_ret, intercept_ret = np.polyfit(Xr.values, Yr.values, 1)
    xr_sorted = Xr.sort_values(); yr_line = slope_ret * xr_sorted + intercept_ret
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

    # =======================================================
    # (ADD) JAPANESE MARGIN CHART (Dual Axis)
    # =======================================================
    if ticker.endswith(".T"):
        st.markdown("---")
        st.subheader(f"■ 信用残・逆日歩（日証金データ: {ticker}）")
        
        with st.spinner("日証金データを取得中（IRBank/Minkabu/Karauri/Yahoo/Kabutan）..."):
            margin_df = fetch_jp_margin_data_robust(ticker)
            
        if not margin_df.empty:
            # グラフ描画
            fig, ax1 = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor("#0b0c0e")
            ax1.set_facecolor("#0b0c0e")
            
            # X軸（日付）
            dates = margin_df["Date"]
            
            # 左軸：信用残（面グラフ）
            if "MarginBuy" in margin_df.columns and "MarginSell" in margin_df.columns:
                ax1.fill_between(dates, margin_df["MarginBuy"], color="#4169E1", alpha=0.3, label="融資残（Buying）")
                ax1.plot(dates, margin_df["MarginBuy"], color="#4169E1", linewidth=1.5)
                
                ax1.fill_between(dates, margin_df["MarginSell"], color="#CD5C5C", alpha=0.3, label="貸株残（Selling）")
                ax1.plot(dates, margin_df["MarginSell"], color="#CD5C5C", linewidth=1.5)
                
                ax1.set_ylabel("信用残高（株）", color="white")
                ax1.tick_params(axis='y', colors="white")
                ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            ax1.set_xlabel("Date", color="white")
            ax1.tick_params(axis='x', colors="white")
            ax1.grid(color="#333333", linestyle="--", alpha=0.5)

            # 右軸：逆日歩（棒グラフ）
            has_hibu = "Hibush" in margin_df.columns and margin_df["Hibush"].sum() > 0
            
            if has_hibu:
                ax2 = ax1.twinx()
                colors = ["#FFD700" if v > 0 else "#333333" for v in margin_df["Hibush"]]
                ax2.bar(dates, margin_df["Hibush"], color=colors, alpha=0.6, width=0.6, label="逆日歩")
                ax2.set_ylabel("逆日歩（円）", color="#FFD700")
                ax2.tick_params(axis='y', colors="#FFD700")
                
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                
                if JP_FONT:
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor="#0b0c0e", labelcolor="white", prop=JP_FONT)
                else:
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", facecolor="#0b0c0e", labelcolor="white")
            else:
                if JP_FONT:
                    ax1.legend(loc="upper left", facecolor="#0b0c0e", labelcolor="white", prop=JP_FONT)
                else:
                    ax1.legend(loc="upper left", facecolor="#0b0c0e", labelcolor="white")

            st.pyplot(fig)
            st.caption("※データソース: IRBank / Minkabu / Karauri / Yahoo / Kabutan (自動切替)。直近のデータのみ表示されます。")
            
            with st.expander("詳細データを見る"):
                st.dataframe(margin_df.sort_values("Date", ascending=False), use_container_width=True)
        else:
            st.warning("日証金データが取得できませんでした（全ソース巡回後も不可）。")


# -------------------------------------------------------
# Streamlit app (ADMIN)
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Out-stander Admin", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0c0e !important; color: #ffffff !important; }
        div[data-testid="stMarkdownContainer"] p, label { color: #ffffff !important; }
        input.st-ai, input.st-ah, div[data-baseweb="input"] { background-color: #1a1c1f !important; color: #ffffff !important; border-color: #444 !important; }
        input { color: #ffffff !important; }
        input::placeholder { color: rgba(255,255,255,0.45) !important; }
        div[data-baseweb="input"] svg { fill: #ffffff !important; }
        div[data-testid="stFormSubmitButton"] button { background-color: #222428 !important; color: #ffffff !important; border: 1px solid #555 !important; }
        div[data-testid="stFormSubmitButton"] button:hover { background-color: #444 !important; border-color: #888 !important; color: #ffffff !important; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
        .custom-error { background-color: #141518; border: 1px solid #2a2c30; border-radius: 12px; padding: 14px 18px; color: #ffffff; font-size: 0.95rem; margin-top: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def show_error_black(msg: str):
        st.markdown(f"""<div class="custom-error">{msg}</div>""", unsafe_allow_html=True)

    st.markdown("## Out-stander（管理者用）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("input_form"):
        ticker = st.text_input("Ticker", value="", placeholder="例: NVDA / 0700.HK / 7203.T")
        today = date.today()
        default_start = today - timedelta(days=220)
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Start", default_start)
        with col2: end_date = st.date_input("End", today)
        submitted = st.form_submit_button("Run")

    if not submitted: st.stop()
    if not ticker.strip():
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

    try:
        price_series = fetch_price_series_cached(ticker.strip(), start_date, end_date)
    except Exception:
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

    if len(price_series) < 30:
        show_error_black("Tickerが無効、または価格データが取得できません。")
        st.stop()

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
    neg_res = fit_lppl_negative_bubble_cached(key, vals, idx_int, peak_date_int=peak_date_int, min_points=10, min_drop_ratio=0.03)

    tc_up_date = pd.Timestamp(bubble_res["tc_date"])
    end_ts = pd.Timestamp(end_date)
    down_tc_date = pd.Timestamp(neg_res["tc_date"]) if neg_res.get("ok") else None

    signal_label, score = compute_signal_and_score(tc_up_date, end_ts, down_tc_date)

    # -----------------------------------------------------------
    # MAIN CHART RENDER with OVERLAY (Modified)
    # -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    
    # Plot Data
    ax.plot(price_series.index, price_series.values, color="gray", label=ticker.strip())
    ax.plot(price_series.index, bubble_res["price_fit"], color="orange", label="上昇モデル")
    ax.axvline(bubble_res["tc_date"], color="red", linestyle="--", label=f"t_c（上昇） {pd.Timestamp(bubble_res['tc_date']).date()}")
    ax.axvline(peak_date, color="white", linestyle=":", label=f"ピーク {pd.Timestamp(peak_date).date()}")
    
    if neg_res.get("ok"):
        down = neg_res["down_series"]
        ax.plot(down.index, down.values, color="cyan", label="下落（ピーク以降）")
        ax.plot(down.index, neg_res["price_fit_down"], "--", color="green", label="下落モデル")
        ax.axvline(neg_res["tc_date"], color="green", linestyle="--", label=f"t_c（下落） {pd.Timestamp(neg_res['tc_date']).date()}")
    
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price (Adj Close preferred)", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333")
    
    # Legend
    # 日本語フォントがあれば適用
    if JP_FONT:
        ax.legend(facecolor="#0b0c0e", labelcolor="white", loc='lower right', prop=JP_FONT)
    else:
        ax.legend(facecolor="#0b0c0e", labelcolor="white", loc='lower right')

    # ★ ADD OVERLAY HERE (Score & Signal on Chart)
    draw_score_overlay(ax, score, signal_label)
    
    # ★ ADD LOGO HERE (Text simulation)
    draw_logo_overlay(ax)

    st.pyplot(fig)
    # -----------------------------------------------------------

    if signal_label == "HIGH":
        risk_label = "High"; risk_color = "#ff4d4f"
    elif signal_label == "CAUTION":
        risk_label = "Caution"; risk_color = "#ffc53d"
    else:
        risk_label = "Safe"; risk_color = "#52c41a"

    # HTML Card (Backup view below chart)
    col_score, col_gain = st.columns(2)
    with col_score:
        score_card_html = f"""<div style="background-color: #141518; border: 1px solid #2a2c30; border-radius: 12px; padding: 18px 20px 16px 20px; margin-top: 8px;">
            <div style="font-size: 0.85rem; color: #a0a2a8; margin-bottom: 6px;">スコア詳細（カード表示）</div>
            <div style="display: flex; align-items: baseline; gap: 12px;">
                <div style="font-size: 40px; font-weight: 700; color: #f5f5f5;">{score}</div>
                <div style="padding: 2px 10px; border-radius: 999px; background-color: {risk_color}33; color: {risk_color}; font-size: 0.85rem; font-weight: 600;">{risk_label}</div>
            </div></div>"""
        st.markdown(score_card_html, unsafe_allow_html=True)
    with col_gain:
        gain_card_html = f"""<div style="background-color: #141518; border: 1px solid #2a2c30; border-radius: 12px; padding: 18px 20px 16px 20px; margin-top: 8px;">
            <div style="font-size: 0.85rem; color: #a0a2a8; margin-bottom: 6px;">上昇倍率（開始→ピーク）</div>
            <div style="font-size: 36px; font-weight: 700; color: #f5f5f5; line-height: 1.1;">{gain:.2f}x</div>
            <div style="margin-top: 6px; display: inline-block; padding: 2px 10px; border-radius: 999px; background-color: #102915; color: #52c41a; font-size: 0.85rem; font-weight: 500;">{gain_pct:+.1f}%</div>
        </div>"""
        st.markdown(gain_card_html, unsafe_allow_html=True)

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
        ["① R²（対数空間）", r2, "バブル判定の条件：R²≥0.65。"],
        ["② m（最重要）", m, "バブル的：m∈[0.25,0.70]（典型0.3〜0.6）。"],
        ["③ ω", omega, "バブル的：ω∈[6,13]。ω≥18は短周期すぎ→バブル“風”（注意）。"],
        ["結論（バブル判定）", verdict, "上の判定フローの最終結論。"],
        ["t_c（日付・近似）", str(tc_up_norm.date()), "tcが近い/通過＝終盤の可能性。"],
        ["t_cまでの残日数", days_to_tc, "終盤度（短い/通過ほど終盤リスク↑）。"],
        ["RMSE（対数空間）", rmse, "フィットの安定度（小さいほど安定）。"],
        ["|C/B|", c_over_b, "≥2.0は“それっぽいフィット”の疑いが増える。"],
        ["2π/ω", log_period, "周期の別表現。"],
        ["フィット期間 N", N, "期間N。"]
    ]
    st.dataframe(pd.DataFrame(admin_rows, columns=["指標", "値", "解説"]), use_container_width=True)
    st.markdown("#### 生パラメータ（A, B, C, m, tc, ω, φ）")
    raw_params = bubble_res.get("params", np.array([], dtype=float)).astype(float)
    if raw_params.size == 7:
        raw_df = pd.DataFrame([raw_params], columns=["A", "B", "C", "m", "tc", "omega", "phi"])
    else:
        raw_df = pd.DataFrame([raw_params])
    st.dataframe(raw_df, use_container_width=True)
    st.markdown("#### フィット品質フラグ（簡易サニティチェック）")
    binfo = bubble_res.get("bounds_info", {})
    if raw_params.size == 7:
        A_, B_, C_, m_, tc_, omega_, phi_ = [float(x) for x in raw_params]
    else:
        A_, B_, C_, m_, tc_, omega_, phi_ = [np.nan]*7
    def near_bound(x, lo, hi, tol=0.02):
        if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi == lo: return False
        r = (x - lo) / (hi - lo)
        return (r < tol) or (r > 1 - tol)
    flags = []
    if np.isfinite(m_) and (m_ < 0.2 or m_ > 0.8): flags.append("m が一般的な“バブル的帯域”から外れています。")
    if near_bound(m_, binfo.get("m_low", np.nan), binfo.get("m_high", np.nan)): flags.append("m が境界付近です（境界解の疑い）。")
    if near_bound(tc_, binfo.get("tc_low", np.nan), binfo.get("tc_high", np.nan)): flags.append("tc が境界付近です。")
    if near_bound(omega_, binfo.get("omega_low", np.nan), binfo.get("omega_high", np.nan)): flags.append("ω が境界付近です。")
    if np.isfinite(c_over_b) and c_over_b > 2.0: flags.append("|C/B| が大きいです（振動過多）。")
    if flags: st.write(pd.DataFrame({"フラグ": flags}))
    else: st.write("単純なヒューリスティックでは明確な赤信号は見当たりません。")

    st.markdown("### 投資判断向けの解釈（管理者のみ）")
    summary, bullets = admin_interpretation_text(bubble_res, end_date)
    with st.expander("投資判断向けの解釈メモ（管理者のみ）", expanded=True):
        st.markdown(f"**要約**：{summary}")
        st.markdown("**ポイント**")
        st.markdown("\n".join([f"- {b}" for b in bullets]))
        st.markdown("---")
        st.markdown("**使い方（実務）**")
        st.markdown("- バブル判定は **R²（信用）×m（形）×ω（自然さ）** の3点で判断。\n- tc は『バブルかどうか』ではなく『終盤かどうか』。")

    st.markdown("---")
    with st.expander("Mid-term Quant 判定表（A〜J + Threshold + Judgement）", expanded=False):
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1: bench = st.text_input("Benchmark（指数）", value="ACWI", help="例: ACWI / ^GSPC / ^N225")
        with col_b2: mid_window = st.number_input("WINDOW（判定表）", min_value=5, max_value=120, value=20, step=1)
        try:
            df_mid = build_midterm_quant_table(ticker=ticker.strip(), bench=bench.strip(), start_date=start_date, end_date=end_date, window=int(mid_window))
            st.dataframe(df_mid[["Block", "Metric", "Value", "Threshold", "Judgement", "Note"]], use_container_width=True, hide_index=True)
        except Exception as e:
            show_error_black(f"判定表の生成に失敗しました: {e}")

    st.markdown("---")
    with st.expander("グラフ（Index/Deviation/Regression/Vol/Drawdown）", expanded=False):
        gcol1, gcol2, gcol3 = st.columns([2, 1, 1])
        with gcol1: graph_bench = st.text_input("Benchmark（グラフ用）", value="ACWI")
        with gcol2: graph_window = st.number_input("WINDOW（グラフ用）", min_value=5, max_value=120, value=20, step=1)
        with gcol3: trading_days = st.number_input("Trading days/year", min_value=200, max_value=365, value=252, step=1)
        try:
            prices_pair = fetch_prices_pair_cached(ticker=ticker.strip(), bench=graph_bench.strip(), start_date=start_date, end_date=end_date)
            st.caption(f"Data: {prices_pair.index.min().date()} to {prices_pair.index.max().date()} | rows={len(prices_pair)}")
            render_graph_pack_from_prices(prices=prices_pair, ticker=ticker.strip(), bench=graph_bench.strip(), window=int(graph_window), trading_days=int(trading_days))
        except Exception as e:
            show_error_black(f"グラフ表示に失敗しました: {e}")

if __name__ == "__main__":
    main()
