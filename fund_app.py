# fund_app.py
import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from auth_gate import require_admin_token

# =========================
# SEC settings
# =========================
# SECはUser-Agent必須（連絡先を含むのが推奨）
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip() or "YourName your.email@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Ticker->CIK 対応表（SEC公式JSON）
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


def _to_musd(x: float) -> float:
    """USDを百万USDに変換（表示用）"""
    return x / 1_000_000.0


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def fetch_ticker_cik_map() -> dict:
    """
    SEC公式の ticker -> cik を取得
    返り値: {"aapl": "0000320193", ...}
    """
    r = requests.get(TICKER_CIK_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()

    out = {}
    # company_tickers.json は { "0": {"cik_str":..., "ticker":..., "title":...}, "1": ... } 形式
    for _, v in data.items():
        t = str(v.get("ticker", "")).strip().lower()
        cik = str(v.get("cik_str", "")).strip()
        if t and cik.isdigit():
            out[t] = cik.zfill(10)
    return out


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_company_facts(cik10: str) -> dict:
    """
    SEC Company Facts (XBRL) を取得
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _extract_fy_series_usd(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    """
    us-gaap の tag から FY (年次) の USD を抽出
    戻り: columns=["fy","end","val","form"]
    """
    us = facts_json.get("facts", {}).get("us-gaap", {})
    node = us.get(xbrl_tag, {}).get("units", {}).get("USD", [])
    rows = []
    for x in node:
        # fp == FY を年次として採用。10-Kが多い
        if x.get("fp") == "FY" and x.get("fy") and x.get("end") and x.get("val") is not None:
            rows.append(
                {
                    "fy": int(x["fy"]),
                    "end": pd.to_datetime(x["end"]),
                    "val": _safe_float(x["val"]),
                    "form": str(x.get("form", "")),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["fy", "end", "val", "form"])

    df = pd.DataFrame(rows).dropna(subset=["val"]).sort_values("fy")
    # FYが重複することがあるので、最新（後に出た）を優先して1つに
    df = df.drop_duplicates(subset=["fy"], keep="last")
    return df


def _pick_first_available_tag(facts_json: dict, candidates: list[str]) -> tuple[str | None, pd.DataFrame]:
    """
    複数候補タグのうち、データが取れた最初のタグを採用
    """
    for tag in candidates:
        df = _extract_fy_series_usd(facts_json, tag)
        if not df.empty:
            return tag, df
    return None, pd.DataFrame(columns=["fy", "end", "val", "form"])


def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    """
    年次PL（売上・営業利益・税前利益・純利益）を作る
    戻り: (table_df, meta)
    table_df columns:
      ["FY","End","Revenue(M$)","OpIncome(M$)","Pretax(M$)","NetIncome(M$)"]
    """
    # タグ候補（企業差を吸収するため複数持つ）
    revenue_tags = ["Revenues", "SalesRevenueNet"]
    op_income_tags = ["OperatingIncomeLoss"]
    pretax_tags = ["IncomeBeforeIncomeTaxes", "PretaxIncome", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems"]
    net_income_tags = ["NetIncomeLoss"]

    meta = {}

    tag, df_rev = _pick_first_available_tag(facts_json, revenue_tags)
    meta["revenue_tag"] = tag

    tag, df_op = _pick_first_available_tag(facts_json, op_income_tags)
    meta["op_income_tag"] = tag

    tag, df_pt = _pick_first_available_tag(facts_json, pretax_tags)
    meta["pretax_tag"] = tag

    tag, df_ni = _pick_first_available_tag(facts_json, net_income_tags)
    meta["net_income_tag"] = tag

    # FYで結合（存在する年だけ揃える：MVPは内積でOK）
    df = df_rev[["fy", "end", "val"]].rename(columns={"val": "revenue"})
    for name, d in [
        ("op_income", df_op),
        ("pretax", df_pt),
        ("net_income", df_ni),
    ]:
        if d.empty:
            df[name] = np.nan
        else:
            df = df.merge(d[["fy", "val"]].rename(columns={"val": name}), on="fy", how="left")

    # 表示用整形
    df = df.sort_values("fy")
    out = pd.DataFrame(
        {
            "FY": df["fy"].astype(int),
            "End": df["end"].dt.date.astype(str),
            "Revenue(M$)": df["revenue"].map(_to_musd),
            "OpIncome(M$)": df["op_income"].map(_to_musd),
            "Pretax(M$)": df["pretax"].map(_to_musd),
            "NetIncome(M$)": df["net_income"].map(_to_musd),
        }
    )

    return out, meta


def plot_pl_annual(table: pd.DataFrame, title: str):
    """
    売上（棒・右軸）＋利益（折れ線・左軸）
    """
    df = table.copy()

    # 末尾N年だけ見たい場合はここで絞れる（まずは全表示）
    x = df["FY"].astype(str).tolist()

    revenue = df["Revenue(M$)"].to_numpy(dtype=float)
    op = df["OpIncome(M$)"].to_numpy(dtype=float)
    pt = df["Pretax(M$)"].to_numpy(dtype=float)
    ni = df["NetIncome(M$)"].to_numpy(dtype=float)

    fig, ax_left = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax_left.set_facecolor("#0b0c0e")

    # 利益：折れ線（左軸）
    ax_left.plot(x, op, label="Operating Income", marker="o")
    ax_left.plot(x, pt, label="Pretax Income", marker="o")
    ax_left.plot(x, ni, label="Net Income", marker="o")
    ax_left.set_ylabel("Profit (Million USD)", color="white")
    ax_left.tick_params(colors="white")
    ax_left.grid(color="#333333")
    ax_left.set_title(title, color="white")

    # 売上：棒（右軸）
    ax_right = ax_left.twinx()
    ax_right.bar(x, revenue, alpha=0.25, label="Revenue")
    ax_right.set_ylabel("Revenue (Million USD)", color="white")
    ax_right.tick_params(colors="white")

    # 凡例（左だけでもOKだが分かりやすく）
    ax_left.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")

    st.pyplot(fig)


def render(authed_email: str):
    st.set_page_config(page_title="Fundamentals (Staging)", layout="wide")

    # 既存のadmin_app.pyに寄せた簡易ダーク背景（必要なら拡張）
    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0c0e !important; color: #ffffff !important; }
        div[data-testid="stMarkdownContainer"] p, label { color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 長期ファンダ（年次PL / 最小実装）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("fund_form"):
        ticker = st.text_input("Ticker（米国株）", value="AVGO", placeholder="例: AAPL / NVDA / AVGO")
        years = st.slider("表示する年数（末尾N年）", min_value=3, max_value=15, value=10)
        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    t = ticker.strip().lower()
    if not t:
        st.error("Tickerを入力してください。")
        st.stop()

    # ticker -> cik
    try:
        m = fetch_ticker_cik_map()
        cik10 = m.get(t)
        if not cik10:
            st.error("このTickerのCIKが見つかりませんでした（米国株のTickerか確認してください）。")
            st.stop()
    except Exception as e:
        st.error(f"Ticker→CIKの取得に失敗しました: {e}")
        st.stop()

    # company facts
    try:
        facts = fetch_company_facts(cik10)
        company_name = facts.get("entityName", ticker.upper())
    except Exception as e:
        st.error(f"SECデータ取得に失敗しました: {e}")
        st.stop()

    # build table
    table, meta = build_pl_annual_table(facts)

    if table.empty:
        st.error("年次PLデータが取得できませんでした（Revenueタグが見つからない等）。")
        st.write("debug:", meta)
        st.stop()

    # 末尾N年に絞る
    table_disp = table.tail(int(years)).reset_index(drop=True)

    # Plot
    plot_pl_annual(table_disp, title=f"{company_name} ({ticker.upper()}) - Income Statement (Annual)")

    # Table
    st.markdown("### 年次PL（百万USD）")
    st.dataframe(
        table_disp.style.format(
            {
                "Revenue(M$)": "{:,.0f}",
                "OpIncome(M$)": "{:,.0f}",
                "Pretax(M$)": "{:,.0f}",
                "NetIncome(M$)": "{:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # メタ情報（使ってるタグ）
    with st.expander("デバッグ情報（採用したXBRLタグ）", expanded=False):
        st.write(meta)
        st.write(f"CIK: {cik10}")
        st.write(f"SEC_USER_AGENT: {SEC_USER_AGENT}")


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()

