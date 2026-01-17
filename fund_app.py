# fund_app.py
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from auth_gate import require_admin_token

# =========================
# SEC settings
# =========================
# SECはUser-Agent推奨（連絡先を含める）
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip() or "YourName your.email@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Ticker -> CIK 対応表（SEC公式JSON）
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


def _extract_annual_series_usd(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    """
    us-gaap の tag から「年次相当」を抽出（USD）
    - form == 10-K のみ採用（年次報告書）
    - fp in {"FY","CY","Q4"} を年次として扱う
    戻り: columns=["fy","end","val","fp","form"]
    """
    us = facts_json.get("facts", {}).get("us-gaap", {})
    node = us.get(xbrl_tag, {}).get("units", {}).get("USD", [])
    rows = []

    for x in node:
        form = str(x.get("form", "")).upper()
        fp = str(x.get("fp", "")).upper()

        if (
            form == "10-K"
            and fp in {"FY", "CY", "Q4"}
            and x.get("fy")
            and x.get("end")
            and x.get("val") is not None
        ):
            rows.append(
                {
                    "fy": int(x["fy"]),
                    "end": pd.to_datetime(x["end"]),
                    "val": _safe_float(x["val"]),
                    "fp": fp,
                    "form": form,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["fy", "end", "val", "fp", "form"])

    df = pd.DataFrame(rows).dropna(subset=["val"])

    # 同一年が複数ある場合：endが遅いもの（≒最新/確定）を採用
    df = df.sort_values(["fy", "end"]).drop_duplicates(subset=["fy"], keep="last")
    df = df.sort_values("fy")
    return df


def _pick_first_available_tag(facts_json: dict, candidates: list[str]) -> tuple[str | None, pd.DataFrame]:
    """
    複数候補タグのうち、データが取れた最初のタグを採用
    """
    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag)
        if not df.empty:
            return tag, df
    return None, pd.DataFrame(columns=["fy", "end", "val", "fp", "form"])


def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    """
    年次PL（売上・営業利益・税前利益・純利益）を作る
    戻り: (table_df, meta)
    """
    revenue_tags = ["Revenues", "SalesRevenueNet"]
    op_income_tags = ["OperatingIncomeLoss"]
    pretax_tags = [
        "IncomeBeforeIncomeTaxes",
        "PretaxIncome",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems",
    ]
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

    if df_rev.empty:
        return pd.DataFrame(), meta

    # revenue を軸に FY で結合（MVPは left join でOK）
    df = df_rev[["fy", "end", "val", "fp"]].rename(columns={"val": "revenue", "fp": "fp_rev"})

    for name, d, fp_col in [
        ("op_income", df_op, "fp_op"),
        ("pretax", df_pt, "fp_pt"),
        ("net_income", df_ni, "fp_ni"),
    ]:
        if d.empty:
            df[name] = np.nan
            df[fp_col] = None
        else:
            tmp = d[["fy", "val", "fp"]].rename(columns={"val": name, "fp": fp_col})
            df = df.merge(tmp, on="fy", how="left")

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

    # デバッグ用：実際に採用された fp
    meta["fp_sample"] = {
        "revenue": df.get("fp_rev", pd.Series(dtype=object)).dropna().unique().tolist(),
        "op_income": df.get("fp_op", pd.Series(dtype=object)).dropna().unique().tolist(),
        "pretax": df.get("fp_pt", pd.Series(dtype=object)).dropna().unique().tolist(),
        "net_income": df.get("fp_ni", pd.Series(dtype=object)).dropna().unique().tolist(),
    }

    return out, meta


def plot_pl_annual(table: pd.DataFrame, title: str):
    """
    売上（棒・右軸）＋利益（折れ線・左軸）
    暗背景でも“くっきり”見える版
    """
    df = table.copy()
    x = df["FY"].astype(str).tolist()

    revenue = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()
    pt = df["Pretax(M$)"].astype(float).to_numpy()
    ni = df["NetIncome(M$)"].astype(float).to_numpy()

    fig, ax_left = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax_left.set_facecolor("#0b0c0e")

    # 折れ線（利益）
    ax_left.plot(x, op, label="Operating Income", linewidth=2.5, marker="o", markersize=6)
    ax_left.plot(x, pt, label="Pretax Income", linewidth=2.5, marker="o", markersize=6)
    ax_left.plot(x, ni, label="Net Income", linewidth=2.5, marker="o", markersize=6)

    ax_left.set_ylabel("Profit (Million USD)", color="white")
    ax_left.tick_params(colors="white")
    ax_left.grid(color="#333333", alpha=0.6)

    # 棒（売上）
    ax_right = ax_left.twinx()
    ax_right.bar(x, revenue, alpha=0.55, width=0.6, label="Revenue")
    ax_right.set_ylabel("Revenue (Million USD)", color="white")
    ax_right.tick_params(colors="white")

    ax_left.set_title(title, color="white")
    ax_left.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    ax_left.tick_params(axis="x", rotation=0, colors="white")

    st.pyplot(fig)


def render(authed_email: str):
    st.set_page_config(page_title="Fundamentals (Staging)", layout="wide")

    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0c0e !important; color: #ffffff !important; }
        div[data-testid="stMarkdownContainer"] p, label { color: #ffffff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 長期ファンダ（年次PL / 年次データ複数年対応版）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("fund_form"):
        ticker = st.text_input(
            "Ticker（米国株）",
            value="AAPL",
            placeholder="例: AAPL / MSFT / GOOGL / AMZN / NVDA / AVGO",
        )
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
        with st.expander("デバッグ", expanded=True):
            st.write(meta)
            st.write(f"CIK: {cik10}")
        st.stop()

    st.caption(f"取得できた年次データ: {len(table)} 年分")

    # 末尾N年に絞る
    table_disp = table.tail(int(years)).reset_index(drop=True)

    # Plot
    plot_pl_annual(
        table_disp,
        title=f"{company_name} ({ticker.upper()}) - Income Statement (Annual)",
    )

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

    with st.expander("デバッグ情報（採用したXBRLタグ）", expanded=False):
        st.write(meta)
        st.write(f"CIK: {cik10}")
        st.write(f"SEC_USER_AGENT: {SEC_USER_AGENT}")


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
