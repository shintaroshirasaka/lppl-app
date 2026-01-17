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
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip() or "YourName your.email@example.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"


def _to_musd(x: float) -> float:
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
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _extract_latest_value(facts_json: dict, tags: list[str]) -> tuple[str | None, float, int, pd.Timestamp | None]:
    """
    指定タグ群から「最新年」の値を1つ返す
    戻り: (採用タグ, 値, 年, end)
    """
    us = facts_json.get("facts", {}).get("us-gaap", {})

    best_year = -1
    best_val = np.nan
    best_tag = None
    best_end = None

    for tag in tags:
        node = us.get(tag, {}).get("units", {}).get("USD", [])
        for x in node:
            form = str(x.get("form", "")).upper()
            end = x.get("end")
            val = x.get("val")
            fy = x.get("fy")

            if not end or val is None:
                continue
            if not form.startswith("10-K"):
                continue

            try:
                end_ts = pd.to_datetime(end, errors="coerce")
                if pd.isna(end_ts):
                    continue
                year = int(fy) if fy else int(end_ts.year)
            except Exception:
                continue

            if year > best_year:
                best_year = year
                best_val = _safe_float(val)
                best_tag = tag
                best_end = end_ts

    return best_tag, best_val, best_year, best_end


# =========================
# BS (Balance Sheet) - Latest year only (simplified)
# =========================
def build_bs_latest_simple(facts_json: dict):
    """
    最新年BSを以下の5区分で取得
    - Current Assets
    - Noncurrent Assets
    - Current Liabilities
    - Noncurrent Liabilities
    - Equity
    """
    meta = {}

    ca_tag, ca, y1, end1 = _extract_latest_value(facts_json, ["AssetsCurrent"])
    nca_tag, nca, y2, end2 = _extract_latest_value(facts_json, ["AssetsNoncurrent"])
    cl_tag, cl, y3, end3 = _extract_latest_value(facts_json, ["LiabilitiesCurrent"])
    ncl_tag, ncl, y4, end4 = _extract_latest_value(facts_json, ["LiabilitiesNoncurrent"])
    eq_tag, eq, y5, end5 = _extract_latest_value(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    )

    year = max(y1, y2, y3, y4, y5)
    # endは最も新しいものを採用
    end = max([d for d in [end1, end2, end3, end4, end5] if d is not None], default=None)
    end_str = str(end.date()) if end is not None else f"{year}-12-31"

    meta.update(
        {
            "bs_year": year,
            "end": end_str,
            "current_assets_tag": ca_tag,
            "noncurrent_assets_tag": nca_tag,
            "current_liabilities_tag": cl_tag,
            "noncurrent_liabilities_tag": ncl_tag,
            "equity_tag": eq_tag,
        }
    )

    snap = pd.DataFrame(
        [
            ["FY", year],
            ["End", end_str],
            ["Current Assets (M$)", _to_musd(ca)],
            ["Noncurrent Assets (M$)", _to_musd(nca)],
            ["Current Liabilities (M$)", _to_musd(cl)],
            ["Noncurrent Liabilities (M$)", _to_musd(ncl)],
            ["Equity (M$)", _to_musd(eq)],
        ],
        columns=["Item", "Value"],
    )

    return snap, meta


def plot_bs_simple(snap: pd.DataFrame, title: str):
    """
    借方（Assets）：
      下：非流動資産
      上：流動資産

    貸方（Liabilities + Equity）：
      下：純資産
      中：非流動負債
      上：流動負債
    """
    vals = dict(zip(snap["Item"], snap["Value"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    # ---- Assets (left) ----
    bottom = 0.0
    # 下：非流動資産
    ax.bar(0, vals["Noncurrent Assets (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Assets")
    bottom += vals["Noncurrent Assets (M$)"]
    # 上：流動資産
    ax.bar(0, vals["Current Assets (M$)"], bottom=bottom, alpha=0.7, label="Current Assets")

    # ---- Liabilities + Equity (right) ----
    bottom = 0.0
    # 下：純資産
    ax.bar(1, vals["Equity (M$)"], bottom=bottom, alpha=0.7, label="Equity")
    bottom += vals["Equity (M$)"]
    # 中：非流動負債
    ax.bar(1, vals["Noncurrent Liabilities (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Liabilities")
    bottom += vals["Noncurrent Liabilities (M$)"]
    # 上：流動負債
    ax.bar(1, vals["Current Liabilities (M$)"], bottom=bottom, alpha=0.7, label="Current Liabilities")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Assets", "Liabilities + Equity"], color="white")
    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")

    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    st.pyplot(fig)


# =========================
# UI
# =========================
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

    st.markdown("## 長期ファンダ（BS：最新年・簡略表示）")
    st.caption(f"認証ユーザー: {authed_email}")

    ticker = st.text_input("Ticker（米国株）", value="AAPL")

    if not ticker.strip():
        st.stop()

    cik10 = fetch_ticker_cik_map().get(ticker.strip().lower())
    if not cik10:
        st.error("CIKが見つかりませんでした。")
        st.stop()

    facts = fetch_company_facts(cik10)
    company_name = facts.get("entityName", ticker.upper())

    # いまはBSだけを確実に動かす版（PLは別ファイルで運用中ならここに統合OK）
    snap, meta = build_bs_latest_simple(facts)
    year = meta.get("bs_year")

    st.caption(f"BS: 最新年 {year}  / End: {meta.get('end')}")

    plot_bs_simple(snap, f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest)")

    st.markdown("### BS（最新年 / 百万USD）")
    snap_disp = snap.copy()

    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(v)
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:,.0f}"
        return str(v)

    snap_disp["Value"] = snap_disp["Value"].map(fmt)
    st.dataframe(snap_disp, use_container_width=True, hide_index=True)

    with st.expander("BSデバッグ（採用タグ）", expanded=False):
        st.write(meta)
        st.write(f"CIK: {cik10}")
        st.write(f"SEC_USER_AGENT: {SEC_USER_AGENT}")


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
