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
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
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


def _extract_annual_series_usd(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    """
    年次相当（USD）を抽出：
    - form: 10-K系（10-K, 10-K/A, 10-K405など）を許可
    - fp: FY/CY/Q4 を優先（ただし完全依存しない）
    - year は fy があれば fy を優先、無ければ end.year
    """
    us = facts_json.get("facts", {}).get("us-gaap", {})
    node = us.get(xbrl_tag, {}).get("units", {}).get("USD", [])
    rows = []

    for x in node:
        form = str(x.get("form", "")).upper().strip()
        fp = str(x.get("fp", "")).upper().strip()
        end = x.get("end")
        val = x.get("val")
        fy_raw = x.get("fy", None)

        if not end or val is None:
            continue
        if not form.startswith("10-K"):
            continue

        end_ts = pd.to_datetime(end, errors="coerce")
        if pd.isna(end_ts):
            continue

        annual_fp = fp in {"FY", "CY", "Q4"}

        if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
            year_key = int(fy_raw)
        else:
            year_key = int(end_ts.year)

        rows.append(
            {
                "year": year_key,
                "end": end_ts,
                "val": _safe_float(val),
                "fp": fp,
                "form": form,
                "fy_raw": fy_raw,
                "annual_fp": int(annual_fp),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "end", "val", "fp", "form", "fy_raw", "annual_fp"])

    df = pd.DataFrame(rows).dropna(subset=["val"])

    # 同一年が複数候補：annual_fp優先→endが新しいもの
    df = df.sort_values(["year", "annual_fp", "end"]).drop_duplicates(subset=["year"], keep="last")
    df = df.sort_values("year")
    return df


def _pick_best_tag_latest_first(facts_json: dict, candidates: list[str]) -> tuple[str | None, pd.DataFrame]:
    """
    タグ選択ルール（長期投資向け）：
      1) 最新年 max(year) が最大のタグを優先
      2) 年数 len(df) が多いタグを優先
      3) 最新 end が新しいタグを優先
    """
    best_tag = None
    best_df = pd.DataFrame(columns=["year", "end", "val", "fp", "form", "fy_raw", "annual_fp"])
    best_max_year = -1
    best_n = -1
    best_last_end = pd.Timestamp("1900-01-01")

    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag)
        if df.empty:
            continue

        max_year = int(df["year"].max())
        n = int(len(df))
        last_end = df["end"].max()

        better = False
        if max_year > best_max_year:
            better = True
        elif max_year == best_max_year:
            if n > best_n:
                better = True
            elif n == best_n and last_end > best_last_end:
                better = True

        if better:
            best_tag = tag
            best_df = df
            best_max_year = max_year
            best_n = n
            best_last_end = last_end

    return best_tag, best_df


def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]
    op_income_tags = ["OperatingIncomeLoss"]
    pretax_tags = [
        "IncomeBeforeIncomeTaxes",
        "PretaxIncome",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems",
    ]
    net_income_tags = ["NetIncomeLoss"]

    meta = {}

    tag, df_rev = _pick_best_tag_latest_first(facts_json, revenue_tags)
    meta["revenue_tag"] = tag

    tag, df_op = _pick_best_tag_latest_first(facts_json, op_income_tags)
    meta["op_income_tag"] = tag

    tag, df_pt = _pick_best_tag_latest_first(facts_json, pretax_tags)
    meta["pretax_tag"] = tag

    tag, df_ni = _pick_best_tag_latest_first(facts_json, net_income_tags)
    meta["net_income_tag"] = tag

    if df_rev.empty:
        return pd.DataFrame(), meta

    df = df_rev[["year", "end", "val", "fp", "form", "fy_raw"]].rename(
        columns={"val": "revenue", "fp": "fp_rev", "form": "form_rev", "fy_raw": "fy_rev"}
    )

    for name, d, fp_col, form_col, fy_col in [
        ("op_income", df_op, "fp_op", "form_op", "fy_op"),
        ("pretax", df_pt, "fp_pt", "form_pt", "fy_pt"),
        ("net_income", df_ni, "fp_ni", "form_ni", "fy_ni"),
    ]:
        if d.empty:
            df[name] = np.nan
            df[fp_col] = None
            df[form_col] = None
            df[fy_col] = None
        else:
            tmp = d[["year", "val", "fp", "form", "fy_raw"]].rename(
                columns={"val": name, "fp": fp_col, "form": form_col, "fy_raw": fy_col}
            )
            df = df.merge(tmp, on="year", how="left")

    df = df.sort_values("year")

    out = pd.DataFrame(
        {
            "FY": df["year"].astype(int),
            "End": df["end"].dt.date.astype(str),
            "Revenue(M$)": df["revenue"].map(_to_musd),
            "OpIncome(M$)": df["op_income"].map(_to_musd),
            "Pretax(M$)": df["pretax"].map(_to_musd),
            "NetIncome(M$)": df["net_income"].map(_to_musd),
        }
    )

    meta["years"] = df["year"].astype(int).tolist()
    meta["fp_sample"] = {
        "revenue": df["fp_rev"].dropna().unique().tolist(),
        "op_income": df["fp_op"].dropna().unique().tolist(),
        "pretax": df["fp_pt"].dropna().unique().tolist(),
        "net_income": df["fp_ni"].dropna().unique().tolist(),
    }
    meta["form_sample"] = {
        "revenue": df["form_rev"].dropna().unique().tolist(),
        "op_income": df["form_op"].dropna().unique().tolist(),
        "pretax": df["form_pt"].dropna().unique().tolist(),
        "net_income": df["form_ni"].dropna().unique().tolist(),
    }
    meta["fy_raw_sample"] = {
        "revenue": df["fy_rev"].dropna().astype(str).unique().tolist(),
        "op_income": df["fy_op"].dropna().astype(str).unique().tolist(),
        "pretax": df["fy_pt"].dropna().astype(str).unique().tolist(),
        "net_income": df["fy_ni"].dropna().astype(str).unique().tolist(),
    }

    return out, meta


def _slice_latest_n_years(table: pd.DataFrame, n_years: int) -> pd.DataFrame:
    """
    最新年度を基準に「最新から過去N年」を切り出す
    """
    latest_year = int(table["FY"].max())
    min_year = latest_year - int(n_years) + 1
    return table[table["FY"] >= min_year].sort_values("FY").reset_index(drop=True)


def plot_pl_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()

    revenue = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()
    pt = df["Pretax(M$)"].astype(float).to_numpy()
    ni = df["NetIncome(M$)"].astype(float).to_numpy()

    fig, ax_left = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax_left.set_facecolor("#0b0c0e")

    ax_left.plot(x, op, label="Operating Income", linewidth=2.5, marker="o", markersize=6)
    ax_left.plot(x, pt, label="Pretax Income", linewidth=2.5, marker="o", markersize=6)
    ax_left.plot(x, ni, label="Net Income", linewidth=2.5, marker="o", markersize=6)

    ax_left.set_ylabel("Profit (Million USD)", color="white")
    ax_left.tick_params(colors="white")
    ax_left.grid(color="#333333", alpha=0.6)

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

    st.markdown("## 長期ファンダ（年次PL / 最新から過去N年表示版）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("fund_form"):
        ticker = st.text_input("Ticker（米国株）", value="AAPL")
        years = st.slider("表示する年数（最新から過去N年）", min_value=3, max_value=15, value=10)
        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    t = ticker.strip().lower()
    if not t:
        st.error("Tickerを入力してください。")
        st.stop()

    m = fetch_ticker_cik_map()
    cik10 = m.get(t)
    if not cik10:
        st.error("このTickerのCIKが見つかりませんでした。")
        st.stop()

    facts = fetch_company_facts(cik10)
    company_name = facts.get("entityName", ticker.upper())

    table, meta = build_pl_annual_table(facts)
    if table.empty:
        st.error("年次PLデータが取得できませんでした。")
        st.write(meta)
        st.stop()

    st.caption(f"取得できた年次データ: {len(table)} 年分（最新年: {int(table['FY'].max())}）")

    # ★最新年度を基準にN年分を切り出す
    table_disp = _slice_latest_n_years(table, int(years))

    plot_pl_annual(table_disp, f"{company_name} ({ticker.upper()}) - Income Statement (Annual)")

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
