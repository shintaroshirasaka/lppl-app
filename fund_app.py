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


def _extract_annual_series_usd(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    """
    年次相当（USD）を抽出：
    - form: 10-K系（10-K, 10-K/A, 10-K405など）
    - year: fyがあればfy、無ければend.year
    - 同一年重複は annual_fp優先→end新しい方
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
    df = df.sort_values(["year", "annual_fp", "end"]).drop_duplicates(subset=["year"], keep="last")
    df = df.sort_values("year")
    return df


def _pick_best_tag_latest_first(facts_json: dict, candidates: list[str]) -> tuple[str | None, pd.DataFrame]:
    """
    タグ選択（長期向け）：
      1) 最新年 max(year) が最大
      2) 年数が多い
      3) 最新 end が新しい
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

        if (max_year > best_max_year) or (
            max_year == best_max_year and (n > best_n or (n == best_n and last_end > best_last_end))
        ):
            best_tag, best_df = tag, df
            best_max_year, best_n, best_last_end = max_year, n, last_end

    return best_tag, best_df


# =========================
# PL: Revenue composite + latest-N slice
# =========================
def _build_revenue_composite(facts_json: dict, revenue_tags: list[str]) -> tuple[pd.DataFrame, dict]:
    tag_series = {}
    tag_years = {}
    for tag in revenue_tags:
        df = _extract_annual_series_usd(facts_json, tag)
        if df.empty:
            continue
        s = df.set_index("year")["val"].sort_index()
        tag_series[tag] = s
        tag_years[tag] = s.index.tolist()

    if not tag_series:
        return pd.DataFrame(), {"revenue_available_tags": []}

    all_years = sorted(set().union(*[s.index for s in tag_series.values()]))

    composite = pd.Series(index=all_years, dtype="float64")
    chosen_tag = pd.Series(index=all_years, dtype="object")

    for y in all_years:
        val = np.nan
        used = None
        for tag in revenue_tags:
            s = tag_series.get(tag)
            if s is None:
                continue
            if y in s.index and np.isfinite(s.loc[y]):
                val = float(s.loc[y])
                used = tag
                break
        composite.loc[y] = val
        chosen_tag.loc[y] = used

    meta = {
        "revenue_available_tags": list(tag_series.keys()),
        "revenue_years_by_tag": tag_years,
        "revenue_chosen_tag_by_year": chosen_tag.dropna().to_dict(),
    }
    out = pd.DataFrame({"year": composite.index.astype(int), "revenue": composite.values})
    return out, meta


def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    revenue_priority = [
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

    df_rev_comp, rev_meta = _build_revenue_composite(facts_json, revenue_priority)
    meta["revenue_composite"] = rev_meta

    tag, df_op = _pick_best_tag_latest_first(facts_json, op_income_tags)
    meta["op_income_tag"] = tag

    tag, df_pt = _pick_best_tag_latest_first(facts_json, pretax_tags)
    meta["pretax_tag"] = tag

    tag, df_ni = _pick_best_tag_latest_first(facts_json, net_income_tags)
    meta["net_income_tag"] = tag

    if df_rev_comp.empty:
        return pd.DataFrame(), meta

    df = df_rev_comp.copy()

    if not df_op.empty:
        df = df.merge(df_op[["year", "end"]], on="year", how="left")
        df = df.merge(df_op[["year", "val"]].rename(columns={"val": "op_income"}), on="year", how="left")
    else:
        df["end"] = pd.NaT
        df["op_income"] = np.nan

    if not df_pt.empty:
        df = df.merge(df_pt[["year", "val"]].rename(columns={"val": "pretax"}), on="year", how="left")
    else:
        df["pretax"] = np.nan

    if not df_ni.empty:
        df = df.merge(df_ni[["year", "val"]].rename(columns={"val": "net_income"}), on="year", how="left")
    else:
        df["net_income"] = np.nan

    df = df.sort_values("year")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["end_str"] = df["end"].dt.date.astype(str)
    df.loc[df["end"].isna(), "end_str"] = df["year"].astype(str) + "-12-31"

    out = pd.DataFrame(
        {
            "FY": df["year"].astype(int),
            "End": df["end_str"],
            "Revenue(M$)": df["revenue"].map(_to_musd),
            "OpIncome(M$)": df["op_income"].map(_to_musd),
            "Pretax(M$)": df["pretax"].map(_to_musd),
            "NetIncome(M$)": df["net_income"].map(_to_musd),
        }
    )

    meta["years"] = out["FY"].astype(int).tolist()
    return out, meta


def _slice_latest_n_years(table: pd.DataFrame, n_years: int) -> pd.DataFrame:
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


# =========================
# BS: latest-year bar + pies A/B
# =========================
def _latest_year_from_assets(facts_json: dict) -> int | None:
    df = _extract_annual_series_usd(facts_json, "Assets")
    if df.empty:
        return None
    return int(df["year"].max())


def _value_for_year(facts_json: dict, tag_candidates: list[str], year: int) -> tuple[str | None, float]:
    for tag in tag_candidates:
        df = _extract_annual_series_usd(facts_json, tag)
        if df.empty:
            continue
        r = df[df["year"] == year]
        if r.empty:
            continue
        v = float(r["val"].iloc[-1])
        if np.isfinite(v):
            return tag, v
    return None, np.nan


def build_bs_latest_simple(facts_json: dict, year: int):
    meta = {}

    ca_tag, ca = _value_for_year(facts_json, ["AssetsCurrent"], year)
    nca_tag, nca = _value_for_year(facts_json, ["AssetsNoncurrent"], year)
    cl_tag, cl = _value_for_year(facts_json, ["LiabilitiesCurrent"], year)
    ncl_tag, ncl = _value_for_year(facts_json, ["LiabilitiesNoncurrent"], year)
    eq_tag, eq = _value_for_year(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        year,
    )

    meta.update(
        {
            "bs_year": year,
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
            ["Current Assets (M$)", _to_musd(ca)],
            ["Noncurrent Assets (M$)", _to_musd(nca)],
            ["Current Liabilities (M$)", _to_musd(cl)],
            ["Noncurrent Liabilities (M$)", _to_musd(ncl)],
            ["Equity (M$)", _to_musd(eq)],
        ],
        columns=["Item", "Value"],
    )

    return snap, meta


def plot_bs_bar(snap: pd.DataFrame, title: str):
    vals = dict(zip(snap["Item"], snap["Value"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    # Assets（下：非流動、上：流動）
    bottom = 0.0
    ax.bar(0, vals["Noncurrent Assets (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Assets")
    bottom += vals["Noncurrent Assets (M$)"]
    ax.bar(0, vals["Current Assets (M$)"], bottom=bottom, alpha=0.7, label="Current Assets")

    # L+E（下：純資産、中：非流動負債、上：流動負債）
    bottom = 0.0
    ax.bar(1, vals["Equity (M$)"], bottom=bottom, alpha=0.7, label="Equity")
    bottom += vals["Equity (M$)"]
    ax.bar(1, vals["Noncurrent Liabilities (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Liabilities")
    bottom += vals["Noncurrent Liabilities (M$)"]
    ax.bar(1, vals["Current Liabilities (M$)"], bottom=bottom, alpha=0.7, label="Current Liabilities")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Assets", "Liabilities + Equity"], color="white")
    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left", bbox_to_anchor=(1.02, 1.0))

    st.pyplot(fig)


def _top3_plus_other(items: list[tuple[str, float]], total: float) -> tuple[list[str], list[float]]:
    pos = [(n, float(v)) for n, v in items if np.isfinite(v) and float(v) > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    top = pos[:3]
    top_sum = sum(v for _, v in top)
    other = max(total - top_sum, 0.0) if np.isfinite(total) else sum(v for _, v in pos[3:])

    labels = [n for n, _ in top]
    sizes = [v for _, v in top]
    if other > 0:
        labels.append("Other")
        sizes.append(other)
    if sum(sizes) <= 0:
        return (["No data"], [1.0])
    return labels, sizes


def build_bs_pies_latest(facts_json: dict, year: int) -> tuple[dict, dict, dict]:
    meta = {"year": year}

    _, total_assets = _value_for_year(facts_json, ["Assets"], year)
    _, total_liab = _value_for_year(facts_json, ["Liabilities"], year)
    _, total_eq = _value_for_year(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        year,
    )
    total_le = (total_liab if np.isfinite(total_liab) else 0.0) + (total_eq if np.isfinite(total_eq) else 0.0)

    # A: Assets top3
    asset_candidates = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue",
                                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
        ("Marketable Securities", ["MarketableSecuritiesCurrent", "MarketableSecuritiesNoncurrent",
                                   "AvailableForSaleSecuritiesCurrent", "AvailableForSaleSecuritiesNoncurrent"]),
        ("Receivables", ["AccountsReceivableNetCurrent", "AccountsReceivableNet"]),
        ("Inventory", ["InventoryNet"]),
        ("PPE (Net)", ["PropertyPlantAndEquipmentNet"]),
        ("Goodwill", ["Goodwill"]),
        ("Intangibles", ["IntangibleAssetsNetExcludingGoodwill"]),
    ]
    asset_items = []
    for name, tags in asset_candidates:
        used, v = _value_for_year(facts_json, tags, year)
        meta[f"asset_{name}_tag"] = used
        if np.isfinite(v):
            asset_items.append((name, v))

    if not np.isfinite(total_assets) or total_assets <= 0:
        total_assets = sum(v for _, v in asset_items if np.isfinite(v) and v > 0)
    a_labels, a_sizes = _top3_plus_other(asset_items, total_assets)

    # B: Liabilities + Equity top3
    le_candidates = [
        ("Current Liabilities", ["LiabilitiesCurrent"]),
        ("Noncurrent Liabilities", ["LiabilitiesNoncurrent"]),
        ("Long-term Debt", ["LongTermDebtNoncurrent",
                            "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                            "LongTermDebt"]),
        ("Retained Earnings", ["RetainedEarningsAccumulatedDeficit"]),
        ("Equity (Total)", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    ]
    le_items = []
    for name, tags in le_candidates:
        used, v = _value_for_year(facts_json, tags, year)
        meta[f"le_{name}_tag"] = used
        if np.isfinite(v) and v > 0:
            le_items.append((name, v))

    if not np.isfinite(total_le) or total_le <= 0:
        total_le = sum(v for _, v in le_items if np.isfinite(v) and v > 0)
    b_labels, b_sizes = _top3_plus_other(le_items, total_le)

    assets_pie = {"labels": a_labels, "sizes": a_sizes, "total": total_assets}
    le_pie = {"labels": b_labels, "sizes": b_sizes, "total": total_le}
    return assets_pie, le_pie, meta


def plot_two_pies(assets_pie: dict, le_pie: dict, year: int):
    col1, col2 = st.columns(2)

    def _pie(ax, labels, sizes, title):
        ax.pie(
            sizes,
            labels=labels,
            autopct=lambda p: f"{p:.0f}%",
            startangle=90,
            textprops={"color": "white"},
        )
        ax.set_title(title, color="white")

    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        fig.patch.set_facecolor("#0b0c0e")
        ax.set_facecolor("#0b0c0e")
        _pie(ax, assets_pie["labels"], assets_pie["sizes"], f"A: Assets Top3 + Other ({year})")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        fig.patch.set_facecolor("#0b0c0e")
        ax.set_facecolor("#0b0c0e")
        _pie(ax, le_pie["labels"], le_pie["sizes"], f"B: Liabilities+Equity Top3 + Other ({year})")
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

    st.markdown("## 長期ファンダ（PL + BS）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("input_form"):
        ticker = st.text_input("Ticker（米国株）", value="AAPL")
        pl_years = st.slider("PL: 表示する年数（最新から過去N年）", min_value=3, max_value=15, value=10)
        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    t = ticker.strip().lower()
    if not t:
        st.error("Tickerを入力してください。")
        st.stop()

    cik10 = fetch_ticker_cik_map().get(t)
    if not cik10:
        st.error("このTickerのCIKが見つかりませんでした。")
        st.stop()

    facts = fetch_company_facts(cik10)
    company_name = facts.get("entityName", ticker.upper())

    tab_pl, tab_bs = st.tabs(["PL（年次）", "BS（最新年）"])

    with tab_pl:
        table, meta = build_pl_annual_table(facts)
        if table.empty:
            st.error("PLデータが取得できませんでした。")
            st.write(meta)
            st.stop()

        table_disp = _slice_latest_n_years(table, int(pl_years))
        st.caption(f"PL: 取得 {len(table_disp)} 年（最新年: {int(table['FY'].max())}）")

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

        with st.expander("PLデバッグ", expanded=False):
            st.write(meta)

    with tab_bs:
        year = _latest_year_from_assets(facts)
        if year is None:
            st.error("Assets（総資産）が取得できませんでした。")
            st.stop()

        snap, meta_bs = build_bs_latest_simple(facts, year)
        st.caption(f"BS: 最新年 {year}")

        # 1) 棒グラフ
        plot_bs_bar(snap, f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest)")

        # 2) 円グラフ A/B
        assets_pie, le_pie, meta_pie = build_bs_pies_latest(facts, year)
        plot_two_pies(assets_pie, le_pie, year)

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

        with st.expander("BSデバッグ", expanded=False):
            st.write({"bs": meta_bs, "pies": meta_pie})


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
