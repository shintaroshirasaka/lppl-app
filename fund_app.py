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
# PL (Income Statement) - Annual (multi-year)
# =========================
def _build_revenue_composite(facts_json: dict, revenue_tags: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Revenueを“年ごとに最良タグで埋める”合成系列
    """
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
        for tag in revenue_tags:  # 優先順
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

    # end は op_income側から持つ（無ければ年末）
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
# BS (Balance Sheet) - Latest year only
# =========================
def build_bs_latest_snapshot(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    """
    BSは最新年の1年だけを返す（Assets / Liabilities / Equity / Cash）
    """
    assets_tags = ["Assets"]
    liabilities_tags = ["Liabilities"]
    equity_tags = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
    cash_tags = ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]

    meta = {}

    tag, df_assets = _pick_best_tag_latest_first(facts_json, assets_tags)
    meta["assets_tag"] = tag

    tag, df_liab = _pick_best_tag_latest_first(facts_json, liabilities_tags)
    meta["liabilities_tag"] = tag

    tag, df_eq = _pick_best_tag_latest_first(facts_json, equity_tags)
    meta["equity_tag"] = tag

    tag, df_cash = _pick_best_tag_latest_first(facts_json, cash_tags)
    meta["cash_tag"] = tag

    if df_assets.empty:
        return pd.DataFrame(), meta

    # できれば 3要素（Assets/Liab/Equity）が揃う年を優先
    years_assets = set(df_assets["year"].astype(int).tolist())
    years_liab = set(df_liab["year"].astype(int).tolist()) if not df_liab.empty else set()
    years_eq = set(df_eq["year"].astype(int).tolist()) if not df_eq.empty else set()

    common = years_assets & years_liab & years_eq
    if common:
        year = max(common)
    else:
        year = int(df_assets["year"].max())

    def pick_val(df: pd.DataFrame, year: int) -> float:
        if df.empty:
            return np.nan
        r = df[df["year"] == year]
        if r.empty:
            return np.nan
        return float(r["val"].iloc[-1])

    def pick_end(df: pd.DataFrame, year: int):
        if df.empty:
            return pd.NaT
        r = df[df["year"] == year]
        if r.empty:
            return pd.NaT
        return pd.to_datetime(r["end"].iloc[-1], errors="coerce")

    assets = pick_val(df_assets, year)
    liab = pick_val(df_liab, year)
    equity = pick_val(df_eq, year)
    cash = pick_val(df_cash, year)

    end = pick_end(df_assets, year)
    end_str = str(end.date()) if pd.notna(end) else f"{year}-12-31"

    snap = pd.DataFrame(
        [
            ["FY", year],
            ["End", end_str],
            ["Assets (M$)", _to_musd(assets) if np.isfinite(assets) else np.nan],
            ["Liabilities (M$)", _to_musd(liab) if np.isfinite(liab) else np.nan],
            ["Equity (M$)", _to_musd(equity) if np.isfinite(equity) else np.nan],
            ["Cash (M$)", _to_musd(cash) if np.isfinite(cash) else np.nan],
        ],
        columns=["Item", "Value"],
    )

    meta["bs_year"] = year
    meta["bs_common_years_exists"] = bool(common)
    return snap, meta


def plot_bs_latest(snap: pd.DataFrame, title: str):
    """
    最新年のBSを「資産」と「負債+資本（積み上げ）」で1枚にする
    """
    get = dict(zip(snap["Item"], snap["Value"]))
    assets = float(get.get("Assets (M$)", np.nan))
    liab = float(get.get("Liabilities (M$)", np.nan))
    equity = float(get.get("Equity (M$)", np.nan))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    x = ["Assets", "Liabilities+Equity"]

    # 左バー：Assets
    ax.bar([0], [assets], alpha=0.55)

    # 右バー：Liabilities + Equity を積み上げ
    ax.bar([1], [liab], alpha=0.55, label="Liabilities")
    ax.bar([1], [equity], bottom=[liab], alpha=0.55, label="Equity")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(x, color="white")
    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white")

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

    st.markdown("## 長期ファンダ（PL 年次 / BS 最新年）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("fund_form"):
        ticker = st.text_input("Ticker（米国株）", value="AAPL")
        years = st.slider("PL: 表示する年数（最新から過去N年）", min_value=3, max_value=15, value=10)
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

    tab_pl, tab_bs = st.tabs(["PL（年次）", "BS（最新年のみ）"])

    with tab_pl:
        table, meta = build_pl_annual_table(facts)
        if table.empty:
            st.error("PLデータが取得できませんでした。")
            st.write(meta)
            st.stop()

        st.caption(f"PL: 取得できた年次データ {len(table)} 年分（最新年: {int(table['FY'].max())}）")

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

        with st.expander("PLデバッグ", expanded=False):
            st.write(meta)

    with tab_bs:
        snap, meta_bs = build_bs_latest_snapshot(facts)
        if snap.empty:
            st.error("BSデータが取得できませんでした（Assetsが取れない等）。")
            st.write(meta_bs)
            st.stop()

        bs_year = int(dict(zip(snap["Item"], snap["Value"]))["FY"])
        st.caption(f"BS: 最新年 {bs_year} のみ表示")

        plot_bs_latest(snap, f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest)")

        st.markdown("### BS（最新年 / 百万USD）")
        # 表を見やすく
        snap_disp = snap.copy()
        # FY/Endは文字列なのでそのまま、数値は整形
        def fmt(v):
            if isinstance(v, (int, np.integer)):
                return str(v)
            if isinstance(v, float) and np.isfinite(v):
                return f"{v:,.0f}"
            return str(v)

        snap_disp["Value"] = snap_disp["Value"].map(fmt)
        st.dataframe(snap_disp, use_container_width=True, hide_index=True)

        with st.expander("BSデバッグ（採用タグなど）", expanded=False):
            st.write(meta_bs)


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
