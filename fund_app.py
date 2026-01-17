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
    年次相当（USD）抽出（10-K系）
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
    タグ選択（最新年優先→件数→end）
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


def _slice_latest_n_years(table: pd.DataFrame, n_years: int) -> pd.DataFrame:
    latest_year = int(table["FY"].max())
    min_year = latest_year - int(n_years) + 1
    return table[table["FY"] >= min_year].sort_values("FY").reset_index(drop=True)


# =========================
# Composite builder (fill by year)
# =========================
def _build_composite_by_year(facts_json: dict, tag_priority: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    tag_priority の順に、年ごとに埋める合成系列を作る
    戻り df: columns=["year","value"]
    """
    tag_series = {}
    tag_years = {}
    for tag in tag_priority:
        df = _extract_annual_series_usd(facts_json, tag)
        if df.empty:
            continue
        s = df.set_index("year")["val"].sort_index()
        tag_series[tag] = s
        tag_years[tag] = s.index.tolist()

    if not tag_series:
        return pd.DataFrame(), {"available_tags": []}

    all_years = sorted(set().union(*[s.index for s in tag_series.values()]))

    composite = pd.Series(index=all_years, dtype="float64")
    chosen_tag = pd.Series(index=all_years, dtype="object")

    for y in all_years:
        val = np.nan
        used = None
        for tag in tag_priority:
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
        "available_tags": list(tag_series.keys()),
        "years_by_tag": tag_years,
        "chosen_tag_by_year": chosen_tag.dropna().to_dict(),
    }
    out = pd.DataFrame({"year": composite.index.astype(int), "value": composite.values})
    return out, meta


# =========================
# PL (annual)
# =========================
def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    # Revenue (既存の強い方針)
    revenue_priority = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]
    # NetIncome（空白対策：複数タグ合成）
    net_income_priority = [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "IncomeLossFromContinuingOperations",
    ]

    op_income_tags = ["OperatingIncomeLoss"]
    pretax_tags = [
        "IncomeBeforeIncomeTaxes",
        "PretaxIncome",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItems",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
    ]

    meta = {}

    rev_df, rev_meta = _build_composite_by_year(facts_json, revenue_priority)
    meta["revenue_composite"] = rev_meta

    ni_df, ni_meta = _build_composite_by_year(facts_json, net_income_priority)
    meta["net_income_composite"] = ni_meta

    tag, df_op = _pick_best_tag_latest_first(facts_json, op_income_tags)
    meta["op_income_tag"] = tag

    tag, df_pt = _pick_best_tag_latest_first(facts_json, pretax_tags)
    meta["pretax_tag"] = tag

    if rev_df.empty:
        return pd.DataFrame(), meta

    # union years
    years = sorted(set(rev_df["year"].tolist()) | set(df_op["year"].tolist()) | set(df_pt["year"].tolist()) | set(ni_df["year"].tolist()))
    if not years:
        return pd.DataFrame(), meta

    # maps
    rev_map = dict(zip(rev_df["year"], rev_df["value"]))
    ni_map = dict(zip(ni_df["year"], ni_df["value"]))
    op_map = dict(zip(df_op["year"], df_op["val"])) if not df_op.empty else {}
    pt_map = dict(zip(df_pt["year"], df_pt["val"])) if not df_pt.empty else {}

    # end date pref: op_income
    end_map = {}
    if not df_op.empty:
        end_map = dict(zip(df_op["year"], df_op["end"]))

    rows = []
    for y in years:
        end = end_map.get(y)
        end_str = str(pd.to_datetime(end).date()) if end is not None and pd.notna(end) else f"{y}-12-31"

        rows.append([
            y,
            end_str,
            _to_musd(rev_map.get(y, np.nan)) if np.isfinite(rev_map.get(y, np.nan)) else np.nan,
            _to_musd(op_map.get(y, np.nan)) if np.isfinite(op_map.get(y, np.nan)) else np.nan,
            _to_musd(pt_map.get(y, np.nan)) if np.isfinite(pt_map.get(y, np.nan)) else np.nan,
            _to_musd(ni_map.get(y, np.nan)) if np.isfinite(ni_map.get(y, np.nan)) else np.nan,
        ])

    out = pd.DataFrame(rows, columns=["FY", "End", "Revenue(M$)", "OpIncome(M$)", "Pretax(M$)", "NetIncome(M$)"])
    meta["years"] = years
    return out, meta


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
# BS (latest year) with fallback for missing noncurrent
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
    """
    missing対策：
      AssetsNoncurrent が無い → Assets - AssetsCurrent で補完
      LiabilitiesNoncurrent が無い → Liabilities - LiabilitiesCurrent で補完
    """
    meta = {"bs_year": year}

    # totals (for fallback)
    assets_tag, assets_total = _value_for_year(facts_json, ["Assets"], year)
    liab_tag, liab_total = _value_for_year(facts_json, ["Liabilities"], year)
    meta["assets_total_tag"] = assets_tag
    meta["liabilities_total_tag"] = liab_tag

    ca_tag, ca = _value_for_year(facts_json, ["AssetsCurrent"], year)
    nca_tag, nca = _value_for_year(facts_json, ["AssetsNoncurrent"], year)

    cl_tag, cl = _value_for_year(facts_json, ["LiabilitiesCurrent"], year)
    ncl_tag, ncl = _value_for_year(facts_json, ["LiabilitiesNoncurrent"], year)

    eq_tag, eq = _value_for_year(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        year,
    )

    # fallback noncurrent assets
    if (not np.isfinite(nca)) and np.isfinite(assets_total) and np.isfinite(ca):
        nca = assets_total - ca
        nca_tag = "CALC:Assets-AssetsCurrent"
    # fallback noncurrent liabilities
    if (not np.isfinite(ncl)) and np.isfinite(liab_total) and np.isfinite(cl):
        ncl = liab_total - cl
        ncl_tag = "CALC:Liabilities-LiabilitiesCurrent"

    meta.update(
        {
            "current_assets_tag": ca_tag,
            "noncurrent_assets_tag": nca_tag,
            "current_liabilities_tag": cl_tag,
            "noncurrent_liabilities_tag": ncl_tag,
            "equity_tag": eq_tag,
        }
    )

    # NaNは0として棒グラフを壊さない
    ca_m = _to_musd(ca) if np.isfinite(ca) else 0.0
    nca_m = _to_musd(nca) if np.isfinite(nca) else 0.0
    cl_m = _to_musd(cl) if np.isfinite(cl) else 0.0
    ncl_m = _to_musd(ncl) if np.isfinite(ncl) else 0.0
    eq_m = _to_musd(eq) if np.isfinite(eq) else 0.0

    snap = pd.DataFrame(
        [
            ["FY", year],
            ["Current Assets (M$)", ca_m],
            ["Noncurrent Assets (M$)", nca_m],
            ["Current Liabilities (M$)", cl_m],
            ["Noncurrent Liabilities (M$)", ncl_m],
            ["Equity (M$)", eq_m],
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


# =========================
# CF (annual) - same years as PL slider
# =========================
def build_cf_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}
    cfo_tags = ["NetCashProvidedByUsedInOperatingActivities"]
    cfi_tags = ["NetCashProvidedByUsedInInvestingActivities"]
    cff_tags = ["NetCashProvidedByUsedInFinancingActivities"]
    capex_tags = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireFixedAssets",
    ]

    tag, df_cfo = _pick_best_tag_latest_first(facts_json, cfo_tags)
    meta["cfo_tag"] = tag
    tag, df_cfi = _pick_best_tag_latest_first(facts_json, cfi_tags)
    meta["cfi_tag"] = tag
    tag, df_cff = _pick_best_tag_latest_first(facts_json, cff_tags)
    meta["cff_tag"] = tag
    tag, df_capex = _pick_best_tag_latest_first(facts_json, capex_tags)
    meta["capex_tag"] = tag

    years = sorted(
        set(df_cfo["year"].astype(int).tolist())
        | set(df_cfi["year"].astype(int).tolist())
        | set(df_cff["year"].astype(int).tolist())
        | set(df_capex["year"].astype(int).tolist())
    )
    if not years:
        return pd.DataFrame(), meta

    def val_at(df: pd.DataFrame, y: int) -> float:
        if df.empty:
            return np.nan
        r = df[df["year"] == y]
        if r.empty:
            return np.nan
        return float(r["val"].iloc[-1])

    rows = []
    for y in years:
        cfo = val_at(df_cfo, y)
        cfi = val_at(df_cfi, y)
        cff = val_at(df_cff, y)
        capex = val_at(df_capex, y)

        capex_out = np.nan
        if np.isfinite(capex):
            capex_out = capex if capex < 0 else -abs(capex)

        fcf = np.nan
        if np.isfinite(cfo) and np.isfinite(capex_out):
            fcf = cfo + capex_out

        end = None
        if not df_cfo.empty and (df_cfo["year"] == y).any():
            end = df_cfo[df_cfo["year"] == y]["end"].iloc[-1]
        end_str = str(pd.to_datetime(end).date()) if end is not None else f"{y}-12-31"

        rows.append([y, end_str, _to_musd(cfo), _to_musd(cfi), _to_musd(cff), _to_musd(fcf)])

    out = pd.DataFrame(rows, columns=["FY", "End", "CFO(M$)", "CFI(M$)", "CFF(M$)", "FCF(M$)"])
    meta["years"] = years
    return out, meta


def plot_cf_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()

    cfo = df["CFO(M$)"].astype(float).to_numpy()
    cfi = df["CFI(M$)"].astype(float).to_numpy()
    cff = df["CFF(M$)"].astype(float).to_numpy()
    fcf = df["FCF(M$)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, cfo, label="CFO (Operating CF)", linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, cfi, label="CFI (Investing CF)", linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, cff, label="CFF (Financing CF)", linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, fcf, label="FCF (CFO + CapEx)", linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    ax.tick_params(axis="x", rotation=0, colors="white")
    st.pyplot(fig)


# =========================
# RPO / Backlog tab (annual) - AVGO first
# =========================
def build_rpo_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    """
    RPO / Contract Liabilities / Deferred Revenue の年次推移
    取れた系列だけ表示する方針。
    """
    meta = {}

    # RPO: 企業によってタグ揺れがあるため候補を複数
    rpo_candidates = [
        "RemainingPerformanceObligation",
        "TransactionPriceAllocatedToRemainingPerformanceObligations",
        "RemainingPerformanceObligationRevenue",
    ]
    # Contract liabilities / deferred revenue
    contract_liab_candidates = [
        "ContractWithCustomerLiability",
        "DeferredRevenue",
    ]
    contract_liab_current_candidates = [
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
    ]

    # それぞれ「最新優先」で1タグ採用
    tag, df_rpo = _pick_best_tag_latest_first(facts_json, rpo_candidates)
    meta["rpo_tag"] = tag
    tag, df_cl = _pick_best_tag_latest_first(facts_json, contract_liab_candidates)
    meta["contract_liab_tag"] = tag
    tag, df_cl_cur = _pick_best_tag_latest_first(facts_json, contract_liab_current_candidates)
    meta["contract_liab_current_tag"] = tag

    years = sorted(set(df_rpo["year"].tolist()) | set(df_cl["year"].tolist()) | set(df_cl_cur["year"].tolist()))
    years = [int(y) for y in years]
    if not years:
        return pd.DataFrame(), meta

    def val_at(df: pd.DataFrame, y: int) -> float:
        if df.empty:
            return np.nan
        r = df[df["year"] == y]
        if r.empty:
            return np.nan
        return float(r["val"].iloc[-1])

    rows = []
    for y in years:
        rpo = val_at(df_rpo, y)
        cl = val_at(df_cl, y)
        clc = val_at(df_cl_cur, y)

        rows.append([y, _to_musd(rpo), _to_musd(cl), _to_musd(clc)])

    out = pd.DataFrame(rows, columns=["FY", "RPO(M$)", "ContractLiab(M$)", "ContractLiabCurrent(M$)"])
    meta["years"] = years
    return out, meta


def plot_rpo_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()

    rpo = df["RPO(M$)"].astype(float).to_numpy()
    cl = df["ContractLiab(M$)"].astype(float).to_numpy()
    clc = df["ContractLiabCurrent(M$)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    # 取れている系列だけ描く（全部NaNなら描かない）
    if np.isfinite(rpo).any():
        ax.plot(x, rpo, label="RPO", linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(cl).any():
        ax.plot(x, cl, label="Contract Liabilities (Total)", linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(clc).any():
        ax.plot(x, clc, label="Contract Liabilities (Current)", linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    ax.tick_params(axis="x", rotation=0, colors="white")
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

    st.markdown("## 長期ファンダ（PL / BS / CF / 受注残・RPO）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("input_form"):
        ticker = st.text_input("Ticker（米国株）", value="AVGO")
        n_years = st.slider("年数（PL/CF/RPO 共通：最新から過去N年）", min_value=3, max_value=15, value=10)
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

    tab_pl, tab_bs, tab_cf, tab_rpo = st.tabs(["PL（年次）", "BS（最新年）", "CF（年次）", "受注残 / RPO"])

    with tab_pl:
        pl_table, pl_meta = build_pl_annual_table(facts)
        if pl_table.empty:
            st.error("PLデータが取得できませんでした。")
            st.write(pl_meta)
            st.stop()

        pl_disp = _slice_latest_n_years(pl_table, int(n_years))
        st.caption(f"PL: 表示 {len(pl_disp)} 年（最新年: {int(pl_table['FY'].max())}）")
        plot_pl_annual(pl_disp, f"{company_name} ({ticker.upper()}) - Income Statement (Annual)")

        st.markdown("### 年次PL（百万USD）")
        st.dataframe(
            pl_disp.style.format(
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
            st.write(pl_meta)

    with tab_bs:
        year = _latest_year_from_assets(facts)
        if year is None:
            st.error("Assets（総資産）が取得できませんでした。")
            st.stop()

        snap, bs_meta = build_bs_latest_simple(facts, year)
        st.caption(f"BS: 最新年 {year}")
        plot_bs_bar(snap, f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest)")

        st.markdown("### BS（最新年 / 百万USD）")
        snap_disp = snap.copy()
        snap_disp["Value"] = snap_disp["Value"].map(lambda v: f"{v:,.0f}" if isinstance(v, (int,float)) and np.isfinite(v) else str(v))
        st.dataframe(snap_disp, use_container_width=True, hide_index=True)

        with st.expander("BSデバッグ", expanded=False):
            st.write(bs_meta)

    with tab_cf:
        cf_table, cf_meta = build_cf_annual_table(facts)
        if cf_table.empty:
            st.error("CFデータが取得できませんでした。")
            st.write(cf_meta)
            st.stop()

        cf_disp = _slice_latest_n_years(cf_table, int(n_years))
        st.caption(f"CF: 表示 {len(cf_disp)} 年（最新年: {int(cf_table['FY'].max())}）")
        plot_cf_annual(cf_disp, f"{company_name} ({ticker.upper()}) - Cash Flow (Annual)")

        st.markdown("### 年次CF（百万USD）")
        st.dataframe(
            cf_disp.style.format(
                {
                    "CFO(M$)": "{:,.0f}",
                    "CFI(M$)": "{:,.0f}",
                    "CFF(M$)": "{:,.0f}",
                    "FCF(M$)": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("CFデバッグ（採用タグ）", expanded=False):
            st.write(cf_meta)
            st.markdown("- **FCF定義**: FCF = CFO + CapEx（CapExは通常マイナス。正符号の場合は -abs に補正）")

    with tab_rpo:
        rpo_table, rpo_meta = build_rpo_annual_table(facts)
        if rpo_table.empty:
            st.warning("この銘柄ではRPO/契約負債（年次）がXBRL上で取得できない可能性があります。")
            st.write(rpo_meta)
            st.stop()

        rpo_disp = _slice_latest_n_years(rpo_table.rename(columns={"FY": "FY"}), int(n_years))
        # _slice_latest_n_yearsは FY 列を前提にしているので合わせる
        rpo_disp = rpo_table.copy()
        rpo_disp = rpo_disp.sort_values("FY")
        # 最新からN年
        latest = int(rpo_disp["FY"].max())
        miny = latest - int(n_years) + 1
        rpo_disp = rpo_disp[rpo_disp["FY"] >= miny].reset_index(drop=True)

        st.caption(f"RPO: 表示 {len(rpo_disp)} 年（最新年: {latest}）")
        plot_rpo_annual(rpo_disp, f"{company_name} ({ticker.upper()}) - RPO / Contract Liabilities (Annual)")

        st.markdown("### 受注残 / RPO（百万USD）")
        st.dataframe(
            rpo_disp.style.format(
                {
                    "RPO(M$)": "{:,.0f}",
                    "ContractLiab(M$)": "{:,.0f}",
                    "ContractLiabCurrent(M$)": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("RPOデバッグ（採用タグ）", expanded=False):
            st.write(rpo_meta)


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
