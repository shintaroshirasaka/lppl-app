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


def _latest_year_from_assets(facts_json: dict) -> tuple[int | None, str | None]:
    df = _extract_annual_series_usd(facts_json, "Assets")
    if df.empty:
        return None, None
    return int(df["year"].max()), "Assets"


def _value_for_year(facts_json: dict, tag_candidates: list[str], year: int) -> tuple[str | None, float]:
    """
    指定yearで値が取れる最初のタグを返す
    """
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


# =========================
# BS (latest year) - simple stacks
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
    year, _ = _latest_year_from_assets(facts_json)
    if year is None:
        return pd.DataFrame(), meta

    # 必要タグ
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
    ax.bar(0, vals["Noncurrent Assets (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Assets")
    bottom += vals["Noncurrent Assets (M$)"]
    ax.bar(0, vals["Current Assets (M$)"], bottom=bottom, alpha=0.7, label="Current Assets")

    # ---- Liabilities + Equity (right) ----
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
# Pie charts: Top3 + Other
# =========================
def _top3_plus_other(items: list[tuple[str, float]], total: float) -> tuple[list[str], list[float]]:
    """
    items: [(name, valueUSD), ...]
    - valueUSDが正のものだけ対象
    - 上位3を抽出し、残りはOther
    """
    pos = [(n, float(v)) for n, v in items if np.isfinite(v) and float(v) > 0]
    pos.sort(key=lambda x: x[1], reverse=True)

    top = pos[:3]
    top_sum = sum(v for _, v in top)
    other = max(total - top_sum, 0.0) if np.isfinite(total) else sum(v for _, v in pos[3:])

    labels = [f"{n}" for n, _ in top]
    sizes = [v for _, v in top]
    if other > 0:
        labels.append("Other")
        sizes.append(other)

    # total=0だとpieが壊れるので保険
    if sum(sizes) <= 0:
        return (["No data"], [1.0])
    return labels, sizes


def build_bs_pies_latest(facts_json: dict, year: int) -> tuple[dict, dict, dict]:
    """
    A: 資産の上位3 + その他
    B: 負債+純資産の上位3 + その他

    ※タグは「取れやすい代表項目」を候補にしておき、
      取れたものの中で上位3を採用する。
    """
    meta = {"year": year}

    # --- Totals ---
    _, total_assets = _value_for_year(facts_json, ["Assets"], year)
    _, total_liab = _value_for_year(facts_json, ["Liabilities"], year)
    _, total_eq = _value_for_year(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        year,
    )
    total_le = (total_liab if np.isfinite(total_liab) else 0.0) + (total_eq if np.isfinite(total_eq) else 0.0)

    meta["total_assets_usd"] = total_assets
    meta["total_liab_usd"] = total_liab
    meta["total_equity_usd"] = total_eq

    # --- Assets candidates (代表項目) ---
    asset_candidates = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue",
                                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
        ("Marketable Securities", ["AvailableForSaleSecuritiesCurrent",
                                   "AvailableForSaleSecuritiesNoncurrent",
                                   "MarketableSecuritiesCurrent",
                                   "MarketableSecuritiesNoncurrent"]),
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

    # 合計に対するOtherは _top3_plus_other で作る

    # --- Liabilities + Equity candidates ---
    # “勘定項目例：利益剰余金”に寄せて、Retained Earningsを候補に入れる。
    le_candidates = [
        # Liabilities side
        ("Current Liabilities", ["LiabilitiesCurrent"]),
        ("Noncurrent Liabilities", ["LiabilitiesNoncurrent"]),
        ("Long-term Debt", ["LongTermDebtNoncurrent",
                            "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                            "LongTermDebt"]),
        # Equity side (component)
        ("Retained Earnings", ["RetainedEarningsAccumulatedDeficit"]),
        # fallback equity total (componentが取れない場合でも何かは出す)
        ("Equity (Total)", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    ]

    le_items = []
    for name, tags in le_candidates:
        used, v = _value_for_year(facts_json, tags, year)
        meta[f"le_{name}_tag"] = used
        # TreasuryStockなど負の項目は円グラフに向かないので「正のものだけ」
        if np.isfinite(v) and v > 0:
            le_items.append((name, v))

    # A pie (assets)
    if not np.isfinite(total_assets) or total_assets <= 0:
        total_assets = sum(v for _, v in asset_items if np.isfinite(v) and v > 0)
    a_labels, a_sizes = _top3_plus_other(asset_items, total_assets)

    # B pie (liab+equity)
    if not np.isfinite(total_le) or total_le <= 0:
        total_le = sum(v for _, v in le_items if np.isfinite(v) and v > 0)
    b_labels, b_sizes = _top3_plus_other(le_items, total_le)

    # ％表示用に返す
    assets_pie = {"labels": a_labels, "sizes": a_sizes, "total": total_assets}
    le_pie = {"labels": b_labels, "sizes": b_sizes, "total": total_le}

    return assets_pie, le_pie, meta


def plot_two_pies(assets_pie: dict, le_pie: dict, year: int):
    """
    棒グラフの下に、A/Bの円グラフを左右に並べる
    """
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

    st.markdown("## 長期ファンダ（BS：棒グラフ＋円グラフA/B）")
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

    # 最新年（Assetsを基準）
    year, _ = _latest_year_from_assets(facts)
    if year is None:
        st.error("Assets（総資産）が取得できませんでした。")
        st.stop()

    # BS 棒グラフ用
    snap, meta_bs = build_bs_latest_simple(facts)
    if snap.empty:
        st.error("BS（流動/非流動/純資産）の取得に失敗しました。")
        st.write(meta_bs)
        st.stop()

    st.caption(f"BS: 最新年 {year}")

    # 1) 棒グラフ
    plot_bs_bar(snap, f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest)")

    # 2) 円グラフ A/B
    assets_pie, le_pie, meta_pie = build_bs_pies_latest(facts, year)
    plot_two_pies(assets_pie, le_pie, year)

    # 表
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

    with st.expander("デバッグ（タグ採用状況）", expanded=False):
        st.write({"bs": meta_bs, "pies": meta_pie})
        st.write(f"CIK: {cik10}")
        st.write(f"SEC_USER_AGENT: {SEC_USER_AGENT}")


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
