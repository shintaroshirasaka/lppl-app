# fund_app.py
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from auth_gate import require_admin_token

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
# BS (Balance Sheet) - Latest year only, with breakdown
# =========================
def _value_at_year(facts_json: dict, candidates: list[str], year: int) -> tuple[str | None, float]:
    """
    candidatesの中から、指定yearで値が取れる最初のタグを返す（値も返す）
    """
    for tag in candidates:
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


def build_bs_latest_breakdown(facts_json: dict, threshold_ratio: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    最新年BSを、内訳つきで返す。
    - threshold_ratio: 総資産に対してこの比率以上の科目を個別表示、それ以外はOtherへ
    戻り:
      assets_breakdown_df: columns=["Group","Item","Value(M$)","ShareOfAssets"]
      le_breakdown_df:     columns=["Group","Item","Value(M$)","ShareOfAssets"]
      meta
    """
    meta = {"threshold_ratio": threshold_ratio}

    # まず最新年を決める（Assetsの最新年）
    assets_tag, df_assets = _pick_best_tag_latest_first(facts_json, ["Assets"])
    meta["assets_tag"] = assets_tag
    if df_assets.empty:
        return pd.DataFrame(), pd.DataFrame(), meta

    bs_year = int(df_assets["year"].max())
    meta["bs_year"] = bs_year

    total_assets = float(df_assets[df_assets["year"] == bs_year]["val"].iloc[-1])
    if not np.isfinite(total_assets) or total_assets <= 0:
        return pd.DataFrame(), pd.DataFrame(), meta

    # --- Assets components (候補タグは複数持つ) ---
    asset_items = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue",
                                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]),
        ("Receivables", ["AccountsReceivableNetCurrent",
                         "AccountsReceivableNet",
                         "ReceivablesNetCurrent"]),
        ("Inventory", ["InventoryNet"]),
        ("Other Current Assets", ["OtherAssetsCurrent"]),
        ("PPE (Net)", ["PropertyPlantAndEquipmentNet"]),
        ("Goodwill", ["Goodwill"]),
        ("Intangibles", ["IntangibleAssetsNetExcludingGoodwill",
                         "IntangibleAssetsNetIncludingGoodwill"]),
        ("Other Noncurrent Assets", ["OtherAssetsNoncurrent",
                                     "OtherAssets",
                                     "NoncurrentAssets"]),
    ]

    # --- Liabilities components ---
    liab_items = [
        ("Current Liabilities", ["LiabilitiesCurrent"]),
        ("Long-term Debt", ["LongTermDebtNoncurrent",
                            "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                            "LongTermDebt"]),
        ("Other Noncurrent Liabilities", ["LiabilitiesNoncurrent",
                                          "OtherLiabilitiesNoncurrent"]),
    ]

    # --- Equity ---
    equity_items = [
        ("Equity", ["StockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    ]

    # 主要3つ（Liabilities/Equity/Assets）も取っておく（整合チェック）
    liab_tag, df_liab = _pick_best_tag_latest_first(facts_json, ["Liabilities"])
    eq_tag, df_eq = _pick_best_tag_latest_first(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    )
    meta["liabilities_tag"] = liab_tag
    meta["equity_tag"] = eq_tag

    # ========== build assets breakdown ==========
    a_rows = []
    a_sum = 0.0
    for name, tags in asset_items:
        used_tag, v = _value_at_year(facts_json, tags, bs_year)
        meta[f"asset_tag_{name}"] = used_tag
        if np.isfinite(v) and v > 0:
            a_sum += v
            a_rows.append((name, v))

    # 合計がTotalAssetsを超えたり不足したりするので、差分はOtherへ吸収
    # ただし “10%ルール”で表示するのは後段で
    residual_assets = total_assets - a_sum
    if np.isfinite(residual_assets):
        a_rows.append(("Other (Residual)", residual_assets))

    assets_df = pd.DataFrame(a_rows, columns=["Item", "ValueUSD"])
    assets_df["ShareOfAssets"] = assets_df["ValueUSD"] / total_assets

    # 10%未満をOtherへまとめる（Residualも含めて）
    big = assets_df[assets_df["ShareOfAssets"] >= threshold_ratio].copy()
    small = assets_df[assets_df["ShareOfAssets"] < threshold_ratio].copy()
    other_val = float(small["ValueUSD"].sum()) if not small.empty else 0.0

    final_assets = big[["Item", "ValueUSD"]].copy()
    if abs(other_val) > 1e-6:
        final_assets = pd.concat([final_assets, pd.DataFrame([["Other (< threshold)", other_val]], columns=["Item","ValueUSD"])], ignore_index=True)

    final_assets["Group"] = "Assets"
    final_assets["Value(M$)"] = final_assets["ValueUSD"].map(_to_musd)
    final_assets["ShareOfAssets"] = final_assets["ValueUSD"] / total_assets
    final_assets = final_assets[["Group","Item","Value(M$)","ShareOfAssets"]]

    # ========== build liabilities+equity breakdown ==========
    # Liabilities total（最新年）
    liab_total = np.nan
    if not df_liab.empty and (df_liab["year"] == bs_year).any():
        liab_total = float(df_liab[df_liab["year"] == bs_year]["val"].iloc[-1])

    eq_total = np.nan
    if not df_eq.empty and (df_eq["year"] == bs_year).any():
        eq_total = float(df_eq[df_eq["year"] == bs_year]["val"].iloc[-1])

    # 内訳：負債（上）を分解
    l_rows = []
    l_sum = 0.0
    for name, tags in liab_items:
        used_tag, v = _value_at_year(facts_json, tags, bs_year)
        meta[f"liab_tag_{name}"] = used_tag
        if np.isfinite(v) and v > 0:
            l_sum += v
            l_rows.append((name, v))

    if np.isfinite(liab_total):
        residual_liab = liab_total - l_sum
        if np.isfinite(residual_liab):
            l_rows.append(("Other Liabilities (Residual)", residual_liab))
    else:
        # Liabilities合計が取れない場合：内訳合計を合計扱いにする
        liab_total = l_sum

    liab_df = pd.DataFrame(l_rows, columns=["Item","ValueUSD"])
    liab_df["ShareOfAssets"] = liab_df["ValueUSD"] / total_assets

    # Equity（下）は基本1項目（10%未満でも表示してOKだが、要件は「内訳は10%超」）
    e_rows = []
    for name, tags in equity_items:
        used_tag, v = _value_at_year(facts_json, tags, bs_year)
        meta[f"equity_tag_{name}"] = used_tag
        if np.isfinite(v):
            e_rows.append((name, v))

    eq_df = pd.DataFrame(e_rows, columns=["Item","ValueUSD"])
    if eq_df.empty and np.isfinite(eq_total):
        eq_df = pd.DataFrame([["Equity", eq_total]], columns=["Item","ValueUSD"])
    eq_df["ShareOfAssets"] = eq_df["ValueUSD"] / total_assets

    # 10%未満の負債内訳をOtherへ
    big_l = liab_df[liab_df["ShareOfAssets"] >= threshold_ratio].copy()
    small_l = liab_df[liab_df["ShareOfAssets"] < threshold_ratio].copy()
    other_l_val = float(small_l["ValueUSD"].sum()) if not small_l.empty else 0.0

    final_liab = big_l[["Item","ValueUSD"]].copy()
    if abs(other_l_val) > 1e-6:
        final_liab = pd.concat([final_liab, pd.DataFrame([["Other Liabilities (< threshold)", other_l_val]], columns=["Item","ValueUSD"])], ignore_index=True)

    final_liab["Group"] = "Liabilities"
    final_liab["Value(M$)"] = final_liab["ValueUSD"].map(_to_musd)
    final_liab["ShareOfAssets"] = final_liab["ValueUSD"] / total_assets
    final_liab = final_liab[["Group","Item","Value(M$)","ShareOfAssets"]]

    # Equityは“下”に置きたいのでGroupはEquity
    final_eq = eq_df.copy()
    final_eq["Group"] = "Equity"
    final_eq["Value(M$)"] = final_eq["ValueUSD"].map(_to_musd)
    final_eq = final_eq[["Group","Item","Value(M$)","ShareOfAssets"]]

    le_df = pd.concat([final_eq, final_liab], ignore_index=True)

    meta["total_assets_usd"] = total_assets
    meta["liabilities_total_usd"] = liab_total
    meta["equity_total_usd"] = eq_total

    return final_assets, le_df, meta


def plot_bs_stacked(assets_df: pd.DataFrame, le_df: pd.DataFrame, title: str):
    """
    左：資産（内訳積み上げ）
    右：負債（上）＋純資産（下）（内訳積み上げ）
    """
    # 並び順：資産は大きい順（見やすい）
    a = assets_df.copy()
    a["abs"] = a["Value(M$)"].abs()
    a = a.sort_values("abs", ascending=False).drop(columns=["abs"])

    # 右側：Equityを“下”、Liabilitiesを“上”
    eq = le_df[le_df["Group"] == "Equity"].copy()
    li = le_df[le_df["Group"] == "Liabilities"].copy()
    li["abs"] = li["Value(M$)"].abs()
    li = li.sort_values("abs", ascending=False).drop(columns=["abs"])

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    x_assets = 0
    x_le = 1

    # --- Assets stack (bottom-up) ---
    bottom = 0.0
    for _, r in a.iterrows():
        v = float(r["Value(M$)"])
        ax.bar([x_assets], [v], bottom=[bottom], alpha=0.65, label=f"A: {r['Item']}")
        bottom += v

    # --- Liabilities+Equity stack (Equity bottom, Liabilities on top) ---
    bottom = 0.0
    # Equity bottom
    for _, r in eq.iterrows():
        v = float(r["Value(M$)"])
        ax.bar([x_le], [v], bottom=[bottom], alpha=0.65, label=f"E: {r['Item']}")
        bottom += v
    # Liabilities on top
    for _, r in li.iterrows():
        v = float(r["Value(M$)"])
        ax.bar([x_le], [v], bottom=[bottom], alpha=0.65, label=f"L: {r['Item']}")
        bottom += v

    ax.set_xticks([x_assets, x_le])
    ax.set_xticklabels(["Assets", "Liabilities + Equity"], color="white")
    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")

    # 凡例は長くなりやすいので右側へ
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

    st.markdown("## 長期ファンダ（PL 年次 / BS 最新年 内訳つき）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("fund_form"):
        ticker = st.text_input("Ticker（米国株）", value="AAPL")
        years = st.slider("PL: 表示する年数（最新から過去N年）", min_value=3, max_value=15, value=10)
        # BS内訳：総資産比で何%以上を出すか（要件は10%）
        bs_threshold = st.slider("BS: 内訳表示の閾値（総資産比）", min_value=0.05, max_value=0.30, value=0.10, step=0.01)
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
        assets_df, le_df, meta_bs = build_bs_latest_breakdown(facts, threshold_ratio=float(bs_threshold))
        if assets_df.empty or le_df.empty:
            st.error("BSデータが取得できませんでした（Assetsが取れない等）。")
            st.write(meta_bs)
            st.stop()

        st.caption(f"BS: 最新年 {meta_bs.get('bs_year')}（総資産比 {bs_threshold*100:.0f}% 以上を個別表示）")

        plot_bs_stacked(
            assets_df,
            le_df,
            f"{company_name} ({ticker.upper()}) - Balance Sheet (Latest, Breakdown)",
        )

        st.markdown("### 資産内訳（最新年 / 百万USD）")
        a_disp = assets_df.copy()
        a_disp["Value(M$)"] = a_disp["Value(M$)"].astype(float).map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "")
        a_disp["ShareOfAssets"] = a_disp["ShareOfAssets"].map(lambda x: f"{x*100:.1f}%" if np.isfinite(x) else "")
        st.dataframe(a_disp, use_container_width=True, hide_index=True)

        st.markdown("### 負債・純資産内訳（最新年 / 百万USD）")
        le_disp = le_df.copy()
        le_disp["Value(M$)"] = le_disp["Value(M$)"].astype(float).map(lambda v: f"{v:,.0f}" if np.isfinite(v) else "")
        le_disp["ShareOfAssets"] = le_disp["ShareOfAssets"].map(lambda x: f"{x*100:.1f}%" if np.isfinite(x) else "")
        st.dataframe(le_disp, use_container_width=True, hide_index=True)

        with st.expander("BSデバッグ（採用タグなど）", expanded=False):
            st.write(meta_bs)
            st.write(f"CIK: {cik10}")
            st.write(f"SEC_USER_AGENT: {SEC_USER_AGENT}")


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
