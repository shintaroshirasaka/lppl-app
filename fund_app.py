import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import re

# ※ auth_gateファイルが同階層にある前提です
try:
    from auth_gate import require_admin_token
except ImportError:
    # テスト用にダミー関数を用意
    def require_admin_token():
        return "debug@example.com"

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
    try:
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
    except Exception as e:
        st.error(f"Tickerマップの取得に失敗しました: {e}")
        return {}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_company_facts(cik10: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# =========================
# XBRL annual extractor (USD)
# =========================
def _extract_annual_series_usd(facts_json: dict, xbrl_tag: str, include_segments: bool = False) -> pd.DataFrame:
    """
    年次相当（USD）抽出（10-K系）
    """
    facts_root = facts_json.get("facts", {})
    if not facts_root:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "filed", "segment"])

    node = []
    
    # 既知のプレフィックスを探す
    for prefix in ["us-gaap", "srt", "ifrs-full", "dei"]:
        if prefix in facts_root and xbrl_tag in facts_root[prefix]:
            node = facts_root[prefix][xbrl_tag].get("units", {}).get("USD", [])
            break
    
    # 見つからなければカスタムタグを探す
    if not node:
        for k in facts_root.keys():
            if k not in ["us-gaap", "srt", "ifrs-full", "dei"]:
                if xbrl_tag in facts_root[k]:
                    node = facts_root[k][xbrl_tag].get("units", {}).get("USD", [])
                    if node:
                        break
    
    rows = []
    for x in node:
        form = str(x.get("form", "")).upper().strip()
        fp = str(x.get("fp", "")).upper().strip()
        end = x.get("end")
        val = x.get("val")
        fy_raw = x.get("fy", None)
        filed = x.get("filed")
        segment_obj = x.get("segment")

        if not end or val is None:
            continue
        # 10-K, 20-F (ADR), 40-F などを許容
        if not (form.startswith("10-K") or form.startswith("20-F") or form.startswith("40-F")):
            continue

        end_ts = pd.to_datetime(end, errors="coerce")
        filed_ts = pd.to_datetime(filed, errors="coerce")
        if pd.isna(end_ts):
            continue

        # ▼▼ 修正箇所：Q4を除外し、FY/CYのみを年次として扱うように変更 ▼▼
        annual_fp = fp in {"FY", "CY"}
        # ▲▲ 修正ここまで ▲▲
        
        if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
            year_key = int(fy_raw)
        else:
            year_key = int(end_ts.year)

        rows.append({
            "year": year_key,
            "end": end_ts,
            "val": _safe_float(val),
            "annual_fp": int(annual_fp),
            "filed": filed_ts,
            "segment": segment_obj
        })

    if not rows:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "filed", "segment"])

    df = pd.DataFrame(rows).dropna(subset=["val"])
    
    # ここで annual_fp=1 のものだけにフィルタリングすることで、Q4データの混入を防ぐ
    df = df[df["annual_fp"] == 1]

    if not include_segments:
        # 連結データのみ
        df = df[df["segment"].isnull()]
        # 重複排除（最新のfiled優先）
        df = df.sort_values(["year", "filed"]).drop_duplicates(subset=["year"], keep="last")
        df = df.sort_values("year")
        return df
    else:
        return df


def _pick_best_tag_latest_first_usd(facts_json: dict, candidates: list[str]) -> tuple[str | None, pd.DataFrame]:
    """
    固定リストからタグを選択（最新年優先）
    """
    best_tag = None
    best_df = pd.DataFrame(columns=["year", "end", "val", "annual_fp"])
    best_max_year = -1
    best_n = -1
    best_last_end = pd.Timestamp("1900-01-01")

    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False)
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


def _find_best_tag_dynamic(facts_json: dict, keywords: list[str], exclude_keywords: list[str] = None, must_end_with: str = None) -> tuple[str | None, pd.DataFrame]:
    """
    XBRL内の全タグをスキャンし、キーワードに合致し、かつ最新データを持つタグを探す
    """
    if exclude_keywords is None:
        exclude_keywords = []
    
    # 検索対象の全タグをリストアップ
    all_tags = []
    facts_root = facts_json.get("facts", {})
    for prefix in facts_root:
        for tag in facts_root[prefix]:
            all_tags.append(tag)
            
    # キーワードマッチング
    candidates = []
    for tag in all_tags:
        tag_lower = tag.lower()
        
        # 必須条件チェック
        if not any(k.lower() in tag_lower for k in keywords):
            continue
        # 除外キーワードチェック
        if any(ek.lower() in tag_lower for ek in exclude_keywords):
            continue
        # 末尾チェック
        if must_end_with and not tag_lower.endswith(must_end_with.lower()):
            continue
            
        candidates.append(tag)
    
    if not candidates:
        return None, pd.DataFrame()

    # データ抽出と評価
    best_tag = None
    best_df = pd.DataFrame(columns=["year"])
    best_score = (-1, -1) # (max_year, count)

    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False)
        if df.empty:
            continue
            
        max_year = int(df["year"].max())
        count = len(df)
        
        # 最新年を最優先、次にデータ数
        score = (max_year, count)
        
        if score > best_score:
            best_score = score
            best_tag = tag
            best_df = df
            
    return best_tag, best_df


def _slice_latest_n_years(table: pd.DataFrame, n_years: int) -> pd.DataFrame:
    if table.empty:
        return table
    if "FY" in table.columns:
        col = "FY"
    elif "year" in table.columns:
        col = "year"
    else:
        return table

    latest_year = int(table[col].max())
    min_year = latest_year - int(n_years) + 1
    return table[table[col] >= min_year].sort_values(col).reset_index(drop=True)


# =========================
# Composite (USD) fill-by-year
# =========================
def _build_composite_by_year_usd(facts_json: dict, tag_priority: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    優先順位の高いタグリストから、年度ごとに最も確からしい値を合成する
    """
    tag_series = {}
    tag_years = {}
    for tag in tag_priority:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False)
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
# Segment Analysis Helpers
# =========================
def _parse_segment_info(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        seg = r["segment"]
        if not seg:
            continue
        
        dims = []
        if isinstance(seg, list):
            dims = seg
        elif isinstance(seg, dict):
            dims = [seg]
        
        for d in dims:
            dim_name = d.get("dimension")
            mem_name = d.get("value")
            if dim_name and mem_name:
                rows.append({
                    "year": r["year"],
                    "filed": r["filed"],
                    "val": r["val"],
                    "dimension": dim_name,
                    "member": mem_name
                })
    
    if not rows:
        return pd.DataFrame(columns=["year", "val", "dimension", "member"])
    
    out = pd.DataFrame(rows)
    # 重複排除: 最新のfiledを優先
    out = out.sort_values(["year", "dimension", "member", "filed"]).drop_duplicates(subset=["year", "dimension", "member"], keep="last")
    return out


def build_segment_table(facts_json: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # (7) セグメント: タグ候補を拡充
    revenue_tags = [
        "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax", 
        "SalesToExternalCustomers", "RevenuesNetOfInterestExpense", 
        "SalesRevenueGoodsNet", "SalesRevenueServicesNet", 
        "SegmentReportingInformationRevenue", "NetOperatingRevenues"
    ]
    
    all_seg_rows = []
    
    for tag in revenue_tags:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=True)
        if df.empty:
            continue
        parsed = _parse_segment_info(df)
        if not parsed.empty:
            parsed["src_tag"] = tag
            all_seg_rows.append(parsed)
            
    if not all_seg_rows:
        return pd.DataFrame(), pd.DataFrame()
    
    merged = pd.concat(all_seg_rows, ignore_index=True)
    # 値が大きい方を優先（詳細な数値が取れている場合を想定）
    merged = merged.sort_values("val", ascending=False).drop_duplicates(subset=["year", "dimension", "member"], keep="first")
    
    def classify_axis(dim_name: str) -> str:
        d = dim_name.lower()
        business_keywords = [
            "businesssegment", "segmentreporting", "productor", "statementbusinesssegmentsaxis", 
            "statementoperatingactivitiessegmentaxis", "consolidationitems", "segmentdomain", 
            "operatingsegments", "concentrationrisk"
        ]
        geo_keywords = [
            "geographical", "geoaxis", "statementgeographicalaxis", "entitywideinfo", "reportablegeographical"
        ]

        if any(x in d for x in business_keywords):
            return "Business"
        if any(x in d for x in geo_keywords):
            return "Geography"
        return "Other"

    merged["axis_type"] = merged["dimension"].apply(classify_axis)
    
    def clean_member(m: str) -> str:
        if ":" in m:
            m = m.split(":")[-1]
        if m.endswith("Member"):
            m = m[:-6]
        return m

    merged["member_clean"] = merged["member"].apply(clean_member)
    
    # 除外キーワード
    exclude_keywords = ["total", "elimination", "adjust", "consolidation", "allother", "intersegment"]
    
    def is_excluded(m_clean):
        m_lower = m_clean.lower()
        if any(ek in m_lower for ek in exclude_keywords):
            return True
        return False

    merged = merged[~merged["member_clean"].apply(is_excluded)]

    def pivot_data(axis_type):
        subset = merged[merged["axis_type"] == axis_type]
        if subset.empty:
            return pd.DataFrame()
        
        pivoted = subset.pivot(index="year", columns="member_clean", values="val")
        pivoted = pivoted.sort_index()
        return pivoted

    return pivot_data("Business"), pivot_data("Geography")


def plot_stacked_bar(df: pd.DataFrame, title: str):
    if df.empty:
        st.info(f"{title}: データが見つかりませんでした。")
        return

    df_m = df / 1_000_000.0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")
    
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(df_m.columns)))
    
    df_m.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_ylabel("Revenue (Million USD)", color="white")
    ax.tick_params(colors="white", axis='x', rotation=0)
    ax.tick_params(colors="white", axis='y')
    ax.grid(color="#333333", alpha=0.6, axis="y")
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    st.pyplot(fig)


# =========================
# PL (annual) - MODIFIED
# =========================
def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    # (1) 売上高: NetOperatingRevenues 等を追加して過去データを補完
    revenue_priority = [
        "RevenueFromContractWithCustomerExcludingAssessedTax", 
        "Revenues", 
        "NetOperatingRevenues", # コカ・コーラの過去データ
        "SalesRevenueGoodsNet", # 消費財メーカー
        "SalesRevenueNet", 
        "SalesRevenue",
        "OperatingRevenue",
        "OperatingRevenues"
    ]
    
    # (3) 純利益: 株主帰属分を最優先に変更（他社アプリとの整合性向上）
    net_income_priority = [
        "NetIncomeLossAvailableToCommonStockholdersBasic", # 最優先
        "NetIncomeLoss", 
        "ProfitLoss", 
        "IncomeLossFromContinuingOperations"
    ]
    
    # 営業利益
    op_income_priority = ["OperatingIncomeLoss", "OperatingIncome"]
    
    # (2) 税前利益は削除

    meta = {}
    
    # コンポジット処理
    rev_df, rev_meta = _build_composite_by_year_usd(facts_json, revenue_priority)
    meta["revenue_composite"] = rev_meta
    
    ni_df, ni_meta = _build_composite_by_year_usd(facts_json, net_income_priority)
    meta["net_income_composite"] = ni_meta
    
    op_df, op_meta = _build_composite_by_year_usd(facts_json, op_income_priority)
    meta["op_income_composite"] = op_meta

    if rev_df.empty:
        return pd.DataFrame(), meta

    rev_map = dict(zip(rev_df["year"].astype(int), rev_df["value"].astype(float)))
    ni_map = dict(zip(ni_df["year"].astype(int), ni_df["value"].astype(float))) if not ni_df.empty else {}
    op_map = dict(zip(op_df["year"].astype(int), op_df["value"].astype(float))) if not op_df.empty else {}
    
    # 日付取得用（Pretax削除に伴い、データが豊富なRevenueタグ等から代表して取得）
    tag_for_date, df_date = _pick_best_tag_latest_first_usd(facts_json, ["Revenues", "NetOperatingRevenues", "NetIncomeLoss"])
    end_map = {}
    if not df_date.empty:
        end_map = dict(zip(df_date["year"].astype(int), df_date["end"]))

    years = sorted(set(rev_map.keys()) | set(op_map.keys()) | set(ni_map.keys()))
    rows = []
    for y in years:
        end = end_map.get(y)
        end_str = str(pd.to_datetime(end).date()) if end is not None and pd.notna(end) else f"{y}-12-31"
        rows.append([
            y, end_str,
            _to_musd(rev_map.get(y, np.nan)),
            _to_musd(op_map.get(y, np.nan)),
            # 税前利益(Pretax)を削除
            _to_musd(ni_map.get(y, np.nan))
        ])

    # カラム定義からPretaxを削除
    out = pd.DataFrame(rows, columns=["FY", "End", "Revenue(M$)", "OpIncome(M$)", "NetIncome(M$)"])
    out = out.sort_values("FY").reset_index(drop=True)
    meta["years"] = years
    return out, meta


def plot_pl_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    revenue = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()
    # Pretax削除
    ni = df["NetIncome(M$)"].astype(float).to_numpy()

    fig, ax_left = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax_left.set_facecolor("#0b0c0e")

    ax_left.plot(x, op, label="Operating Income", linewidth=2.5, marker="o", markersize=6)
    # Pretax描画削除
    ax_left.plot(x, ni, label="Net Income (Attributable)", linewidth=2.5, marker="o", markersize=6)

    ax_left.set_ylabel("Profit (Million USD)", color="white")
    ax_left.tick_params(colors="white")
    ax_left.grid(color="#333333", alpha=0.6)

    ax_right = ax_left.twinx()
    ax_right.bar(x, revenue, alpha=0.55, width=0.6, label="Revenue")
    ax_right.set_ylabel("Revenue (Million USD)", color="white")
    ax_right.tick_params(colors="white")

    ax_left.set_title(title, color="white")
    ax_left.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


def plot_operating_margin(table: pd.DataFrame, title: str):
    df = table.copy()
    # ソートを保証
    df = df.sort_values("FY")
    x = df["FY"].astype(str).tolist()
    rev = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()

    margin = np.full_like(rev, np.nan, dtype=float)
    for i in range(len(rev)):
        if np.isfinite(rev[i]) and rev[i] != 0 and np.isfinite(op[i]):
            margin[i] = op[i] / rev[i] * 100.0

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, margin, linewidth=2.5, marker="o", markersize=6, label="Operating Margin (%)")
    ax.set_ylabel("%", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(f"{title} (GAAP Base)", color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


# =========================
# BS (latest) - FIXED PIE CHART LOGIC
# =========================
def _latest_year_from_assets(facts_json: dict) -> int | None:
    df = _extract_annual_series_usd(facts_json, "Assets")
    if df.empty:
        return None
    return int(df["year"].max())


def _value_for_year_usd(facts_json: dict, tag_candidates: list[str], year: int) -> tuple[str | None, float]:
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
    meta = {"bs_year": year}

    assets_tag, assets_total = _value_for_year_usd(facts_json, ["Assets"], year)
    liab_tag, liab_total = _value_for_year_usd(facts_json, ["Liabilities", "LiabilitiesAndStockholdersEquity"], year)
    
    ca_tag, ca = _value_for_year_usd(facts_json, ["AssetsCurrent"], year)
    nca_tag, nca = _value_for_year_usd(facts_json, ["AssetsNoncurrent"], year)

    cl_tag, cl = _value_for_year_usd(facts_json, ["LiabilitiesCurrent"], year)
    ncl_tag, ncl = _value_for_year_usd(facts_json, ["LiabilitiesNoncurrent"], year)

    eq_tag, eq = _value_for_year_usd(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"],
        year,
    )

    # 1. 資産の補完
    if (not np.isfinite(nca)) and np.isfinite(assets_total) and np.isfinite(ca):
        nca = assets_total - ca
        nca_tag = "CALC:Assets-AssetsCurrent"

    # 2. 負債合計の補完
    if (not np.isfinite(liab_total)):
        if np.isfinite(assets_total) and np.isfinite(eq):
            liab_total = assets_total - eq
            liab_tag = "CALC:Assets-Equity"
        elif np.isfinite(cl) and np.isfinite(ncl):
            liab_total = cl + ncl
            liab_tag = "CALC:LiabilitiesCurrent+Noncurrent"
            
    # 3. (1) 負債合計が計算値（資産-資本）と大きく乖離している場合、計算値を採用（利益剰余金過大表示対策）
    if np.isfinite(assets_total) and np.isfinite(eq):
         calc_liab = assets_total - eq
         if np.isfinite(liab_total) and abs(liab_total - calc_liab) > (assets_total * 0.05):
              liab_total = calc_liab
              liab_tag = "CALC:Assets-Equity (Override)"

    # 4. 負債内訳の補完
    if (not np.isfinite(ncl)) and np.isfinite(liab_total) and np.isfinite(cl):
        ncl = liab_total - cl
        ncl_tag = "CALC:Liabilities-LiabilitiesCurrent"
    elif (not np.isfinite(cl)) and np.isfinite(liab_total) and np.isfinite(ncl):
        cl = liab_total - ncl
        cl_tag = "CALC:Liabilities-LiabilitiesNoncurrent"

    meta.update({
        "assets_total_tag": assets_tag,
        "liabilities_total_tag": liab_tag,
        "current_assets_tag": ca_tag,
        "noncurrent_assets_tag": nca_tag,
        "current_liabilities_tag": cl_tag,
        "noncurrent_liabilities_tag": ncl_tag,
        "equity_tag": eq_tag,
    })

    ca_m = _to_musd(ca) if np.isfinite(ca) else 0.0
    nca_m = _to_musd(nca) if np.isfinite(nca) else 0.0
    cl_m = _to_musd(cl) if np.isfinite(cl) else 0.0
    ncl_m = _to_musd(ncl) if np.isfinite(ncl) else 0.0
    eq_m = _to_musd(eq) if np.isfinite(eq) else 0.0

    snap = pd.DataFrame([
        ["FY", year],
        ["Current Assets (M$)", ca_m],
        ["Noncurrent Assets (M$)", nca_m],
        ["Current Liabilities (M$)", cl_m],
        ["Noncurrent Liabilities (M$)", ncl_m],
        ["Equity (M$)", eq_m],
    ], columns=["Item", "Value"])
    return snap, meta


def plot_bs_bar(snap: pd.DataFrame, title: str):
    vals = dict(zip(snap["Item"], snap["Value"]))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    bottom = 0.0
    ax.bar(0, vals["Noncurrent Assets (M$)"], bottom=bottom, alpha=0.7, label="Noncurrent Assets")
    bottom += vals["Noncurrent Assets (M$)"]
    ax.bar(0, vals["Current Assets (M$)"], bottom=bottom, alpha=0.7, label="Current Assets")

    bottom = 0.0
    ax.bar(1, vals["Equity (M$)"], bottom=bottom, alpha=0.7, label="Equity")
    bottom += vals["Equity (M$)"],
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


# ▼▼ 修正箇所：Otherが消える問題を修正 ▼▼
def _top3_plus_other(items: list[tuple[str, float]], total: float) -> tuple[list[str], list[float]]:
    # 正の値だけ抽出してソート
    pos = [(n, float(v)) for n, v in items if np.isfinite(v) and float(v) > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    
    top = pos[:3]
    top_sum = sum(v for _, v in top)
    
    # 内訳合計(sum_all_pos)が報告上のTotal(total)を超えている場合（自己株式等の影響）に対応
    # 大きい方を円グラフの全体として採用する
    sum_all_pos = sum(v for _, v in pos)
    effective_total = max(total, sum_all_pos) if (np.isfinite(total) and total > 0) else sum_all_pos
    
    # 合計との差額をOtherとして算出
    other = max(effective_total - top_sum, 0.0)

    labels = [n for n, _ in top]
    sizes = [v for _, v in top]
    
    if other > 0:
        labels.append("Other")
        sizes.append(other)
        
    if not sizes:
        return ["No data"], [1.0]
        
    return labels, sizes
# ▲▲ 修正ここまで ▲▲


def build_bs_pies_latest(facts_json: dict, year: int) -> tuple[dict, dict, dict]:
    meta = {"year": year}

    _, total_assets = _value_for_year_usd(facts_json, ["Assets"], year)
    _, total_liab = _value_for_year_usd(facts_json, ["Liabilities"], year)
    _, total_eq = _value_for_year_usd(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"],
        year,
    )
    
    # (1) 負債合計の優先補完（資産-資本）
    if not np.isfinite(total_liab) and np.isfinite(total_assets) and np.isfinite(total_eq):
        total_liab = total_assets - total_eq

    total_le = (total_liab if np.isfinite(total_liab) else 0.0) + (total_eq if np.isfinite(total_eq) else 0.0)
    
    # ▼▼ 修正箇所: 負債＋純資産の合計を総資産に合わせる（Balance Sheetの等式 A = L + E を強制）▼▼
    # これにより、積み上げ項目が足りない場合や少数株主持分等が漏れている場合でも、
    # 差額が自動的に「Other」として円グラフに表示されるようになります。
    if np.isfinite(total_assets) and total_assets > 0:
        total_le = total_assets
    # ▲▲ 修正ここまで ▲▲

    # A: Assets Breakdown
    asset_candidates = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "CashAndCashEquivalents"]),
        ("Receivables", ["AccountsReceivableNetCurrent", "AccountsReceivableNet", "ReceivablesNetCurrent"]),
        ("Inventory", ["InventoryNet", "Inventory"]),
        ("PPE (Net)", [
            "PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetNet",
            "PropertyPlantAndEquipment"
        ]),
        ("Goodwill", ["Goodwill"]),
        ("Intangibles", ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet", "IntangibleAssetsNet"]),
    ]
    asset_items = []
    for name, tags in asset_candidates:
        used, v = _value_for_year_usd(facts_json, tags, year)
        meta[f"asset_{name}_tag"] = used
        if np.isfinite(v):
            asset_items.append((name, v))
            
    # 総資産データがない場合のバックアップ
    sum_asset_items = sum(v for _, v in asset_items if np.isfinite(v) and v > 0)
    if not np.isfinite(total_assets) or total_assets <= 0:
        total_assets = sum_asset_items
    
    a_labels, a_sizes = _top3_plus_other(asset_items, total_assets)

    # B: Liabilities + Equity Breakdown
    le_items = []
    
    # --- Liabilities ---
    liab_candidates = [
        ("Accounts Payable", ["AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayable"]),
        ("Deferred Revenue", ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent", "DeferredRevenue"]),
        ("Short-term Debt", ["ShortTermBorrowings", "CommercialPaper", "DebtCurrent", "LongTermDebtCurrent"]),
        ("Long-term Debt", ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"]),
        ("Other Liabilities", ["OtherLiabilitiesNoncurrent", "OtherLiabilities"])
    ]
    sum_liab_found = 0.0
    for name, tags in liab_candidates:
        used, v = _value_for_year_usd(facts_json, tags, year)
        meta[f"le_{name}_tag"] = used
        if np.isfinite(v) and v > 0:
            le_items.append((name, v))
            sum_liab_found += v
            
    # その他負債（計算値）
    if np.isfinite(total_liab) and total_liab > sum_liab_found:
        other_liab = total_liab - sum_liab_found
        if other_liab > 0:
            le_items.append(("Other Liabilities (Calc)", other_liab))

    # --- Equity ---
    eq_candidates = [
        ("Retained Earnings", ["RetainedEarningsAccumulatedDeficit"]),
        ("Add. Paid-in Capital", ["AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock"]),
        ("Common Stock", ["CommonStockValue"]),
        ("Treasury Stock", ["TreasuryStockValue"])
    ]
    sum_eq_found = 0.0
    for name, tags in eq_candidates:
        used, v = _value_for_year_usd(facts_json, tags, year)
        meta[f"le_{name}_tag"] = used
        if np.isfinite(v) and v > 0:
            le_items.append((name, v))
            sum_eq_found += v
        elif name == "Treasury Stock" and np.isfinite(v) and v < 0:
             sum_eq_found += v

    # その他資本（計算値）
    if np.isfinite(total_eq) and total_eq > sum_eq_found:
        other_eq = total_eq - sum_eq_found
        if other_eq > 0:
            le_items.append(("Other Equity (Calc)", other_eq))

    # 負債・純資産合計の調整（内訳合計より小さい場合は内訳合計を採用）
    sum_le_items = sum(v for _, v in le_items if np.isfinite(v) and v > 0)
    if not np.isfinite(total_le) or total_le <= 0:
        total_le = sum_le_items
        
    b_labels, b_sizes = _top3_plus_other(le_items, total_le)

    return (
        {"labels": a_labels, "sizes": a_sizes, "total": total_assets},
        {"labels": b_labels, "sizes": b_sizes, "total": total_le},
        meta,
    )


def plot_two_pies(assets_pie: dict, le_pie: dict, year: int):
    col1, col2 = st.columns(2)

    def _pie(ax, labels, sizes, title):
        ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%", startangle=90, textprops={"color": "white"})
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
        _pie(ax, le_pie["labels"], le_pie["sizes"], f"B: L+E Top3 + Other ({year})")
        st.pyplot(fig)


# =========================
# CF (annual)
# =========================
def build_cf_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}
    cfo_tags = ["NetCashProvidedByUsedInOperatingActivities"]
    cfi_tags = ["NetCashProvidedByUsedInInvestingActivities"]
    cff_tags = ["NetCashProvidedByUsedInFinancingActivities"]
    capex_tags = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsToAcquireFixedAssets"]

    tag, df_cfo = _pick_best_tag_latest_first_usd(facts_json, cfo_tags)
    meta["cfo_tag"] = tag
    tag, df_cfi = _pick_best_tag_latest_first_usd(facts_json, cfi_tags)
    meta["cfi_tag"] = tag
    tag, df_cff = _pick_best_tag_latest_first_usd(facts_json, cff_tags)
    meta["cff_tag"] = tag
    tag, df_capex = _pick_best_tag_latest_first_usd(facts_json, capex_tags)
    meta["capex_tag"] = tag

    years = sorted(set(df_cfo["year"].tolist()) | set(df_cfi["year"].tolist()) | set(df_cff["year"].tolist()) | set(df_capex["year"].tolist()))
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

        rows.append([y, _to_musd(cfo), _to_musd(cfi), _to_musd(cff), _to_musd(fcf)])

    out = pd.DataFrame(rows, columns=["FY", "CFO(M$)", "CFI(M$)", "CFF(M$)", "FCF(M$)"])
    meta["years"] = years
    return out, meta


def plot_cf_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    cfo = df["CFO(M$)"].astype(float).to_numpy()
    # cfi = df["CFI(M$)"].astype(float).to_numpy()
    # cff = df["CFF(M$)"].astype(float).to_numpy()
    fcf = df["FCF(M$)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, cfo, label="CFO (Operating)", linewidth=2.5, marker="o", markersize=6)
    # ax.plot(x, cfi, label="CFI", linewidth=2.5, marker="o", markersize=6)
    # ax.plot(x, cff, label="CFF", linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, fcf, label="FCF (Free)", linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


# =========================
# RPO tab - FIXED (3)
# =========================
def build_rpo_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}
    
    # 1. RPO (受注残) の検索
    rpo_keywords = ["RemainingPerformanceObligation", "PerformanceObligation", "TransactionPriceAllocated", "Backlog"]
    rpo_exclude = ["Satisfied", "Recognized", "Billings"]
    
    tag_rpo, df_rpo = _find_best_tag_dynamic(
        facts_json, 
        keywords=rpo_keywords, 
        exclude_keywords=rpo_exclude
    )
    meta["rpo_tag"] = tag_rpo

    # 2. 契約負債 (Contract Liabilities) の検索
    cl_keywords = ["ContractWithCustomerLiability", "DeferredRevenue", "DeferredIncome", "CustomerAdvances", "UnearnedRevenue"]
    
    base_exclude = [
        "Tax", "Benefit", "Expense",       # 税金・費用
        "IncreaseDecrease", "ChangeIn",    # 増減（CF項目）
        "Recognized", "Satisfied",         # 認識額（PL項目）
        "Billings", "CumulativeEffect"     # その他
    ]
    
    # Total用
    cl_exclude_total = base_exclude + ["Current", "Noncurrent"]
    tag_cl, df_cl = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_total
    )
    meta["contract_liab_tag"] = tag_cl
    
    # Current用
    cl_exclude_current = base_exclude + ["Noncurrent"]
    tag_clc, df_clc = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_current,
        must_end_with="Current"
    )
    meta["contract_liab_current_tag"] = tag_clc
    
    # Noncurrent用
    cl_exclude_noncurrent = base_exclude
    tag_cln, df_cln = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_noncurrent,
        must_end_with="Noncurrent"
    )
    meta["contract_liab_noncurrent_tag"] = tag_cln

    # --- 集計ロジック ---
    years = sorted(
        set(df_rpo["year"].tolist() if not df_rpo.empty else []) | 
        set(df_cl["year"].tolist() if not df_cl.empty else []) | 
        set(df_clc["year"].tolist() if not df_clc.empty else []) |
        set(df_cln["year"].tolist() if not df_cln.empty else [])
    )
    years = [int(y) for y in years]
    
    current_year = pd.Timestamp.now().year
    if years and max(years) < current_year - 2:
        meta["warning"] = f"Latest data is from {max(years)}. Tags might be discontinued."

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
        cl_total = val_at(df_cl, y)
        cl_curr = val_at(df_clc, y)
        cl_non = val_at(df_cln, y)

        # Totalがない場合、Current + Noncurrent で補完
        if (not np.isfinite(cl_total)) and np.isfinite(cl_curr) and np.isfinite(cl_non):
            cl_total = cl_curr + cl_non
        
        # TotalもNoncurrentもなく、Currentだけある場合、Total = Current とみなす
        if (not np.isfinite(cl_total)) and np.isfinite(cl_curr) and tag_cln is None:
             cl_total = cl_curr

        rows.append([
            y, 
            _to_musd(rpo), 
            _to_musd(cl_total), 
            _to_musd(cl_curr)
        ])

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

    if np.isfinite(rpo).any():
        ax.plot(x, rpo, label="RPO / Backlog", linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(cl).any():
        ax.plot(x, cl, label="Contract Liabilities (Total)", linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(clc).any():
        ax.plot(x, clc, label="Contract Liabilities (Current)", linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("Million USD", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


# =========================
# Ratios tab: ROA / ROE / Inventory Turnover - FIXED (4)
# =========================
def _annual_series_map_usd(facts_json: dict, tag_candidates: list[str]) -> tuple[str | None, dict]:
    tag, df = _pick_best_tag_latest_first_usd(facts_json, tag_candidates)
    m = dict(zip(df["year"].astype(int), df["val"].astype(float))) if not df.empty else {}
    return tag, m


def build_ratios_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}

    # 純利益: 基礎データとして株主帰属分を優先（EPSとの整合性）
    ni_priority = ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss", "IncomeLossFromContinuingOperations"]
    ni_df, ni_meta = _build_composite_by_year_usd(facts_json, ni_priority)
    meta["net_income_composite"] = ni_meta
    ni_map = dict(zip(ni_df["year"].astype(int), ni_df["value"].astype(float))) if not ni_df.empty else {}

    assets_tag, assets_map = _annual_series_map_usd(facts_json, ["Assets"])
    meta["assets_tag"] = assets_tag

    eq_tag, eq_map = _annual_series_map_usd(
        facts_json, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"]
    )
    meta["equity_tag"] = eq_tag

    inv_tag, inv_map = _annual_series_map_usd(facts_json, ["InventoryNet", "InventoryGross", "Inventory"])
    meta["inventory_tag"] = inv_tag

    # (4) 回転率: CostOfSales, CostOfGoodsSoldを追加
    cogs_candidates = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfSales", "CostOfGoodsSold"]
    cogs_tag, cogs_map = _annual_series_map_usd(facts_json, cogs_candidates)
    meta["cogs_tag"] = cogs_tag

    years = sorted(set(ni_map.keys()) | set(assets_map.keys()) | set(eq_map.keys()) | set(inv_map.keys()) | set(cogs_map.keys()))
    if not years:
        return pd.DataFrame(), meta

    def avg_two(m: dict, y: int) -> float:
        v = m.get(y, np.nan)
        vp = m.get(y - 1, np.nan)
        if np.isfinite(v) and np.isfinite(vp):
            return (v + vp) / 2.0
        return v if np.isfinite(v) else np.nan

    rows = []
    for y in years:
        ni = ni_map.get(y, np.nan)
        a_avg = avg_two(assets_map, y)
        e_avg = avg_two(eq_map, y)
        inv_avg = avg_two(inv_map, y)
        cogs = cogs_map.get(y, np.nan)

        roa = (ni / a_avg) if np.isfinite(ni) and np.isfinite(a_avg) and a_avg != 0 else np.nan
        roe = (ni / e_avg) if np.isfinite(ni) and np.isfinite(e_avg) and e_avg != 0 else np.nan
        inv_turn = (cogs / inv_avg) if np.isfinite(cogs) and np.isfinite(inv_avg) and inv_avg != 0 else np.nan

        rows.append([y, roa * 100 if np.isfinite(roa) else np.nan, roe * 100 if np.isfinite(roe) else np.nan, inv_turn])

    out = pd.DataFrame(rows, columns=["FY", "ROA(%)", "ROE(%)", "InventoryTurnover(x)"])
    meta["years"] = years
    return out, meta


def plot_roa_roe(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    roa = df["ROA(%)"].astype(float).to_numpy()
    roe = df["ROE(%)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, roa, label="ROA (%)", linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, roe, label="ROE (%)", linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("%", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


def plot_inventory_turnover(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    it = df["InventoryTurnover(x)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, it, label="Inventory Turnover (x)", linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("x", color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
    st.pyplot(fig)


# =========================
# EPS (unit-aware) + STOCK SPLIT ADJUSTMENT - FIXED (5)
# =========================
def _get_split_adjustment_factor(ticker: str, date_index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    factors = pd.Series(1.0, index=date_index)
    if not ticker:
        return factors, {}

    try:
        # Tickerを大文字に強制変換して問い合わせ
        t = yf.Ticker(ticker.upper())
        splits = t.splits
        
        split_debug = {}
        if splits.empty:
            return factors, {"msg": "No splits found by yfinance"}

        split_debug = {str(d.date()): r for d, r in splits.items()}

        for split_date, ratio in splits.items():
            split_date = pd.to_datetime(split_date).tz_localize(None)
            mask = factors.index.tz_localize(None) < split_date
            if ratio > 0:
                factors.loc[mask] *= (1.0 / ratio)
                
        return factors, split_debug
    except Exception as e:
        return factors, {"error": str(e)}


def _extract_eps_series_any_unit(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    us = facts_json.get("facts", {}).get("us-gaap", {})
    tag_obj = us.get(xbrl_tag, {})
    units = tag_obj.get("units", {})
    if not units:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "unit", "filed"])

    rows = []
    for unit_key, node in units.items():
        for x in node:
            form = str(x.get("form", "")).upper().strip()
            fp = str(x.get("fp", "")).upper().strip()
            end = x.get("end")
            val = x.get("val")
            fy_raw = x.get("fy", None)
            filed = x.get("filed")

            if not end or val is None or not filed:
                continue
            if not form.startswith("10-K"):
                continue

            end_ts = pd.to_datetime(end, errors="coerce")
            filed_ts = pd.to_datetime(filed, errors="coerce")

            if pd.isna(end_ts) or pd.isna(filed_ts):
                continue

            # EPSでも同様にQ4混入を防ぐため、annual_fpを厳密化
            annual_fp = fp in {"FY", "CY"}

            if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
                year_key = int(fy_raw)
            else:
                year_key = int(end_ts.year)

            rows.append({
                "year": year_key,
                "end": end_ts,
                "val": _safe_float(val),
                "annual_fp": int(annual_fp),
                "unit": unit_key,
                "filed": filed_ts
            })

    if not rows:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "unit", "filed"])

    df = pd.DataFrame(rows).dropna(subset=["val"])
    
    # 年次(FY/CY)のみ抽出
    df = df[df["annual_fp"] == 1]

    # (5) EPS: 単位判定の緩和
    def unit_score(u: str) -> int:
        u2 = u.lower().replace(" ", "")
        if "usd/shares" in u2 or "usd/share" in u2 or "usd/shr" in u2 or "usdper" in u2:
            return 3
        if "shares" in u2 or "share" in u2:
            return 2
        if "usd" in u2:
            return 1
        return 0

    df["score"] = df["unit"].apply(unit_score)
    best_score = df["score"].max()
    df_best = df[df["score"] == best_score].copy()

    # 最新のfiled優先
    df_best = df_best.sort_values(["year", "filed"]).drop_duplicates(subset=["year"], keep="last")
    df_best["unit_best"] = df_best["unit"]
    return df_best[["year", "end", "val", "unit_best"]]


def build_eps_table(facts_json: dict, ticker_symbol: str = "") -> tuple[pd.DataFrame, dict]:
    meta = {}
    # (3) EPS: Dilutedを最優先
    eps_tags = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]

    best_df = pd.DataFrame()
    best_tag = None

    for tag in eps_tags:
        df = _extract_eps_series_any_unit(facts_json, tag)
        if df.empty:
            continue
        # 優先順位順に走査し、より多くのデータ(または最新)があれば採用
        if best_df.empty:
            best_df = df
            best_tag = tag
        else:
            if int(df["year"].max()) > int(best_df["year"].max()) or (
                int(df["year"].max()) == int(best_df["year"].max()) and len(df) > len(best_df)
            ):
                best_df = df
                best_tag = tag

    meta["eps_tag"] = best_tag
    if best_df.empty:
        meta["eps_composite"] = {"available_tags": []}
        return pd.DataFrame(), meta

    # 分割調整
    if ticker_symbol:
        best_df["end"] = pd.to_datetime(best_df["end"])
        factors, split_debug = _get_split_adjustment_factor(ticker_symbol, pd.DatetimeIndex(best_df["end"]))
        best_df["val_adjusted"] = best_df["val"] * factors.values
        meta["split_adjusted"] = True
        meta["split_history"] = split_debug 
    else:
        best_df["val_adjusted"] = best_df["val"]
        meta["split_adjusted"] = False

    meta["eps_unit"] = best_df["unit_best"].iloc[-1]
    out = pd.DataFrame({"FY": best_df["year"].astype(int), "EPS": best_df["val_adjusted"].astype(float)})
    meta["years"] = out["FY"].tolist()
    return out, meta


def plot_eps(table: pd.DataFrame, title: str, unit_label: str = "USD/share"):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    eps = df["EPS"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0b0c0e")
    ax.set_facecolor("#0b0c0e")

    ax.plot(x, eps, label="EPS (Adjusted, Diluted)", linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel(unit_label, color="white")
    ax.tick_params(colors="white")
    ax.grid(color="#333333", alpha=0.6)
    ax.set_title(title, color="white")
    ax.legend(facecolor="#0b0c0e", labelcolor="white", loc="upper left")
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

    st.markdown("## 長期ファンダ（PL / BS / CF / セグメント / 受注残・RPO / 回転率 / EPS）")
    st.caption(f"認証ユーザー: {authed_email}")

    with st.form("input_form"):
        ticker = st.text_input("Ticker（米国株）", value="AVGO")
        n_years = st.slider("期間（共通：最新から過去N年）", min_value=3, max_value=15, value=10)
        submitted = st.form_submit_button("Run")

    if not submitted:
        st.stop()

    t = ticker.strip().lower()
    if not t:
        st.error("Tickerを入力してください。")
        st.stop()

    cik_map = fetch_ticker_cik_map()
    cik10 = cik_map.get(t)
    if not cik10:
        st.error("このTickerのCIKが見つかりませんでした。")
        st.stop()

    facts = fetch_company_facts(cik10)
    company_name = facts.get("entityName", ticker.upper())

    tab_pl, tab_bs, tab_cf, tab_seg, tab_rpo, tab_turn, tab_eps = st.tabs(
        ["PL（年次）", "BS（最新年）", "CF（年次）", "セグメント（売上）", "受注残 / RPO", "回転率", "EPS"]
    )

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
        # Pretaxを削除し、純利益をAttributableベースで表示
        st.dataframe(
            pl_disp.style.format({"Revenue(M$)": "{:,.0f}", "OpIncome(M$)": "{:,.0f}", "NetIncome(M$)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### 営業利益率の推移（%）")
        plot_operating_margin(pl_disp, f"{company_name} ({ticker.upper()}) - Operating Margin (%)")
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
        assets_pie, le_pie, pie_meta = build_bs_pies_latest(facts, year)
        plot_two_pies(assets_pie, le_pie, year)
        with st.expander("BSデバッグ（タグ採用状況）", expanded=False):
            st.write({"bs": bs_meta, "pies": pie_meta})

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
            cf_disp.style.format({"CFO(M$)": "{:,.0f}", "CFI(M$)": "{:,.0f}", "CFF(M$)": "{:,.0f}", "FCF(M$)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("CFデバッグ（採用タグ）", expanded=False):
            st.write(cf_meta)

    with tab_seg:
        st.caption("注: セグメント情報は企業がXBRL標準タクソノミ（Dimension）を使用している場合のみ表示されます。")
        biz_df, geo_df = build_segment_table(facts)
        
        col1, col2 = st.columns(2)
        with col1:
            if not biz_df.empty:
                biz_disp = biz_df[biz_df.index >= int(biz_df.index.max()) - n_years + 1]
                plot_stacked_bar(biz_disp, f"{company_name} - Revenue by Business Segment")
                st.dataframe(biz_disp.style.format("{:,.0f}"), use_container_width=True)
            else:
                st.info("事業セグメント情報が見つかりませんでした。")
        
        with col2:
            if not geo_df.empty:
                geo_disp = geo_df[geo_df.index >= int(geo_df.index.max()) - n_years + 1]
                plot_stacked_bar(geo_disp, f"{company_name} - Revenue by Geography")
                st.dataframe(geo_disp.style.format("{:,.0f}"), use_container_width=True)
            else:
                st.info("地域セグメント情報が見つかりませんでした。")

    with tab_rpo:
        rpo_table, rpo_meta = build_rpo_annual_table(facts)
        if rpo_table.empty:
            st.warning("この銘柄ではRPO/契約負債（年次）がXBRL上で取得できない可能性があります。")
            st.write(rpo_meta)
        else:
            rpo_disp = _slice_latest_n_years(rpo_table, int(n_years))
            st.caption(f"RPO: 表示 {len(rpo_disp)} 年（最新年: {int(rpo_table['FY'].max())}）")
            plot_rpo_annual(rpo_disp, f"{company_name} ({ticker.upper()}) - RPO / Contract Liabilities (Annual)")
            st.dataframe(
                rpo_disp.style.format({"RPO(M$)": "{:,.0f}", "ContractLiab(M$)": "{:,.0f}", "ContractLiabCurrent(M$)": "{:,.0f}"}),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("RPOデバッグ（採用タグ）", expanded=False):
                st.write(rpo_meta)

    with tab_turn:
        rat_table, rat_meta = build_ratios_table(facts)
        if rat_table.empty:
            st.error("回転率・ROA/ROEの計算に必要なデータが取得できませんでした。")
            st.write(rat_meta)
            st.stop()
        rat_disp = _slice_latest_n_years(rat_table, int(n_years))
        st.caption(f"回転率: 表示 {len(rat_disp)} 年（最新年: {int(rat_table['FY'].max())}）")
        st.markdown("### ROA / ROE（%）推移")
        plot_roa_roe(rat_disp, f"{company_name} ({ticker.upper()}) - ROA / ROE (%)")
        st.markdown("### 棚卸資産回転率（回）推移")
        plot_inventory_turnover(rat_disp, f"{company_name} ({ticker.upper()}) - Inventory Turnover (x)")
        st.dataframe(
            rat_disp.style.format({"ROA(%)": "{:.2f}", "ROE(%)": "{:.2f}", "InventoryTurnover(x)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("回転率デバッグ（採用タグ）", expanded=False):
            st.write(rat_meta)

    with tab_eps:
        eps_table, eps_meta = build_eps_table(facts, ticker_symbol=t)
        if eps_table.empty:
            st.error("EPSが取得できませんでした。")
            st.write(eps_meta)
        else:
            eps_disp = _slice_latest_n_years(eps_table, int(n_years))
            unit_label = eps_meta.get("eps_unit", "USD/share")
            split_note = " (Split Adjusted)" if eps_meta.get("split_adjusted") else ""
            st.caption(f"EPS: 表示 {len(eps_disp)} 年（最新年: {int(eps_table['FY'].max())}） / unit: {unit_label}{split_note}")
            plot_eps(eps_disp, f"{company_name} ({ticker.upper()}) - EPS (Diluted)", unit_label="USD/share")
            st.dataframe(eps_disp.style.format({"EPS": "{:.2f}"}), use_container_width=True, hide_index=True)
            with st.expander("EPSデバッグ", expanded=False):
                st.write(eps_meta)


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
