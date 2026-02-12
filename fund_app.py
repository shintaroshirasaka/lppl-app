import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
import streamlit as st
import yfinance as yf
import re

# =======================================================
# FONT & STYLE SETUP (Luxury / HNWI Design)
# =======================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# --- Luxury Color Palette ---
HNWI_BG = "#050505"       # Deepest Black
HNWI_AX_BG = "#0b0c0e"    # Off-Black for Axes
TEXT_COLOR = "#F0F0F0"    # Off-White text
TICK_COLOR = "#888888"    # Grey ticks
GRID_COLOR = "#333333"    # Subtle grid

# Data Colors
C_GOLD = "#C5A059"        # Main Gold
C_SILVER = "#A0A0A0"      # Secondary Silver
C_BRONZE = "#cd7f32"      # Tertiary Bronze
C_BLUE = "#4682B4"        # Steel Blue (Muted)
C_SLATE = "#708090"       # Slate Grey

# Custom Colormap for Stacked Bars
LUXURY_CMAP = ListedColormap([C_GOLD, C_SILVER, C_BLUE, C_BRONZE, C_SLATE, "#8B4513", "#556B2F"])

# =======================================================
# VISUALIZATION HELPERS
# =======================================================
def draw_logo_overlay(ax):
    """Adds the OUT-STANDER watermark logo."""
    ax.text(0.98, 0.03, "OUT-STANDER", transform=ax.transAxes,
            fontsize=20, color='#3d3320', fontweight='bold',
            fontname='serif', ha='right', va='bottom', zorder=0, alpha=0.9)

def style_hnwi_ax(ax, title=None, dual_y=False):
    """Applies the luxury styling to an axes object."""
    ax.set_facecolor(HNWI_AX_BG)
    if title:
        # Minimalist Title styling
        ax.set_title(title, color=TEXT_COLOR, fontweight='normal', fontname='serif', pad=15)
    
    ax.tick_params(colors=TICK_COLOR, which='both')
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    
    # Minimalist Grid (Dotted)
    ax.grid(color=GRID_COLOR, linestyle=":", linewidth=0.5, alpha=0.5)
    
    # Spine styling (Open look)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    
    if dual_y:
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_color(GRID_COLOR)
    else:
        ax.spines['right'].set_visible(False)
        
    draw_logo_overlay(ax)

# =======================================================
# AUTH GATE (Dummy for compatibility)
# =======================================================
try:
    from auth_gate import require_admin_token
except ImportError:
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
        st.error(f"Failed to fetch Ticker map: {e}")
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
def _extract_annual_series_usd(facts_json: dict, xbrl_tag: str, include_segments: bool = False, min_duration: int = 0) -> pd.DataFrame:
    facts_root = facts_json.get("facts", {})
    if not facts_root:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "filed", "segment"])

    node = []
    for prefix in ["us-gaap", "srt", "ifrs-full", "dei"]:
        if prefix in facts_root and xbrl_tag in facts_root[prefix]:
            node = facts_root[prefix][xbrl_tag].get("units", {}).get("USD", [])
            break
    
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
        start = x.get("start")
        val = x.get("val")
        fy_raw = x.get("fy", None)
        filed = x.get("filed")
        segment_obj = x.get("segment")

        if not end or val is None:
            continue
        if not (form.startswith("10-K") or form.startswith("20-F") or form.startswith("40-F")):
            continue

        end_ts = pd.to_datetime(end, errors="coerce")
        filed_ts = pd.to_datetime(filed, errors="coerce")
        start_ts = pd.to_datetime(start, errors="coerce")

        if pd.isna(end_ts):
            continue

        if min_duration > 0:
            if pd.isna(start_ts):
                continue
            days = (end_ts - start_ts).days
            if days < min_duration:
                continue

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
            "filed": filed_ts,
            "segment": segment_obj
        })

    if not rows:
        return pd.DataFrame(columns=["year", "end", "val", "annual_fp", "filed", "segment"])

    df = pd.DataFrame(rows).dropna(subset=["val"])
    df = df[df["annual_fp"] == 1]

    if not include_segments:
        df = df[df["segment"].isnull()]
        df = df.sort_values(["year", "filed"]).drop_duplicates(subset=["year"], keep="last")
        df = df.sort_values("year")
        return df
    else:
        return df


def _pick_best_tag_latest_first_usd(facts_json: dict, candidates: list[str], min_duration: int = 0) -> tuple[str | None, pd.DataFrame]:
    best_tag = None
    best_df = pd.DataFrame(columns=["year", "end", "val", "annual_fp"])
    best_max_year = -1
    best_n = -1
    best_last_end = pd.Timestamp("1900-01-01")

    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False, min_duration=min_duration)
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


def _find_best_tag_dynamic(facts_json: dict, keywords: list[str], exclude_keywords: list[str] = None, must_end_with: str = None, min_duration: int = 0) -> tuple[str | None, pd.DataFrame]:
    if exclude_keywords is None:
        exclude_keywords = []
    
    all_tags = []
    facts_root = facts_json.get("facts", {})
    for prefix in facts_root:
        for tag in facts_root[prefix]:
            all_tags.append(tag)
            
    candidates = []
    for tag in all_tags:
        tag_lower = tag.lower()
        if not any(k.lower() in tag_lower for k in keywords):
            continue
        if any(ek.lower() in tag_lower for ek in exclude_keywords):
            continue
        if must_end_with and not tag_lower.endswith(must_end_with.lower()):
            continue
        candidates.append(tag)
    
    if not candidates:
        return None, pd.DataFrame()

    best_tag = None
    best_df = pd.DataFrame(columns=["year"])
    best_score = (-1, -1)

    for tag in candidates:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False, min_duration=min_duration)
        if df.empty:
            continue
            
        max_year = int(df["year"].max())
        count = len(df)
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


def _build_composite_by_year_usd(facts_json: dict, tag_priority: list[str], min_duration: int = 0) -> tuple[pd.DataFrame, dict]:
    tag_series = {}
    tag_years = {}
    for tag in tag_priority:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=False, min_duration=min_duration)
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
    out = out.sort_values(["year", "dimension", "member", "filed"]).drop_duplicates(subset=["year", "dimension", "member"], keep="last")
    return out


def build_segment_table(facts_json: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    revenue_tags = [
        "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax", 
        "SalesToExternalCustomers", "RevenuesNetOfInterestExpense", 
        "SalesRevenueGoodsNet", "SalesRevenueServicesNet", 
        "SegmentReportingInformationRevenue", "NetOperatingRevenues"
    ]
    
    all_seg_rows = []
    
    for tag in revenue_tags:
        df = _extract_annual_series_usd(facts_json, tag, include_segments=True, min_duration=350)
        if df.empty:
            continue
        parsed = _parse_segment_info(df)
        if not parsed.empty:
            parsed["src_tag"] = tag
            all_seg_rows.append(parsed)
            
    if not all_seg_rows:
        return pd.DataFrame(), pd.DataFrame()
    
    merged = pd.concat(all_seg_rows, ignore_index=True)
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
        st.info(f"{title}: Data not found.")
        return

    df_m = df / 1_000_000.0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)
    
    # Use Luxury Colormap instead of default
    df_m.plot(kind="bar", stacked=True, ax=ax, colormap=LUXURY_CMAP, width=0.7)
    
    ax.set_ylabel("Revenue (Million USD)", color=TEXT_COLOR)
    ax.tick_params(colors=TICK_COLOR, axis='x', rotation=0)
    ax.tick_params(colors=TICK_COLOR, axis='y')
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    st.pyplot(fig)


# =========================
# PL (annual) - MODIFIED
# =========================
def build_pl_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    revenue_priority = [
        "RevenueFromContractWithCustomerExcludingAssessedTax", 
        "Revenues", 
        "NetOperatingRevenues",
        "SalesRevenueGoodsNet",
        "SalesRevenueNet", 
        "SalesRevenue",
        "OperatingRevenue",
        "OperatingRevenues"
    ]
    
    net_income_priority = [
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "NetIncomeLoss", 
        "ProfitLoss", 
        "IncomeLossFromContinuingOperations"
    ]
    
    op_income_priority = ["OperatingIncomeLoss", "OperatingIncome"]
    
    meta = {}
    
    rev_df, rev_meta = _build_composite_by_year_usd(facts_json, revenue_priority, min_duration=350)
    meta["revenue_composite"] = rev_meta
    
    ni_df, ni_meta = _build_composite_by_year_usd(facts_json, net_income_priority, min_duration=350)
    meta["net_income_composite"] = ni_meta
    
    op_df, op_meta = _build_composite_by_year_usd(facts_json, op_income_priority, min_duration=350)
    meta["op_income_composite"] = op_meta

    if rev_df.empty:
        return pd.DataFrame(), meta

    rev_map = dict(zip(rev_df["year"].astype(int), rev_df["value"].astype(float)))
    ni_map = dict(zip(ni_df["year"].astype(int), ni_df["value"].astype(float))) if not ni_df.empty else {}
    op_map = dict(zip(op_df["year"].astype(int), op_df["value"].astype(float))) if not op_df.empty else {}
    
    tag_for_date, df_date = _pick_best_tag_latest_first_usd(facts_json, ["Revenues", "NetOperatingRevenues", "NetIncomeLoss"], min_duration=350)
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
            _to_musd(ni_map.get(y, np.nan))
        ])

    out = pd.DataFrame(rows, columns=["FY", "End", "Revenue(M$)", "OpIncome(M$)", "NetIncome(M$)"])
    out = out.sort_values("FY").reset_index(drop=True)
    meta["years"] = years
    return out, meta


def plot_pl_annual(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    revenue = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()
    ni = df["NetIncome(M$)"].astype(float).to_numpy()

    fig, ax_left = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(HNWI_BG)
    # Apply styling to main axis
    style_hnwi_ax(ax_left, title=title, dual_y=True)

    # Plot Income Lines (Gold & Bronze)
    ax_left.plot(x, op, label="Operating Income", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)
    ax_left.plot(x, ni, label="Net Income", color=C_BRONZE, linewidth=2.5, marker="o", markersize=6)

    ax_left.set_ylabel("Profit (Million USD)", color=TEXT_COLOR)

    # Secondary Axis for Revenue (Bars)
    ax_right = ax_left.twinx()
    ax_right.bar(x, revenue, color=C_SILVER, alpha=0.3, width=0.6, label="Revenue")
    
    # Manually style the twin axis to match
    ax_right.set_ylabel("Revenue (Million USD)", color=TEXT_COLOR)
    ax_right.tick_params(colors=TICK_COLOR, which='both')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_color(GRID_COLOR)
    ax_right.spines['bottom'].set_visible(False)

    # Legend handling
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    
    st.pyplot(fig)


def plot_operating_margin(table: pd.DataFrame, title: str):
    df = table.copy()
    df = df.sort_values("FY")
    x = df["FY"].astype(str).tolist()
    rev = df["Revenue(M$)"].astype(float).to_numpy()
    op = df["OpIncome(M$)"].astype(float).to_numpy()

    margin = np.full_like(rev, np.nan, dtype=float)
    for i in range(len(rev)):
        if np.isfinite(rev[i]) and rev[i] != 0 and np.isfinite(op[i]):
            margin[i] = op[i] / rev[i] * 100.0

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, margin, color=C_GOLD, linewidth=2.5, marker="o", markersize=6, label="Operating Margin (%)")
    ax.set_ylabel("%", color=TEXT_COLOR)
    st.pyplot(fig)


# =========================
# BS (latest)
# =========================
def _latest_year_from_assets(facts_json: dict) -> int | None:
    df = _extract_annual_series_usd(facts_json, "Assets", min_duration=0)
    if df.empty:
        return None
    return int(df["year"].max())


def _value_for_year_usd(facts_json: dict, tag_candidates: list[str], year: int) -> tuple[str | None, float]:
    for tag in tag_candidates:
        df = _extract_annual_series_usd(facts_json, tag, min_duration=0)
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

    if (not np.isfinite(nca)) and np.isfinite(assets_total) and np.isfinite(ca):
        nca = assets_total - ca
        nca_tag = "CALC:Assets-AssetsCurrent"

    if (not np.isfinite(liab_total)):
        if np.isfinite(assets_total) and np.isfinite(eq):
            liab_total = assets_total - eq
            liab_tag = "CALC:Assets-Equity"
        elif np.isfinite(cl) and np.isfinite(ncl):
            liab_total = cl + ncl
            liab_tag = "CALC:LiabilitiesCurrent+Noncurrent"
            
    if np.isfinite(assets_total) and np.isfinite(eq):
         calc_liab = assets_total - eq
         if np.isfinite(liab_total) and abs(liab_total - calc_liab) > (assets_total * 0.05):
             liab_total = calc_liab
             liab_tag = "CALC:Assets-Equity (Override)"

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
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    # Assets
    bottom = 0.0
    ax.bar(0, vals["Noncurrent Assets (M$)"], bottom=bottom, color=C_SLATE, alpha=0.9, label="Noncurrent Assets", width=0.6)
    bottom += vals["Noncurrent Assets (M$)"]
    ax.bar(0, vals["Current Assets (M$)"], bottom=bottom, color=C_BLUE, alpha=0.9, label="Current Assets", width=0.6)

    # Liabilities + Equity
    bottom = 0.0
    ax.bar(1, vals["Equity (M$)"], bottom=bottom, color=C_GOLD, alpha=0.9, label="Equity", width=0.6)
    bottom += vals["Equity (M$)"]
    ax.bar(1, vals["Noncurrent Liabilities (M$)"], bottom=bottom, color=C_BRONZE, alpha=0.9, label="Noncurrent Liab", width=0.6)
    bottom += vals["Noncurrent Liabilities (M$)"]
    ax.bar(1, vals["Current Liabilities (M$)"], bottom=bottom, color="#8B4513", alpha=0.9, label="Current Liab", width=0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Assets", "Liab + Equity"], color=TEXT_COLOR)
    ax.set_ylabel("Million USD", color=TEXT_COLOR)
    
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    st.pyplot(fig)


def _top3_plus_other(items: list[tuple[str, float]], total: float) -> tuple[list[str], list[float]]:
    pos = [(n, float(v)) for n, v in items if np.isfinite(v) and float(v) > 0]
    pos.sort(key=lambda x: x[1], reverse=True)
    
    top = pos[:3]
    top_sum = sum(v for _, v in top)
    
    sum_all_pos = sum(v for _, v in pos)
    effective_total = max(total, sum_all_pos) if (np.isfinite(total) and total > 0) else sum_all_pos
    
    other = max(effective_total - top_sum, 0.0)

    labels = [n for n, _ in top]
    sizes = [v for _, v in top]
    
    if other > 0:
        labels.append("Other")
        sizes.append(other)
        
    if not sizes:
        return ["No data"], [1.0]
        
    return labels, sizes


def build_bs_pies_latest(facts_json: dict, year: int) -> tuple[dict, dict, dict]:
    meta = {"year": year}

    _, total_assets = _value_for_year_usd(facts_json, ["Assets"], year)
    _, total_liab = _value_for_year_usd(facts_json, ["Liabilities"], year)
    _, total_eq = _value_for_year_usd(
        facts_json,
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"],
        year,
    )
    
    if not np.isfinite(total_liab) and np.isfinite(total_assets) and np.isfinite(total_eq):
        total_liab = total_assets - total_eq

    total_le = (total_liab if np.isfinite(total_liab) else 0.0) + (total_eq if np.isfinite(total_eq) else 0.0)
    
    if np.isfinite(total_assets) and total_assets > 0:
        total_le = total_assets

    asset_candidates = [
        ("Cash & Equiv", ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "CashAndCashEquivalents"]),
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
            
    sum_asset_items = sum(v for _, v in asset_items if np.isfinite(v) and v > 0)
    if not np.isfinite(total_assets) or total_assets <= 0:
        total_assets = sum_asset_items
    
    a_labels, a_sizes = _top3_plus_other(asset_items, total_assets)

    le_items = []
    
    liab_candidates = [
        ("Acct Payable", ["AccountsPayableCurrent", "AccountsPayableTradeCurrent", "AccountsPayable"]),
        ("Def Revenue", ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent", "DeferredRevenue"]),
        ("ST Debt", ["ShortTermBorrowings", "CommercialPaper", "DebtCurrent", "LongTermDebtCurrent"]),
        ("LT Debt", ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"]),
        ("Other Liab", ["OtherLiabilitiesNoncurrent", "OtherLiabilities"])
    ]
    sum_liab_found = 0.0
    for name, tags in liab_candidates:
        used, v = _value_for_year_usd(facts_json, tags, year)
        meta[f"le_{name}_tag"] = used
        if np.isfinite(v) and v > 0:
            le_items.append((name, v))
            sum_liab_found += v
            
    if np.isfinite(total_liab) and total_liab > sum_liab_found:
        other_liab = total_liab - sum_liab_found
        if other_liab > 0:
            le_items.append(("Other Liab (Calc)", other_liab))

    eq_candidates = [
        ("Retained Earnings", ["RetainedEarningsAccumulatedDeficit"]),
        ("Paid-in Capital", ["AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock"]),
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

    if np.isfinite(total_eq) and total_eq > sum_eq_found:
        other_eq = total_eq - sum_eq_found
        if other_eq > 0:
            le_items.append(("Other Eq (Calc)", other_eq))

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
    
    # Luxury colors for Pie
    colors = [C_GOLD, C_SILVER, C_BLUE, C_BRONZE, C_SLATE]

    def _pie(ax, labels, sizes, title):
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct=lambda p: f"{p:.0f}%", 
            startangle=90, colors=colors,
            textprops={"color": TEXT_COLOR}
        )
        ax.set_title(title, color=TEXT_COLOR, fontname='serif')
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('#000000') # Black text on colored slices
            autotext.set_fontweight('bold')

    with col1:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        fig.patch.set_facecolor(HNWI_BG)
        style_hnwi_ax(ax)
        ax.axis('equal') # Remove axes for pie
        _pie(ax, assets_pie["labels"], assets_pie["sizes"], f"Asset Breakdown ({year})")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        fig.patch.set_facecolor(HNWI_BG)
        style_hnwi_ax(ax)
        ax.axis('equal')
        _pie(ax, le_pie["labels"], le_pie["sizes"], f"Liab & Equity Breakdown ({year})")
        st.pyplot(fig)


# =========================
# CF (annual) - FIXED: Use composite for Capex to handle tag changes across years
# =========================
def build_cf_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}
    cfo_tags = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]
    cfi_tags = [
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations",
    ]
    cff_tags = [
        "NetCashProvidedByUsedInFinancingActivities",
        "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations",
    ]
    capex_tags = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireFixedAssets",
    ]

    cfo_df, cfo_meta = _build_composite_by_year_usd(facts_json, cfo_tags, min_duration=350)
    meta["cfo_composite"] = cfo_meta
    cfo_map = dict(zip(cfo_df["year"].astype(int), cfo_df["value"].astype(float))) if not cfo_df.empty else {}

    cfi_df, cfi_meta = _build_composite_by_year_usd(facts_json, cfi_tags, min_duration=350)
    meta["cfi_composite"] = cfi_meta
    cfi_map = dict(zip(cfi_df["year"].astype(int), cfi_df["value"].astype(float))) if not cfi_df.empty else {}

    cff_df, cff_meta = _build_composite_by_year_usd(facts_json, cff_tags, min_duration=350)
    meta["cff_composite"] = cff_meta
    cff_map = dict(zip(cff_df["year"].astype(int), cff_df["value"].astype(float))) if not cff_df.empty else {}

    capex_df, capex_meta = _build_composite_by_year_usd(facts_json, capex_tags, min_duration=350)
    meta["capex_composite"] = capex_meta
    capex_map = dict(zip(capex_df["year"].astype(int), capex_df["value"].astype(float))) if not capex_df.empty else {}

    years = sorted(set(cfo_map.keys()) | set(cfi_map.keys()) | set(cff_map.keys()) | set(capex_map.keys()))
    years = [int(y) for y in years]
    if not years:
        return pd.DataFrame(), meta

    rows = []
    for y in years:
        cfo = cfo_map.get(y, np.nan)
        cfi = cfi_map.get(y, np.nan)
        cff = cff_map.get(y, np.nan)
        capex_raw = capex_map.get(y, np.nan)

        capex_out = np.nan
        if np.isfinite(capex_raw):
            capex_out = capex_raw if capex_raw < 0 else -abs(capex_raw)

        fcf = np.nan
        if np.isfinite(cfo) and np.isfinite(capex_out):
            fcf = cfo + capex_out

        rows.append([y, _to_musd(cfo), _to_musd(cfi), _to_musd(cff), _to_musd(capex_out), _to_musd(fcf)])

    out = pd.DataFrame(rows, columns=["FY", "CFO(M$)", "CFI(M$)", "CFF(M$)", "Capex(M$)", "FCF(M$)"])
    meta["years"] = years
    return out, meta


def plot_cf_cfo_capex(table: pd.DataFrame, title: str):
    """Chart 1: CFO (positive) and Capex (negative) trend lines."""
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    cfo = df["CFO(M$)"].astype(float).to_numpy()
    capex = df["Capex(M$)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, cfo, label="CFO (Operating)", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, capex, label="Capex (Negative)", color=C_BLUE, linewidth=2.5, marker="o", markersize=6)

    ax.axhline(y=0, color=GRID_COLOR, linewidth=0.8, linestyle="-")
    ax.set_ylabel("Million USD", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


def plot_cf_fcf(table: pd.DataFrame, title: str):
    """Chart 2: FCF trend with zero line."""
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    fcf = df["FCF(M$)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, fcf, label="FCF (Free Cash Flow)", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)

    # Fill positive/negative regions
    fcf_series = pd.Series(fcf, index=range(len(fcf)))
    x_idx = np.arange(len(x))
    ax.fill_between(x_idx, fcf, 0, where=(fcf >= 0), color=C_GOLD, alpha=0.10, interpolate=True)
    ax.fill_between(x_idx, fcf, 0, where=(fcf < 0), color=C_BRONZE, alpha=0.15, interpolate=True)

    ax.axhline(y=0, color=C_SILVER, linewidth=1.0, linestyle="-", alpha=0.7)
    ax.set_ylabel("Million USD", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


# =========================
# RPO tab
# =========================
def build_rpo_annual_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}
    
    rpo_keywords = ["RemainingPerformanceObligation", "PerformanceObligation", "TransactionPriceAllocated", "Backlog"]
    rpo_exclude = ["Satisfied", "Recognized", "Billings"]
    
    tag_rpo, df_rpo = _find_best_tag_dynamic(
        facts_json, 
        keywords=rpo_keywords, 
        exclude_keywords=rpo_exclude,
        min_duration=0 
    )
    meta["rpo_tag"] = tag_rpo

    cl_keywords = ["ContractWithCustomerLiability", "DeferredRevenue", "DeferredIncome", "CustomerAdvances", "UnearnedRevenue"]
    
    base_exclude = [
        "Tax", "Benefit", "Expense",
        "IncreaseDecrease", "ChangeIn",
        "Recognized", "Satisfied",
        "Billings", "CumulativeEffect"
    ]
    
    cl_exclude_total = base_exclude + ["Current", "Noncurrent"]
    tag_cl, df_cl = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_total,
        min_duration=0
    )
    meta["contract_liab_tag"] = tag_cl
    
    cl_exclude_current = base_exclude + ["Noncurrent"]
    tag_clc, df_clc = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_current,
        must_end_with="Current",
        min_duration=0
    )
    meta["contract_liab_current_tag"] = tag_clc
    
    cl_exclude_noncurrent = base_exclude
    tag_cln, df_cln = _find_best_tag_dynamic(
        facts_json, 
        keywords=cl_keywords, 
        exclude_keywords=cl_exclude_noncurrent,
        must_end_with="Noncurrent",
        min_duration=0
    )
    meta["contract_liab_noncurrent_tag"] = tag_cln

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

        if (not np.isfinite(cl_total)) and np.isfinite(cl_curr) and np.isfinite(cl_non):
            cl_total = cl_curr + cl_non
        
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
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    if np.isfinite(rpo).any():
        ax.plot(x, rpo, label="RPO / Backlog", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(cl).any():
        ax.plot(x, cl, label="Contract Liabilities (Total)", color=C_SILVER, linewidth=2.5, marker="o", markersize=6)
    if np.isfinite(clc).any():
        ax.plot(x, clc, label="Contract Liabilities (Current)", color=C_BRONZE, linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("Million USD", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


# =========================
# Ratios tab: ROA / ROE / Inventory Turnover - FIXED (4)
# =========================
def _annual_series_map_usd(facts_json: dict, tag_candidates: list[str], min_duration: int = 0) -> tuple[str | None, dict]:
    tag, df = _pick_best_tag_latest_first_usd(facts_json, tag_candidates, min_duration=min_duration)
    m = dict(zip(df["year"].astype(int), df["val"].astype(float))) if not df.empty else {}
    return tag, m


def build_ratios_table(facts_json: dict) -> tuple[pd.DataFrame, dict]:
    meta = {}

    ni_priority = ["NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLoss", "ProfitLoss", "IncomeLossFromContinuingOperations"]
    ni_df, ni_meta = _build_composite_by_year_usd(facts_json, ni_priority, min_duration=350)
    meta["net_income_composite"] = ni_meta
    ni_map = dict(zip(ni_df["year"].astype(int), ni_df["value"].astype(float))) if not ni_df.empty else {}

    assets_tag, assets_map = _annual_series_map_usd(facts_json, ["Assets"], min_duration=0)
    meta["assets_tag"] = assets_tag

    eq_tag, eq_map = _annual_series_map_usd(
        facts_json, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"],
        min_duration=0
    )
    meta["equity_tag"] = eq_tag

    inv_tag, inv_map = _annual_series_map_usd(facts_json, ["InventoryNet", "InventoryGross", "Inventory"], min_duration=0)
    meta["inventory_tag"] = inv_tag

    cogs_candidates = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfSales", "CostOfGoodsSold"]
    cogs_tag, cogs_map = _annual_series_map_usd(facts_json, cogs_candidates, min_duration=350)
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
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, roa, label="ROA (%)", color=C_SILVER, linewidth=2.5, marker="o", markersize=6)
    ax.plot(x, roe, label="ROE (%)", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)

    ax.set_ylabel("%", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


def plot_inventory_turnover(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    it = df["InventoryTurnover(x)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, it, label="Inventory Turnover (x)", color=C_BLUE, linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("x", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


# =========================
# EPS (unit-aware) + STOCK SPLIT ADJUSTMENT - FIXED (5)
# =========================
def _get_split_adjustment_factor(ticker: str, date_index: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    factors = pd.Series(1.0, index=date_index)
    if not ticker:
        return factors, {}

    try:
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


def _extract_eps_series_any_unit(facts_json: dict, xbrl_tag: str, min_duration: int = 0) -> pd.DataFrame:
    """
    Extract EPS (Supports min_duration)
    """
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
            start = x.get("start")
            val = x.get("val")
            fy_raw = x.get("fy", None)
            filed = x.get("filed")

            if not end or val is None or not filed:
                continue
            if not form.startswith("10-K"):
                continue

            end_ts = pd.to_datetime(end, errors="coerce")
            start_ts = pd.to_datetime(start, errors="coerce")
            filed_ts = pd.to_datetime(filed, errors="coerce")

            if pd.isna(end_ts) or pd.isna(filed_ts):
                continue

            if min_duration > 0:
                if pd.isna(start_ts):
                    continue
                days = (end_ts - start_ts).days
                if days < min_duration:
                    continue 

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
    df = df[df["annual_fp"] == 1]

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

    df_best = df_best.sort_values(["year", "filed"]).drop_duplicates(subset=["year"], keep="last")
    df_best["unit_best"] = df_best["unit"]
    return df_best[["year", "end", "val", "unit_best"]]


def build_eps_table(facts_json: dict, ticker_symbol: str = "") -> tuple[pd.DataFrame, dict]:
    meta = {}
    eps_tags = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]

    best_df = pd.DataFrame()
    best_tag = None

    for tag in eps_tags:
        df = _extract_eps_series_any_unit(facts_json, tag, min_duration=350)
        if df.empty:
            continue
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
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, eps, label="EPS (Adjusted, Diluted)", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel(unit_label, color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


# =========================
# UI
# =========================
def render(authed_email: str):
    st.set_page_config(page_title="Fundamentals (Luxury Edition)", layout="wide")
    # Apply Luxury CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap');
        .stApp { background-color: #050505 !important; color: #f0f0f0 !important; font-family: 'Times New Roman', serif; }
        div[data-testid="stMarkdownContainer"] p, label, h1, h2, h3 { color: #f0f0f0 !important; font-family: 'Times New Roman', serif !important; }
        input.st-ai, input.st-ah, div[data-baseweb="input"] { background-color: #111111 !important; color: #C5A059 !important; border-color: #333 !important; }
        div[data-testid="stFormSubmitButton"] button { background-color: #1a1a1a !important; color: #C5A059 !important; border: 1px solid #333 !important; font-family: 'Times New Roman', serif; }
        div[data-testid="stFormSubmitButton"] button:hover { background-color: #C5A059 !important; border-color: #C5A059 !important; color: #000000 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Fundamental Analysis")
    st.caption(f"Authenticated User: {authed_email}")

    with st.form("input_form"):
        ticker = st.text_input("Ticker (US Stock)", value="MSFT")
        n_years = st.slider("Period (Years)", min_value=3, max_value=15, value=10)
        submitted = st.form_submit_button("Run Analysis")

    if not submitted:
        st.stop()

    t = ticker.strip().lower()
    if not t:
        st.error("Please enter a Ticker.")
        st.stop()

    cik_map = fetch_ticker_cik_map()
    cik10 = cik_map.get(t)
    if not cik10:
        st.error("CIK not found for this Ticker.")
        st.stop()

    facts = fetch_company_facts(cik10)
    company_name = facts.get("entityName", ticker.upper())

    tab_pl, tab_bs, tab_cf, tab_seg, tab_rpo, tab_turn, tab_eps = st.tabs(
        ["PL (Annual)", "BS (Latest)", "CF (Annual)", "Segments", "Backlog", "Ratios", "EPS"]
    )

    with tab_pl:
        pl_table, pl_meta = build_pl_annual_table(facts)
        if pl_table.empty:
            st.error("Could not retrieve PL data.")
            st.write(pl_meta)
            st.stop()
        pl_disp = _slice_latest_n_years(pl_table, int(n_years))
        st.caption(f"Showing {len(pl_disp)} years")
        
        # Simplified Title
        plot_pl_annual(pl_disp, f"Income Statement")
        
        st.markdown("### Annual PL (Million USD)")
        st.dataframe(
            pl_disp.style.format({"Revenue(M$)": "{:,.0f}", "OpIncome(M$)": "{:,.0f}", "NetIncome(M$)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("### Operating Margin Trend")
        plot_operating_margin(pl_disp, f"Operating Margin")
        with st.expander("PL Debug", expanded=False):
            st.write(pl_meta)

    with tab_bs:
        year = _latest_year_from_assets(facts)
        if year is None:
            st.error("Could not retrieve Assets.")
            st.stop()
        snap, bs_meta = build_bs_latest_simple(facts, year)
        st.caption(f"Latest Year {year}")
        plot_bs_bar(snap, f"Balance Sheet Summary")
        assets_pie, le_pie, pie_meta = build_bs_pies_latest(facts, year)
        plot_two_pies(assets_pie, le_pie, year)
        with st.expander("BS Debug", expanded=False):
            st.write({"bs": bs_meta, "pies": pie_meta})

    with tab_cf:
        cf_table, cf_meta = build_cf_annual_table(facts)
        if cf_table.empty:
            st.error("Could not retrieve CF data.")
            st.write(cf_meta)
            st.stop()
        cf_disp = _slice_latest_n_years(cf_table, int(n_years))
        st.caption(f"Showing {len(cf_disp)} years")
        st.markdown("### CFO & Capex Trend")
        plot_cf_cfo_capex(cf_disp, f"CFO and Capex (Negative)")
        st.markdown("### Free Cash Flow Trend")
        plot_cf_fcf(cf_disp, f"FCF with Zero Line")
        st.markdown("### Annual CF (Million USD)")
        st.dataframe(
            cf_disp.style.format({"CFO(M$)": "{:,.0f}", "CFI(M$)": "{:,.0f}", "CFF(M$)": "{:,.0f}", "Capex(M$)": "{:,.0f}", "FCF(M$)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("CF Debug", expanded=False):
            st.write(cf_meta)

    with tab_seg:
        st.caption("Revenue Segments")
        biz_df, geo_df = build_segment_table(facts)
        
        col1, col2 = st.columns(2)
        with col1:
            if not biz_df.empty:
                biz_disp = biz_df[biz_df.index >= int(biz_df.index.max()) - n_years + 1]
                plot_stacked_bar(biz_disp, f"Revenue by Business")
                st.dataframe(biz_disp.style.format("{:,.0f}"), use_container_width=True)
            else:
                st.info("No Business Segment info found.")
        
        with col2:
            if not geo_df.empty:
                geo_disp = geo_df[geo_df.index >= int(geo_df.index.max()) - n_years + 1]
                plot_stacked_bar(geo_disp, f"Revenue by Geography")
                st.dataframe(geo_disp.style.format("{:,.0f}"), use_container_width=True)
            else:
                st.info("No Geography Segment info found.")

    with tab_rpo:
        rpo_table, rpo_meta = build_rpo_annual_table(facts)
        if rpo_table.empty:
            st.warning("RPO data unavailable.")
        else:
            rpo_disp = _slice_latest_n_years(rpo_table, int(n_years))
            st.caption(f"Showing {len(rpo_disp)} years")
            plot_rpo_annual(rpo_disp, f"RPO & Contract Liabilities")
            st.dataframe(
                rpo_disp.style.format({"RPO(M$)": "{:,.0f}", "ContractLiab(M$)": "{:,.0f}", "ContractLiabCurrent(M$)": "{:,.0f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with tab_turn:
        rat_table, rat_meta = build_ratios_table(facts)
        if rat_table.empty:
            st.error("Data unavailable for Ratios.")
            st.stop()
        rat_disp = _slice_latest_n_years(rat_table, int(n_years))
        st.caption(f"Showing {len(rat_disp)} years")
        st.markdown("### Return Ratios")
        plot_roa_roe(rat_disp, f"ROA / ROE")
        st.markdown("### Efficiency")
        plot_inventory_turnover(rat_disp, f"Inventory Turnover")
        st.dataframe(
            rat_disp.style.format({"ROA(%)": "{:.2f}", "ROE(%)": "{:.2f}", "InventoryTurnover(x)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with tab_eps:
        eps_table, eps_meta = build_eps_table(facts, ticker_symbol=t)
        if eps_table.empty:
            st.error("Could not retrieve EPS.")
        else:
            eps_disp = _slice_latest_n_years(eps_table, int(n_years))
            unit_label = eps_meta.get("eps_unit", "USD/share")
            st.caption(f"Showing {len(eps_disp)} years")
            plot_eps(eps_disp, f"EPS (Diluted)", unit_label=unit_label)
            st.dataframe(eps_disp.style.format({"EPS": "{:.2f}"}), use_container_width=True, hide_index=True)


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
