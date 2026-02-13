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
import time

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
    """Adds the OUT-STANDER watermark logo with outline for visibility."""
    import matplotlib.patheffects as pe
    ax.text(0.98, 0.03, "OUT-STANDER", transform=ax.transAxes,
            fontsize=20, color='#a09080', fontweight='bold',
            fontname='serif', ha='right', va='bottom', zorder=5, alpha=0.7,
            path_effects=[
                pe.withStroke(linewidth=3, foreground='#0a0a0a'),
            ])

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
SEC_HEADERS_GENERIC = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
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
# Segment Analysis (XBRL Filing-based)
# =========================

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _get_10k_filings_list(cik10: str, max_filings: int = 8) -> list[dict]:
    """Get recent 10-K filing metadata from SEC submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    report_dates = recent.get("reportDate", [])

    results = []
    for i, form in enumerate(forms):
        if form.upper() in ("10-K", "10-K/A", "20-F", "20-F/A", "40-F"):
            results.append({
                "accession": accessions[i] if i < len(accessions) else "",
                "filing_date": dates[i] if i < len(dates) else "",
                "report_date": report_dates[i] if i < len(report_dates) else "",
                "form": form,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
            })
            if len(results) >= max_filings:
                break
    return results


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_xbrl_instance(cik10: str, accession: str, primary_doc: str = "") -> tuple[str | None, str]:
    """Download XBRL instance XML for a filing. Returns (xml_text, debug_msg)."""
    cik_int = str(int(cik10))
    acc_nd = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nd}/"
    debug = f"acc={accession} primary_doc={primary_doc}\nbase={base}\n"

    # Build file candidates from primaryDocument
    file_candidates = []
    if primary_doc:
        stem = primary_doc
        for ext in [".htm", ".html"]:
            if stem.lower().endswith(ext):
                stem = stem[:-len(ext)]
                break
        file_candidates.append((stem + "_htm.xml", "_htm.xml"))
        file_candidates.append((primary_doc, "primaryDoc(iXBRL)"))

    # Try direct file URLs (use SEC_HEADERS_GENERIC to avoid wrong Host header)
    for fname, label in file_candidates:
        url = base + fname
        try:
            time.sleep(0.12)
            r = requests.get(url, headers=SEC_HEADERS_GENERIC, timeout=60)
            if r.status_code == 200 and len(r.text) > 1000:
                debug += f"✓ {label}: {len(r.text)} chars\n"
                return r.text, debug
            else:
                debug += f"✗ {label}: {r.status_code}\n"
        except Exception as e:
            debug += f"✗ {label}: {e}\n"

    # Fallback: try index.json
    idx_url = base + "index.json"
    try:
        time.sleep(0.12)
        r = requests.get(idx_url, headers=SEC_HEADERS_GENERIC, timeout=30)
        if r.status_code == 200:
            items = r.json().get("directory", {}).get("item", [])
            file_list = [item.get("name", "") for item in items if isinstance(item, dict)]
            htm_files = [f for f in file_list if f.endswith("_htm.xml")]
            debug += f"index.json OK: {len(file_list)} files, htm_xml={htm_files[:3]}\n"
            if htm_files:
                target_url = base + htm_files[0]
                time.sleep(0.12)
                r2 = requests.get(target_url, headers=SEC_HEADERS_GENERIC, timeout=90)
                if r2.status_code == 200:
                    debug += f"✓ via index.json: {htm_files[0]} ({len(r2.text)} chars)\n"
                    return r2.text, debug
        else:
            debug += f"✗ index.json: {r.status_code}\n"
    except Exception as e:
        debug += f"✗ index.json: {e}\n"

    debug += "All candidates failed\n"
    return None, debug


def _diagnose_xbrl(xml_text: str) -> dict:
    """Diagnostic: analyze XBRL XML to understand its structure."""
    info = {}
    info["total_chars"] = len(xml_text)

    # Check if it's standard XBRL or iXBRL
    info["has_xbrli_context"] = "xbrli:context" in xml_text or "<context " in xml_text
    info["has_ix_nonfraction"] = "ix:nonFraction" in xml_text or "ix:nonfraction" in xml_text

    # Count contexts (all, not just dimensional)
    ctx_all = re.findall(r'<(?:[\w-]+:)?context\s+id="([^"]+)"', xml_text)
    info["total_contexts"] = len(ctx_all)
    info["sample_ctx_ids"] = ctx_all[:5]

    # Count dimensional contexts
    dim_ctx = re.findall(r'<(?:[\w-]+:)?explicitMember\s+dimension="([^"]+)"', xml_text)
    info["dimensional_contexts"] = len(dim_ctx)
    info["unique_dimensions"] = list(set(dim_ctx))[:10]

    # Find revenue-related tags (broad search)
    # First: case-insensitive search for ANY occurrence of "revenue" near a tag
    rev_raw = re.findall(r'[<"\s](\w*[Rr]evenue\w*)', xml_text[:500000])
    rev_unique = list(set(rev_raw))[:20]
    info["revenue_strings_in_xml"] = rev_unique

    # Look for tags containing "Revenue" with contextRef (handle hyphenated namespaces)
    rev_ctx = re.findall(r'<([\w-]+:\w*[Rr]evenue\w*)\s[^>]*contextRef', xml_text[:500000])
    info["revenue_tags_with_contextref"] = list(set(rev_ctx))[:10]

    # Show first 300 chars around first "Revenue" occurrence
    idx = xml_text.lower().find("revenue")
    if idx >= 0:
        start = max(0, idx - 50)
        end = min(len(xml_text), idx + 250)
        info["revenue_context_snippet"] = xml_text[start:end]
    else:
        info["revenue_context_snippet"] = "NO 'revenue' found in XML"

    # Sample: find any fact with contextRef in a dimensional context
    # First get dimensional context IDs
    ctx_re = re.compile(
        r'<(?:[\w-]+:)?context\s+id="([^"]+)"[^>]*>(.*?)</(?:[\w-]+:)?context>', re.DOTALL
    )
    dim_re = re.compile(
        r'<(?:[\w-]+:)?explicitMember\s+dimension="([^"]+)"[^>]*>([^<]+)</(?:[\w-]+:)?explicitMember>'
    )
    dim_ctx_ids = set()
    dim_ctx_samples = []
    for m in ctx_re.finditer(xml_text):
        cid, body = m.group(1), m.group(2)
        dims = dim_re.findall(body)
        if dims:
            dim_ctx_ids.add(cid)
            if len(dim_ctx_samples) < 3:
                dim_ctx_samples.append({"id": cid, "dims": dims, "body_snippet": body[:200]})
    info["dim_ctx_count"] = len(dim_ctx_ids)
    info["dim_ctx_samples"] = dim_ctx_samples

    # Check: do revenue facts reference dimensional contexts?
    rev_kw = ["revenue", "sales"]
    rev_with_dim = []
    fact_re = re.compile(
        r'<(?:([\w-]+):)?(\w+)\s+([^>]*?)contextRef="([^"]+)"([^>]*?)>([^<]+)</(?:[\w-]+:)?\2>'
    )
    for m in fact_re.finditer(xml_text):
        tag = m.group(2)
        if any(kw in tag.lower() for kw in rev_kw):
            ctx_ref = m.group(4)
            if ctx_ref in dim_ctx_ids and len(rev_with_dim) < 5:
                rev_with_dim.append({"tag": tag, "ctx": ctx_ref, "val": m.group(6).strip()[:30]})
    info["revenue_facts_in_dim_ctx"] = rev_with_dim

    # Also check iXBRL pattern
    ix_rev = []
    ix_re = re.compile(r'<ix:nonFraction[^>]*name="([^"]*(?:[Rr]evenue|[Ss]ales)[^"]*)"[^>]*contextRef="([^"]+)"[^>]*>([^<]*)</ix:nonFraction>', re.IGNORECASE)
    for m in ix_re.finditer(xml_text):
        name, ctx, val = m.group(1), m.group(2), m.group(3)
        if ctx in dim_ctx_ids and len(ix_rev) < 5:
            ix_rev.append({"name": name, "ctx": ctx, "val": val.strip()[:30]})
    info["ix_revenue_in_dim_ctx"] = ix_rev

    # Broader: find ANY tags in dimensional contexts (sample 5)
    any_dim_facts = []
    for m in fact_re.finditer(xml_text):
        ctx_ref = m.group(4)
        if ctx_ref in dim_ctx_ids and len(any_dim_facts) < 5:
            any_dim_facts.append({"tag": m.group(2), "ctx": ctx_ref, "val": m.group(6).strip()[:20]})
    info["any_facts_in_dim_ctx"] = any_dim_facts

    return info


def _parse_segment_facts_from_xbrl(xml_text: str) -> list[dict]:
    """Extract dimensioned revenue facts from XBRL/iXBRL via regex."""

    # --- Step 1: Parse contexts with dimensional segment info ---
    ctx_re = re.compile(
        r'<(?:[\w-]+:)?context\s+id="([^"]+)"[^>]*>(.*?)</(?:[\w-]+:)?context>', re.DOTALL
    )
    dim_re = re.compile(
        r'<(?:[\w-]+:)?explicitMember\s+dimension="([^"]+)"[^>]*>([^<]+)</(?:[\w-]+:)?explicitMember>'
    )
    start_re = re.compile(r'<(?:[\w-]+:)?startDate>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?startDate>')
    end_re = re.compile(r'<(?:[\w-]+:)?endDate>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?endDate>')
    instant_re = re.compile(r'<(?:[\w-]+:)?instant>(\d{4}-\d{2}-\d{2})</(?:[\w-]+:)?instant>')

    contexts = {}
    for m in ctx_re.finditer(xml_text):
        cid, body = m.group(1), m.group(2)
        dims = dim_re.findall(body)
        if not dims:
            continue

        s = start_re.search(body)
        e = end_re.search(body)
        ins = instant_re.search(body)
        start_date = s.group(1) if s else None
        end_date = e.group(1) if e else (ins.group(1) if ins else None)

        dur = 0
        if start_date and end_date:
            try:
                dur = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            except Exception:
                pass

        contexts[cid] = {"dims": dims, "start": start_date, "end": end_date, "dur": dur}

    if not contexts:
        return []

    # --- Step 2: Find revenue facts referencing dimensional contexts ---
    revenue_tags = {
        "revenues", "revenue",
        "revenuefromcontractwithcustomerexcludingassessedtax",
        "revenuesfromcontractwithcustomerexcludingassessedtax",
        "revenuefromcontractwithcustomerincludingassessedtax",
        "revenuesfromcontractwithcustomerincludingassessedtax",
        "salesrevenuenet", "salestoexternalcustomers",
        "netoperatingrevenues", "salesrevenueservicesnet", "salesrevenue",
        "revenuesnetofinterestexpense",
        "revenuesfromexternalcustomer", "revenuesfromexternalcustomers",
        "revenuefromexternalcustomers",
        "segmentreportinginformationrevenue",
    }

    results = []

    # --- Pattern A: Standard XBRL ---
    # <ns:Tag contextRef="..." ...>value</ns:Tag>
    # NOTE: namespace prefix can contain hyphens (e.g. us-gaap:), so use [\w-]+ not \w+
    fact_re = re.compile(
        r'<(?:([\w-]+):)?(\w+)\s+([^>]*?)contextRef="([^"]+)"([^>]*?)>([^<]+)</(?:[\w-]+:)?\2>'
    )
    for m in fact_re.finditer(xml_text):
        tag = m.group(2)
        if tag.lower() not in revenue_tags:
            continue
        ctx_ref = m.group(4)
        if ctx_ref not in contexts:
            continue
        val_str = m.group(6).strip().replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            continue
        if val == 0:
            continue
        _collect_segment_result(results, tag, ctx_ref, val, contexts)

    # --- Pattern B: Inline XBRL (iXBRL) ---
    # <ix:nonFraction contextRef="..." name="us-gaap:Revenues" ...>value</ix:nonFraction>
    ixbrl_re = re.compile(
        r'<ix:nonFraction\s+([^>]*?)contextRef="([^"]+)"([^>]*?)name="([^"]+)"([^>]*?)>([^<]*)</ix:nonFraction>',
        re.IGNORECASE
    )
    # Also match with name before contextRef
    ixbrl_re2 = re.compile(
        r'<ix:nonFraction\s+([^>]*?)name="([^"]+)"([^>]*?)contextRef="([^"]+)"([^>]*?)>([^<]*)</ix:nonFraction>',
        re.IGNORECASE
    )

    for pattern, ctx_grp, name_grp, val_grp in [(ixbrl_re, 2, 4, 6), (ixbrl_re2, 4, 2, 6)]:
        for m in pattern.finditer(xml_text):
            name_full = m.group(name_grp)
            # name_full is like "us-gaap:Revenues"
            tag = name_full.split(":")[-1] if ":" in name_full else name_full
            if tag.lower() not in revenue_tags:
                continue
            ctx_ref = m.group(ctx_grp)
            if ctx_ref not in contexts:
                continue
            val_str = m.group(val_grp).strip().replace(",", "").replace(" ", "")
            if not val_str or val_str == "-":
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            if val == 0:
                continue

            # Handle iXBRL scale attribute
            full_tag = m.group(0)
            scale_m = re.search(r'scale="(-?\d+)"', full_tag)
            if scale_m:
                val = val * (10 ** int(scale_m.group(1)))
            # Handle sign attribute
            sign_m = re.search(r'sign="-"', full_tag)
            if sign_m:
                val = -abs(val)

            _collect_segment_result(results, tag, ctx_ref, val, contexts)

    return results


def _collect_segment_result(results: list, tag: str, ctx_ref: str, val: float, contexts: dict):
    """Helper to add a segment fact to results list.
    
    Only uses single-dimension contexts to avoid cross-sectional pollution.
    """
    ctx = contexts[ctx_ref]
    if ctx["dur"] < 330:
        return
    # Strict: only single-dimension contexts (avoids cross-dimensional intersections)
    if len(ctx["dims"]) != 1:
        return

    dim, member = ctx["dims"][0]
    end_str = ctx["end"]
    year = int(end_str[:4]) if end_str else 0
    results.append({
        "tag": tag,
        "dimension": dim,
        "member": member,
        "end_date": end_str,
        "year": year,
        "value": val,
    })


def _classify_dimension(dim_str: str) -> str:
    """Classify XBRL dimension axis into Segment / Product / Geography / Skip.
    
    Returns:
        "Segment"   - Official business segments (StatementBusinessSegmentsAxis)
        "Product"   - Product/Service breakdown (ProductOrServiceAxis, etc.)
        "Geography" - Geographic breakdown (StatementGeographicalAxis, etc.)
        "Skip"      - Irrelevant axes (tax, hedging, debt, etc.)
    """
    d = dim_str.lower()
    
    # --- Geography ---
    geo_kw = [
        "statementgeographicalaxis", "geographicalaxis",
        "entitywideinformationrevenuefrommajorcustomer",  # sometimes used for geo
    ]
    if any(k in d for k in geo_kw):
        return "Geography"
    
    # --- Official Business Segments ---
    seg_kw = [
        "statementbusinesssegmentsaxis",
        "segmentreportingaxis",
    ]
    if any(k in d for k in seg_kw):
        return "Segment"
    
    # --- Product/Service breakdown ---
    prod_kw = [
        "productorserviceaxis",
        "productsorservices",
        "statementoperatingsegmentsaxis",
    ]
    if any(k in d for k in prod_kw):
        return "Product"
    
    # --- Skip everything else (tax, hedging, debt, legal entity, etc.) ---
    return "Skip"


def _clean_xbrl_member(m: str) -> str:
    """Clean XBRL member name to human-readable form."""
    if ":" in m:
        m = m.split(":")[-1]
    if m.endswith("Member"):
        m = m[:-6]
    # CamelCase → spaces
    m = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', m)
    m = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', m)
    return m.strip()


def build_segment_table(cik10: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build revenue segment tables from individual 10-K XBRL filings.
    
    Returns: (segment_pivot, product_pivot, geo_pivot, meta)
    """
    meta = {}

    filings = _get_10k_filings_list(cik10)
    meta["filings_found"] = len(filings)
    if not filings:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    all_facts = []
    fetched = 0
    fetch_debug = []
    for f in filings:
        xml, dbg = _fetch_xbrl_instance(cik10, f["accession"], f.get("primary_doc", ""))
        fetch_debug.append(dbg)
        if xml is None:
            continue
        fetched += 1
        facts = _parse_segment_facts_from_xbrl(xml)
        meta[f"filing_{fetched}_facts"] = len(facts)
        all_facts.extend(facts)
        if fetched == 1:
            meta["diag"] = _diagnose_xbrl(xml)

    meta["filings_fetched"] = fetched
    meta["total_facts"] = len(all_facts)
    meta["fetch_debug"] = fetch_debug

    if not all_facts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    df = pd.DataFrame(all_facts)
    df["axis_type"] = df["dimension"].apply(_classify_dimension)
    df["member_clean"] = df["member"].apply(_clean_xbrl_member)

    # Drop irrelevant axes
    df = df[df["axis_type"] != "Skip"]

    # Exclude totals / eliminations / corporate
    exclude_kw = [
        "total", "elimination", "adjust", "consolidation", "allother",
        "intersegment", "corporate", "reconciling", "unallocated",
    ]
    mask = ~df["member_clean"].str.lower().apply(
        lambda x: any(k in x.lower() for k in exclude_kw)
    )
    df = df[mask]

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta

    # Deduplicate: for each year/axis_type/member, keep largest value
    df = df.sort_values("value", ascending=False).drop_duplicates(
        subset=["year", "axis_type", "member_clean"], keep="first"
    )

    # Convert to M$
    df["value_m"] = df["value"].apply(_to_musd)

    meta["unique_dimensions"] = df["dimension"].unique().tolist()
    meta["segment_members"] = sorted(df[df["axis_type"] == "Segment"]["member_clean"].unique().tolist())
    meta["product_members"] = sorted(df[df["axis_type"] == "Product"]["member_clean"].unique().tolist())
    meta["geo_members"] = sorted(df[df["axis_type"] == "Geography"]["member_clean"].unique().tolist())

    def _make_pivot(sub_df: pd.DataFrame) -> pd.DataFrame:
        if sub_df.empty:
            return pd.DataFrame()
        pivot = sub_df.pivot_table(index="year", columns="member_clean", values="value_m", aggfunc="first")
        # Drop columns that are >70% NaN (sparse/discontinued segments)
        thresh = len(pivot) * 0.3
        pivot = pivot.dropna(axis=1, thresh=int(max(thresh, 1)))
        # Drop columns where the latest year has no data (discontinued)
        if not pivot.empty:
            latest_year = pivot.index.max()
            pivot = pivot.loc[:, pivot.loc[latest_year].notna()]
        return pivot.sort_index()

    seg_pivot = _make_pivot(df[df["axis_type"] == "Segment"])
    prod_pivot = _make_pivot(df[df["axis_type"] == "Product"])
    geo_pivot = _make_pivot(df[df["axis_type"] == "Geography"])

    return seg_pivot, prod_pivot, geo_pivot, meta


def plot_stacked_bar(df: pd.DataFrame, title: str):
    if df.empty:
        st.info(f"{title}: Data not found.")
        return

    n_cols = len(df.columns)
    # Extended luxury palette for more segments
    palette = [C_GOLD, C_SILVER, C_BLUE, C_BRONZE, C_SLATE,
               "#8B4513", "#556B2F", "#8B008B", "#2F4F4F", "#B8860B",
               "#4169E1", "#CD853F", "#6B8E23", "#9370DB", "#20B2AA"]
    colors = palette[:n_cols] if n_cols <= len(palette) else palette * (n_cols // len(palette) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    df.plot(kind="bar", stacked=True, ax=ax, color=colors[:n_cols], width=0.7, edgecolor='none')

    ax.set_ylabel("Revenue (Million USD)", color=TEXT_COLOR)
    ax.tick_params(colors=TICK_COLOR, axis='x', rotation=0)
    ax.tick_params(colors=TICK_COLOR, axis='y')
    
    # Format y-axis with comma separator
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Legend: outside right, compact
    ax.legend(
        facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR,
        loc="upper left", bbox_to_anchor=(1.0, 1.0),
        frameon=False, fontsize=8,
    )
    fig.tight_layout()
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


def plot_roa(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    roa = df["ROA(%)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, roa, label="ROA (%)", color=C_SILVER, linewidth=2.5, marker="o", markersize=6)
    ax.set_ylabel("%", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


def plot_roe(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["FY"].astype(str).tolist()
    roe = df["ROE(%)"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

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
    out = pd.DataFrame({
        "FY": best_df["year"].astype(int),
        "End": best_df["end"].values,
        "EPS": best_df["val_adjusted"].astype(float),
    })
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
# PER (Price-to-Earnings Ratio) - Split-Adjusted
# =========================
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_adj_close_at_dates(ticker_symbol: str, dates: list[str]) -> dict:
    """Fetch split-adjusted close prices at given dates using yfinance.
    
    For each target date, searches a window of +/- 5 business days
    to handle weekends and holidays.
    Returns {date_str: price} dict.
    """
    if not ticker_symbol or not dates:
        return {}

    try:
        tk = yf.Ticker(ticker_symbol.upper())
        price_map = {}

        all_dates = [pd.to_datetime(d) for d in dates]
        global_start = min(all_dates) - pd.Timedelta(days=10)
        global_end = max(all_dates) + pd.Timedelta(days=10)

        hist = tk.history(start=global_start.strftime("%Y-%m-%d"),
                          end=global_end.strftime("%Y-%m-%d"),
                          auto_adjust=True)

        if hist.empty:
            return {}

        hist.index = hist.index.tz_localize(None)

        for d_str in dates:
            target = pd.to_datetime(d_str)
            mask = (hist.index >= target - pd.Timedelta(days=7)) & (hist.index <= target + pd.Timedelta(days=7))
            candidates = hist.loc[mask]
            if candidates.empty:
                continue
            idx = (candidates.index - target).map(abs)
            closest_idx = idx.argmin()
            price_map[d_str] = float(candidates["Close"].iloc[closest_idx])

        return price_map
    except Exception as e:
        return {}


def build_per_table(eps_table: pd.DataFrame, ticker_symbol: str) -> tuple[pd.DataFrame, dict]:
    """Build PER table from EPS table with End dates."""
    meta = {}

    if eps_table.empty or "End" not in eps_table.columns:
        return pd.DataFrame(), meta

    df = eps_table.copy()
    df["End"] = pd.to_datetime(df["End"])
    date_strs = [d.strftime("%Y-%m-%d") for d in df["End"]]

    price_map = _fetch_adj_close_at_dates(ticker_symbol, date_strs)
    meta["prices_fetched"] = len(price_map)
    meta["prices_requested"] = len(date_strs)

    rows = []
    for _, r in df.iterrows():
        fy = int(r["FY"])
        eps = float(r["EPS"])
        d_str = r["End"].strftime("%Y-%m-%d")
        price = price_map.get(d_str, np.nan)

        per = np.nan
        if np.isfinite(price) and np.isfinite(eps) and eps > 0:
            per = price / eps

        rows.append([fy, price, eps, per])

    out = pd.DataFrame(rows, columns=["FY", "Price", "EPS", "PER"])
    meta["years_with_per"] = int(out["PER"].dropna().shape[0])
    return out, meta


def plot_per(table: pd.DataFrame, title: str):
    df = table.dropna(subset=["PER"]).copy()
    if df.empty:
        st.info("PER data not available.")
        return

    x = df["FY"].astype(str).tolist()
    per = df["PER"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.plot(x, per, label="PER (Trailing)", color=C_GOLD, linewidth=2.5, marker="o", markersize=6)

    # Reference lines
    ax.axhline(y=15, color=C_SILVER, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(y=25, color=C_SILVER, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_ylabel("x", color=TEXT_COLOR)
    ax.legend(facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)
    st.pyplot(fig)


# =========================
# Quarterly PL & EPS
# =========================
def _extract_all_periods_usd(facts_json: dict, xbrl_tag: str) -> pd.DataFrame:
    """Extract all 10-Q and 10-K entries for a tag (USD unit), with duration info."""
    facts_root = facts_json.get("facts", {})
    if not facts_root:
        return pd.DataFrame()

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
        segment = x.get("segment")

        if not end or not start or val is None:
            continue
        if not (form.startswith("10-K") or form.startswith("10-Q") or form.startswith("20-F") or form.startswith("40-F")):
            continue
        if segment is not None:
            continue

        end_ts = pd.to_datetime(end, errors="coerce")
        start_ts = pd.to_datetime(start, errors="coerce")
        filed_ts = pd.to_datetime(filed, errors="coerce")

        if pd.isna(end_ts) or pd.isna(start_ts):
            continue

        days = (end_ts - start_ts).days

        if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
            fy = int(fy_raw)
        else:
            fy = int(end_ts.year)

        rows.append({
            "fy": fy,
            "fp": fp,
            "start": start_ts,
            "end": end_ts,
            "days": days,
            "val": _safe_float(val),
            "filed": filed_ts,
            "form": form,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["val"])
    return df


def _build_individual_quarters(facts_json: dict, tag_priority: list[str]) -> pd.DataFrame:
    """Build individual quarter values (Q1-Q4) from composite tags.
    
    Q1-Q3: extracted from 10-Q with duration 60-100 days.
    Q4: derived as FY_annual - (Q1 + Q2 + Q3).
    Returns DataFrame with columns: [fy, q, end, val, source]
    """
    all_periods = pd.DataFrame()
    used_tag = None

    for tag in tag_priority:
        df = _extract_all_periods_usd(facts_json, tag)
        if df.empty:
            continue
        if all_periods.empty or df["fy"].max() > all_periods["fy"].max():
            all_periods = df
            used_tag = tag
        elif df["fy"].max() == all_periods["fy"].max() and len(df) > len(all_periods):
            all_periods = df
            used_tag = tag

    if all_periods.empty:
        return pd.DataFrame(columns=["fy", "q", "end", "val", "source"])

    # Individual quarters from 10-Q (duration 60-100 days)
    q_mask = (all_periods["days"] >= 60) & (all_periods["days"] <= 100) & (all_periods["fp"].isin({"Q1", "Q2", "Q3"}))
    q_data = all_periods[q_mask].copy()

    # Map fp to quarter number
    fp_to_q = {"Q1": 1, "Q2": 2, "Q3": 3}
    q_data["q"] = q_data["fp"].map(fp_to_q)
    q_data = q_data.sort_values(["fy", "q", "filed"]).drop_duplicates(subset=["fy", "q"], keep="last")

    # Annual totals from 10-K (duration > 330 days)
    fy_mask = (all_periods["days"] >= 330) & (all_periods["fp"] == "FY")
    fy_data = all_periods[fy_mask].copy()
    fy_data = fy_data.sort_values(["fy", "filed"]).drop_duplicates(subset=["fy"], keep="last")
    fy_map = dict(zip(fy_data["fy"].astype(int), fy_data["val"].astype(float)))
    fy_end_map = dict(zip(fy_data["fy"].astype(int), fy_data["end"]))

    # Build result
    result_rows = []

    # Add Q1-Q3
    for _, r in q_data.iterrows():
        result_rows.append({
            "fy": int(r["fy"]),
            "q": int(r["q"]),
            "end": r["end"],
            "val": float(r["val"]),
            "source": "direct",
        })

    # Derive Q4 = FY - (Q1 + Q2 + Q3)
    q_pivot = q_data.groupby("fy").apply(
        lambda g: {int(r["q"]): float(r["val"]) for _, r in g.iterrows()}
    )

    for fy_year, fy_total in fy_map.items():
        if not np.isfinite(fy_total):
            continue
        q_dict = q_pivot.get(fy_year, {})
        q1 = q_dict.get(1, np.nan)
        q2 = q_dict.get(2, np.nan)
        q3 = q_dict.get(3, np.nan)

        if np.isfinite(q1) and np.isfinite(q2) and np.isfinite(q3):
            q4_val = fy_total - (q1 + q2 + q3)
            q4_end = fy_end_map.get(fy_year)
            result_rows.append({
                "fy": fy_year,
                "q": 4,
                "end": q4_end,
                "val": q4_val,
                "source": "derived_FY-Q123",
            })

    if not result_rows:
        return pd.DataFrame(columns=["fy", "q", "end", "val", "source"])

    out = pd.DataFrame(result_rows)
    out["quarter_label"] = out["fy"].astype(str) + "Q" + out["q"].astype(str)
    out = out.sort_values(["fy", "q"]).reset_index(drop=True)
    return out


def _extract_quarterly_eps(facts_json: dict, tag_priority: list[str]) -> pd.DataFrame:
    """Extract quarterly EPS from per-share units (10-Q + 10-K derived Q4)."""
    us = facts_json.get("facts", {}).get("us-gaap", {})

    all_periods = pd.DataFrame()
    used_tag = None

    for tag in tag_priority:
        tag_obj = us.get(tag, {})
        units = tag_obj.get("units", {})
        if not units:
            continue

        rows = []
        for unit_key, node in units.items():
            u_lower = unit_key.lower().replace(" ", "")
            if not ("usd/share" in u_lower or "usd/shr" in u_lower or "usdper" in u_lower):
                continue

            for x in node:
                form = str(x.get("form", "")).upper().strip()
                fp = str(x.get("fp", "")).upper().strip()
                end = x.get("end")
                start = x.get("start")
                val = x.get("val")
                fy_raw = x.get("fy", None)
                filed = x.get("filed")
                segment = x.get("segment")

                if not end or not start or val is None:
                    continue
                if not (form.startswith("10-K") or form.startswith("10-Q")):
                    continue
                if segment is not None:
                    continue

                end_ts = pd.to_datetime(end, errors="coerce")
                start_ts = pd.to_datetime(start, errors="coerce")
                filed_ts = pd.to_datetime(filed, errors="coerce")

                if pd.isna(end_ts) or pd.isna(start_ts):
                    continue

                days = (end_ts - start_ts).days

                if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
                    fy = int(fy_raw)
                else:
                    fy = int(end_ts.year)

                rows.append({
                    "fy": fy, "fp": fp, "start": start_ts, "end": end_ts,
                    "days": days, "val": _safe_float(val), "filed": filed_ts,
                })

        if not rows:
            continue

        df = pd.DataFrame(rows).dropna(subset=["val"])
        if df.empty:
            continue

        if all_periods.empty or df["fy"].max() > all_periods["fy"].max():
            all_periods = df
            used_tag = tag
        elif df["fy"].max() == all_periods["fy"].max() and len(df) > len(all_periods):
            all_periods = df
            used_tag = tag

    if all_periods.empty:
        return pd.DataFrame(columns=["fy", "q", "end", "val", "source"])

    # Individual quarters (60-100 days)
    q_mask = (all_periods["days"] >= 60) & (all_periods["days"] <= 100) & (all_periods["fp"].isin({"Q1", "Q2", "Q3"}))
    q_data = all_periods[q_mask].copy()
    fp_to_q = {"Q1": 1, "Q2": 2, "Q3": 3}
    q_data["q"] = q_data["fp"].map(fp_to_q)
    q_data = q_data.sort_values(["fy", "q", "filed"]).drop_duplicates(subset=["fy", "q"], keep="last")

    # Annual EPS (FY, > 330 days)
    fy_mask = (all_periods["days"] >= 330) & (all_periods["fp"] == "FY")
    fy_data = all_periods[fy_mask].copy()
    fy_data = fy_data.sort_values(["fy", "filed"]).drop_duplicates(subset=["fy"], keep="last")
    fy_map = dict(zip(fy_data["fy"].astype(int), fy_data["val"].astype(float)))
    fy_end_map = dict(zip(fy_data["fy"].astype(int), fy_data["end"]))

    result_rows = []
    for _, r in q_data.iterrows():
        result_rows.append({
            "fy": int(r["fy"]), "q": int(r["q"]), "end": r["end"],
            "val": float(r["val"]), "source": "direct",
        })

    # Q4 EPS = FY EPS - (Q1 + Q2 + Q3) EPS
    q_pivot = q_data.groupby("fy").apply(
        lambda g: {int(r["q"]): float(r["val"]) for _, r in g.iterrows()}
    )

    for fy_year, fy_total in fy_map.items():
        if not np.isfinite(fy_total):
            continue
        q_dict = q_pivot.get(fy_year, {})
        q1 = q_dict.get(1, np.nan)
        q2 = q_dict.get(2, np.nan)
        q3 = q_dict.get(3, np.nan)
        if np.isfinite(q1) and np.isfinite(q2) and np.isfinite(q3):
            q4_val = fy_total - (q1 + q2 + q3)
            result_rows.append({
                "fy": fy_year, "q": 4, "end": fy_end_map.get(fy_year),
                "val": q4_val, "source": "derived_FY-Q123",
            })

    if not result_rows:
        return pd.DataFrame(columns=["fy", "q", "end", "val", "source"])

    out = pd.DataFrame(result_rows)
    out["quarter_label"] = out["fy"].astype(str) + "Q" + out["q"].astype(str)
    out = out.sort_values(["fy", "q"]).reset_index(drop=True)
    return out


def build_quarterly_pl_table(facts_json: dict, n_quarters: int = 8) -> tuple[pd.DataFrame, dict]:
    meta = {}

    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "NetOperatingRevenues",
        "SalesRevenueGoodsNet",
        "SalesRevenueNet",
        "SalesRevenue",
    ]
    op_income_tags = ["OperatingIncomeLoss", "OperatingIncome"]
    pretax_tags = [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ]

    rev_q = _build_individual_quarters(facts_json, revenue_tags)
    meta["rev_quarters"] = len(rev_q) if not rev_q.empty else 0

    op_q = _build_individual_quarters(facts_json, op_income_tags)
    meta["op_quarters"] = len(op_q) if not op_q.empty else 0

    pt_q = _build_individual_quarters(facts_json, pretax_tags)
    meta["pretax_quarters"] = len(pt_q) if not pt_q.empty else 0

    # Merge all on quarter_label
    if rev_q.empty:
        return pd.DataFrame(), meta

    base = rev_q[["fy", "q", "quarter_label", "end"]].copy()
    base["Revenue(M$)"] = rev_q["val"].apply(_to_musd)

    if not op_q.empty:
        op_map = dict(zip(op_q["quarter_label"], op_q["val"].apply(_to_musd)))
        base["OpIncome(M$)"] = base["quarter_label"].map(op_map)
    else:
        base["OpIncome(M$)"] = np.nan

    if not pt_q.empty:
        pt_map = dict(zip(pt_q["quarter_label"], pt_q["val"].apply(_to_musd)))
        base["PreTaxIncome(M$)"] = base["quarter_label"].map(pt_map)
    else:
        base["PreTaxIncome(M$)"] = np.nan

    base = base.sort_values(["fy", "q"]).reset_index(drop=True)

    # Slice to latest n_quarters
    if len(base) > n_quarters:
        base = base.iloc[-n_quarters:].reset_index(drop=True)

    out = base[["quarter_label", "Revenue(M$)", "OpIncome(M$)", "PreTaxIncome(M$)"]].copy()
    out.rename(columns={"quarter_label": "Quarter"}, inplace=True)
    meta["total_quarters"] = len(out)
    return out, meta


def build_quarterly_eps_table(facts_json: dict, ticker_symbol: str = "", n_quarters: int = 8) -> tuple[pd.DataFrame, dict]:
    meta = {}

    eps_tags = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]
    eps_q = _extract_quarterly_eps(facts_json, eps_tags)

    if eps_q.empty:
        return pd.DataFrame(), meta

    # Split adjustment
    if ticker_symbol:
        eps_q["end"] = pd.to_datetime(eps_q["end"])
        factors, split_debug = _get_split_adjustment_factor(ticker_symbol, pd.DatetimeIndex(eps_q["end"]))
        eps_q["val"] = eps_q["val"] * factors.values
        meta["split_adjusted"] = True
    else:
        meta["split_adjusted"] = False

    eps_q = eps_q.sort_values(["fy", "q"]).reset_index(drop=True)

    if len(eps_q) > n_quarters:
        eps_q = eps_q.iloc[-n_quarters:].reset_index(drop=True)

    out = pd.DataFrame({
        "Quarter": eps_q["quarter_label"],
        "EPS": eps_q["val"].astype(float),
    })
    meta["total_quarters"] = len(out)
    return out, meta


def plot_quarterly_bars(table: pd.DataFrame, value_col: str, title: str, color=None):
    if color is None:
        color = C_GOLD
    df = table.copy()
    x = df["Quarter"].tolist()
    vals = df[value_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    colors = [C_GOLD if v >= 0 else C_BRONZE for v in vals]
    ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)

    ax.axhline(y=0, color=GRID_COLOR, linewidth=0.8, linestyle="-")
    ax.set_ylabel("Million USD", color=TEXT_COLOR)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


def plot_quarterly_bars_with_margin(table: pd.DataFrame, income_col: str, revenue_col: str, title: str, margin_label: str = "Margin (%)"):
    """Bar chart (left axis: income) + Line chart (right axis: margin %)."""
    df = table.copy()
    x = df["Quarter"].tolist()
    income = df[income_col].astype(float).to_numpy()
    revenue = df[revenue_col].astype(float).to_numpy()

    margin = np.full_like(income, np.nan, dtype=float)
    for i in range(len(income)):
        if np.isfinite(income[i]) and np.isfinite(revenue[i]) and revenue[i] != 0:
            margin[i] = income[i] / revenue[i] * 100.0

    fig, ax_left = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax_left, title=title, dual_y=True)

    # Bars: income (left axis)
    bar_colors = [C_GOLD if v >= 0 else C_BRONZE for v in income]
    ax_left.bar(x, income, color=bar_colors, alpha=0.85, width=0.6, label=income_col.replace("(M$)", ""))
    ax_left.axhline(y=0, color=GRID_COLOR, linewidth=0.8, linestyle="-")
    ax_left.set_ylabel("Million USD", color=TEXT_COLOR)
    ax_left.tick_params(axis='x', rotation=45)

    # Line: margin (right axis)
    ax_right = ax_left.twinx()
    ax_right.plot(x, margin, color=C_SILVER, linewidth=2.0, marker="o", markersize=5, label=margin_label)
    ax_right.set_ylabel("%", color=TEXT_COLOR)
    ax_right.tick_params(colors=TICK_COLOR, which='both')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_color(GRID_COLOR)
    ax_right.spines['bottom'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)

    st.pyplot(fig)


def plot_quarterly_eps_chart(table: pd.DataFrame, title: str):
    df = table.copy()
    x = df["Quarter"].tolist()
    eps = df["EPS"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    colors = [C_GOLD if v >= 0 else C_BRONZE for v in eps]
    ax.bar(x, eps, color=colors, alpha=0.85, width=0.6)

    ax.axhline(y=0, color=GRID_COLOR, linewidth=0.8, linestyle="-")
    ax.set_ylabel("USD / share", color=TEXT_COLOR)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)


# =========================
# Auto DCF Valuation
# =========================
def _get_shares_outstanding(facts_json: dict) -> tuple[float, str | None]:
    """Get diluted shares outstanding from SEC EDGAR (dei or us-gaap)."""
    candidates = [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("dei", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ]
    facts_root = facts_json.get("facts", {})

    best_val = np.nan
    best_tag = None
    best_year = -1

    for prefix, tag in candidates:
        tag_obj = facts_root.get(prefix, {}).get(tag, {})
        units = tag_obj.get("units", {})

        node = units.get("shares", []) or units.get("USD", [])
        if not node:
            continue

        for x in node:
            form = str(x.get("form", "")).upper().strip()
            fp = str(x.get("fp", "")).upper().strip()
            val = x.get("val")
            fy_raw = x.get("fy", None)
            filed = x.get("filed")

            if val is None:
                continue
            if not (form.startswith("10-K") or form.startswith("20-F") or form.startswith("40-F") or form.startswith("10-Q")):
                continue

            if isinstance(fy_raw, (int, np.integer)) or (isinstance(fy_raw, str) and str(fy_raw).isdigit()):
                fy = int(fy_raw)
            else:
                continue

            v = _safe_float(val)
            if np.isfinite(v) and v > 0 and fy > best_year:
                best_val = v
                best_tag = f"{prefix}:{tag}"
                best_year = fy

    return best_val, best_tag


def _get_net_cash(facts_json: dict, year: int) -> tuple[float, dict]:
    """Calculate net cash = cash - total debt for a given year."""
    meta = {}

    cash_tags = [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalentsAtCarryingValue",
        "CashAndCashEquivalents",
    ]
    _, cash = _value_for_year_usd(facts_json, cash_tags, year)
    meta["cash"] = _to_musd(cash) if np.isfinite(cash) else np.nan

    st_debt_tags = ["ShortTermBorrowings", "CommercialPaper", "DebtCurrent", "LongTermDebtCurrent"]
    _, st_debt = _value_for_year_usd(facts_json, st_debt_tags, year)

    lt_debt_tags = ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"]
    _, lt_debt = _value_for_year_usd(facts_json, lt_debt_tags, year)

    total_debt = 0.0
    if np.isfinite(st_debt):
        total_debt += st_debt
    if np.isfinite(lt_debt):
        total_debt += lt_debt
    meta["total_debt"] = _to_musd(total_debt)

    net_cash = np.nan
    if np.isfinite(cash):
        net_cash = cash - total_debt
    meta["net_cash"] = _to_musd(net_cash) if np.isfinite(net_cash) else np.nan

    return net_cash, meta


def build_auto_dcf(facts_json: dict, ticker_symbol: str) -> tuple[dict, dict]:
    """Fully automated DCF valuation using historical data.

    Fixed assumptions:
        WACC = 10%, Terminal growth = 2.5%
    Derived from historical data (3-year averages):
        Revenue CAGR, CFO margin, Capex/Revenue ratio
    """
    WACC = 0.085
    TERMINAL_G = 0.035
    PROJECTION_YEARS = 10
    meta = {"wacc": WACC * 100, "terminal_g": TERMINAL_G * 100}

    # --- Gather historical data ---
    # Revenue
    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "NetOperatingRevenues", "SalesRevenueNet", "SalesRevenue",
    ]
    rev_df, _ = _build_composite_by_year_usd(facts_json, revenue_tags, min_duration=350)
    if rev_df.empty or len(rev_df) < 3:
        return {}, {"error": "Insufficient revenue data for DCF (need >= 3 years)"}
    rev_map = dict(zip(rev_df["year"].astype(int), rev_df["value"].astype(float)))

    # CFO
    cfo_tags = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]
    cfo_df, _ = _build_composite_by_year_usd(facts_json, cfo_tags, min_duration=350)
    cfo_map = dict(zip(cfo_df["year"].astype(int), cfo_df["value"].astype(float))) if not cfo_df.empty else {}

    # Capex
    capex_tags = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsToAcquireFixedAssets",
    ]
    capex_df, _ = _build_composite_by_year_usd(facts_json, capex_tags, min_duration=350)
    capex_map = dict(zip(capex_df["year"].astype(int), capex_df["value"].astype(float))) if not capex_df.empty else {}

    # --- Calculate 3-year averages ---
    sorted_years = sorted(rev_map.keys())
    latest_year = sorted_years[-1]
    last3 = [y for y in sorted_years if y >= latest_year - 2]

    # Revenue CAGR (3yr)
    y_start = last3[0]
    y_end = last3[-1]
    rev_start = rev_map.get(y_start, np.nan)
    rev_end = rev_map.get(y_end, np.nan)
    n_cagr = y_end - y_start
    if np.isfinite(rev_start) and np.isfinite(rev_end) and rev_start > 0 and rev_end > 0 and n_cagr > 0:
        rev_cagr = (rev_end / rev_start) ** (1.0 / n_cagr) - 1.0
    else:
        rev_cagr = 0.05
    meta["revenue_cagr_3yr"] = rev_cagr * 100

    # CFO margin (3yr avg)
    cfo_margins = []
    for y in last3:
        r = rev_map.get(y, np.nan)
        c = cfo_map.get(y, np.nan)
        if np.isfinite(r) and np.isfinite(c) and r > 0:
            cfo_margins.append(c / r)
    avg_cfo_margin = float(np.mean(cfo_margins)) if cfo_margins else 0.30
    meta["avg_cfo_margin"] = avg_cfo_margin * 100

    # Capex/Revenue ratio (3yr avg)
    capex_ratios = []
    for y in last3:
        r = rev_map.get(y, np.nan)
        cx = capex_map.get(y, np.nan)
        if np.isfinite(r) and np.isfinite(cx) and r > 0:
            capex_ratios.append(abs(cx) / r)
    avg_capex_ratio = float(np.mean(capex_ratios)) if capex_ratios else 0.10
    meta["avg_capex_ratio"] = avg_capex_ratio * 100

    # Base revenue (latest year)
    base_revenue = rev_end
    meta["base_revenue_m"] = _to_musd(base_revenue)
    meta["base_year"] = int(latest_year)

    # --- 10-year FCF projection ---
    # Revenue growth linearly decays from rev_cagr to terminal_g
    projection = []
    prev_revenue = base_revenue
    for i in range(1, PROJECTION_YEARS + 1):
        t = i / PROJECTION_YEARS
        growth = rev_cagr * (1 - t) + TERMINAL_G * t
        revenue = prev_revenue * (1 + growth)
        cfo = revenue * avg_cfo_margin
        capex = revenue * avg_capex_ratio
        fcf = cfo - capex
        projection.append({
            "Year": f"Y{i} ({int(latest_year) + i})",
            "Revenue(M$)": _to_musd(revenue),
            "CFO(M$)": _to_musd(cfo),
            "Capex(M$)": _to_musd(-capex),
            "FCF(M$)": _to_musd(fcf),
            "Growth(%)": growth * 100,
        })
        prev_revenue = revenue

    proj_df = pd.DataFrame(projection)

    # --- Discount FCFs ---
    pv_fcfs = 0.0
    for i, row in enumerate(projection):
        fcf_raw = row["FCF(M$)"] * 1_000_000
        pv = fcf_raw / ((1 + WACC) ** (i + 1))
        pv_fcfs += pv

    # Terminal value
    last_fcf = projection[-1]["FCF(M$)"] * 1_000_000
    if WACC > TERMINAL_G and last_fcf > 0:
        terminal_value = last_fcf * (1 + TERMINAL_G) / (WACC - TERMINAL_G)
    else:
        terminal_value = 0.0
    pv_terminal = terminal_value / ((1 + WACC) ** PROJECTION_YEARS)

    ev = pv_fcfs + pv_terminal

    # --- Net cash & shares ---
    bs_year = _latest_year_from_assets(facts_json)
    if bs_year is None:
        bs_year = int(latest_year)
    net_cash_raw, nc_meta = _get_net_cash(facts_json, bs_year)
    meta["net_cash_detail"] = nc_meta

    if not np.isfinite(net_cash_raw):
        net_cash_raw = 0.0

    equity_value = ev + net_cash_raw

    shares_raw, shares_tag = _get_shares_outstanding(facts_json)
    meta["shares_tag"] = shares_tag
    if not np.isfinite(shares_raw) or shares_raw <= 0:
        return {}, {"error": "Could not retrieve shares outstanding"}
    meta["shares_outstanding"] = shares_raw

    fair_price = equity_value / shares_raw
    meta["fair_price"] = fair_price

    # Current price and PER
    current_price = np.nan
    current_eps = np.nan
    fair_per = np.nan
    current_per = np.nan
    try:
        tk = yf.Ticker(ticker_symbol.upper())
        hist = tk.history(period="5d", auto_adjust=True)
        if not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
    except Exception:
        pass

    # Get latest EPS
    eps_table, eps_meta = build_eps_table(facts_json, ticker_symbol=ticker_symbol)
    if not eps_table.empty:
        current_eps = float(eps_table["EPS"].iloc[-1])

    if np.isfinite(current_eps) and current_eps > 0:
        fair_per = fair_price / current_eps
        if np.isfinite(current_price):
            current_per = current_price / current_eps

    # --- Build result ---
    result = {
        "projection": proj_df,
        "pv_fcfs_m": _to_musd(pv_fcfs),
        "terminal_value_m": _to_musd(terminal_value),
        "pv_terminal_m": _to_musd(pv_terminal),
        "enterprise_value_m": _to_musd(ev),
        "net_cash_m": _to_musd(net_cash_raw),
        "equity_value_m": _to_musd(equity_value),
        "shares": shares_raw,
        "fair_price": fair_price,
        "current_price": current_price,
        "upside_pct": ((fair_price / current_price) - 1) * 100 if np.isfinite(current_price) and current_price > 0 else np.nan,
        "current_eps": current_eps,
        "fair_per": fair_per,
        "current_per": current_per,
    }
    return result, meta


def plot_dcf_fcf_projection(proj_df: pd.DataFrame, title: str):
    """Chart: projected FCF bars with revenue line."""
    df = proj_df.copy()
    x = df["Year"].tolist()
    fcf = df["FCF(M$)"].astype(float).to_numpy()
    revenue = df["Revenue(M$)"].astype(float).to_numpy()

    fig, ax_left = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax_left, title=title, dual_y=True)

    bar_colors = [C_GOLD if v >= 0 else C_BRONZE for v in fcf]
    ax_left.bar(x, fcf, color=bar_colors, alpha=0.85, width=0.6, label="FCF")
    ax_left.set_ylabel("FCF (Million USD)", color=TEXT_COLOR)
    ax_left.tick_params(axis='x', rotation=45)

    ax_right = ax_left.twinx()
    ax_right.plot(x, revenue, color=C_SILVER, linewidth=2.0, marker="o", markersize=5, label="Revenue")
    ax_right.set_ylabel("Revenue (Million USD)", color=TEXT_COLOR)
    ax_right.tick_params(colors=TICK_COLOR, which='both')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['right'].set_color(GRID_COLOR)
    ax_right.spines['bottom'].set_visible(False)

    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, facecolor=HNWI_AX_BG, labelcolor=TEXT_COLOR, loc="upper left", frameon=False)

    st.pyplot(fig)


def plot_dcf_waterfall(result: dict, title: str):
    """Horizontal bar showing EV breakdown: PV of FCFs + PV of Terminal + Net Cash."""
    pv_fcfs = result["pv_fcfs_m"]
    pv_term = result["pv_terminal_m"]
    net_cash = result["net_cash_m"]
    equity = result["equity_value_m"]

    labels = ["PV of FCFs", "PV of Terminal", "Net Cash", "Equity Value"]
    values = [pv_fcfs, pv_term, net_cash, equity]
    colors = [C_GOLD, C_SILVER, C_BLUE, C_BRONZE]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(HNWI_BG)
    style_hnwi_ax(ax, title=title)

    ax.barh(labels, values, color=colors, alpha=0.85, height=0.5)
    ax.set_xlabel("Million USD", color=TEXT_COLOR)

    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:,.0f}", va='center', color=TEXT_COLOR, fontsize=9)

    ax.invert_yaxis()
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
        ticker = st.text_input("Ticker (US Stock)", value="")
        n_years = st.slider("Period (Years)", min_value=3, max_value=15, value=15)
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

    tab_pl, tab_bs, tab_cf, tab_seg, tab_rpo, tab_turn, tab_eps, tab_qtly, tab_dcf = st.tabs(
        ["PL (Annual)", "BS (Latest)", "CF (Annual)", "Segments", "Backlog", "Ratios", "EPS", "Quarterly", "DCF"]
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
        st.caption("Revenue Segments (from XBRL filings)")
        with st.spinner("Fetching & parsing XBRL filings for segment data..."):
            seg_df, prod_df, geo_df, seg_meta = build_segment_table(cik10)

        # --- 1. Official Segments (StatementBusinessSegmentsAxis) ---
        st.markdown("### Reporting Segments")
        if not seg_df.empty:
            seg_disp = seg_df[seg_df.index >= int(seg_df.index.max()) - n_years + 1]
            plot_stacked_bar(seg_disp, "Revenue by Reporting Segment")
            st.dataframe(seg_disp.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("No official reporting segment data found (StatementBusinessSegmentsAxis).")

        # --- 2. Product/Service Breakdown ---
        st.markdown("### Product / Service Breakdown")
        if not prod_df.empty:
            prod_disp = prod_df[prod_df.index >= int(prod_df.index.max()) - n_years + 1]
            plot_stacked_bar(prod_disp, "Revenue by Product / Service")
            st.dataframe(prod_disp.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("No product/service breakdown found (ProductOrServiceAxis).")

        # --- 3. Geography ---
        st.markdown("### Geography")
        if not geo_df.empty:
            geo_disp = geo_df[geo_df.index >= int(geo_df.index.max()) - n_years + 1]
            plot_stacked_bar(geo_disp, "Revenue by Geography")
            st.dataframe(geo_disp.style.format("{:,.0f}"), use_container_width=True)
        else:
            st.info("No geography segment data found.")

        # Debug info
        with st.expander("🔍 Segment Debug Info"):
            st.write(f"**10-K filings found:** {seg_meta.get('filings_found', 0)}")
            st.write(f"**XBRL instances fetched:** {seg_meta.get('filings_fetched', 0)}")
            st.write(f"**Segment revenue facts extracted:** {seg_meta.get('total_facts', 0)}")
            if "unique_dimensions" in seg_meta:
                st.write(f"**Dimensions used:** {seg_meta['unique_dimensions']}")
            if "segment_members" in seg_meta:
                st.write(f"**Reporting segments:** {seg_meta['segment_members']}")
            if "product_members" in seg_meta:
                st.write(f"**Product/Service:** {seg_meta['product_members']}")
            if "geo_members" in seg_meta:
                st.write(f"**Geography:** {seg_meta['geo_members']}")
            if "diag" in seg_meta:
                st.write("---")
                st.write("**XBRL Diagnostic (Filing 1):**")
                diag = seg_meta["diag"]
                st.write(f"- Total contexts: {diag.get('total_contexts', 0)}")
                st.write(f"- Dimensional contexts: {diag.get('dim_ctx_count', 0)}")
                st.write(f"- Unique dimensions: {diag.get('unique_dimensions', [])}")
                st.write(f"- Revenue tags with contextRef: {diag.get('revenue_tags_with_contextref', [])}")
                st.write(f"- Revenue facts in dim ctx (XBRL): {diag.get('revenue_facts_in_dim_ctx', [])}")
                st.write(f"- Revenue facts in dim ctx (iXBRL): {diag.get('ix_revenue_in_dim_ctx', [])}")
            if "fetch_debug" in seg_meta:
                st.write("---")
                st.write("**Fetch log (per filing):**")
                for i, dbg in enumerate(seg_meta["fetch_debug"]):
                    st.code(f"[Filing {i+1}]\n{dbg}")

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

        # --- ROA Summary ---
        st.markdown("### ROA")
        plot_roa(rat_disp, f"ROA (Return on Assets)")
        roa_vals = rat_disp["ROA(%)"].dropna()
        if len(roa_vals) >= 2:
            avg_roa = roa_vals.mean()
            first_roa = roa_vals.iloc[0]
            last_roa = roa_vals.iloc[-1]
            n_roa = len(roa_vals) - 1
            if first_roa > 0 and last_roa > 0 and n_roa > 0:
                cagr_roa = ((last_roa / first_roa) ** (1.0 / n_roa) - 1.0) * 100.0
            else:
                cagr_roa = np.nan
            col_a, col_b = st.columns(2)
            col_a.metric("Average ROA", f"{avg_roa:.2f}%")
            if np.isfinite(cagr_roa):
                col_b.metric("CAGR (ROA)", f"{cagr_roa:+.2f}% / yr")

        # --- ROE Summary ---
        st.markdown("### ROE")
        plot_roe(rat_disp, f"ROE (Return on Equity)")
        roe_vals = rat_disp["ROE(%)"].dropna()
        if len(roe_vals) >= 2:
            avg_roe = roe_vals.mean()
            first_roe = roe_vals.iloc[0]
            last_roe = roe_vals.iloc[-1]
            n_roe = len(roe_vals) - 1
            if first_roe > 0 and last_roe > 0 and n_roe > 0:
                cagr_roe = ((last_roe / first_roe) ** (1.0 / n_roe) - 1.0) * 100.0
            else:
                cagr_roe = np.nan
            col_c, col_d = st.columns(2)
            col_c.metric("Average ROE", f"{avg_roe:.2f}%")
            if np.isfinite(cagr_roe):
                col_d.metric("CAGR (ROE)", f"{cagr_roe:+.2f}% / yr")

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

            st.markdown("### EPS")
            plot_eps(eps_disp, f"EPS (Diluted)", unit_label=unit_label)

            eps_vals = eps_disp["EPS"].dropna()
            if len(eps_vals) >= 2:
                avg_eps = eps_vals.mean()
                first_eps = eps_vals.iloc[0]
                last_eps = eps_vals.iloc[-1]
                n_eps = len(eps_vals) - 1
                if first_eps > 0 and last_eps > 0 and n_eps > 0:
                    cagr_eps = ((last_eps / first_eps) ** (1.0 / n_eps) - 1.0) * 100.0
                else:
                    cagr_eps = np.nan
                col_e, col_f = st.columns(2)
                col_e.metric("Average EPS", f"{avg_eps:.2f}")
                if np.isfinite(cagr_eps):
                    col_f.metric("CAGR (EPS)", f"{cagr_eps:+.2f}% / yr")

            # --- PER ---
            st.markdown("### PER (Trailing)")
            per_table, per_meta = build_per_table(eps_disp, ticker_symbol=t)
            if per_table.empty or per_table["PER"].dropna().empty:
                st.info("PER data not available (stock price could not be retrieved).")
            else:
                per_disp = per_table.dropna(subset=["PER"])
                plot_per(per_disp, f"PER Trend (FY-End Price / EPS)")

                per_vals = per_disp["PER"].dropna()
                if len(per_vals) >= 1:
                    avg_per = per_vals.mean()
                    med_per = per_vals.median()
                    latest_per = per_vals.iloc[-1]
                    cagr_per = np.nan
                    if len(per_vals) >= 2:
                        first_per = per_vals.iloc[0]
                        n_per = len(per_vals) - 1
                        if first_per > 0 and latest_per > 0 and n_per > 0:
                            cagr_per = ((latest_per / first_per) ** (1.0 / n_per) - 1.0) * 100.0
                    col_g, col_h, col_i, col_j = st.columns(4)
                    col_g.metric("Average PER", f"{avg_per:.1f}x")
                    col_h.metric("Median PER", f"{med_per:.1f}x")
                    col_i.metric("Latest PER", f"{latest_per:.1f}x")
                    if np.isfinite(cagr_per):
                        col_j.metric("CAGR (PER)", f"{cagr_per:+.1f}% / yr")

                st.dataframe(
                    per_disp[["FY", "Price", "EPS", "PER"]].style.format(
                        {"Price": "{:,.2f}", "EPS": "{:.2f}", "PER": "{:.1f}"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            # --- EPS Data Table ---
            st.markdown("### EPS Data")
            eps_display_cols = eps_disp[["FY", "EPS"]].copy()
            st.dataframe(eps_display_cols.style.format({"EPS": "{:.2f}"}), use_container_width=True, hide_index=True)

    with tab_qtly:
        n_q = st.slider("Quarters to show", min_value=4, max_value=20, value=8, key="q_slider")

        st.markdown("### Quarterly Income Statement")
        q_pl, q_pl_meta = build_quarterly_pl_table(facts, n_quarters=n_q)
        if q_pl.empty:
            st.warning("Quarterly PL data not available.")
        else:
            st.caption(f"Showing {len(q_pl)} quarters")

            plot_quarterly_bars(q_pl, "Revenue(M$)", "Quarterly Revenue")

            if q_pl["OpIncome(M$)"].notna().any():
                plot_quarterly_bars_with_margin(q_pl, "OpIncome(M$)", "Revenue(M$)", "Quarterly Operating Income & Margin", "Op Margin (%)")

            if q_pl["PreTaxIncome(M$)"].notna().any():
                plot_quarterly_bars_with_margin(q_pl, "PreTaxIncome(M$)", "Revenue(M$)", "Quarterly Pre-Tax Income & Margin", "Pre-Tax Margin (%)")

            st.dataframe(
                q_pl.style.format({
                    "Revenue(M$)": "{:,.0f}",
                    "OpIncome(M$)": "{:,.0f}",
                    "PreTaxIncome(M$)": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Quarterly EPS (Diluted)")
        q_eps, q_eps_meta = build_quarterly_eps_table(facts, ticker_symbol=t, n_quarters=n_q)
        if q_eps.empty:
            st.warning("Quarterly EPS data not available.")
        else:
            plot_quarterly_eps_chart(q_eps, "Quarterly EPS (Split-Adjusted)")
            st.dataframe(
                q_eps.style.format({"EPS": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

    with tab_dcf:
        st.markdown("### Auto DCF Valuation")
        st.caption("⚠️ Simplified estimate — WACC 8.5%, Terminal Growth 3.5%, historical averages projected. Not investment advice.")

        dcf_result, dcf_meta = build_auto_dcf(facts, ticker_symbol=t)

        if not dcf_result:
            st.warning(f"DCF calculation failed: {dcf_meta.get('error', 'Unknown error')}")
        else:
            # --- Assumptions ---
            st.markdown("#### Assumptions (Auto-derived)")
            c1, c2, c3 = st.columns(3)
            c1.metric("WACC", f"{dcf_meta.get('wacc', 0):.1f}%")
            c2.metric("Terminal Growth", f"{dcf_meta.get('terminal_g', 0):.1f}%")
            c3.metric("Base Year", str(dcf_meta.get("base_year", "N/A")))

            c4, c5, c6 = st.columns(3)
            c4.metric("Revenue CAGR (3yr)", f"{dcf_meta.get('revenue_cagr_3yr', 0):.1f}%")
            c5.metric("Avg CFO Margin (3yr)", f"{dcf_meta.get('avg_cfo_margin', 0):.1f}%")
            c6.metric("Avg Capex/Rev (3yr)", f"{dcf_meta.get('avg_capex_ratio', 0):.1f}%")

            st.markdown("---")

            # --- Key Results ---
            st.markdown("#### Valuation Results")
            r1, r2, r3 = st.columns(3)
            fair_p = dcf_result.get("fair_price", np.nan)
            cur_p = dcf_result.get("current_price", np.nan)
            upside = dcf_result.get("upside_pct", np.nan)
            r1.metric("Fair Value / share", f"${fair_p:,.1f}" if np.isfinite(fair_p) else "N/A")
            r2.metric("Current Price", f"${cur_p:,.1f}" if np.isfinite(cur_p) else "N/A")
            r3.metric(
                "Upside / Downside",
                f"{upside:+.1f}%" if np.isfinite(upside) else "N/A",
                delta=f"{upside:+.1f}%" if np.isfinite(upside) else None,
                delta_color="normal",
            )

            r4, r5, r6 = st.columns(3)
            fair_per = dcf_result.get("fair_per", np.nan)
            cur_per = dcf_result.get("current_per", np.nan)
            cur_eps = dcf_result.get("current_eps", np.nan)
            r4.metric("DCF Fair PER", f"{fair_per:.1f}x" if np.isfinite(fair_per) else "N/A")
            r5.metric("Current PER", f"{cur_per:.1f}x" if np.isfinite(cur_per) else "N/A")
            r6.metric("Latest EPS", f"${cur_eps:.2f}" if np.isfinite(cur_eps) else "N/A")

            st.markdown("---")

            # --- EV Breakdown ---
            st.markdown("#### Enterprise Value Breakdown")
            ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
            ev_c1.metric("PV of FCFs", f"{dcf_result['pv_fcfs_m']:,.0f} M$")
            ev_c2.metric("PV of Terminal", f"{dcf_result['pv_terminal_m']:,.0f} M$")
            ev_c3.metric("Net Cash", f"{dcf_result['net_cash_m']:,.0f} M$")
            ev_c4.metric("Equity Value", f"{dcf_result['equity_value_m']:,.0f} M$")

            # Terminal as % of EV
            ev_total = dcf_result.get("enterprise_value_m", 0)
            if ev_total > 0:
                tv_pct = dcf_result["pv_terminal_m"] / ev_total * 100
                st.caption(f"Terminal Value accounts for {tv_pct:.0f}% of Enterprise Value")

            plot_dcf_waterfall(dcf_result, "EV Breakdown (Million USD)")

            st.markdown("---")

            # --- FCF Projection Chart & Table ---
            st.markdown("#### 10-Year FCF Projection")
            proj_df = dcf_result["projection"]
            plot_dcf_fcf_projection(proj_df, "Projected FCF & Revenue")

            st.dataframe(
                proj_df.style.format({
                    "Revenue(M$)": "{:,.0f}",
                    "CFO(M$)": "{:,.0f}",
                    "Capex(M$)": "{:,.0f}",
                    "FCF(M$)": "{:,.0f}",
                    "Growth(%)": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


def main():
    authed_email = require_admin_token()
    render(authed_email)


if __name__ == "__main__":
    main()
