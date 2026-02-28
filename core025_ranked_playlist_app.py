
import streamlit as st
import pandas as pd
import unicodedata
import gzip, base64, io
import datetime
import difflib
import zipfile
import json

# Embedded rule files (gz+base64). This eliminates Streamlit Cloud path issues.
_EMBED_WEIGHTS_B64_GZ = """H4sIAHTHoWkC/42Q0W5bIQyG7/csaLINNnDRt9g9ojkkQTtLogNJXr/mJK3SdtEGkkE2/r/fbI5LMduS+1nP48lc8nwu5lQ3v81xntK11N2+m0Pa197SLnetpVFNu3oph9SXXLt5za3M9VDWipnK3HO6tPSeNodyfVdae1tfzEbB4/KD2LRSppRf21S3W/PyYpzRJI0AP1k0WKboEL04DxwCaYrgdr2Xgr4HGB3EH5L73NL1uLR+ynVJo6rivxYdkHgALA4AClIES8QBxA2mJfYRAzOBFYxRU6gvrEevUKdrdMMQecS101LyNBjhPsAwyu7z/vs4GInRWgkx2MhejX0fp53/pDm3PtWdfrpirBJWzhC9GUcSiRJuE6zGffDWgkXxjiCswvQvYRz+NVgVcD6QQJDIwoT+mX9ii1GiheiE4v/5hztmCHz5JX6CARRw5AN4F0ToEfMGUwA9YMsCAAA="""
_EMBED_TIEPACK_B64_GZ = """H4sIAHTHoWkC/3XNwQrCMBAE0LvfModmJcd+S4jJpi5WGrKJ/r5JBUGxpxmWx2zYCiOxr63nlvHwa2NkCTc8WZZr3bvTWhA6HeVEFsocnb9olJQwzzCgfjWYRkxk/xkC2bex30ZzYR8HsUcz2u5u9VqjLFKHPB/Kz5j5/fcC5Awqw+wAAAA="""

def _read_embedded_csv(b64_gz_text: str) -> pd.DataFrame:
    raw = gzip.decompress(base64.b64decode(b64_gz_text.encode("ascii")))
    return pd.read_csv(io.BytesIO(raw))

import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional



def _canon_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept Lottery Post exports that may use different header names.
    We try to map to required canonical columns:
      - Draw Date
      - State
      - Game
      - Results
    """
    colmap = {_canon_col(c): c for c in df.columns}

    # candidates for each required field (canonicalized)
    candidates = {
        "Draw Date": [
            "drawdate", "drawdateutc", "date", "drawdatedate", "drawingdate", "drawndate", "draw_date"
        ],
        "State": [
            "state", "jurisdiction", "province", "region", "location"
        ],
        "Game": [
            "game", "gamename", "lotterygame", "draw", "drawing", "game_name"
        ],
        "Results": [
            "results", "result", "winningnumbers", "winningnumber", "numbers", "winnumbers", "winning", "win"
        ],
    }

    resolved = {}
    for need, opts in candidates.items():
        found = None
        for o in opts:
            if o in colmap:
                found = colmap[o]
                break
        if found:
            resolved[need] = found

    # If already correct, nothing to do
    if all(k in df.columns for k in ["Draw Date", "State", "Game", "Results"]):
        return df

    # Attempt partial mapping, and keep original cols
    df2 = df.copy()
    for need, src in resolved.items():
        if need not in df2.columns and src in df2.columns:
            df2[need] = df2[src]

    return df2

# --- Stream key normalization (prevents "string mismatch" joins) ---
def canon_stream(s: str) -> str:
    """Canonicalize Stream labels like "State | Game" to a stable ASCII key."""
    if s is None:
        return ""
    s = str(s)
    # Normalize unicode (accents, smart quotes, etc.)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    s = s.replace('–', '-').replace('—', '-')
    # Normalize separators and whitespace
    s = s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    # Normalize the pipe separator if present
    if '|' in s:
        parts = [re.sub(r'\s+', ' ', p).strip() for p in s.split('|')]
        if len(parts) >= 2:
            left = parts[0].lower()
            right = ' | '.join(parts[1:]).lower()
            s = f"{left}|{right}"
        else:
            s = s.lower()
    else:
        s = s.lower()
    return s


# ------------------------------
# Walk-forward play-method helpers (50 streams/day by winner-heavy percentile bins)
# ------------------------------

def _wf_pct_bin_label(pct_0_1: float, bin_size_pct: int = 5) -> str:
    """Return a stable percentile-bin label like '00-05%' for pct in [0,1]."""
    try:
        pct = float(pct_0_1)
    except Exception:
        pct = 0.0
    pct = 0.0 if pct != pct else max(0.0, min(1.0, pct))  # NaN-safe clamp
    bs = int(bin_size_pct) if int(bin_size_pct) > 0 else 5
    lo = int((pct * 100.0) // bs) * bs
    hi = min(100, lo + bs)
    return f"{lo:02d}-{hi:02d}%"


def _wf_suggest_bins_greedy(per_event: pd.DataFrame,
                            per_stream: pd.DataFrame,
                            cap_streams: int = 50,
                            bin_size_pct: int = 5) -> list[str]:
    """Greedy pick bins by winner frequency until average daily candidate streams reaches ~cap."""
    if per_event is None or per_event.empty or per_stream is None or per_stream.empty:
        return []

    pe = per_event.copy()
    ps = per_stream.copy()

    if "Percentile" not in pe.columns or "Percentile" not in ps.columns:
        return []

    pe["PctBin"] = pe["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    ps["PctBin"] = ps["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))

    pe_in = pe[pe.get("InUniverse", 0).astype(int) == 1].copy()
    if pe_in.empty:
        return []

    win_counts = pe_in["PctBin"].value_counts().to_dict()

    # Candidate count per day per bin
    if "PlayDate" not in ps.columns:
        return []
    daily_bin_counts = ps.groupby(["PlayDate", "PctBin"]).size().reset_index(name="n")
    avg_bin = daily_bin_counts.groupby("PctBin")["n"].mean().to_dict()

    bins_sorted = sorted(win_counts.keys(), key=lambda b: (-win_counts.get(b, 0), b))
    selected = []
    cur_avg = 0.0
    target = float(max(1, int(cap_streams)))

    for b in bins_sorted:
        selected.append(b)
        cur_avg = sum(avg_bin.get(x, 0.0) for x in selected)
        if cur_avg >= target:
            break

    return selected



def _wf_select_cap50_by_bins(per_stream: pd.DataFrame,
                             selected_bins: list[str],
                             cap_streams: int = 50,
                             bin_size_pct: int = 5,
                             fill_to_cap: bool = True) -> pd.DataFrame:
    """Mark Selected50=1 for each day using selected percentile bins, then trim/fill to exact cap.

    Tie-breakers are deterministic and aligned to the ranked playlist:
      - Prefer lower Rank (Rank=1 is best)
      - Then stable StreamKey
    """
    if per_stream is None or per_stream.empty:
        return per_stream.copy() if per_stream is not None else pd.DataFrame()

    df = per_stream.copy()
    if "PlayDate" not in df.columns:
        return df

    # Bin label for each stream/day row
    if "Percentile" in df.columns:
        df["PctBin"] = df["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    else:
        df["PctBin"] = ""

    df["Selected50"] = 0

    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers (Rank ascending is best)
    sort_by = []
    sort_asc = []
    if "Rank" in df.columns:
        sort_by.append("Rank")
        sort_asc.append(True)
    elif "StreamScore" in df.columns:
        sort_by.append("StreamScore")
        sort_asc.append(False)

    if "StreamKey" in df.columns:
        sort_by.append("StreamKey")
        sort_asc.append(True)
    else:
        sort_by.append(df.columns[0])
        sort_asc.append(True)

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        cap_d = min(cap, len(g))

        cand = g[g["PctBin"].isin(selected_bins)] if selected_bins else g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_by, ascending=sort_asc).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_by, ascending=sort_asc).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df

    df["PctBin"] = df["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    df["Selected50"] = 0

    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers
    sort_cols = []
    if "StreamScore" in df.columns:
        sort_cols.append("StreamScore")
    if "Rank" in df.columns:
        sort_cols.append("Rank")
    sort_cols.append("StreamKey" if "StreamKey" in df.columns else df.columns[0])

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        cap_d = min(cap, len(g))

        cand = g[g["PctBin"].isin(selected_bins)] if selected_bins else g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_cols, ascending=[False]*len(sort_cols[:-1]) + [True]).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_cols, ascending=[False]*len(sort_cols[:-1]) + [True]).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df



# ------------------------------
# Walk-forward play-method helpers (50 streams/day by historically best winner-producing rank rows)
# ------------------------------

def _wf_suggest_rank_rows_topk(per_event: pd.DataFrame,
                               cap_streams: int = 50) -> list[int]:
    """Pick the top-K rank rows (non-contiguous) with the most in-universe winners."""
    if per_event is None or per_event.empty:
        return []

    pe = per_event.copy()
    if "Rank" not in pe.columns:
        return []

    # Use only in-universe events (where the winning stream was in that day's ranked universe)
    try:
        pe_in = pe[pe.get("InUniverse", 0).astype(int) == 1].copy()
    except Exception:
        pe_in = pe[pe.get("InUniverse", 0) == 1].copy()

    pe_in["Rank"] = pd.to_numeric(pe_in["Rank"], errors="coerce")
    pe_in = pe_in.dropna(subset=["Rank"])
    if pe_in.empty:
        return []

    counts = pe_in["Rank"].astype(int).value_counts()
    # sort by wins desc, then rank asc
    ordered = sorted(counts.index.tolist(), key=lambda r: (-int(counts.get(r, 0)), int(r)))
    k = int(cap_streams) if int(cap_streams) > 0 else 50
    selected = ordered[:k]
    return [int(x) for x in selected]


def _wf_select_cap50_by_ranks(per_stream: pd.DataFrame,
                              selected_ranks: list[int],
                              cap_streams: int = 50,
                              fill_to_cap: bool = True) -> pd.DataFrame:
    """Mark Selected50=1 for each day using selected Rank rows, then trim/fill to exact cap."""
    if per_stream is None or per_stream.empty:
        return per_stream.copy() if per_stream is not None else pd.DataFrame()

    df = per_stream.copy()
    if "PlayDate" not in df.columns:
        return df

    df["Selected50"] = 0
    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers (Rank ascending is best)
    sort_by = []
    sort_asc = []
    if "Rank" in df.columns:
        sort_by.append("Rank")
        sort_asc.append(True)
    elif "StreamScore" in df.columns:
        sort_by.append("StreamScore")
        sort_asc.append(False)

    if "StreamKey" in df.columns:
        sort_by.append("StreamKey")
        sort_asc.append(True)
    else:
        sort_by.append(df.columns[0])
        sort_asc.append(True)

    # Normalize selected_ranks
    sel = set(int(x) for x in selected_ranks) if selected_ranks else set()

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        cap_d = min(cap, len(g))

        if sel and "Rank" in g.columns:
            cand = g[pd.to_numeric(g["Rank"], errors="coerce").fillna(-1).astype(int).isin(sel)]
        else:
            cand = g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_by, ascending=sort_asc).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_by, ascending=sort_asc).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df


def _wf_augment_event_date_with_play50(per_event: pd.DataFrame,
                                      per_date: pd.DataFrame,
                                      per_stream_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Selected50 + CapturedMember fields to per_event, and daily selection/capture stats to per_date."""
    pe = per_event.copy() if per_event is not None else pd.DataFrame()
    pdx = per_date.copy() if per_date is not None else pd.DataFrame()

    if pe.empty or per_stream_selected is None or per_stream_selected.empty:
        return pe, pdx

    # Merge Selected50 and stream-level play members back onto per_event
    key_cols = ["PlayDate", "StreamKey"] if "StreamKey" in pe.columns else ["PlayDate", "WinningStreamKey"]
    stream_key_col = "StreamKey" if "StreamKey" in per_stream_selected.columns else "WinningStreamKey"

    ps = per_stream_selected.copy()
    if "StreamKey" not in ps.columns and "WinningStreamKey" in ps.columns:
        ps = ps.rename(columns={"WinningStreamKey": "StreamKey"})
    if "StreamKey" in ps.columns:
        ps_small = ps[["PlayDate", "StreamKey", "Selected50", "PlayMembers", "PlayMember", "Top1", "Top2", "FlipRec"]].copy()
    else:
        ps_small = ps[["PlayDate", stream_key_col, "Selected50"]].copy()

    # Normalize per_event stream key column
    if "WinningStreamKey" in pe.columns and "StreamKey" not in pe.columns:
        pe = pe.rename(columns={"WinningStreamKey": "StreamKey"})
    if "StreamKey" not in pe.columns:
        return pe, pdx

    pe = pe.merge(ps_small, how="left", on=["PlayDate", "StreamKey"], suffixes=("", "_stream"))

    pe["Selected50"] = pe["Selected50"].fillna(0).astype(int)

    def _member_set(row):
        s = row.get("PlayMembers", None)
        if isinstance(s, str) and s.strip():
            parts = [p.strip() for p in s.split("+") if p.strip()]
            return set(parts)
        # fallback to PlayMember (Top1)
        pm = row.get("PlayMember", None)
        if isinstance(pm, str) and pm.strip():
            return {pm.strip()}
        t1 = row.get("Top1", None)
        if isinstance(t1, str) and t1.strip():
            return {t1.strip()}
        return set()

    pe["_PlaySet"] = pe.apply(_member_set, axis=1)
    pe["CapturedMember"] = 0
    if "WinningMember" in pe.columns:
        pe["CapturedMember"] = pe.apply(lambda r: 1 if (int(r.get("InUniverse", 0)) == 1 and int(r.get("Selected50", 0)) == 1 and r.get("WinningMember") in r.get("_PlaySet", set())) else 0, axis=1)

    pe["CapturedBy"] = ""
    if "WinningMember" in pe.columns:
        def _captured_by(r):
            if int(r.get("CapturedMember", 0)) != 1:
                return ""
            if r.get("WinningMember") == r.get("Top1"):
                return "TOP1"
            if r.get("WinningMember") == r.get("Top2"):
                return "TOP2"
            return "PLAYSET"
        pe["CapturedBy"] = pe.apply(_captured_by, axis=1)

    pe.drop(columns=["_PlaySet"], inplace=True, errors="ignore")

    # Daily selection stats
    if not pdx.empty and "PlayDate" in pdx.columns and "PlayDate" in ps.columns:
        sel_counts = ps.groupby("PlayDate")["Selected50"].sum().reset_index(name="SelectedStreams")
        pdx = pdx.merge(sel_counts, how="left", on="PlayDate")
        pdx["SelectedStreams"] = pdx["SelectedStreams"].fillna(0).astype(int)

        # Daily capture stats (for hit-events)
        if "HitEvents" in pdx.columns:
            day_hits = pe.groupby("PlayDate").agg(
                HitEvents=("PlayDate", "size"),
                HitStreams_Selected50=("Selected50", "sum"),
                HitMembers_Captured=("CapturedMember", "sum"),
            ).reset_index()
            pdx = pdx.drop(columns=["HitEvents"], errors="ignore").merge(day_hits, how="left", on="PlayDate")
            pdx["HitEvents"] = pdx["HitEvents"].fillna(0).astype(int)
            pdx["HitStreams_Selected50"] = pdx["HitStreams_Selected50"].fillna(0).astype(int)
            pdx["HitMembers_Captured"] = pdx["HitMembers_Captured"].fillna(0).astype(int)

    return pe, pdx

st.set_page_config(page_title="Core 025 — Live Ranked Streams + Lab Backtest", layout="wide")

# ----------------------------
# Robust parsing for Lottery Post "Results"
# ----------------------------
_DIG4_RE = re.compile(r"(\d)[^\d]+(\d)[^\d]+(\d)[^\d]+(\d)")

def extract_4digits(val) -> Optional[str]:
    """Extract a clean 4-digit result string (preserving leading zeros).

    Strict on purpose: we do NOT "borrow" digits from Fireball/Wildball/Bonus/etc.
    If we can't confidently parse a 4-digit result, return None and exclude the row.
    """
    if val is None or pd.isna(val):
        return None

    s = str(val).strip()
    if not s:
        return None

    # Strip anything after comma and after common add-on keywords to avoid capturing extra digits.
    s0 = s.split(",", 1)[0].strip()
    s0 = re.split(r"\b(?:wild\s*ball|fireball|bonus|multiplier)\b", s0, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    m = _DIG4_RE.search(s0)
    if m:
        out = "".join(m.groups())
        return out if len(out) == 4 else None

    # Accept a standalone 4-digit chunk if present.
    m2 = re.search(r"\b(\d{4})\b", s0)
    if m2:
        return m2.group(1)

    return None

def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    # Try to map variant headers into canonical names
    df = _map_columns(df)

    # Required columns (Lottery Post export style)
    needed = ["Draw Date", "State", "Game", "Results"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # Helpful debug: show available columns
        raise ValueError(
            f"History file missing columns: {missing}. Need: {needed}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out["Draw Date"] = pd.to_datetime(out["Draw Date"], errors="coerce")
    out["result_raw"] = out["Results"].astype(str)
    out["result"] = out["result_raw"].apply(extract_4digits)

    # diagnostics
    parse_fail = int(out["result"].isna().sum())
    out.attrs["parse_fail_count"] = parse_fail
    if parse_fail:
        bad = out.loc[out["result"].isna(), ["Draw Date","State","Game","Results"]].head(10)
        out.attrs["parse_fail_examples"] = bad.to_dict(orient="records")

    # drop bad rows
    out = out.dropna(subset=["Draw Date", "State", "Game", "result"]).copy()

    out["stream"] = out["State"].astype(str).str.strip() + " | " + out["Game"].astype(str).str.strip()
    out = out.sort_values(["stream", "Draw Date"]).reset_index(drop=True)

    # within-stream order index
    out["stream_idx"] = out.groupby("stream").cumcount()
    return out

# ----------------------------
# Core 025 helpers

# ----------------------------
TARGETS = {"0025": "0025", "0225": "0225", "0255": "0255"}
TARGET_SET = set(TARGETS.keys())

def as_member(result4: str) -> Optional[str]:
    """Map a 4-digit result to its 025-family member if it is a permutation of {0,0,2,5}, {0,2,2,5}, {0,2,5,5}."""
    # Count digits
    if not result4 or len(result4) != 4 or not result4.isdigit():
        return None
    from collections import Counter
    c = Counter(result4)
    # canonical patterns
    patt = tuple(sorted(c.items()))
    # 0025 => 0x2,2x1,5x1
    if c.get("0",0)==2 and c.get("2",0)==1 and c.get("5",0)==1:
        return "0025"
    # 0225 => 0x1,2x2,5x1
    if c.get("0",0)==1 and c.get("2",0)==2 and c.get("5",0)==1:
        return "0225"
    # 0255 => 0x1,2x1,5x2
    if c.get("0",0)==1 and c.get("2",0)==1 and c.get("5",0)==2:
        return "0255"
    return None

def seed_digits(seed4: str) -> List[int]:
    return [int(ch) for ch in seed4]

def seed_sum(seed4: str) -> int:
    return sum(seed_digits(seed4))

def seed_spread(seed4: str) -> int:
    d = seed_digits(seed4)
    return max(d) - min(d)

def seed_absdiff(seed4: str) -> int:
    d = seed_digits(seed4)
    # definition used in your weight table: abs( (d0+d1) - (d2+d3) )
    return abs((d[0]+d[1]) - (d[2]+d[3]))

WORST_PAIRS_025 = {("3","9"), ("5","5"), ("2","6"), ("2","9"), ("7","9")}
def seed_has_worstpair_025(seed4: str) -> bool:
    # unordered digit pairs from 4 digits (6 combos)
    digs = list(seed4)
    pairs = []
    for i in range(4):
        for j in range(i+1,4):
            a,b = digs[i], digs[j]
            pairs.append(tuple(sorted((a,b))))
    return any(p in WORST_PAIRS_025 for p in pairs)

def seed_sum_lastdigit(seed4: str) -> int:
    return seed_sum(seed4) % 10

# Feature compute
def compute_features(seed4: str) -> Dict[str, object]:
    return {
        "seed_sum": seed_sum(seed4),
        "seed_spread": seed_spread(seed4),
        "seed_absdiff": seed_absdiff(seed4),
        "seed_has_worstpair_025": seed_has_worstpair_025(seed4),
        "seed_sum_lastdigit": seed_sum_lastdigit(seed4),
        "seed_has_9": ("9" in seed4),
    }

# ----------------------------
# Rule engine
# ----------------------------
@dataclass
class Rule:
    feature: str
    op: str
    value: object
    pick: str
    weight: float
    source: str  # "base" or "tie"

    @classmethod
    def from_row(cls, r) -> "Rule":
        """
        Build a Rule from a pandas row.
        Supports both weight column names: 'weight' (tie-pack) or 'new_weight' (base weights).
        Normalizes pick to 4-digit member strings: 25->0025, 225->0225, 255->0255.
        Parses boolean and numeric 'value' where possible.
        """
        def _norm_pick(p):
            if p is None:
                return ""
            s = str(p).strip()
            # handle floats like 25.0
            if re.fullmatch(r"\d+(\.0+)?", s):
                s = str(int(float(s)))
            # normalize to 4 digits
            if s.isdigit() and len(s) <= 4:
                s = s.zfill(4)
            return s

        def _parse_value(v):
            if v is None:
                return v
            if isinstance(v, bool):
                return v
            s = str(v).strip()
            if s.lower() == "true":
                return True
            if s.lower() == "false":
                return False
            # numeric?
            if re.fullmatch(r"-?\d+", s):
                try:
                    return int(s)
                except Exception:
                    return s
            if re.fullmatch(r"-?\d+\.\d+", s):
                try:
                    return float(s)
                except Exception:
                    return s
            return v

        feature = str(r.get("feature", "")).strip()
        op = str(r.get("op", "==")).strip()
        value = _parse_value(r.get("value", None))
        pick = _norm_pick(r.get("pick", ""))

        if "weight" in r.index:
            w = r.get("weight", 0)
        elif "new_weight" in r.index:
            w = r.get("new_weight", 0)
        elif "old_weight" in r.index:
            w = r.get("old_weight", 0)
        else:
            w = 0

        try:
            weight = float(w)
        except Exception:
            weight = 0.0

        source = str(r.get("source", "")).strip() if "source" in r.index else ""
        if source == "":
            source = "tie" if ("weight" in r.index and "new_weight" not in r.index) else "base"

        return cls(feature=feature, op=op, value=value, pick=pick, weight=weight, source=source)

def op_match(actual, op: str, value) -> bool:
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "<=":
        return actual <= value
    if op == "<":
        return actual < value
    if op == ">=":
        return actual >= value
    if op == ">":
        return actual > value
    # fallback string compare
    return str(actual) == str(value)

def load_rules(weights_csv: str, tie_csv: str, max_rules: int, min_support: int,
               weights_up=None, tie_up=None) -> Tuple[List[Rule], List[Rule], Dict]:
    """
    Load rule-weights + tie-pack, with clear provenance so audits can confirm what file was used.

    Priority order (both files):
      1) CSV at provided path
      2) Uploaded file in the UI
      3) Embedded fallback bundled in the app
    """
    import hashlib

    def _sha256(b: bytes) -> str:
        try:
            return hashlib.sha256(b).hexdigest()
        except Exception:
            return ""

    def _read_csv_bytes_from_path(p: str) -> Optional[bytes]:
        if not p:
            return None
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read()
        except Exception:
            return None
        return None

    def _read_csv_bytes_from_upload(up) -> Optional[bytes]:
        if up is None:
            return None
        try:
            # Streamlit UploadedFile supports getvalue()
            return up.getvalue()
        except Exception:
            try:
                # Fallback: read()
                return up.read()
            except Exception:
                return None

    def _read_csv_bytes_from_embedded(b64_gz: str) -> bytes:
        return gzip.decompress(base64.b64decode(b64_gz.encode("ascii")))

    meta: Dict = {
        "weights": {"source": None, "rows": 0, "sha256": ""},
        "tie_pack": {"source": None, "rows": 0, "sha256": ""},
    }

    # ---------------- Weights ----------------
    w_bytes = _read_csv_bytes_from_path(weights_csv)
    w_source = "path" if w_bytes is not None else None

    if w_bytes is None:
        w_bytes = _read_csv_bytes_from_upload(weights_up)
        w_source = "upload" if w_bytes is not None else None

    if w_bytes is None:
        w_bytes = _read_csv_bytes_from_embedded(_EMBED_WEIGHTS_B64_GZ)
        w_source = "embedded"

    meta["weights"]["source"] = w_source
    meta["weights"]["sha256"] = _sha256(w_bytes)

    w = pd.read_csv(io.BytesIO(w_bytes)).copy()
    if "support" in w.columns:
        w = w[w["support"].fillna(0).astype(int) >= int(min_support)].copy()

    if "priority" in w.columns:
        w = w.sort_values(["priority"], ascending=True)

    base_rules: List[Rule] = [Rule.from_row(r) for _, r in w.head(int(max_rules)).iterrows()]
    meta["weights"]["rows"] = int(len(w))

    # ---------------- Tie-pack ----------------
    t_bytes = _read_csv_bytes_from_path(tie_csv)
    t_source = "path" if t_bytes is not None else None

    if t_bytes is None:
        t_bytes = _read_csv_bytes_from_upload(tie_up)
        t_source = "upload" if t_bytes is not None else None

    if t_bytes is None:
        t_bytes = _read_csv_bytes_from_embedded(_EMBED_TIEPACK_B64_GZ)
        t_source = "embedded"

    meta["tie_pack"]["source"] = t_source
    meta["tie_pack"]["sha256"] = _sha256(t_bytes)

    t = pd.read_csv(io.BytesIO(t_bytes)).copy()
    if "priority" in t.columns:
        t = t.sort_values(["priority"], ascending=True)

    tie_rules: List[Rule] = [Rule.from_row(r) for _, r in t.iterrows()]
    meta["tie_pack"]["rows"] = int(len(t))

    return base_rules, tie_rules, meta
def score_seed(seed4: str, base_rules: List[Rule], tie_rules: List[Rule]) -> Dict[str, object]:
    feats = compute_features(seed4)

    # base scoring
    scores = {m: 0.0 for m in TARGET_SET}
    fired = []
    for rule in base_rules:
        if rule.feature not in feats:
            continue
        if op_match(feats[rule.feature], rule.op, rule.value):
            scores[rule.pick] += rule.weight
            fired.append((rule.source, rule.feature, rule.op, rule.value, rule.pick, rule.weight))

    # rank
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top1, top1s = ordered[0]
    top2, top2s = ordered[1]
    top3, top3s = ordered[2]
    base_gap = top1s - top2s

    tie_fired = 0
    tie_break_fired = []

    # if tie or tiny gap, apply tie-pack to try to separate
    if base_gap == 0:
        tie_scores = {m: 0.0 for m in TARGET_SET}
        for rule in tie_rules:
            if rule.feature not in feats:
                continue
            if op_match(feats[rule.feature], rule.op, rule.value):
                tie_scores[rule.pick] += rule.weight
                tie_break_fired.append((rule.source, rule.feature, rule.op, rule.value, rule.pick, rule.weight))
        # apply tie_scores as secondary key
        ordered2 = sorted(scores.items(), key=lambda kv: (kv[1], tie_scores[kv[0]], kv[0]), reverse=True)
        tie_fired = 1 if any(v>0 for v in tie_scores.values()) else 0
        top1, top1s = ordered2[0]
        top2, top2s = ordered2[1]
        top3, top3s = ordered2[2]

        # if still tied even after tie_scores, that's a dead tie
        dead_tie = 1 if (top1s == top2s and tie_scores[top1] == tie_scores[top2]) else 0
    else:
        dead_tie = 0

    # coverage rules CR1/CR2
    forced = None
    if dead_tie == 1:
        ssum = feats["seed_sum"]
        forced = "0225" if ssum <= 11 else "0025"
        top1 = forced

    return {
        "seed": seed4,
        "features": feats,
        "scores": scores,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "base_gap": base_gap,
        "tie_fired": tie_fired,
        "dead_tie": dead_tie,
        "forced_pick": forced,
        "base_rules_fired": fired,
        "tie_rules_fired": tie_break_fired,
    }

# ----------------------------
# Stream cadence features
# ----------------------------
def compute_cadence_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    """
    For each stream and each row, compute:
      - is_025: whether this result is 025-family
      - draws_since_last_025: within stream, draws since last 025 (excluding current row)
      - rolling_025_30: count of 025 hits in prior 30 draws (excluding current)
    """
    df = hist.copy()
    df["member"] = df["result"].apply(as_member)
    df["is_025"] = df["member"].notna().astype(int)

    # draws since last 025
    def _since_last(s: pd.Series) -> pd.Series:
        last = -1
        out = []
        for i, v in enumerate(s.values):
            out.append(i - last if last != -1 else 9999)
            if v == 1:
                last = i
        return pd.Series(out, index=s.index)

    df["draws_since_last_025"] = df.groupby("stream")["is_025"].apply(_since_last).reset_index(level=0, drop=True)

    # rolling prior-30 025 count
    df["rolling_025_30"] = (
        df.groupby("stream")["is_025"]
          .apply(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
          .reset_index(level=0, drop=True)
          .fillna(0)
          .astype(int)
    )
    return df

def stream_score_row(seed_score: Dict[str, object], cadence_row: Dict[str, object],
                     w_match=1.0, w_gap=0.6, w_roll=0.25, w_since=0.15,
                     since_mode: str = "due",
                     cooldown_k=2, cooldown_penalty=1.0) -> float:
    """
    StreamScore for ranking all streams (full list):
      - match_strength: total base points (sum of top member scores)
      - gap: top1 - top2 (after tie-pack/coverage)
      - rolling_025_30: prior 30 hit count
      - draws_since_last_025: larger means "more due" (weak effect)
      - cooldown: if last 025 was within last k draws, downrate.
    """
    scores = seed_score["scores"]
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1, top1s = ordered[0]
    top2, top2s = ordered[1]
    match_strength = float(top1s)

    gap = float(top1s - top2s)

    roll = float(cadence_row.get("rolling_025_30", 0))
    since = float(cadence_row.get("draws_since_last_025", 9999))
    since_norm = min(since, 50.0) / 50.0  # 0..1 where 1 is most 'due'

    if since_mode == "recent":

        since_component = 1.0 - since_norm

    elif since_mode == "blend":

        # U-shape: reward both extremes (very recent OR very due)

        since_component = abs(since_norm - 0.5) * 2.0

    else:

        # legacy

        since_component = since_norm

    score = (w_match * match_strength) + (w_gap * gap) + (w_roll * roll) + (w_since * since_component)

    # cooldown
    if since <= cooldown_k:
        score -= cooldown_penalty

    # tie risk downrate
    if seed_score["dead_tie"] == 1:
        score -= 0.5
    elif seed_score["tie_fired"] == 1 and seed_score["base_gap"] == 0:
        score -= 0.2

    return score

# ----------------------------
# UI
# ----------------------------
st.title("Core 025 — Full Ranked Stream Playlist (Live) + Lab Backtest")

with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Upload Lottery Post export (CSV or TXT)", type=["csv","txt"])
    
    st.caption("Required columns (CSV or TXT): Draw Date, State, Game, Results")

    st.divider()
    st.header("Walk-forward (no-lookahead) dates")
    st.caption("Predictions NEVER use results after End date.")
    # These date inputs are rendered AFTER the history file is parsed, so we can safely
    # compute sensible defaults from the uploaded history without triggering Streamlit
    # session_state widget-key mutation errors.
    wf_dates_box = st.container()

    st.divider()
    st.header("Rule Settings (member engine)")
    gate_no9 = st.checkbox("Gate: seed has NO digit 9 (recommended)", value=True)
    max_rules = st.slider("Max base rules used", 1, 20, 12)
    min_support = st.slider("Min support (n_hits_gate) per base rule", 1, 50, 6)
    st.caption("Tie-pack rules always used as tie-breakers when base scores tie.")

    st.divider()
    st.header("Stream Rank Settings (live)")
    w_match = st.slider("Weight: trait match strength", 0.0, 3.0, 1.0, 0.05)
    w_gap   = st.slider("Weight: #1–#2 gap", 0.0, 3.0, 0.6, 0.05)
    w_roll  = st.slider("Weight: rolling 30-draw 025 hits", 0.0, 2.0, 0.25, 0.05)
    w_since = st.slider("Weight: draws since last 025 (due)", 0.0, 2.0, 0.15, 0.05)
    since_mode = st.selectbox(
        "Draws since last 025 mode",
        options=["due", "recent", "blend"],
        index=0,
        format_func=lambda x: {
            "due": "Due (higher = better) — legacy",
            "recent": "Recent (lower = better) — momentum",
            "blend": "Blend extremes (recent + very due) — U-shape",
        }[x],
        help="Only changes how the draws-since component is scored. Default keeps legacy behavior.",
    )
    cooldown_k = st.slider("Cooldown (draws) after a 025 hit", 0, 10, 2)
    cooldown_penalty = st.slider("Cooldown penalty", 0.0, 3.0, 1.0, 0.05)

    st.divider()

    st.subheader("Actions")
    c1, c2 = st.columns(2)
    if c1.button("Clear cache", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # Clear internal cached artifacts (do NOT remove widget keys after widgets are created)
        for _k in ["_cached_live_rows", "_cached_playlist", "_inputs_sig"]:
            if _k in st.session_state:
                del st.session_state[_k]
        # Force a rebuild on next run
        st.session_state["playlist_force_rebuild"] = st.session_state.get("playlist_force_rebuild", 0) + 1
        st.success("Cache cleared.")
        st.rerun()

    if c2.button("Rebuild playlist", use_container_width=True):
        # Bump a counter that participates in the cache signature
        st.session_state["playlist_force_rebuild"] = st.session_state.get("playlist_force_rebuild", 0) + 1
        for _k in ["_cached_playlist"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.success("Rebuilding playlist…")
        st.rerun()

    st.header("Files")
    weights_csv = st.text_input("Weights CSV path", value="025_rule_weights_rarityaware_detail_v1.csv")
    tie_csv = st.text_input("Tie-pack CSV path", value="025_tie_pack_v1.csv")

    # If running on Streamlit Cloud and these files are not bundled in the repo,
    # upload them here (preferred over fighting path issues).
    weights_file_up = st.file_uploader(
        "Upload weights CSV (025_rule_weights_rarityaware_detail_v1.csv)",
        type=["csv"],
        key="weights_up",
    )
    tiepack_file_up = st.file_uploader(
        "Upload tie-pack CSV (025_tie_pack_v1.csv)",
        type=["csv"],
        key="tiepack_up",
    )

tab_live, tab_lab = st.tabs(["LIVE: Full ranked stream list", "LAB: Historical 025 hit-events"])

if hist_file is None:
    st.info("Upload your Lottery Post export CSV in the sidebar to begin.")
    st.stop()

# Load history

def _peek_first_line(uploaded) -> str:
    try:
        b = uploaded.getvalue()
        s = b[:5000].decode("utf-8", errors="ignore")
        return s.splitlines()[0] if s.splitlines() else ""
    except Exception:
        return ""

def _read_txt(f):
    """
    Robust reader for LotteryPost tab-delimited exports that may contain:
      - headerless 4-column rows: date, state, game, result
      - occasional stray/continuation lines (e.g., 'Fireball: 9') or broken records
    We prefer tab-delimited parsing and skip malformed lines.
    """
    first = _peek_first_line(f)
    headerish = bool(
        re.search(r"date", first, re.I)
        and re.search(r"state", first, re.I)
        and re.search(r"game", first, re.I)
        and re.search(r"result", first, re.I)
    )

    f.seek(0)
    if headerish:
        try:
            return pd.read_csv(f, sep="	", engine="python", dtype=str, on_bad_lines="skip")
        except Exception:
            f.seek(0)
            return pd.read_csv(f, sep=None, engine="python", dtype=str, on_bad_lines="skip")
    else:
        try:
            return pd.read_csv(
                f,
                sep="	",
                engine="python",
                header=None,
                names=["Draw Date", "State", "Game", "Results"],
                dtype=str,
                on_bad_lines="skip",
            )
        except Exception:
            f.seek(0)
            df0 = pd.read_csv(f, sep=None, engine="python", header=None, dtype=str, on_bad_lines="skip")
            if df0.shape[1] >= 4:
                df0 = df0.iloc[:, :4].copy()
                df0.columns = ["Draw Date", "State", "Game", "Results"]
            return df0

try:
    filename = getattr(hist_file, "name", "").lower()

    if filename.endswith(".txt") or filename.endswith(".tsv"):
        raw = _read_txt(hist_file)
    else:
        raw = pd.read_csv(hist_file)

    hist = normalize_history(raw)

    if len(hist) == 0:
        st.error(
            "No valid history rows were parsed after normalization. This usually means the TXT/TSV has broken/extra lines or the date/result formats are not being recognized."
        )
        st.caption(
            "Tip: Your file should be tab-delimited with 4 columns per row: date, state, game, result (e.g., 'Thu, Mar 14, 2024	Missouri	Pick 4 Midday	9-3-1-9')."
        )
        with st.expander("Show first 50 raw rows (as parsed)"):
            st.dataframe(raw.head(50), use_container_width=True)
        st.stop()
except Exception as e:
    st.error(f"Failed to read/normalize history: {e}")
    st.stop()
# Diagnostics
st.write(f"Loaded history rows (parsed): {len(hist):,}")
dmin, dmax = hist["Draw Date"].min(), hist["Draw Date"].max()

st.write(f"History date span: {dmin.date()} → {dmax.date()}")

# ---- Walk-forward date defaults + validation ----
_default_start = dmin.date()
_default_end = dmax.date()
_default_play = (_default_end + datetime.timedelta(days=1))

# Render the date inputs AFTER history is parsed so defaults are based on this upload,
# without mutating widget-backed session_state keys.
with wf_dates_box:
    wf_start = st.date_input("Start date (training ends at or before this)", value=st.session_state.get("wf_start", _default_start), key="wf_start")
    wf_end = st.date_input("End date (last training day)", value=st.session_state.get("wf_end", _default_end), key="wf_end")
    wf_play = st.date_input("Play date (prediction target day)", value=st.session_state.get("wf_play", _default_play), key="wf_play")

# No-lookahead guard: End date must be BEFORE Play date
if wf_end >= wf_play:
    st.warning("No-lookahead guard: End date should be BEFORE Play date. Adjusting End date to Play date - 1 day.")
    wf_end = (pd.Timestamp(wf_play) - pd.Timedelta(days=1)).date()

# Ensure start <= end and clamp to file span
if wf_start > wf_end:
    st.warning("Start date was after End date. Adjusting Start date to End date.")
    wf_start = wf_end

if wf_start < _default_start:
    wf_start = _default_start
if wf_end > _default_end:
    wf_end = _default_end

st.caption(f"Walk-forward window in effect: {wf_start} → {wf_end} (Play date: {wf_play}). Model cannot see results after {wf_end}.")

# If user requested a rebuild, clear cached computations so the playlist is rebuilt deterministically
if st.session_state.get("playlist_force_rebuild", 0) > 0:
    try:
        st.cache_data.clear()
    except Exception:
        pass
pf = getattr(hist, "attrs", {}).get("parse_fail_count", 0)
if pf:
    st.warning(f"Result-parse failures: {pf:,} rows could not be parsed into 4 digits and were ignored.")
    ex = getattr(hist, "attrs", {}).get("parse_fail_examples", [])
    if ex:
        with st.expander("Show parse-failure examples"):
            st.dataframe(pd.DataFrame(ex), use_container_width=True)

# Load rules (from local files next to app, or user can type full path)
try:
    base_rules, tie_rules, rules_meta = load_rules(weights_csv, tie_csv, max_rules=max_rules, min_support=min_support, weights_up=weights_file_up, tie_up=tiepack_file_up)
except Exception as e:
    st.error(f"Failed to load rules from '{weights_csv}' and '{tie_csv}': {e}")
    st.stop()

st.caption(
    f"Rule files in effect — Weights: {rules_meta['weights']['source']} (rows={rules_meta['weights']['rows']}) | "
    f"Tie-pack: {rules_meta['tie_pack']['source']} (rows={rules_meta['tie_pack']['rows']})"
)
with st.expander("Rule file provenance (sha256 hashes)"):
    st.code(
        f"Weights source: {rules_meta['weights']['source']}\n"
        f"Weights sha256: {rules_meta['weights']['sha256']}\n"
        f"Tie-pack source: {rules_meta['tie_pack']['source']}\n"
        f"Tie-pack sha256: {rules_meta['tie_pack']['sha256']}\n"
    )

with st.expander("Active base rules (after support + max_rules filters)"):
    if base_rules:
        st.dataframe(pd.DataFrame([{
            "feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": r.weight
        } for r in base_rules]), use_container_width=True)
    else:
        st.warning("No base rules active under the current settings. Lower min_support or increase max_rules.")

with st.expander("Tie-pack rules"):
    st.dataframe(pd.DataFrame([{
        "feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": r.weight
    } for r in tie_rules]), use_container_width=True)

# Cadence metrics (for live)

# Apply walk-forward window to history (NO LOOKAHEAD)
hist_wf = hist[(hist["Draw Date"].dt.date >= wf_start) & (hist["Draw Date"].dt.date <= wf_end)].copy()

# Re-sort and re-index per stream after filtering
hist_wf = hist_wf.sort_values(["stream", "Draw Date"]).reset_index(drop=True)
hist_wf["stream_idx"] = hist_wf.groupby("stream").cumcount()

# Cadence metrics computed ONLY on visible history
hist_cad = compute_cadence_metrics(hist_wf)

# ----------------------------
# LIVE MODE
# ----------------------------
with tab_live:
    st.subheader("Full ranked stream list (most → least likely to produce a 025-family hit next)")

    # Latest seed per stream
    latest = hist_cad.sort_values(["stream", "Draw Date"]).groupby("stream").tail(1).copy()
    latest = latest.rename(columns={"result": "seed_result", "Draw Date": "seed_date"})

    # Gate no9 (locked core gate)
    pre_gate_latest = latest.copy()
    gated_out_live = pre_gate_latest.iloc[0:0].copy()
    if gate_no9:
        _mask9 = pre_gate_latest["seed_result"].astype(str).str.contains("9")
        gated_out_live = pre_gate_latest[_mask9].copy()
        latest = pre_gate_latest[~_mask9].copy()
    else:
        latest = pre_gate_latest

    # Visibility: which streams were excluded by the NO9 gate today
    if gate_no9:
        with st.expander(f"Show streams gated out by NO9 today ({len(gated_out_live):,})", expanded=False):
            if gated_out_live.empty:
                st.write("None.")
            else:
                g = gated_out_live.rename(columns={"stream": "Stream", "seed_date": "SeedDate", "seed_result": "Seed"}).copy()
                g["StreamKey"] = g["Stream"].apply(canon_stream)
                g["SeedDate"] = g["SeedDate"].apply(lambda x: x.date() if pd.notna(x) else None)
                show_cols = [c for c in ["Stream", "StreamKey", "SeedDate", "Seed"] if c in g.columns]
                st.dataframe(g[show_cols].sort_values(["Stream"]), use_container_width=True)

    rows = []
    for _, r in latest.iterrows():
        seed4 = r["seed_result"]
        s = score_seed(seed4, base_rules, tie_rules)

        cadence_row = {
            "draws_since_last_025": int(r.get("draws_since_last_025", 9999)),
            "rolling_025_30": int(r.get("rolling_025_30", 0)),
        }
        ss = stream_score_row(
            s, cadence_row,
            w_match=w_match, w_gap=w_gap, w_roll=w_roll, w_since=w_since,
            since_mode=since_mode,
            cooldown_k=cooldown_k, cooldown_penalty=cooldown_penalty
        )

        feats = s["features"]
        rows.append({
            "Stream": r["stream"],
            "SeedDate": r["seed_date"].date() if pd.notna(r["seed_date"]) else None,
            "Seed": seed4,
            "PredictedMember": s["top1"],
            "Top2": s["top2"],
            "Top3": s["top3"],
            "Top1Score": max(s["scores"].values()),
            "BaseGap(#1-#2)": float(s["base_gap"]),
            "TieFired": int(s["tie_fired"]),
            "DeadTie": int(s["dead_tie"]),
                                "CadenceRolling025_30": int(cadence_row.get("rolling_025_30", 0)),
                                "CadenceDrawsSinceLast025": int(cadence_row.get("draws_since_last_025", 9999)),
            "ForcedPick": s["forced_pick"] if s["forced_pick"] else "",
            "SinceLast025(draws)": cadence_row["draws_since_last_025"],
            "Rolling025_30": cadence_row["rolling_025_30"],
            "SeedSum": feats["seed_sum"],
            "SeedSpread": feats["seed_spread"],
            "SeedAbsDiff": feats["seed_absdiff"],
            "WorstPair025": feats["seed_has_worstpair_025"],
            "StreamScore": float(ss),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["StreamKey"] = out["Stream"].apply(canon_stream)
    if out.empty:
        st.warning("No streams left after gating/filtering. Try turning off gate_no9.")
    else:
        out = out.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=False).reset_index(drop=True)
        out.insert(0, "Rank", np.arange(1, len(out)+1))

        # Tier labels (no cutoff): A strongest, then B, then C
        # thresholds derived from relative distribution each run
        qA = out["StreamScore"].quantile(0.20)
        qB = out["StreamScore"].quantile(0.50)

        def tier(s):
            if s >= qA:
                return "A"
            if s >= qB:
                return "B"
            return "C"
        out["Tier"] = out["StreamScore"].apply(tier)

        # --- Flip recommendation (Top-2 trigger) ---
        # This is the same "Top-2 recommended" segment we used in the earlier analysis:
        # base_gap == 0 AND tie_fired == 1 AND dead_tie == 0
        out["FlipRec"] = ((out["BaseGap(#1-#2)"] == 0) & (out["TieFired"] == 1) & (out["DeadTie"] == 0)).astype(int)
        # Legacy single-member suggestion (kept for continuity): FlipRec means "prefer TOP2"
        out["PlayPick"] = np.where(out["FlipRec"] == 1, "TOP2", "TOP1")
        out["PlayMember"] = np.where(out["FlipRec"] == 1, out["Top2"], out["PredictedMember"])
        # NEW default play policy (requested): when TieFired (FlipRec=1), play TOP1 + TOP2 (2 members)
        out["PlayPlan"] = np.where(out["FlipRec"] == 1, "TOP1+TOP2", "TOP1")
        out["PlayMembers"] = np.where(
            out["FlipRec"] == 1,
            out["PredictedMember"].astype(str) + " + " + out["Top2"].astype(str),
            out["PredictedMember"].astype(str),
        )
        out["PlayCount"] = np.where(out["FlipRec"] == 1, 2, 1).astype(int)

        # Put the play columns right next to the prediction columns for visibility
        _cols = list(out.columns)
        for c in ["PlayPlan", "PlayMembers", "PlayCount", "PlayPick", "PlayMember", "FlipRec"]:
            if c in _cols:
                _cols.remove(c)
        # Insert after PredictedMember (or after Seed if that col is missing)
        if 'PredictedMember' in _cols:
            _i = _cols.index('PredictedMember') + 1
        elif 'Seed' in _cols:
            _i = _cols.index('Seed') + 1
        else:
            _i = min(5, len(_cols))
        _cols[_i:_i] = ["PlayPlan", "PlayMembers", "PlayCount", "PlayPick", "PlayMember", "FlipRec"]
        out = out[_cols]

        st.caption("Full list ranked. Tier A ≈ top 20% by StreamScore; Tier B ≈ next 30%; Tier C ≈ bottom 50% (no cutoff). "
           "FlipRec=1 means the app recommends playing TOP1 + TOP2 for that stream (PlayMembers; PlayCount=2).")

        def _style_flip(df):
            # Bold the play columns (and Top2) when FlipRec fires
            def _row_style(row):
                if int(row.get("FlipRec", 0)) == 1:
                    return [
                        "font-weight:700;" if col in ("PlayPlan", "PlayMembers", "PlayCount", "PlayMember", "PlayPick", "Top2") else ""
                        for col in df.columns
                    ]
                return ["" for _ in df.columns]

            return df.style.apply(_row_style, axis=1)

        _render = _style_flip(out)
        st.dataframe(_render, use_container_width=True, height=650)

        # Export
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download LIVE ranked streams (CSV)", data=csv_bytes, file_name="core025_live_ranked_streams.csv", mime="text/csv")


# ----------------------------
# LAB MODE (historical 025 hit-events)
# ----------------------------
with tab_lab:
    st.subheader("Historical 025 hit-events (seed → next winner member)")

    # Build transitions within stream
    df = hist_wf.copy()
    df["member"] = df["result"].apply(as_member)
    df["is_025"] = df["member"].notna().astype(int)

    # seed = previous result in same stream
    df["seed"] = df.groupby("stream")["result"].shift(1)
    df["seed_date"] = df.groupby("stream")["Draw Date"].shift(1)
    df = df.dropna(subset=["seed", "member"]).copy()

    # apply gate to seeds
    if gate_no9:
        df = df[~df["seed"].astype(str).str.contains("9")].copy()

    st.write(f"Total gated 025 hit-events: {len(df):,}")

    # Score each seed and compare to actual member
    lab_rows = []
    for _, r in df.iterrows():
        seed4 = r["seed"]
        s = score_seed(seed4, base_rules, tie_rules)
        actual = r["member"]
        hit1 = int(s["top1"] == actual)
        hit2 = int(actual in {s["top1"], s["top2"]})

        lab_rows.append({
            "Stream": r["stream"],
            "SeedDate": r["seed_date"].date() if pd.notna(r["seed_date"]) else None,
            "Seed": seed4,
            "WinnerDate": r["Draw Date"].date() if pd.notna(r["Draw Date"]) else None,
            "ActualWinnerMember": actual,
            "Top1": s["top1"],
            "Top2": s["top2"],
            "Top3": s["top3"],
            "HitTop1": hit1,
            "HitTop2": hit2,
            "TieFired": int(s["tie_fired"]),
            "DeadTie": int(s["dead_tie"]),
            "ForcedPick": s["forced_pick"] if s["forced_pick"] else "",
            "SeedSum": s["features"]["seed_sum"],
            "SeedAbsDiff": s["features"]["seed_absdiff"],
            "SeedSpread": s["features"]["seed_spread"],
            "WorstPair025": s["features"]["seed_has_worstpair_025"],
        })

    lab = pd.DataFrame(lab_rows)
    if lab.empty:
        st.warning("No 025 hit-events found under current settings.")
    else:
        top1_acc = lab["HitTop1"].mean()
        top2_acc = lab["HitTop2"].mean()
        st.write(f"Top-1 accuracy: {top1_acc*100:.2f}%  |  Top-2 capture: {top2_acc*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Actual winner distribution (gated hit-events):")
            st.dataframe(lab["ActualWinnerMember"].value_counts().rename_axis("member").reset_index(name="count"), use_container_width=True)
        with col2:
            st.write("Top-1 pick distribution:")
            st.dataframe(lab["Top1"].value_counts().rename_axis("pick").reset_index(name="count"), use_container_width=True)

        st.dataframe(lab.sort_values(["WinnerDate","Stream"]), use_container_width=True, height=650)

        lab_csv = lab.to_csv(index=False).encode("utf-8")
        st.download_button("Download LAB hit-events (CSV)", data=lab_csv, file_name="core025_lab_hitevents.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Walk-forward Tier Coverage Study (StreamScore percentiles)")
        st.caption(
            "Automates 200+ play-dates: for each date, builds the ranked playlist using ONLY history before that date, "
            "then records where the actual 025-family winning stream landed (Tier/Rank/Percentile)."
        )

        # Choose date range based on available history
        _all_dates = pd.to_datetime(hist_wf["Draw Date"]).dt.date.unique().tolist()
        _all_dates = sorted([d for d in _all_dates if pd.notna(d)])
        if _all_dates:
            _min_d, _max_d = _all_dates[0], _all_dates[-1]
        else:
            _min_d, _max_d = datetime.date.today(), datetime.date.today()

        cA, cB, cC, cD = st.columns([1,1,1,1])
        with cA:
            tierA_pct = st.number_input("Tier A = top % by StreamScore", min_value=1, max_value=90, value=20, step=1, key="tierA_pct")
        with cB:
            tierB_pct = st.number_input("Tier B = next %", min_value=1, max_value=99, value=30, step=1, key="tierB_pct")
        with cC:
            wf_start = st.date_input("Start date", value=_min_d, min_value=_min_d, max_value=_max_d, key="wf_tier_start")
        with cD:
            wf_end = st.date_input("End date", value=_max_d, min_value=_min_d, max_value=_max_d, key="wf_tier_end")

        # Validate percents
        if tierA_pct + tierB_pct >= 100:
            st.error("Tier A% + Tier B% must be < 100 (Tier C is the remainder).")
        else:
            
            selection_mode = st.selectbox(
                "Play-50 selection strategy",
                options=[
                    "Percentile bins (winner-heavy zones)",
                    "Best historical winner-producing rank rows",
                ],
                index=0,
                key="wf_selection_mode",
            )
            cap_streams = st.number_input("Daily stream cap (play method)", min_value=1, max_value=500, value=50, step=1, key="wf_cap_streams")
            bin_size_pct = st.selectbox(
                "Percentile bin size (row-zones) — used only in Percentile-bins mode",
                options=[1, 2, 5, 10],
                index=2,
                key="wf_bin_size_pct",
            )
            fill_to_cap = st.checkbox("Fill to exact cap using best remaining streams", value=True, key="wf_fill_to_cap")
            cRun1, cRun2 = st.columns([1, 1])
            with cRun1:
                run_btn = st.button("Run Walk-forward Tier Study", type="primary", key="run_wf_tier_study")
            with cRun2:
                compare_btn = st.button("Run Gate ON vs OFF Comparison (NO9)", type="secondary", key="run_wf_tier_compare")
            # Cached prior WF outputs (keeps downloads stable across reruns and prevents NameError if UI renders cached views)
            _wf_cache = st.session_state.get("_wf_tier_cache", {}) if isinstance(st.session_state.get("_wf_tier_cache", {}), dict) else {}
            per_event = _wf_cache.get("per_event", pd.DataFrame())
            per_date  = _wf_cache.get("per_date", pd.DataFrame())
            per_stream = _wf_cache.get("per_stream", pd.DataFrame())
            wf_runs = st.session_state.get("_wf_tier_cache_runs", {}) if isinstance(st.session_state.get("_wf_tier_cache_runs", {}), dict) else {}


            wf_sig_base = (
                str(selection_mode),
                str(wf_start), str(wf_end),
                float(tierA_pct), float(tierB_pct),
                int(min_support), int(max_rules),
                json.dumps([{"feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": float(r.weight)} for r in base_rules], sort_keys=True, default=str),
                json.dumps([{"feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": float(r.weight)} for r in tie_rules], sort_keys=True, default=str),
                int(cap_streams), int(bin_size_pct), int(fill_to_cap),
            )

            def _wf_tier_compute_base(gate_flag: bool):
                gate_no9 = bool(gate_flag)
                per_stream_rows = []

                # Hit-events = rows where the RESULT is a 025-family member
                events = hist_wf.copy()
                events["Member"] = events["result"].apply(as_member)
                events = events.dropna(subset=["Member"]).copy()
                events["PlayDate"] = pd.to_datetime(events["Draw Date"]).dt.date

                # Restrict to date window
                events = events[(events["PlayDate"] >= wf_start) & (events["PlayDate"] <= wf_end)].copy()

                # No-cheat requirement: playlist on date D uses ONLY history strictly before D
                # Also: if you keep gate_no9 on, apply it to the SEED used for scoring (latest seed before D)

                if events.empty:
                    st.warning("No 025-family hit-events found in that date range.")
                else:
                    # Pre-group events by date for caching playlist build per date
                    events_by_date = {d: g.copy() for d, g in events.groupby("PlayDate")}
                    play_dates = sorted(events_by_date.keys())

                    prog = st.progress(0)
                    status = st.empty()

                    per_event_rows = []
                    per_date_rows = []

                    for i, d in enumerate(play_dates):
                        status.write(f"Building playlist as-of {d} ({i+1}/{len(play_dates)}) …")

                        hist_before = hist_wf[pd.to_datetime(hist_wf["Draw Date"]).dt.date < d].copy()
                        if hist_before.empty:
                            # no prior history — cannot score
                            # Snapshot per-stream-per-day ranked universe for 50-stream play-method analysis
                            try:
                                day_df = idx_map.reset_index().copy()
                                day_df["PlayDate"] = d
                                day_df["UniverseSize"] = int(len(idx_map))
                                day_df["PreGateUniverseSize"] = int(len(pre_gate_keys)) if 'pre_gate_keys' in locals() else int(len(idx_map))
                                day_df["GateExcluded"] = int(len(gated_out_keys)) if 'gated_out_keys' in locals() else 0
                                per_stream_rows.append(day_df)
                            except Exception:
                                pass

                            for _, ev in events_by_date[d].iterrows():
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": ev.get("stream"),
                "WinningStreamKey": canon_stream(ev.get("stream")),
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": None,
                                    "Percentile": None,
                                    "Tier": None,
                                    "FlipRec": None,
                                    "PlayPlan": None,
                                    "PlayMembers": None,
                                    "PlayCount": None,
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
                                    "Reason": "NO_HISTORY_BEFORE_PLAYDATE",
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                })
                            per_date_rows.append({
                                "PlayDate": d,
                                "UniverseSize": 0,
                                "TierA_Size": 0,
                                "TierB_Size": 0,
                                "TierC_Size": 0,
                                "HitEvents": int(len(events_by_date[d])),
                                "Covered_TierA": 0,
                                "Covered_TierAB": 0,
                            })
                            prog.progress(int((i+1)/len(play_dates)*100))
                            continue

                        # Build cadence metrics and latest-per-stream seeds as-of d
                        hist_before = compute_cadence_metrics(hist_before)
                        latest = hist_before.sort_values(["stream", "Draw Date"]).groupby("stream").tail(1).copy()
                        latest['StreamKey'] = latest['stream'].apply(canon_stream)
                        latest_all = latest.copy()  # pre-gate snapshot for diagnostics
                        # Map StreamKey -> seed (pre-gate) for proof columns in diagnostics
                        seed_by_key = dict(zip(
                            latest_all['StreamKey'].dropna().astype(str).tolist(),
                            latest_all['result'].astype(str).tolist()
                        ))
                        seed_has9_by_key = {k: ('9' in str(v)) for k, v in seed_by_key.items()}

                        latest = latest.rename(columns={"result": "seed_result", "Draw Date": "seed_date"})

                        pre_gate_keys = set(latest_all["StreamKey"].dropna().astype(str).tolist())
                        gated_out_keys = set()
                        if gate_no9:
                            latest = latest[~latest["seed_result"].astype(str).str.contains("9")].copy()
                            post_gate_keys = set(latest["StreamKey"].dropna().astype(str).tolist())
                            gated_out_keys = pre_gate_keys - post_gate_keys

                        rows = []
                        for _, r in latest.iterrows():
                            seed4 = r["seed_result"]
                            s = score_seed(seed4, base_rules, tie_rules)
                            cadence_row = {
                                "draws_since_last_025": int(r.get("draws_since_last_025", 9999)),
                                "rolling_025_30": int(r.get("rolling_025_30", 0)),
                            }
                            ss = stream_score_row(
                                s, cadence_row,
                                w_match=w_match, w_gap=w_gap, w_roll=w_roll, w_since=w_since,
                                since_mode=since_mode,
                                cooldown_k=cooldown_k, cooldown_penalty=cooldown_penalty
                            )
                            rows.append({
                                "Stream": r["stream"],
                                "StreamScore": float(ss),
                                "PredictedMember": s["top1"],
                                "Top2": s["top2"],
                                "Top3": s["top3"],
                                "Top1Score": max(s["scores"].values()),
                                "BaseGap(#1-#2)": float(s["base_gap"]),
                                "TieFired": int(s["tie_fired"]),
                                "DeadTie": int(s["dead_tie"]),
                                "CadenceRolling025_30": int(cadence_row.get("rolling_025_30", 0)),
                                "CadenceDrawsSinceLast025": int(cadence_row.get("draws_since_last_025", 9999)),
                            })

                        ranked = pd.DataFrame(rows)
                        if ranked.empty:
                            uni_n = 0
                            # record as missing universe
                            for _, ev in events_by_date[d].iterrows():
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": ev.get("stream"),
                "WinningStreamKey": canon_stream(ev.get("stream")),
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": None,
                                    "Percentile": None,
                                    "Tier": None,
                                    "FlipRec": None,
                                    "PlayPlan": None,
                                    "PlayMembers": None,
                                    "PlayCount": None,
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
                                    "Reason": "UNIVERSE_EMPTY",
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                })
                            per_date_rows.append({
                                "PlayDate": d,
                                "UniverseSize": 0,
                                "TierA_Size": 0,
                                "TierB_Size": 0,
                                "TierC_Size": 0,
                                "HitEvents": int(len(events_by_date[d])),
                                "Covered_TierA": 0,
                                "Covered_TierAB": 0,
                            })
                            prog.progress(int((i+1)/len(play_dates)*100))
                            continue

                        ranked = ranked.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=False).reset_index(drop=True)
                        ranked.insert(0, "Rank", np.arange(1, len(ranked)+1))

                        # FlipRec + PlayMember (same rule as LIVE)
                        ranked["FlipRec"] = ((ranked["BaseGap(#1-#2)"] == 0) & (ranked["TieFired"] == 1) & (ranked["DeadTie"] == 0)).astype(int)
                        # Legacy single-member suggestion (kept): FlipRec means "prefer TOP2"
                        ranked["PlayMember"] = np.where(ranked["FlipRec"] == 1, ranked["Top2"], ranked["PredictedMember"])
                        # NEW default play policy (requested): when TieFired (FlipRec=1), play TOP1 + TOP2
                        ranked["PlayPlan"] = np.where(ranked["FlipRec"] == 1, "TOP1+TOP2", "TOP1")
                        ranked["PlayMembers"] = np.where(
                            ranked["FlipRec"] == 1,
                            ranked["PredictedMember"].astype(str) + " + " + ranked["Top2"].astype(str),
                            ranked["PredictedMember"].astype(str),
                        )
                        ranked["PlayCount"] = np.where(ranked["FlipRec"] == 1, 2, 1).astype(int)

                        # Percentile rank (1 = best). Example: Rank 1 of 80 => 1/80 = 0.0125
                        uni_n = int(len(ranked))
                        ranked["Percentile"] = ranked["Rank"] / uni_n

                        # Tier cutoffs by StreamScore percentiles (requested behavior)
                        # Tier A: top tierA_pct% by StreamScore, Tier B: next tierB_pct%, Tier C: rest
                        qA = ranked["StreamScore"].quantile(1.0 - tierA_pct/100.0)
                        qB = ranked["StreamScore"].quantile(1.0 - (tierA_pct + tierB_pct)/100.0)

                        def _tier_by_score(x):
                            if x >= qA:
                                return "A"
                            if x >= qB:
                                return "B"
                            return "C"

                        ranked["Tier"] = ranked["StreamScore"].apply(_tier_by_score)

                        # Tier sizes (for cost / plays per day)
                        tierA_size = int((ranked["Tier"] == "A").sum())
                        tierB_size = int((ranked["Tier"] == "B").sum())
                        tierC_size = int((ranked["Tier"] == "C").sum())

                        # Evaluate all hit-events on date d
                        covered_A = 0
                        covered_AB = 0

                        ranked = ranked.copy()
                        if "StreamKey" not in ranked.columns:
                            ranked["StreamKey"] = ranked["Stream"].apply(canon_stream)
                        # Precompute key-sets/mappings for out-of-universe diagnostics (avoid NameError)
                        rank_keys = set(ranked["StreamKey"].dropna().astype(str).tolist())
                        rank_display_by_key = {str(k): v for k, v in zip(ranked["StreamKey"], ranked["Stream"]) if pd.notna(k)}
                        latest_display_by_key = {str(k): v for k, v in zip(latest_all["StreamKey"], latest_all["stream"]) if pd.notna(k)}
                        all_keys = set(latest_all["StreamKey"].dropna().astype(str).tolist())
                                                # Reverse-order rank map (sanity-check: is score direction flipped?)
                        rev = ranked.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=True).reset_index(drop=True).copy()
                        rev["Rank_rev"] = np.arange(1, len(rev)+1)
                        rev["Percentile_rev"] = rev["Rank_rev"] / float(len(rev)) if len(rev) else None
                        rev_map = rev.set_index("StreamKey")[["Rank_rev", "Percentile_rev"]]
                        desired_cols = [
                            "Rank", "Percentile", "Tier", "FlipRec", "PlayPlan", "PlayMembers", "PlayCount", "PlayMember",
                            "StreamScore", "PredictedMember", "Top2", "Top3",
                            "Top1Score", "BaseGap(#1-#2)", "TieFired", "DeadTie",
                            "CadenceRolling025_30", "CadenceDrawsSinceLast025"
                        ]
                        idx_map = ranked.set_index("StreamKey").reindex(columns=desired_cols).join(rev_map, how="left")

                        # Snapshot per-stream-per-day ranked universe (for Play-50 reproduction)
                        try:
                            day_df = idx_map.reset_index().copy()
                            # Attach display label + required member columns for downstream selection/capture
                            day_df["Stream"] = day_df["StreamKey"].astype(str).map(rank_display_by_key)
                            if "PredictedMember" in day_df.columns and "Top1" not in day_df.columns:
                                day_df["Top1"] = day_df["PredictedMember"].astype(str)
                            if "PlayDate" not in day_df.columns:
                                day_df["PlayDate"] = d
                            day_df["UniverseSize"] = int(uni_n)
                            day_df["PreGateUniverseSize"] = int(len(pre_gate_keys)) if 'pre_gate_keys' in locals() else int(uni_n)
                            day_df["GateExcluded"] = int(len(gated_out_keys)) if 'gated_out_keys' in locals() else 0
                            per_stream_rows.append(day_df)
                        except Exception:
                            pass


                        # --- Component-only rank maps (LAB diagnostics) ---
                        def _make_rank_map(_df, _col, ascending=False):
                            """Return StreamKey -> 1-based rank when sorting by _col."""
                            if _col not in _df.columns:
                                return {}
                            tmp = _df[["StreamKey", _col]].copy()
                            tmp["StreamKey"] = tmp["StreamKey"].astype(str)
                            tmp[_col] = pd.to_numeric(tmp[_col], errors="coerce")
                            tmp = tmp.sort_values([_col, "StreamKey"], ascending=[ascending, True], na_position="last").reset_index(drop=True)
                            tmp["__rank"] = np.arange(1, len(tmp) + 1)
                            return dict(zip(tmp["StreamKey"].tolist(), tmp["__rank"].astype(int).tolist()))

                        rankmap_top1 = _make_rank_map(ranked, "Top1Score", ascending=False)
                        rankmap_gap = _make_rank_map(ranked, "BaseGap(#1-#2)", ascending=False)
                        rankmap_roll = _make_rank_map(ranked, "CadenceRolling025_30", ascending=False)
                        rankmap_since_due = _make_rank_map(ranked, "CadenceDrawsSinceLast025", ascending=False)
                        rankmap_since_recent = _make_rank_map(ranked, "CadenceDrawsSinceLast025", ascending=True)
                        # SinceBlend (LAB-only): favor extremes in DrawsSinceLast025 (very recent OR very due/never)
                        # Use the better (lower) percentile from asc vs desc orderings to score each stream.
                        stream_keys = ranked["StreamKey"].astype(str).tolist()
                        if uni_n <= 1:
                            rankmap_since_blend = {k: 1 for k in stream_keys}
                        else:
                            _pct_recent = {k: (rankmap_since_recent.get(k, uni_n) / uni_n) for k in stream_keys}
                            _pct_due = {k: (rankmap_since_due.get(k, uni_n) / uni_n) for k in stream_keys}
                            _blend_score = {k: 1.0 - min(_pct_recent[k], _pct_due[k]) for k in stream_keys}
                            _blend_order = sorted(stream_keys, key=lambda k: (-_blend_score[k], k))
                            rankmap_since_blend = {k: i + 1 for i, k in enumerate(_blend_order)}

                        # Capture the day's #1 stream (for driver-audit panels)
                        _top = ranked.iloc[0]
                        top_stream = _top["Stream"]
                        top_stream_key = str(_top.get("StreamKey")) if pd.notna(_top.get("StreamKey")) else canon_stream(top_stream)
                        top_streamscore = float(_top.get("StreamScore"))
                        top_top1score = float(_top.get("Top1Score"))
                        top_basegap = float(_top.get("BaseGap(#1-#2)"))
                        top_roll30 = int(_top.get("CadenceRolling025_30"))
                        top_since = int(_top.get("CadenceDrawsSinceLast025"))
                        top_tiefired = int(_top.get("TieFired"))
                        top_deadtie = int(_top.get("DeadTie"))

                        # Per-day tie/granularity metrics (LAB diagnostics)
                        day_unique_streamscore = int(ranked["StreamScore"].nunique(dropna=True)) if "StreamScore" in ranked.columns else None
                        day_unique_top1score = int(ranked["Top1Score"].nunique(dropna=True)) if "Top1Score" in ranked.columns else None
                        _top_ss_raw = _top.get("StreamScore")
                        _top_t1_raw = _top.get("Top1Score")
                        _top_gap_raw = _top.get("BaseGap(#1-#2)")
                        day_top_score_tie = int((ranked["StreamScore"] == _top_ss_raw).sum()) if "StreamScore" in ranked.columns else None
                        if all(c in ranked.columns for c in ["StreamScore", "Top1Score", "BaseGap(#1-#2)"]):
                            day_top_tuple_tie = int(((ranked["StreamScore"] == _top_ss_raw) & (ranked["Top1Score"] == _top_t1_raw) & (ranked["BaseGap(#1-#2)"] == _top_gap_raw)).sum())
                        else:
                            day_top_tuple_tie = None

                        for _, ev in events_by_date[d].iterrows():
                            stream_raw = ev.get("Stream") if ev.get("Stream") is not None else ev.get("stream")
                            stream_key = canon_stream(stream_raw)
                            if stream_key in idx_map.index:
                                rr = idx_map.loc[stream_key]
                                t = rr["Tier"]
                                prev_seed = seed_by_key.get(stream_key) if stream_key else None
                                prev_seed_has9 = seed_has9_by_key.get(stream_key) if stream_key else None
                                in_uni = 1
                                if t == "A":
                                    covered_A += 1
                                    covered_AB += 1
                                elif t == "B":
                                    covered_AB += 1
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": stream_raw,
                                    "WinningStreamKey": stream_key,
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": int(rr["Rank"]),
                                    "Percentile": float(rr["Percentile"]),
                                    "Tier": t,
                                    "FlipRec": int(rr["FlipRec"]),
                                    "PlayPlan": rr.get("PlayPlan"),
                                    "PlayMembers": rr.get("PlayMembers"),
                                    "PlayCount": int(rr.get("PlayCount")) if pd.notna(rr.get("PlayCount")) else None,
                                    "PlayMember": rr["PlayMember"],
                                    "Reason": "",
                                    "PrevSeed": prev_seed,
                                    "PrevSeedHas9": prev_seed_has9,
                                    "WinnerStreamScore": float(rr["StreamScore"]),
                                    "WinnerTop1Score": float(rr["Top1Score"]),
                                    "WinnerBaseGap": float(rr["BaseGap(#1-#2)"]),
                                    "WinnerRoll025_30": int(rr["CadenceRolling025_30"]),
                                    "WinnerDrawsSinceLast025": int(rr["CadenceDrawsSinceLast025"]),
                                    "WinnerTieFired": int(rr["TieFired"]),
                                    "WinnerDeadTie": int(rr["DeadTie"]),
                                    "Rank_Top1Score": int(rankmap_top1.get(stream_key)) if rankmap_top1 and (stream_key in rankmap_top1) else None,
                                    "Rank_BaseGap": int(rankmap_gap.get(stream_key)) if rankmap_gap and (stream_key in rankmap_gap) else None,
                                    "Rank_Roll025_30": int(rankmap_roll.get(stream_key)) if rankmap_roll and (stream_key in rankmap_roll) else None,
                                    "Rank_Since_due": int(rankmap_since_due.get(stream_key)) if rankmap_since_due and (stream_key in rankmap_since_due) else None,
                                    "Rank_Since_recent": int(rankmap_since_recent.get(stream_key)) if rankmap_since_recent and (stream_key in rankmap_since_recent) else None,
                                    "Rank_Since_blend": int(rankmap_since_blend.get(stream_key)) if rankmap_since_blend and (stream_key in rankmap_since_blend) else None,
                                    "Pct_Top1Score": (float(rankmap_top1.get(stream_key)) / float(uni_n)) if rankmap_top1 and (stream_key in rankmap_top1) and uni_n else None,
                                    "Pct_BaseGap": (float(rankmap_gap.get(stream_key)) / float(uni_n)) if rankmap_gap and (stream_key in rankmap_gap) and uni_n else None,
                                    "Pct_Roll025_30": (float(rankmap_roll.get(stream_key)) / float(uni_n)) if rankmap_roll and (stream_key in rankmap_roll) and uni_n else None,
                                    "Pct_Since_due": (float(rankmap_since_due.get(stream_key)) / float(uni_n)) if rankmap_since_due and (stream_key in rankmap_since_due) and uni_n else None,
                                    "Pct_Since_recent": (float(rankmap_since_recent.get(stream_key)) / float(uni_n)) if rankmap_since_recent and (stream_key in rankmap_since_recent) and uni_n else None,
                                    "Pct_Since_blend": (float(rankmap_since_blend.get(stream_key)) / float(uni_n)) if rankmap_since_blend and (stream_key in rankmap_since_blend) and uni_n else None,
                                    "Rank_rev": (int(rr["Rank_rev"]) if pd.notna(rr.get("Rank_rev")) else None),
                                    "Percentile_rev": (float(rr["Percentile_rev"]) if pd.notna(rr.get("Percentile_rev")) else None),
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                    "UniverseSize": uni_n,
                                    "InUniverse": in_uni,
                                })
                            else:
                                # Out-of-universe diagnostics
                                if not stream_key:
                                    reason = 'EMPTY_STREAM'
                                elif (gate_no9 and (stream_key in gated_out_keys)):
                                    reason = 'GATED_OUT_SEED_HAS_9'
                                elif (all_keys and (stream_key not in all_keys)):
                                    reason = 'NO_HISTORY_BEFORE_PLAYDATE'
                                elif not rank_keys:
                                    reason = 'UNIVERSE_EMPTY'
                                else:
                                    reason = 'NOT_RANKED'
                                best_match_key = None
                                best_match_stream = None
                                match_score = None
                                if stream_key and rank_keys:
                                    cand = difflib.get_close_matches(stream_key, list(rank_keys), n=1, cutoff=0.60)
                                    if cand:
                                        best_match_key = cand[0]
                                        best_match_stream = rank_display_by_key.get(best_match_key) or latest_display_by_key.get(best_match_key)
                                        match_score = round(difflib.SequenceMatcher(a=stream_key, b=best_match_key).ratio(), 3)
                                prev_seed = seed_by_key.get(stream_key) if stream_key else None
                                prev_seed_has9 = seed_has9_by_key.get(stream_key) if stream_key else None
                                per_event_rows.append({
                                    'PlayDate': d,
                                    'WinningStream': stream_raw,
                                    'WinningStreamKey': stream_key,
                                    'WinningResult': ev.get('result'),
                                    'WinningMember': ev.get('Member'),
                                    'PrevSeed': prev_seed,
                                    'PrevSeedHas9': prev_seed_has9,
                                    'WinnerStreamScore': None,
                                    'WinnerTop1Score': None,
                                    'WinnerBaseGap': None,
                                    'WinnerRoll025_30': None,
                                    'WinnerDrawsSinceLast025': None,
                                    'WinnerTieFired': None,
                                    'WinnerDeadTie': None,
                                    'Rank_Top1Score': None,
                                    'Rank_BaseGap': None,
                                    'Rank_Roll025_30': None,
                                    'Rank_Since_due': None,
                                    'Rank_Since_recent': None,
                                    'Pct_Top1Score': None,
                                    'Pct_BaseGap': None,
                                    'Pct_Roll025_30': None,
                                    'Pct_Since_due': None,
                                    'Pct_Since_recent': None,
                                    'Rank_rev': None,
                                    'Percentile_rev': None,
                                    'Rank': None,
                                    'Percentile': None,
                                    'Tier': None,
                                    'FlipRec': None,
                                    'PlayMember': None,
                                    'UniverseSize': uni_n,
                                    'InUniverse': 0,
                                    'Reason': reason,
                                    'BestMatchStream': best_match_stream,
                                    'MatchScore': match_score,
                                })

                        per_date_rows.append({
                            "PlayDate": d,
                            "UniverseSize": uni_n,
                            "TopStream": top_stream,
                            "TopStreamKey": top_stream_key,
                            "TopStreamScore": top_streamscore,
                            "TopTop1Score": top_top1score,
                            "TopBaseGap": top_basegap,
                            "TopRoll025_30": top_roll30,
                            "TopDrawsSinceLast025": top_since,
                            "TopTieFired": top_tiefired,
                            "TopDeadTie": top_deadtie,
                            "UniqueStreamScore": day_unique_streamscore,
                            "UniqueTop1Score": day_unique_top1score,
                            "TopStreamScoreTieCount": day_top_score_tie,
                            "TopTopTupleTieCount": day_top_tuple_tie,
                            "PreGateUniverseSize": int(len(pre_gate_keys)) if "pre_gate_keys" in locals() else None,
                            "GatedOutStreams": int(len(gated_out_keys)) if "gated_out_keys" in locals() else 0,
                            "TierA_Size": tierA_size,
                            "TierB_Size": tierB_size,
                            "TierC_Size": tierC_size,
                            "HitEvents": int(len(events_by_date[d])),
                            "Covered_TierA": int(covered_A),
                            "Covered_TierAB": int(covered_AB),
                        })

                        prog.progress(int((i+1)/len(play_dates)*100))

                    status.empty()

                    per_event = pd.DataFrame(per_event_rows)
                    per_date = pd.DataFrame(per_date_rows)
                    # ---- Keep Pick-4 digits as 4-char text everywhere (prevents "25" vs "0025" confusion) ----
                    def _pad4(v):
                        if pd.isna(v):
                            return ""
                        s = str(v).strip()
                        # keep only digits (defensive)
                        s = re.sub(r"[^0-9]", "", s)
                        if s == "":
                            return ""
                        return s.zfill(4)

                    for _df in (per_event, per_date):
                        for _c in ["PrevSeed", "WinningResult", "PlayMembers", "PlayMember", "WinningMember"]:
                            if _c in _df.columns:
                                _df[_c] = _df[_c].apply(_pad4)
                                # Excel-safe text version (leading apostrophe forces text)
                                _df[_c + "_text"] = _df[_c].apply(lambda x: ("'" + x) if x else "")
                    # Persist results so downloads don't require re-running after a Streamlit rerun (e.g., clicking download).

                # Build per-stream-per-day table
                if per_stream_rows:
                    per_stream = pd.concat(per_stream_rows, ignore_index=True)
                else:
                    per_stream = pd.DataFrame()
                if not per_stream.empty:
                    per_stream["GateNO9"] = int(gate_no9)
                    per_stream["GateMode"] = "GateON" if gate_no9 else "GateOFF"
                # Ensure per_event has a stable StreamKey column for merges
                if not per_event.empty and "WinningStreamKey" in per_event.columns and "StreamKey" not in per_event.columns:
                    per_event["StreamKey"] = per_event["WinningStreamKey"]
                # Tag outputs with gate mode
                for _df in [per_event, per_date]:
                    if _df is not None and not _df.empty:
                        _df["GateNO9"] = int(gate_no9)
                        _df["GateMode"] = "GateON" if gate_no9 else "GateOFF"
                return per_event, per_date, per_stream

            if run_btn or compare_btn:
                wf_runs = {}
                selected_bins = []

                def _build_run_payload(label: str,
                                       pe_aug: pd.DataFrame,
                                       pd_aug: pd.DataFrame,
                                       ps_sel: pd.DataFrame,
                                       selection_mode: str,
                                       selected_bins: list[str],
                                       selected_ranks: list[int]):
                    sig = wf_sig_base + (label,)
                    return {
                        "sig": sig,
                        "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "gate_mode": label,
                        "selection_mode": str(selection_mode),
                        "cap_streams": int(cap_streams),
                        "bin_size_pct": int(bin_size_pct),
                        "selected_bins": list(selected_bins) if selected_bins else [],
                        "selected_ranks": [int(x) for x in (selected_ranks or [])],
                        "per_event": pe_aug,
                        "per_date": pd_aug,
                        "per_stream": ps_sel,
                    }


                
                if compare_btn:
                    # Run Gate ON first to derive selection rule, then apply same rule to Gate OFF
                    pe_on, pd_on, ps_on = _wf_tier_compute_base(True)

                    selected_bins: list[str] = []
                    selected_ranks: list[int] = []

                    if str(selection_mode).startswith("Percentile"):
                        selected_bins = _wf_suggest_bins_greedy(
                            pe_on, ps_on,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                        )
                        ps_on_sel = _wf_select_cap50_by_bins(
                            ps_on, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        selected_ranks = _wf_suggest_rank_rows_topk(pe_on, cap_streams=int(cap_streams))
                        ps_on_sel = _wf_select_cap50_by_ranks(
                            ps_on, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_on_sel is not None and not ps_on_sel.empty:
                        ps_on_sel["SelectionMode"] = str(selection_mode)

                    pe_on_aug, pd_on_aug = _wf_augment_event_date_with_play50(pe_on, pd_on, ps_on_sel)
                    if not pe_on_aug.empty:
                        pe_on_aug["SelectionMode"] = str(selection_mode)
                    if pd_on_aug is not None and not pd_on_aug.empty:
                        pd_on_aug["SelectionMode"] = str(selection_mode)

                    wf_runs["GateON"] = _build_run_payload(
                        "GateON",
                        pe_on_aug, pd_on_aug, ps_on_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )

                    pe_off, pd_off, ps_off = _wf_tier_compute_base(False)

                    if str(selection_mode).startswith("Percentile"):
                        ps_off_sel = _wf_select_cap50_by_bins(
                            ps_off, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        ps_off_sel = _wf_select_cap50_by_ranks(
                            ps_off, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_off_sel is not None and not ps_off_sel.empty:
                        ps_off_sel["SelectionMode"] = str(selection_mode)

                    pe_off_aug, pd_off_aug = _wf_augment_event_date_with_play50(pe_off, pd_off, ps_off_sel)
                    if not pe_off_aug.empty:
                        pe_off_aug["SelectionMode"] = str(selection_mode)
                    if pd_off_aug is not None and not pd_off_aug.empty:
                        pd_off_aug["SelectionMode"] = str(selection_mode)

                    wf_runs["GateOFF"] = _build_run_payload(
                        "GateOFF",
                        pe_off_aug, pd_off_aug, ps_off_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )


                    # Active view defaults to GateON for diagnostics below
                    per_event = pe_on_aug
                    per_date = pd_on_aug
                    per_stream = ps_on_sel

                    comp = []
                    for lbl in ["GateON", "GateOFF"]:
                        pe_x = wf_runs[lbl]["per_event"]
                        pd_x = wf_runs[lbl]["per_date"]
                        total_ev = int(len(pe_x))
                        in_uni = int(pe_x.get("InUniverse", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        covered_ab = int(pe_x.get("Covered_AB", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        selected_hits = int(pe_x.get("Selected50", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        captured = int(pe_x.get("CapturedMember", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        comp.append({
                            "GateMode": lbl,
                            "HitEvents": total_ev,
                            "InUniverse": in_uni,
                            "Covered_AB": covered_ab,
                            "Selected50_HitStreams": selected_hits,
                            "CapturedMembers": captured,
                            "CaptureRate(Captured/InUniverse)": (captured / in_uni) if in_uni else 0.0,
                            "SelectedBins": ", ".join(selected_bins) if selected_bins else "",
                        })

                    st.subheader("Gate ON vs OFF (NO9) comparison")
                    st.dataframe(pd.DataFrame(comp))

                else:
                    
                    gate_flag = bool(gate_no9)
                    label = "GateON" if gate_flag else "GateOFF"
                    pe_1, pd_1, ps_1 = _wf_tier_compute_base(gate_flag)

                    selected_bins: list[str] = []
                    selected_ranks: list[int] = []

                    if str(selection_mode).startswith("Percentile"):
                        selected_bins = _wf_suggest_bins_greedy(
                            pe_1, ps_1,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                        )
                        ps_sel = _wf_select_cap50_by_bins(
                            ps_1, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        selected_ranks = _wf_suggest_rank_rows_topk(pe_1, cap_streams=int(cap_streams))
                        ps_sel = _wf_select_cap50_by_ranks(
                            ps_1, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_sel is not None and not ps_sel.empty:
                        ps_sel["SelectionMode"] = str(selection_mode)

                    pe_aug, pd_aug = _wf_augment_event_date_with_play50(pe_1, pd_1, ps_sel)
                    if not pe_aug.empty:
                        pe_aug["SelectionMode"] = str(selection_mode)
                    if pd_aug is not None and not pd_aug.empty:
                        pd_aug["SelectionMode"] = str(selection_mode)

                    wf_runs[label] = _build_run_payload(
                        label,
                        pe_aug, pd_aug, ps_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )

                    per_event = pe_aug
                    per_date = pd_aug
                    per_stream = ps_sel

                st.session_state["_wf_tier_cache_runs"] = wf_runs
                st.session_state["_wf_tier_cache"] = {
                    "sig": wf_sig_base + (("GateON" if bool(gate_no9) else "GateOFF") if not compare_btn else "GateON",),
                    "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "per_event": per_event,
                    "per_date": per_date,
                    "per_stream": per_stream,
                    "cap_streams": int(cap_streams),
                    "bin_size_pct": int(bin_size_pct),
                    "selection_mode": str(selection_mode),
                    "selected_bins": list(selected_bins) if "selected_bins" in locals() else [],
                    "selected_ranks": [int(x) for x in (selected_ranks or [])] if "selected_ranks" in locals() else [],
                }


                total_events = int(len(per_event))
                in_universe = int(per_event["InUniverse"].sum())
                covered_A = int((per_event["Tier"] == "A").sum())
                covered_AB = int(per_event["Tier"].isin(["A","B"]).sum())

                st.write("### Summary")
                st.write(f"Hit-events analyzed: **{total_events:,}**  |  Winner streams present in universe: **{in_universe:,}**")
                st.write(f"Tier A coverage: **{covered_A}/{total_events} = {covered_A/total_events*100:.2f}%**")
                st.write(f"Tier A+B coverage: **{covered_AB}/{total_events} = {covered_AB/total_events*100:.2f}%**")

                # --- Gate impact (NO9) ---
                gate_excluded = int((per_event['Reason'] == 'GATED_OUT_SEED_HAS_9').sum()) if 'Reason' in per_event.columns else 0
                no_history = int((per_event['Reason'] == 'NO_HISTORY_BEFORE_PLAYDATE').sum()) if 'Reason' in per_event.columns else 0
                st.write('### Gate impact (NO9)')
                if total_events > 0:
                    st.write(f"Winner events excluded by gate (seed has 9): **{gate_excluded:,}/{total_events:,} = {gate_excluded/total_events*100:.2f}%**")
                    st.write(f"Winner events with no prior history (first-day/new stream): **{no_history:,}/{total_events:,} = {no_history/total_events*100:.2f}%**")
                if 'GatedOutStreams' in per_date.columns and len(per_date) > 0:
                    avg_removed = float(per_date['GatedOutStreams'].mean())
                    med_removed = float(per_date['GatedOutStreams'].median())
                    avg_pre = float(per_date['PreGateUniverseSize'].mean()) if 'PreGateUniverseSize' in per_date.columns else float('nan')
                    avg_post = float(per_date['UniverseSize'].mean()) if 'UniverseSize' in per_date.columns else float('nan')
                    st.write(f"Avg streams removed per day by gate: **{avg_removed:.2f}** (median **{med_removed:.0f}**) | Avg universe size: pre-gate **{avg_pre:.1f}**, post-gate **{avg_post:.1f}**")

                # --- Ranking performance within playable universe (non-gated) ---
                st.write('### Ranking performance within playable universe')
                playable = per_event[(per_event.get('Reason','') == '')].copy() if total_events > 0 else per_event.iloc[0:0].copy()
                n_playable = int(len(playable))
                if n_playable == 0:
                    st.write('No playable (non-gated) winner events in the selected window.')
                else:
                    top1 = int((playable['Rank'] == 1).sum())
                    top2 = int((playable['Rank'] <= 2).sum())
                    top3 = int((playable['Rank'] <= 3).sum())
                    avg_rank = float(playable['Rank'].mean())
                    med_rank = float(playable['Rank'].median())
                    p90_rank = float(playable['Rank'].quantile(0.90))
                    st.write(f"Playable winner events: **{n_playable:,}** | Top1: **{top1}/{n_playable} = {top1/n_playable*100:.2f}%** | Top2: **{top2}/{n_playable} = {top2/n_playable*100:.2f}%** | Top3: **{top3}/{n_playable} = {top3/n_playable*100:.2f}%**")
                    st.write(f"Rank stats (playable winners): avg **{avg_rank:.2f}**, median **{med_rank:.0f}**, 90th pct **{p90_rank:.0f}**")


                st.markdown("---")
                st.subheader("Model sanity + lift + driver audit (LAB-only)")

                # 1) Direction sanity-check: are we ranking the correct end of StreamScore?
                st.write("### 1) Direction sanity-check (normal vs reversed StreamScore ordering)")
                if n_playable == 0:
                    st.info("No playable winner events available for sanity-check in this window.")
                else:
                    # Normal
                    _n = playable.copy()
                    _n = _n.dropna(subset=["Rank"])
                    top1_n = int((_n["Rank"] == 1).sum())
                    top2_n = int((_n["Rank"] <= 2).sum())
                    top3_n = int((_n["Rank"] <= 3).sum())
                    avg_n = float(_n["Rank"].mean())
                    med_n = float(_n["Rank"].median())
                    p90_n = float(_n["Rank"].quantile(0.90))

                    # Reversed
                    _r = playable.copy()
                    _r = _r.dropna(subset=["Rank_rev"])
                    top1_r = int((_r["Rank_rev"] == 1).sum())
                    top2_r = int((_r["Rank_rev"] <= 2).sum())
                    top3_r = int((_r["Rank_rev"] <= 3).sum())
                    avg_r = float(_r["Rank_rev"].mean())
                    med_r = float(_r["Rank_rev"].median())
                    p90_r = float(_r["Rank_rev"].quantile(0.90))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Normal (higher StreamScore = better)**")
                        st.write(f"Top1: **{top1_n}/{n_playable} = {top1_n/n_playable*100:.2f}%**")
                        st.write(f"Top2: **{top2_n}/{n_playable} = {top2_n/n_playable*100:.2f}%**")
                        st.write(f"Top3: **{top3_n}/{n_playable} = {top3_n/n_playable*100:.2f}%**")
                        st.write(f"Rank stats: avg **{avg_n:.2f}**, median **{med_n:.0f}**, p90 **{p90_n:.0f}**")
                    with c2:
                        st.write("**Reversed (lower StreamScore = better)**")
                        st.write(f"Top1: **{top1_r}/{n_playable} = {top1_r/n_playable*100:.2f}%**")
                        st.write(f"Top2: **{top2_r}/{n_playable} = {top2_r/n_playable*100:.2f}%**")
                        st.write(f"Top3: **{top3_r}/{n_playable} = {top3_r/n_playable*100:.2f}%**")
                        st.write(f"Rank stats: avg **{avg_r:.2f}**, median **{med_r:.0f}**, p90 **{p90_r:.0f}**")

                # 2) Lift check: where do playable winners land by StreamScore percentile?
                st.write("### 2) Score lift check (winner placement by StreamScore percentile)")
                if n_playable == 0:
                    st.info("No playable winner events available for lift check in this window.")
                else:
                    def _bin_from_pct(p):
                        if pd.isna(p):
                            return None
                        if p <= 0.20:
                            return "Top 20%"
                        if p <= 0.50:
                            return "20–50%"
                        if p <= 0.80:
                            return "50–80%"
                        return "Bottom 20%"

                    lift = playable.copy()
                    lift["PctBin_Normal"] = lift["Percentile"].apply(_bin_from_pct)
                    lift["PctBin_Reversed"] = lift["Percentile_rev"].apply(_bin_from_pct) if "Percentile_rev" in lift.columns else None

                    def _summ_bin(colname):
                        vc = lift[colname].value_counts(dropna=False)
                        order = ["Top 20%", "20–50%", "50–80%", "Bottom 20%"]
                        out = []
                        for k in order:
                            c = int(vc.get(k, 0))
                            out.append({"Bin": k, "Count": c, "Pct": (c / n_playable * 100.0)})
                        return pd.DataFrame(out)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Normal ordering**")
                        st.dataframe(_summ_bin("PctBin_Normal"), use_container_width=True, height=190)
                    with c2:
                        st.write("**Reversed ordering**")
                        if "Percentile_rev" in lift.columns:
                            st.dataframe(_summ_bin("PctBin_Reversed"), use_container_width=True, height=190)
                        else:
                            st.info("Reversed percentiles not available (this indicates Rank_rev was not recorded).")

                    # Expected random baseline (for context)
                    st.caption("Random baseline (no signal): roughly 20% / 30% / 30% / 20% across the bins.")

                # 3) Driver audit: compare #1 stream vs winner stream on miss events
                st.write("### 3) Driver audit (why the day’s #1 stream beat the winner on miss events)")
                if n_playable == 0:
                    st.info("No playable winner events available for driver audit in this window.")
                else:
                    bad_rank_threshold = st.number_input(
                        "Define a miss as Winner Rank >",
                        min_value=1,
                        max_value=500,
                        value=25,
                        step=1,
                        key="bad_rank_threshold_driver_audit",
                    )

                    miss = playable[(playable["Rank"].notna()) & (playable["Rank"] > bad_rank_threshold)].copy()
                    if miss.empty:
                        st.success("No miss events under this threshold in the selected window.")
                    else:
                        # Join in the day’s #1 stream details from per-date summary
                        join_cols = [
                            "PlayDate", "UniverseSize",
                            "TopStream", "TopStreamKey", "TopStreamScore",
                            "TopTop1Score", "TopBaseGap", "TopRoll025_30", "TopDrawsSinceLast025",
                            "TopTieFired", "TopDeadTie"
                        ]
                        join_cols = [c for c in join_cols if c in per_date.columns]
                        miss = miss.merge(per_date[join_cols], on="PlayDate", how="left")

                        # Helper to compute StreamScore component contributions (matches the stream_score_row formula)
                        def _since_component(x):
                            try:
                                xv = float(x)
                            except Exception:
                                return 0.0
                            since_norm_local = min(max(xv, 0.0), 50.0) / 50.0
                            if since_mode == "recent":
                                return 1.0 - since_norm_local
                            if since_mode == "blend":
                                return abs(since_norm_local - 0.5) * 2.0
                            return since_norm_local


                        def _score_parts(match_strength, gap, roll, since, tie_fired, dead_tie):
                            base = (w_match * float(match_strength)) + (w_gap * float(gap)) + (w_roll * float(roll)) + (w_since * float(_since_component(since)))
                            cooldown = float(cooldown_penalty) if float(since) <= float(cooldown_k) else 0.0
                            tie_pen = 0.0
                            if int(dead_tie) == 1:
                                tie_pen = 0.5
                            elif int(tie_fired) == 1 and float(gap) == 0.0:
                                tie_pen = 0.2
                            return {
                                "match": w_match * float(match_strength),
                                "gap": w_gap * float(gap),
                                "roll": w_roll * float(roll),
                                "since": w_since * float(_since_component(since)),
                                "cooldown_pen": cooldown,
                                "tie_pen": tie_pen,
                                "base_sum": base,
                                "final_calc": base - cooldown - tie_pen,
                            }

                        # Winner parts
                        miss["W_match"] = miss["WinnerTop1Score"]
                        miss["W_gap"] = miss["WinnerBaseGap"]
                        miss["W_roll"] = miss["WinnerRoll025_30"]
                        miss["W_since"] = miss["WinnerDrawsSinceLast025"]
                        miss["W_tiefired"] = miss["WinnerTieFired"]
                        miss["W_deadtie"] = miss["WinnerDeadTie"]

                        # Top parts
                        miss["T_match"] = miss["TopTop1Score"]
                        miss["T_gap"] = miss["TopBaseGap"]
                        miss["T_roll"] = miss["TopRoll025_30"]
                        miss["T_since"] = miss["TopDrawsSinceLast025"]
                        miss["T_tiefired"] = miss["TopTieFired"]
                        miss["T_deadtie"] = miss["TopDeadTie"]

                        # Compute contribution deltas
                        deltas = []
                        for _, row in miss.iterrows():
                            wp = _score_parts(row["W_match"], row["W_gap"], row["W_roll"], row["W_since"], row["W_tiefired"], row["W_deadtie"])
                            tp = _score_parts(row["T_match"], row["T_gap"], row["T_roll"], row["T_since"], row["T_tiefired"], row["T_deadtie"])
                            deltas.append({
                                "delta_match": tp["match"] - wp["match"],
                                "delta_gap": tp["gap"] - wp["gap"],
                                "delta_roll": tp["roll"] - wp["roll"],
                                "delta_since": tp["since"] - wp["since"],
                                "delta_cooldown_pen": tp["cooldown_pen"] - wp["cooldown_pen"],
                                "delta_tie_pen": tp["tie_pen"] - wp["tie_pen"],
                                "delta_streamscore": float(row["TopStreamScore"]) - float(row["WinnerStreamScore"]),
                            })
                        deltas_df = pd.DataFrame(deltas)
                        miss = pd.concat([miss.reset_index(drop=True), deltas_df.reset_index(drop=True)], axis=1)

                        # Which component most often drove the gap?
                        comp_cols = ["delta_match", "delta_gap", "delta_roll", "delta_since", "delta_cooldown_pen", "delta_tie_pen"]
                        miss["PrimaryDriver"] = miss[comp_cols].abs().idxmax(axis=1).str.replace("delta_", "", regex=False)

                        driver_counts = miss["PrimaryDriver"].value_counts().rename_axis("Driver").reset_index(name="Count")
                        driver_counts["Pct"] = (driver_counts["Count"] / len(miss) * 100.0)

                        st.write(f"Miss events in window: **{len(miss):,}** (Winner Rank > {bad_rank_threshold})")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.write("Primary driver frequency:")
                            st.dataframe(driver_counts, use_container_width=True, height=260)
                        with c2:
                            st.write("Average deltas (Top stream minus Winner stream):")
                            avg_d = miss[["delta_streamscore"] + comp_cols].mean().to_frame("AvgDelta").reset_index().rename(columns={"index": "Metric"})
                            st.dataframe(avg_d, use_container_width=True, height=260)

                        st.write("Worst miss events (highest winner ranks):")
                        show_cols = [
                            "PlayDate", "WinningStream", "Rank", "WinnerStreamScore",
                            "TopStream", "TopStreamScore",
                            "delta_streamscore", "PrimaryDriver",
                            "delta_match", "delta_gap", "delta_roll", "delta_since",
                            "delta_cooldown_pen", "delta_tie_pen",
                        ]
                        show_cols = [c for c in show_cols if c in miss.columns]
                        st.dataframe(
                            miss.sort_values(["Rank"], ascending=False)[show_cols].head(60),
                            use_container_width=True,
                            height=520
                        )

                # --- 4) Component lift leaderboard (LAB-only) ---
                st.write("### 4) Component lift leaderboard (winner placement by single component ranks)")
                if playable.empty:
                    st.info("No playable winner events available for component lift (all winners were out-of-universe or gated).")
                else:
                    def _kpi_from_rankcol(_df, _label, _col):
                        if _col not in _df.columns:
                            return None
                        x = pd.to_numeric(_df[_col], errors="coerce").dropna()
                        n = int(len(x))
                        if n == 0:
                            return None
                        top1 = int((x == 1).sum())
                        top2 = int((x <= 2).sum())
                        top3 = int((x <= 3).sum())
                        return {
                            "Component": _label,
                            "N": n,
                            "Top1": top1,
                            "Top1%": (top1 / n * 100.0),
                            "Top2": top2,
                            "Top2%": (top2 / n * 100.0),
                            "Top3": top3,
                            "Top3%": (top3 / n * 100.0),
                            "AvgRank": float(x.mean()),
                            "MedianRank": float(x.median()),
                            "P90Rank": float(x.quantile(0.90)),
                        }

                    comp_rows = []
                    comp_rows.append(_kpi_from_rankcol(playable, "StreamScore (normal)", "Rank"))
                    comp_rows.append(_kpi_from_rankcol(playable, "StreamScore (reversed)", "Rank_rev"))
                    comp_rows.append(_kpi_from_rankcol(playable, "Top1Score only (desc)", "Rank_Top1Score"))
                    comp_rows.append(_kpi_from_rankcol(playable, "BaseGap only (desc)", "Rank_BaseGap"))
                    comp_rows.append(_kpi_from_rankcol(playable, "CadenceRolling025_30 only (desc)", "Rank_Roll025_30"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 only (due: desc)", "Rank_Since_due"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 only (recent: asc)", "Rank_Since_recent"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 SinceBlend (extremes)", "Rank_Since_blend"))
                    comp_rows = [r for r in comp_rows if r is not None]
                    if not comp_rows:
                        st.info("Component lift leaderboard unavailable (missing component rank columns).")
                    else:
                        comp_df = pd.DataFrame(comp_rows)
                        comp_df = comp_df.sort_values(["Top3%", "Top2%", "Top1%"], ascending=False)
                        st.dataframe(comp_df, use_container_width=True, height=320)
                        st.caption("Interpretation: higher Top1/Top2/Top3% indicates the component alone places winners nearer the top. This is LAB-only diagnostics; LIVE ranking remains unchanged.")

                # --- 5) Tie + granularity diagnostics (LAB-only) ---
                st.write("### 5) Tie + granularity diagnostics (how often scores are tied)")
                if per_date.empty or ("UniqueStreamScore" not in per_date.columns):
                    st.info("Tie diagnostics unavailable (missing per-day tie/granularity metrics).")
                else:
                    tie = per_date.copy()
                    tie["TopStreamScoreTieCount"] = pd.to_numeric(tie.get("TopStreamScoreTieCount"), errors="coerce")
                    tie["TopTopTupleTieCount"] = pd.to_numeric(tie.get("TopTopTupleTieCount"), errors="coerce")
                    tie["UniqueStreamScore"] = pd.to_numeric(tie.get("UniqueStreamScore"), errors="coerce")
                    tie["UniqueTop1Score"] = pd.to_numeric(tie.get("UniqueTop1Score"), errors="coerce")
                    tie["TopScoreTied"] = (tie["TopStreamScoreTieCount"].fillna(0) > 1).astype(int)
                    tie["TopTupleTied"] = (tie["TopTopTupleTieCount"].fillna(0) > 1).astype(int)

                    c1, c2 = st.columns(2)
                    with c1:
                        n_days = int(len(tie))
                        st.write(f"Days analyzed: **{n_days:,}**")
                        st.write(f"Days with top StreamScore tie: **{int(tie['TopScoreTied'].sum()):,}/{n_days:,} = {tie['TopScoreTied'].mean()*100.0:.2f}%**")
                        st.write(f"Days with top (StreamScore,Top1Score,Gap) tie: **{int(tie['TopTupleTied'].sum()):,}/{n_days:,} = {tie['TopTupleTied'].mean()*100.0:.2f}%**")
                        st.write(f"Unique StreamScores/day: avg **{tie['UniqueStreamScore'].mean():.1f}**, median **{tie['UniqueStreamScore'].median():.0f}**")
                        st.write(f"Unique Top1Scores/day: avg **{tie['UniqueTop1Score'].mean():.1f}**, median **{tie['UniqueTop1Score'].median():.0f}**")

                    with c2:
                        dist = tie["TopStreamScoreTieCount"].value_counts(dropna=True).sort_index().reset_index()
                        dist.columns = ["TopStreamScoreTieCount", "Days"]
                        st.write("Distribution: how many streams share the day's #1 StreamScore")
                        st.dataframe(dist, use_container_width=True, height=220)

                    st.write("Lowest-granularity days (fewest unique StreamScores):")
                    show_cols = ["PlayDate", "UniverseSize", "UniqueStreamScore", "UniqueTop1Score", "TopStream", "TopStreamScore", "TopStreamScoreTieCount", "TopTopTupleTieCount"]
                    show_cols = [c for c in show_cols if c in tie.columns]
                    st.dataframe(tie.sort_values(["UniqueStreamScore", "UniqueTop1Score"], ascending=True)[show_cols].head(50), use_container_width=True, height=340)
                # Plays per day (average tier sizes)
                if not per_date.empty and per_date["UniverseSize"].sum() > 0:
                    avg_uni = per_date["UniverseSize"].mean()
                    avg_A = per_date["TierA_Size"].mean()
                    avg_AB = (per_date["TierA_Size"] + per_date["TierB_Size"]).mean()
                    st.write(
                        f"Average streams/day in universe: **{avg_uni:.1f}**  |  "
                        f"Tier A plays/day: **{avg_A:.1f}**  |  Tier A+B plays/day: **{avg_AB:.1f}**"
                    )

                st.write("### Winner rank distribution (first 200 rows)")
                st.dataframe(per_event.sort_values(["PlayDate","Rank"], na_position="last").head(200), use_container_width=True, height=420)

            # ============================================================
            # Out-of-universe diagnostics table
            # ============================================================
            st.subheader('Out-of-universe diagnostics (why some winners were not ranked)')
            if isinstance(per_event, pd.DataFrame) and ('InUniverse' in per_event.columns):
                out_diag = per_event[per_event['InUniverse'] == 0].copy()
            else:
                out_diag = per_event.iloc[0:0].copy()

            if out_diag.empty:
                st.success('No out-of-universe winners in this view.')
            else:
                # Ensure key columns exist even for older rows
                if 'WinningStreamKey' not in out_diag.columns:
                    out_diag['WinningStreamKey'] = out_diag['WinningStream'].apply(canon_stream)
                if 'Reason' not in out_diag.columns:
                    out_diag['Reason'] = 'NOT_RANKED'
                if 'BestMatchStream' not in out_diag.columns:
                    out_diag['BestMatchStream'] = None
                if 'MatchScore' not in out_diag.columns:
                    out_diag['MatchScore'] = None

                diag_cols = [
                    'PlayDate', 'WinningStream', 'WinningStreamKey', 'WinningResult', 'WinningMember',
                    'PrevSeed', 'PrevSeedHas9',
                    'UniverseSize', 'Reason', 'BestMatchStream', 'MatchScore'
                ]
                diag_cols = [c for c in diag_cols if c in out_diag.columns]
                st.dataframe(out_diag[diag_cols].sort_values(['Reason','PlayDate'], ascending=[True, False]), use_container_width=True)

                st.caption(
                    'Interpretation: '
                    'GATED_OUT_SEED_HAS_9 = excluded by the core gate (seed contains 9). '
                    'NO_HISTORY_BEFORE_PLAYDATE = stream label/key was not present in history prior to that play date (new stream or label drift). '
                    'UNIVERSE_EMPTY = no ranked streams could be built for that play date.'
                )

                csv_bytes = out_diag[diag_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download out-of-universe diagnostics CSV',
                    data=csv_bytes,
                    file_name=f'core025_out_of_universe_diagnostics_{datetime.date.today().isoformat()}.csv',
                    mime='text/csv',
                    use_container_width=True,
                )


                # Coverage curve by top X% (optional quick view)
                # Ensure total_events exists even after reruns (download clicks cause reruns)
                total_events = int(len(per_event)) if isinstance(per_event, pd.DataFrame) else 0

                st.write("### Coverage curve by top-% cutoff (by Rank)")
                cut_pcts = [5,10,15,20,25,30,40,50]
                curve = []
                for p in cut_pcts:
                    # covered if Rank <= ceil(p% of universe size)
                    cov = 0
                    for _, row in per_event.dropna(subset=["Rank","UniverseSize"]).iterrows():
                        n = int(row["UniverseSize"])
                        k = max(1, int(np.ceil(n * (p/100.0))))
                        if int(row["Rank"]) <= k:
                            cov += 1
                    curve.append({
                        "Top%": p,
                        "Covered": cov,
                        "Total": total_events,
                        "Coverage%": (cov/total_events*100.0) if total_events else 0.0,
                    })
                curve_df = pd.DataFrame(curve)
                st.dataframe(curve_df, use_container_width=True)

                st.markdown("---")
                st.subheader("Walk-forward Row Percentile Map (row-by-row)")
                st.caption(
                    "This is the row-based view: for each winning 025-family hit-event, we record the exact row (rank) "
                    "where the winning stream landed in the full ranked playlist for that day (walk-forward, no cheating). "
                    "Use this to decide: 'Play rows 1..K' instead of percent chunks."
                )

                per_event_in = per_event[(per_event["InUniverse"] == 1) & per_event["Rank"].notna() & per_event["UniverseSize"].notna()].copy()
                if per_event_in.empty:
                    st.warning("No in-universe events to build a row map (all winners were out-of-universe in this window).")
                else:
                    total_in = int(len(per_event_in))
                    max_rank = int(per_event_in["Rank"].max())

                    # Exact row win map (how often the winner is at each row)
                    row_map = (
                        per_event_in.groupby("Rank")
                        .size()
                        .rename("Wins")
                        .reset_index()
                        .sort_values("Rank")
                    )
                    row_map["Win%_of_events"] = row_map["Wins"] / total_events * 100.0
                    row_map["Win%_of_in_universe"] = row_map["Wins"] / total_in * 100.0

                    st.write("### Exact Row Win Map (top 50 rows shown)")
                    st.dataframe(row_map.head(50), use_container_width=True, height=520)

                    # Coverage curve for playing rows 1..K (fixed plays/day = K)
                    st.write("### Coverage curve: play rows 1..K")
                    K_max = int(min(50, max_rank))
                    cov_rows = []
                    wins_by_rank = dict(zip(row_map["Rank"].astype(int), row_map["Wins"].astype(int)))
                    running = 0
                    for k in range(1, K_max + 1):
                        running += wins_by_rank.get(k, 0)
                        cov_rows.append({
                            "TopRows(K)": k,
                            "Covered": int(running),
                            "TotalEvents": int(total_events),
                            "Coverage%": (running / total_events * 100.0) if total_events else 0.0,
                            "Coverage_InUniverse%": (running / total_in * 100.0) if total_in else 0.0,
                        })
                    cov_rows_df = pd.DataFrame(cov_rows)

                    if str(selection_mode).startswith("Best historical"):
                        _suggested_rows = _wf_suggest_rank_rows_topk(per_event, cap_streams=int(cap_streams))
                        st.markdown(
                            "**Suggested top rows (non-contiguous):** " + (", ".join([str(x) for x in _suggested_rows]) if _suggested_rows else "(none)")
                        )
                    st.dataframe(cov_rows_df, use_container_width=True, height=520)

                    # Percentile bin map (smooth view)
                    st.write("### Percentile Bin Map (5% bins)")
                    per_event_in["PctBin"] = (np.floor(per_event_in["Percentile"] * 20) / 20).clip(0, 0.95)
                    per_event_in["PctBinLabel"] = per_event_in["PctBin"].apply(lambda x: f"{int(x*100)}–{int(x*100+5)}%")
                    bin_map = (
                        per_event_in.groupby("PctBinLabel")
                        .size()
                        .rename("Wins")
                        .reset_index()
                    )
                    # Ensure bins are ordered
                    bin_order = [f"{i}–{i+5}%" for i in range(0, 100, 5)]
                    bin_map["PctBinLabel"] = pd.Categorical(bin_map["PctBinLabel"], categories=bin_order, ordered=True)
                    bin_map = bin_map.sort_values("PctBinLabel")
                    bin_map["Win%_of_events"] = bin_map["Wins"] / total_events * 100.0
                    bin_map["Win%_of_in_universe"] = bin_map["Wins"] / total_in * 100.0
                    st.dataframe(bin_map, use_container_width=True, height=520)

                    st.caption(
                        "Tip: If you want a single stable rule, pick a K from the 'rows 1..K' curve. "
                        "Row-based rules keep daily play counts fixed even when the number of streams varies."
                    )

                # Downloads
                st.download_button(
                    "Download per-event tier results (CSV)",
                    data=per_event.to_csv(index=False).encode("utf-8"),
                    file_name="core025_walkforward_tier_per_event.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download per-date tier summary (CSV)",
                    data=per_date.to_csv(index=False).encode("utf-8"),
                    file_name="core025_walkforward_tier_per_date.csv",
                    mime="text/csv",
                )
            if per_stream is not None and not per_stream.empty:
                st.download_button(
                    "Download per-stream (per day)",
                    data=per_stream.to_csv(index=False),
                    file_name="core025_walkforward_tier_per_stream.csv",
                    mime="text/csv",
                )


    # ----------------------------------------------------------------------------------
    # Cached downloads: clicking a Streamlit download button triggers a rerun.
    # Without caching, users would need to re-run the walk-forward each time.
    # This section keeps the last walk-forward tables available for repeated downloads.
    # ----------------------------------------------------------------------------------
    wf_cache = st.session_state.get("_wf_tier_cache")
    if (
        isinstance(wf_cache, dict)
        and isinstance(wf_cache.get("per_event"), pd.DataFrame)
        and isinstance(wf_cache.get("per_date"), pd.DataFrame)
    ):
        with st.expander("Downloads (cached — no need to re-run Walk-forward)", expanded=False):
            st.caption(
                f"Cached from last run at {wf_cache.get('ran_at','(unknown time)')}. "
                "After any rerun (including downloads), these files remain available."
            )

            per_event_c = wf_cache["per_event"]
            per_date_c = wf_cache["per_date"]
            per_stream_c = wf_cache.get("per_stream", pd.DataFrame())

            # Out-of-universe diagnostics (derived)
            if "InUniverse" in per_event_c.columns:
                try:
                    out_diag_c = per_event_c[per_event_c["InUniverse"].astype(int) == 0].copy()
                except Exception:
                    out_diag_c = per_event_c[per_event_c["InUniverse"] == 0].copy()
            else:
                out_diag_c = per_event_c.iloc[0:0].copy()

            # Rebuild Component Lift table (derived) for export convenience
            component_lift_df = None
            try:
                playable_c = per_event_c.copy()
                if "InUniverse" in playable_c.columns:
                    playable_c = playable_c[playable_c["InUniverse"].astype(int) == 1]
                if "Reason" in playable_c.columns:
                    playable_c = playable_c[playable_c["Reason"].fillna("").astype(str).str.strip() == ""]
                components = [
                    ("StreamScore (normal)", "Rank"),
                    ("StreamScore (reversed)", "Rank_rev"),
                    ("Top1Score only (desc)", "Rank_Top1Score"),
                    ("BaseGap only (desc)", "Rank_BaseGap"),
                    ("CadenceRolling025_30 only (desc)", "Rank_Roll025_30"),
                    ("DrawsSinceLast025 only (due: desc)", "Rank_Since_due"),
                    ("DrawsSinceLast025 only (recent: asc)", "Rank_Since_recent"),
                    ("DrawsSinceLast025 SinceBlend (extremes)", "Rank_Since_blend"),
                ]
                rows = []
                N = int(len(playable_c))
                for comp_name, rank_col in components:
                    if rank_col not in playable_c.columns or N == 0:
                        continue
                    r = pd.to_numeric(playable_c[rank_col], errors="coerce")
                    r = r.dropna().astype(int)
                    if len(r) == 0:
                        continue
                    rows.append(
                        {
                            "Component": comp_name,
                            "N": N,
                            "Top1": int((r <= 1).sum()),
                            "Top1%": float((r <= 1).sum() / N * 100.0),
                            "Top2": int((r <= 2).sum()),
                            "Top2%": float((r <= 2).sum() / N * 100.0),
                            "Top3": int((r <= 3).sum()),
                            "Top3%": float((r <= 3).sum() / N * 100.0),
                            "AvgRank": float(r.mean()),
                            "MedianRank": float(r.median()),
                            "P90Rank": float(r.quantile(0.90)),
                        }
                    )
                if rows:
                    component_lift_df = pd.DataFrame(rows).sort_values(
                        by=["Top3%", "Top2%", "Top1%"], ascending=False
                    )
            except Exception:
                component_lift_df = None

            # Rebuild Tie diagnostics summary (derived) for export convenience
            tie_summary_df = None
            try:
                if "TopStreamScoreTieCount" in per_date_c.columns:
                    days_total = int(len(per_date_c))
                    ties = int((pd.to_numeric(per_date_c["TopStreamScoreTieCount"], errors="coerce") > 1).sum())
                    tie_summary_df = pd.DataFrame(
                        [
                            {
                                "DaysAnalyzed": days_total,
                                "DaysWithTopStreamScoreTie": ties,
                                "TieRate%": (ties / days_total * 100.0) if days_total else 0.0,
                                "UniqueStreamScoresPerDay_Avg": float(pd.to_numeric(per_date_c.get("UniqueStreamScore"), errors="coerce").mean())
                                if "UniqueStreamScore" in per_date_c.columns
                                else None,
                                "UniqueStreamScoresPerDay_Median": float(pd.to_numeric(per_date_c.get("UniqueStreamScore"), errors="coerce").median())
                                if "UniqueStreamScore" in per_date_c.columns
                                else None,
                                "UniqueTop1ScoresPerDay_Avg": float(pd.to_numeric(per_date_c.get("UniqueTop1Score"), errors="coerce").mean())
                                if "UniqueTop1Score" in per_date_c.columns
                                else None,
                                "UniqueTop1ScoresPerDay_Median": float(pd.to_numeric(per_date_c.get("UniqueTop1Score"), errors="coerce").median())
                                if "UniqueTop1Score" in per_date_c.columns
                                else None,
                            }
                        ]
                    )
            except Exception:
                tie_summary_df = None

            # Individual table downloads
            st.download_button(
                "Download walk-forward per-event (CSV)",
                data=per_event_c.to_csv(index=False).encode("utf-8"),
                file_name="core025_walkforward_tier_per_event.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download walk-forward per-date (CSV)",
                data=per_date_c.to_csv(index=False).encode("utf-8"),
                file_name="core025_walkforward_tier_per_date.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download out-of-universe diagnostics (CSV)",
                data=out_diag_c.to_csv(index=False).encode("utf-8"),
                file_name="core025_out_of_universe_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if isinstance(component_lift_df, pd.DataFrame) and len(component_lift_df) > 0:
                st.download_button(
                    "Download component lift leaderboard (CSV)",
                    data=component_lift_df.to_csv(index=False).encode("utf-8"),
                    file_name="core025_component_lift_leaderboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if isinstance(tie_summary_df, pd.DataFrame) and len(tie_summary_df) > 0:
                st.download_button(
                    "Download tie diagnostics summary (CSV)",
                    data=tie_summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="core025_tie_diagnostics_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # One-click bundle
            try:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr("core025_walkforward_tier_per_event.csv", per_event_c.to_csv(index=False))
                    z.writestr("core025_walkforward_tier_per_date.csv", per_date_c.to_csv(index=False))
                    if per_stream_c is not None and not per_stream_c.empty:
                        z.writestr("core025_walkforward_tier_per_stream.csv", per_stream_c.to_csv(index=False))
                    z.writestr("core025_out_of_universe_diagnostics.csv", out_diag_c.to_csv(index=False))
                    if isinstance(component_lift_df, pd.DataFrame) and len(component_lift_df) > 0:
                        z.writestr("core025_component_lift_leaderboard.csv", component_lift_df.to_csv(index=False))
                    if isinstance(tie_summary_df, pd.DataFrame) and len(tie_summary_df) > 0:
                        z.writestr("core025_tie_diagnostics_summary.csv", tie_summary_df.to_csv(index=False))
                st.download_button(
                    "Download ALL cached LAB tables (.zip)",
                    data=buf.getvalue(),
                    file_name="core025_lab_tables.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as _zip_e:
                st.info(f"Zip bundle unavailable: {_zip_e}")

            wf_runs_cache = st.session_state.get("_wf_tier_cache_runs", {})
            if isinstance(wf_runs_cache, dict) and wf_runs_cache:
                with st.expander("Downloads (Gate ON vs OFF caches)", expanded=False):
                    st.caption("Available cached runs:")
                    st.write(", ".join(sorted(list(wf_runs_cache.keys()))))
                    # Individual downloads
                    
                    for _lbl, _cache in wf_runs_cache.items():
                        _sel_mode = str(_cache.get("selection_mode", ""))
                        _bins_list = _cache.get("selected_bins", []) or []
                        _ranks_list = _cache.get("selected_ranks", []) or []
                        _rule = ", ".join(_bins_list) if _bins_list else (", ".join([str(x) for x in _ranks_list]) if _ranks_list else "(none)")
                        st.markdown(
                            f"**{_lbl}** — ran_at: {_cache.get('ran_at','')}  |  cap: {_cache.get('cap_streams','')}  |  mode: {_sel_mode}  |  bin: {_cache.get('bin_size_pct','')}  |  rule: {_rule}"
                        )
                        _pe = _cache.get("per_event", pd.DataFrame())
                        _pd = _cache.get("per_date", pd.DataFrame())
                        _ps = _cache.get("per_stream", pd.DataFrame())
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1:
                            st.download_button(
                                f"Download per-event ({_lbl})",
                                data=_pe.to_csv(index=False),
                                file_name=f"core025_walkforward_tier_per_event_{_lbl}.csv",
                                mime="text/csv",
                            )
                        with c2:
                            st.download_button(
                                f"Download per-date ({_lbl})",
                                data=_pd.to_csv(index=False),
                                file_name=f"core025_walkforward_tier_per_date_{_lbl}.csv",
                                mime="text/csv",
                            )
                        with c3:
                            if _ps is not None and not _ps.empty:
                                st.download_button(
                                    f"Download per-stream ({_lbl})",
                                    data=_ps.to_csv(index=False),
                                    file_name=f"core025_walkforward_tier_per_stream_{_lbl}.csv",
                                    mime="text/csv",
                                )

                    # Combined zip of all cached runs
                    try:
                        zbuf2 = io.BytesIO()
                        with zipfile.ZipFile(zbuf2, mode="w", compression=zipfile.ZIP_DEFLATED) as z2:
                            for _lbl, _cache in wf_runs_cache.items():
                                _pe = _cache.get("per_event", pd.DataFrame())
                                _pd = _cache.get("per_date", pd.DataFrame())
                                _ps = _cache.get("per_stream", pd.DataFrame())
                                z2.writestr(f"core025_walkforward_tier_per_event_{_lbl}.csv", _pe.to_csv(index=False))
                                z2.writestr(f"core025_walkforward_tier_per_date_{_lbl}.csv", _pd.to_csv(index=False))
                                if _ps is not None and not _ps.empty:
                                    z2.writestr(f"core025_walkforward_tier_per_stream_{_lbl}.csv", _ps.to_csv(index=False))
                        zbuf2.seek(0)
                        st.download_button(
                            "Download ALL cached runs (zip)",
                            data=zbuf2.getvalue(),
                            file_name="core025_walkforward_gate_runs_bundle.zip",
                            mime="application/zip",
                        )
                    except Exception:
                        st.info("Gate-runs zip bundle unavailable.")
