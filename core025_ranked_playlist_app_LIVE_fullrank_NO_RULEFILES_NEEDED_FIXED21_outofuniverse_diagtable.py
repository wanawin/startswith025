
import streamlit as st
import pandas as pd
import unicodedata
import gzip, base64, io
import datetime
import difflib

# Embedded rule files (gz+base64). This eliminates Streamlit Cloud path issues.
_EMBED_WEIGHTS_B64_GZ = """H4sIAN/+m2kC/32QTW4DIQyF9z0LqrCNDSxyi+4R7ZAEdZpEA0muX5Okv1ICkgc9M997+G2/FLMuuR/1uz+YU56PxRzq27vZz1M6l7rZdrNL29pb2uSuvTS6aVNPZZf6kms3r7mVue7KpWOmMvecTi19yWZXzjfSE7JppUwpv7aprtdmtTLOqIij2GcWLcQYHYAX5y2HgENDez3fesHgN2qbWzrvl9YPuS7JqqzQl0WfgTzABAMMAhgtIXKw4oYXIfsIgRktCcSoEugN8uDVy+ni3zbtsJQ8DXa4BcbBdX/3nfgQkYFIQgwU2YNqP+DjR5pz61Pd6CyVT4q+GAzaNSmgSJRwjXxJ6oMnsgTiHdoA+AgII7AW0h+d10Q2SGRhBH83MDJBlEg2OsEI/Ihvb/wxkH/zYLpnYEGsQx+sd0EEvRp8AuaOk4ONAgAA"""
_EMBED_TIEPACK_B64_GZ = """H4sIAN/+m2kC/22MQQqAMBAD774ll1Z69C1ltVtdVCxuq99XQUTEUyAzSbesjMiUy5lLwkZTYSTpRuws/ZAr66DMwVOrQWJE08DAnq35QxbWvZGmlSlcxH1GWmY/keYgveRLqL/CMzX36QHfvAsArQAAAA=="""

def _read_embedded_csv(b64_gz_text: str) -> pd.DataFrame:
    raw = gzip.decompress(base64.b64decode(b64_gz_text.encode("ascii")))
    return pd.read_csv(io.BytesIO(raw))

import numpy as np
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


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


st.set_page_config(page_title="Core 025 — Live Ranked Streams + Lab Backtest", layout="wide")

# ----------------------------
# Robust parsing for Lottery Post "Results"
# ----------------------------
_DIG4_RE = re.compile(r"(\d)[^\d]+(\d)[^\d]+(\d)[^\d]+(\d)")

def extract_4digits(result_text: str) -> Optional[str]:
    """
    Extract first 4 digits from a Lottery Post Results cell, even if it contains modifiers like:
      '0-2-5-2, Wild Ball: 4'
      '0-5-0-2, Fireball: 0'
    Returns '0252' or None.
    """
    if result_text is None:
        return None
    s = str(result_text).strip()
    m = _DIG4_RE.search(s)
    if m:
        return "".join(m.groups())
    digs = re.findall(r"\d", s)
    if len(digs) >= 4:
        return "".join(digs[:4])
    return None

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
               weights_up=None, tie_up=None) -> Tuple[List[Rule], List[Rule]]:
    # Weights: path -> upload -> embedded fallback
    w = None
    try:
        w = pd.read_csv(weights_csv)
    except Exception:
        if weights_up is not None:
            try:
                w = pd.read_csv(weights_up)
            except Exception:
                w = None
    if w is None:
        w = _read_embedded_csv(_EMBED_WEIGHTS_B64_GZ)

    # Tie-pack: path -> upload -> embedded fallback
    t = None
    try:
        t = pd.read_csv(tie_csv)
    except Exception:
        if tie_up is not None:
            try:
                t = pd.read_csv(tie_up)
            except Exception:
                t = None
    if t is None:
        t = _read_embedded_csv(_EMBED_TIEPACK_B64_GZ)

    w = w.copy()
    t = t.copy()

    if "support" in w.columns:
        w = w[w["support"].fillna(0).astype(int) >= int(min_support)].copy()

    if "priority" in w.columns:
        w = w.sort_values(["priority"], ascending=True)

    base_rules: List[Rule] = []
    for _, r in w.head(int(max_rules)).iterrows():
        base_rules.append(Rule.from_row(r))

    if "priority" in t.columns:
        t = t.sort_values(["priority"], ascending=True)

    tie_rules: List[Rule] = []
    for _, r in t.iterrows():
        tie_rules.append(Rule.from_row(r))

    return base_rules, tie_rules

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
    df = hist_wf.copy()
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
                     w_match=1.0, w_gap=0.6, w_roll=0.25, w_since=0.15, cooldown_k=2, cooldown_penalty=1.0) -> float:
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
    since_norm = min(since, 50.0) / 50.0  # 0..1

    score = (w_match * match_strength) + (w_gap * gap) + (w_roll * roll) + (w_since * since_norm)

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
    base_rules, tie_rules = load_rules(weights_csv, tie_csv, max_rules=max_rules, min_support=min_support, weights_up=weights_file_up, tie_up=tiepack_file_up)
except Exception as e:
    st.error(f"Failed to load rules from '{weights_csv}' and '{tie_csv}': {e}")
    st.stop()

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

    # Gate no9
    if gate_no9:
        latest = latest[~latest["seed_result"].astype(str).str.contains("9")].copy()

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
        out["PlayPick"] = np.where(out["FlipRec"] == 1, "TOP2", "TOP1")
        out["PlayMember"] = np.where(out["FlipRec"] == 1, out["Top2"], out["PredictedMember"])

        # Put the play columns right next to the prediction columns for visibility
        _cols = list(out.columns)
        for c in ["PlayPick", "PlayMember", "FlipRec"]:
            if c in _cols:
                _cols.remove(c)
        # Insert after PredictedMember (or after Seed if that col is missing)
        if 'PredictedMember' in _cols:
            _i = _cols.index('PredictedMember') + 1
        elif 'Seed' in _cols:
            _i = _cols.index('Seed') + 1
        else:
            _i = min(5, len(_cols))
        _cols[_i:_i] = ["PlayPick", "PlayMember", "FlipRec"]
        out = out[_cols]

        st.caption("Full list ranked. Tier A ≈ top 20% by StreamScore; Tier B ≈ next 30%; Tier C ≈ bottom 50% (no cutoff). "
           "FlipRec=1 means the app recommends playing TOP2 (PlayMember) instead of TOP1 for that stream.")

        def _style_flip(df):
            # Bold the play columns (and Top2) when FlipRec fires
            def _row_style(row):
                if int(row.get("FlipRec", 0)) == 1:
                    return [
                        "font-weight:700;" if col in ("PlayMember", "PlayPick", "Top2") else ""
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
            run_btn = st.button("Run Walk-forward Tier Study", type="primary", key="run_wf_tier_study")

            if run_btn:
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
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
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
                        latest = latest.rename(columns={"result": "seed_result", "Draw Date": "seed_date"})

                        if gate_no9:
                            latest = latest[~latest["seed_result"].astype(str).str.contains("9")].copy()

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
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
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
                        ranked["PlayMember"] = np.where(ranked["FlipRec"] == 1, ranked["Top2"], ranked["PredictedMember"])

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
                        idx_map = ranked.set_index("StreamKey")[["Rank", "Percentile", "Tier", "FlipRec", "PlayMember"]]
                        for _, ev in events_by_date[d].iterrows():
                            stream_raw = ev.get("Stream") if ev.get("Stream") is not None else ev.get("stream")
                            stream_key = canon_stream(stream_raw)
                            if stream_key in idx_map.index:
                                rr = idx_map.loc[stream_key]
                                t = rr["Tier"]
                                in_uni = 1
                                if t == "A":
                                    covered_A += 1
                                    covered_AB += 1
                                elif t == "B":
                                    covered_AB += 1
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": stream_raw,
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": int(rr["Rank"]),
                                    "Percentile": float(rr["Percentile"]),
                                    "Tier": t,
                                    "FlipRec": int(rr["FlipRec"]),
                                    "PlayMember": rr["PlayMember"],
                                    "UniverseSize": uni_n,
                                    "InUniverse": in_uni,
                                })
                            else:
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": stream_raw,
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": None,
                                    "Percentile": None,
                                    "Tier": None,
                                    "FlipRec": None,
                                    "PlayMember": None,
                                    "UniverseSize": uni_n,
                                    "InUniverse": 0,
                                })

                        per_date_rows.append({
                            "PlayDate": d,
                            "UniverseSize": uni_n,
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

                    total_events = int(len(per_event))
                    in_universe = int(per_event["InUniverse"].sum())
                    covered_A = int((per_event["Tier"] == "A").sum())
                    covered_AB = int(per_event["Tier"].isin(["A","B"]).sum())

                    st.write("### Summary")
                    st.write(f"Hit-events analyzed: **{total_events:,}**  |  Winner streams present in universe: **{in_universe:,}**")
                    st.write(f"Tier A coverage: **{covered_A}/{total_events} = {covered_A/total_events*100:.2f}%**")
                    st.write(f"Tier A+B coverage: **{covered_AB}/{total_events} = {covered_AB/total_events*100:.2f}%**")

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

                    # --- Out-of-universe diagnostics ---
                    st.write("### Out-of-universe diagnostics")
                    st.caption("A hit-event is *out of universe* when the winning stream for that playdate is not present in that playdate's ranked universe. This can be a real schedule gap (stream not drawing that day) or a labeling mismatch.")

                    _hist_stream_series = hist.get("Stream", pd.Series(dtype=str)).astype(str)
                    _global_keys = _hist_stream_series.apply(canon_stream)
                    global_universe_keys = set(k for k in _global_keys.unique().tolist() if k)

                    try:
                        rep_stream_by_key = (
                            pd.DataFrame({"Stream": _hist_stream_series, "StreamKey": _global_keys})
                              .assign(_cnt=1)
                              .groupby(["StreamKey", "Stream"], dropna=False)["_cnt"].sum()
                              .reset_index()
                              .sort_values(["StreamKey", "_cnt"], ascending=[True, False])
                              .drop_duplicates(subset=["StreamKey"], keep="first")
                              .set_index("StreamKey")["Stream"]
                              .to_dict()
                        )
                    except Exception:
                        rep_stream_by_key = {}

                    oou = per_event.copy()
                    if "InUniverse" in oou.columns:
                        oou = oou[oou["InUniverse"] == 0].copy()
                    else:
                        oou = oou.iloc[0:0].copy()

                    if oou.empty:
                        st.info("No out-of-universe hit-events found under the current settings.")
                    else:
                        def _best_match(key: str, candidates):
                            if not key or not candidates:
                                return ("", "", 0.0)
                            best_key = ""
                            best_score = 0.0
                            for c in candidates:
                                r = difflib.SequenceMatcher(a=key, b=c).ratio()
                                if r > best_score:
                                    best_key, best_score = c, r
                            return (best_key, rep_stream_by_key.get(best_key, ""), float(best_score))

                        oou["WinningStreamKey"] = oou.get("WinningStreamKey", "").fillna("").astype(str)
                        oou["InGlobalUniverse"] = oou["WinningStreamKey"].apply(lambda k: 1 if k in global_universe_keys else 0)

                        def _reason(row):
                            k = row.get("WinningStreamKey", "")
                            if not k or k == 'None':
                                return 'CANON_EMPTY'
                            if row.get("InGlobalUniverse", 0) == 0:
                                return 'UNKNOWN_STREAMKEY (likely naming mismatch)'
                            return 'NOT_IN_TODAYS_UNIVERSE (schedule/outside ranked universe)'

                        oou["Reason"] = oou.apply(_reason, axis=1)

                        cand_keys = list(global_universe_keys)
                        bm = oou.apply(lambda r: _best_match(r.get("WinningStreamKey", ""), cand_keys) if r.get("InGlobalUniverse", 0)==0 else ("", "", 0.0), axis=1)
                        oou["BestMatchKey"] = [t[0] for t in bm]
                        oou["BestMatchStream"] = [t[1] for t in bm]
                        oou["MatchScore"] = [t[2] for t in bm]

                        show_cols = []
                        for c in ["PlayDate", "WinningStream", "WinningStreamKey", "Reason", "InGlobalUniverse", "BestMatchStream", "MatchScore", "WinningResult", "WinningMember"]:
                            if c in oou.columns:
                                show_cols.append(c)

                        st.dataframe(oou[show_cols].reset_index(drop=True), use_container_width=True, height=380)

                        try:
                            _oou_csv = oou.to_csv(index=False).encode('utf-8')
                            st.download_button("Download out-of-universe diagnostics (CSV)", data=_oou_csv, file_name="out_of_universe_diagnostics.csv", mime="text/csv")
                        except Exception:
                            pass

                        st.markdown("**Interpretation:**\n\n- **NOT_IN_TODAYS_UNIVERSE** usually means the stream didn't draw on that playdate (so it cannot be ranked that day).\n- **UNKNOWN_STREAMKEY** indicates a stream-label mismatch; use `BestMatchStream` to identify what it should have matched.")


                    # Coverage curve by top X% (optional quick view)
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