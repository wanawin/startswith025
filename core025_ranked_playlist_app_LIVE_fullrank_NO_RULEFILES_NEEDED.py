
import streamlit as st
import pandas as pd
import gzip, base64, io

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
class Rule:
    feature: str
    op: str
    value: object
    pick: str
    weight: float
    source: str  # "base" or "tie"

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
try:
    # Read CSV or TXT (auto-detect delimiter for TXT; supports headerless 4-col exports)
    filename = getattr(hist_file, "name", "").lower()

    def _read_txt(f):
        # 1) try normal auto-detect with header
        try:
            df0 = pd.read_csv(f, sep=None, engine="python")
        except Exception:
            f.seek(0)
            try:
                df0 = pd.read_csv(f, sep="\t", engine="python")
            except Exception:
                f.seek(0)
                df0 = pd.read_csv(f)
        return df0

    if filename.endswith(".txt"):
        raw = _read_txt(hist_file)

        # If the file has NO header, pandas may treat the first data row as headers.
        # Detect that case: required cols missing AND columns look like actual data values.
        req = {"Draw Date","State","Game","Results"}
        cols = list(raw.columns)

        def _looks_like_date(s):
            s = str(s)
            return bool(re.search(r"(\b\d{4}-\d{2}-\d{2}\b)|(\b[A-Za-z]{3,9}\b.*\b\d{4}\b)", s))

        headerless_suspect = (not req.issubset(set(cols))) and len(cols) in (4,5) and _looks_like_date(cols[0])

        if headerless_suspect:
            # Re-read as headerless with 4 columns.
            hist_file.seek(0)
            # Try auto-detect delimiter again but header=None
            try:
                raw = pd.read_csv(hist_file, sep=None, engine="python", header=None)
            except Exception:
                hist_file.seek(0)
                try:
                    raw = pd.read_csv(hist_file, sep="\t", engine="python", header=None)
                except Exception:
                    hist_file.seek(0)
                    raw = pd.read_csv(hist_file, header=None)

            # If we got 4+ columns, keep first 4
            if raw.shape[1] >= 4:
                raw = raw.iloc[:, :4].copy()
                raw.columns = ["Draw Date","State","Game","Results"]
    else:
        raw = pd.read_csv(hist_file)

    hist = normalize_history(raw)
except Exception as e:
    st.error(f"Failed to read/normalize history: {e}")
    st.stop()

# Diagnostics
st.write(f"Loaded history rows (parsed): {len(hist):,}")
dmin, dmax = hist["Draw Date"].min(), hist["Draw Date"].max()
st.write(f"History date span: {dmin.date()} → {dmax.date()}")
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
hist_cad = compute_cadence_metrics(hist)

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

        st.caption("Full list ranked. Tier A ≈ top 20% by StreamScore; Tier B ≈ next 30%; Tier C ≈ bottom 50% (no cutoff).")
        st.dataframe(out, use_container_width=True, height=650)

        # Export
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download LIVE ranked streams (CSV)", data=csv_bytes, file_name="core025_live_ranked_streams.csv", mime="text/csv")

# ----------------------------
# LAB MODE (historical 025 hit-events)
# ----------------------------
with tab_lab:
    st.subheader("Historical 025 hit-events (seed → next winner member)")

    # Build transitions within stream
    df = hist.copy()
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
