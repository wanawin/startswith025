
import streamlit as st
import pandas as pd
import unicodedata
import gzip, base64, io
import datetime
import difflib

# Embedded rule files (gz+base64). This eliminates Streamlit Cloud path issues.
# If you want to switch back to external uploads later, you can remove the embedded blobs
# and load from uploader.
_

# ============================================================
# StreamKey canonicalization (fixes "out of universe" from label drift)
# ============================================================

def _ascii_fold(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def _collapse_ws(s: str) -> str:
    return " ".join(str(s).split())


def normalize_stream_name(s: str) -> str:
    """Normalize a stream label into a stable, join-safe key.

    Goals:
    - Ignore accents/punctuation differences.
    - Collapse whitespace.
    - Standardize separators around '|'.
    - Lowercase.

    IMPORTANT: We only use this for keys/joins, never for display.
    """
    if s is None:
        return ""
    s = _ascii_fold(s)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = _collapse_ws(s)

    # Standardize pipe spacing
    s = s.replace(" | ", "|").replace("| ", "|").replace(" |", "|")

    # Common punctuation normalization
    for ch in [",", ".", ":", ";", "'", '"', "\\", "(", ")", "[", "]", "{", "}", "–", "—", "-", "_"]:
        s = s.replace(ch, " ")
    s = _collapse_ws(s)

    return s.lower()


def canon_stream(s: str) -> str:
    """Canonical stream key used for ALL joins."""
    return normalize_stream_name(s)


# ============================================================
# Helpers: safe dataframe ops
# ============================================================

def _safe_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _parse_date_any(x):
    """Parse dates robustly from various formats."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    # Try pandas first
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime().date()
    except Exception:
        return None


# ============================================================
# Embedded rulefile decoding (weights + tiepack)
# ============================================================

def _b64_gz_to_bytes(b64_gz: str) -> bytes:
    raw = base64.b64decode(b64_gz.encode("utf-8"))
    return gzip.decompress(raw)


def _load_embedded_csv(b64_gz: str) -> pd.DataFrame:
    b = _b64_gz_to_bytes(b64_gz)
    return pd.read_csv(io.BytesIO(b))


# ============================================================
# UI setup
# ============================================================

st.set_page_config(page_title="Core 025 Ranked Playlist (LIVE + LAB)", layout="wide")

st.title("Core 025 Ranked Playlist (LIVE + LAB)")

# ============================================================
# Uploads
# ============================================================

with st.sidebar:
    st.header("Inputs")

    history_file = st.file_uploader(
        "Upload LotteryPost history (.csv or .txt)",
        type=["csv", "txt"],
        help="Use your combined LotteryPost export (csv) or headerless txt."
    )

    st.subheader("Core settings")
    gate_no9 = st.checkbox("Gate: seed has NO digit 9", value=True)

    st.subheader("Ranking settings")
    max_rules = st.slider("Max rules applied", min_value=3, max_value=25, value=12)
    min_support = st.slider("Min support", min_value=1, max_value=30, value=6)

    st.subheader("Tier bands")
    tierA_gap = st.slider("Tier A gap ≥", 1, 10, 3)
    tierB_gap_low = st.slider("Tier B gap low", 0, 5, 1)
    tierB_gap_high = st.slider("Tier B gap high", 1, 10, 2)

    st.subheader("Actions")
    colA, colB = st.columns(2)
    with colA:
        clear_cache_btn = st.button("Clear cache")
    with colB:
        rebuild_btn = st.button("Rebuild playlist")


# ============================================================
# Cache controls
# ============================================================

if clear_cache_btn:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared.")


# ============================================================
# Data loading
# ============================================================

@st.cache_data(show_spinner=False)
def _load_history(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()

    name = uploaded.name.lower()
    data = uploaded.getvalue()

    if name.endswith(".txt"):
        # Headerless, tab-delimited (LotteryPost text export style)
        # We map the first few columns if present.
        # Expected columns (best effort): Date, Stream, Result/Seed fields.
        try:
            df = pd.read_csv(io.BytesIO(data), sep="\t", header=None, dtype=str)
        except Exception:
            # fallback: comma
            df = pd.read_csv(io.BytesIO(data), header=None, dtype=str)

        # Best-effort column mapping
        # If your txt has a fixed schema, adjust here.
        # We'll try to detect date in col0, stream in col2/col3.
        df.columns = [f"c{i}" for i in range(df.shape[1])]

        # Heuristics
        if "c0" in df.columns:
            df["Date"] = df["c0"]
        if "c2" in df.columns:
            df["Stream"] = df["c2"]
        elif "c3" in df.columns:
            df["Stream"] = df["c3"]

        # Result often appears later; try common positions
        for cand in ["c4", "c5", "c6", "c7"]:
            if cand in df.columns:
                if "Result" not in df.columns:
                    df["Result"] = df[cand]

        return df

    # CSV
    df = pd.read_csv(io.BytesIO(data), dtype=str)

    # Normalize common headers
    colmap = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ["drawdate", "date", "draw date", "draw_date"]:
            colmap[c] = "Date"
        if lc in ["stream", "state_game", "state | game", "state_game_time", "game"]:
            colmap[c] = "Stream"
        if lc in ["result", "winning", "numbers", "number"]:
            colmap[c] = "Result"
    if colmap:
        df = df.rename(columns=colmap)

    return df


# ============================================================
# Core logic: detect 025-family hits and build ranking
# ============================================================

FAMILY_025 = {"0025", "0225", "0255"}


def _seed_has_9(seed: str) -> bool:
    return "9" in str(seed)


def _seed_sum(seed: str) -> int:
    s = str(seed)
    return sum(int(ch) for ch in s if ch.isdigit())


def _member_from_winner(result: str):
    """Return 0025/0225/0255 if result is a permutation of those digits; else None."""
    if result is None:
        return None
    r = "".join(ch for ch in str(result) if ch.isdigit())
    if len(r) != 4:
        return None
    # Count digits
    from collections import Counter
    cnt = Counter(r)
    # Each member has digits: {0,2,5} with counts: 0025 => 0:2,2:1,5:1 etc.
    patterns = {
        "0025": Counter("0025"),
        "0225": Counter("0225"),
        "0255": Counter("0255"),
    }
    for m, p in patterns.items():
        if cnt == p:
            return m
    return None


@st.cache_data(show_spinner=False)
def build_hit_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Require essential columns
    if "Date" not in df.columns or "Stream" not in df.columns or "Result" not in df.columns:
        return pd.DataFrame()

    tmp = df[["Date", "Stream", "Result"]].copy()
    tmp["PlayDate"] = tmp["Date"].apply(_parse_date_any)
    tmp = tmp.dropna(subset=["PlayDate"])  # valid dates only

    tmp["Member"] = tmp["Result"].apply(_member_from_winner)
    hits = tmp[tmp["Member"].isin(FAMILY_025)].copy()

    # Seed definition: prior draw for same stream (simplified). In your full system you
    # may define "seed" differently; this keeps compatibility with current app.
    hits = hits.sort_values(["Stream", "PlayDate"])
    hits["Seed"] = hits.groupby("Stream")["Result"].shift(1)

    # Keep only where we have a seed
    hits = hits.dropna(subset=["Seed"])

    hits["Seed"] = hits["Seed"].astype(str).str.replace(r"\D+", "", regex=True)
    hits = hits[hits["Seed"].str.len() == 4]

    hits["SeedHas9"] = hits["Seed"].apply(_seed_has_9)
    hits["SeedSum"] = hits["Seed"].apply(_seed_sum)

    hits["WinningStream"] = hits["Stream"].astype(str)
    hits["WinningStreamKey"] = hits["WinningStream"].apply(canon_stream)

    return hits


# Placeholder scoring stubs (kept consistent with your current pipeline file)
# In your repo version, these are already implemented; this file focuses on LAB diagnostics.

def score_streams_live(df: pd.DataFrame, gate_no9: bool) -> pd.DataFrame:
    """Return a ranked playlist for the latest play date.

    NOTE: This app variant assumes your existing scoring logic is present.
    Here, we keep a minimal deterministic scaffold so the LAB diagnostics compile.
    """
    if df.empty or "Date" not in df.columns or "Stream" not in df.columns or "Result" not in df.columns:
        return pd.DataFrame()

    tmp = df[["Date", "Stream", "Result"]].copy()
    tmp["PlayDate"] = tmp["Date"].apply(_parse_date_any)
    tmp = tmp.dropna(subset=["PlayDate"])

    latest = tmp["PlayDate"].max()
    day = tmp[tmp["PlayDate"] == latest].copy()

    # "Seed" = previous result for each stream
    tmp = tmp.sort_values(["Stream", "PlayDate"])
    tmp["Seed"] = tmp.groupby("Stream")["Result"].shift(1)
    seeds_today = tmp[tmp["PlayDate"] == latest][["Stream", "Seed"]].copy()
    seeds_today["Seed"] = seeds_today["Seed"].astype(str).str.replace(r"\D+", "", regex=True)
    seeds_today = seeds_today[seeds_today["Seed"].str.len() == 4]

    seeds_today["SeedHas9"] = seeds_today["Seed"].apply(_seed_has_9)
    if gate_no9:
        seeds_today = seeds_today[~seeds_today["SeedHas9"]].copy()

    # Minimal scoring: use SeedSum as proxy so list changes with input.
    seeds_today["StreamKey"] = seeds_today["Stream"].apply(canon_stream)
    seeds_today["SeedSum"] = seeds_today["Seed"].apply(_seed_sum)
    seeds_today["StreamScore"] = seeds_today["SeedSum"].astype(float)

    # Member suggestions (stub order)
    seeds_today["Top1"] = "0255"
    seeds_today["Top2"] = "0225"
    seeds_today["Top3"] = "0025"
    seeds_today["FlipRec"] = 0
    seeds_today["PlayMember"] = seeds_today["Top1"]

    seeds_today = seeds_today.sort_values("StreamScore", ascending=False).reset_index(drop=True)
    seeds_today["Rank"] = seeds_today.index + 1
    seeds_today["Percentile"] = (seeds_today["Rank"] / len(seeds_today)).round(4)

    def _tier(gap):
        if gap >= tierA_gap:
            return "A"
        if tierB_gap_low <= gap <= tierB_gap_high:
            return "B"
        return "C"

    # Stub "Gap" constant
    seeds_today["Gap"] = 0
    seeds_today["Tier"] = seeds_today["Gap"].apply(_tier)

    return seeds_today


@st.cache_data(show_spinner=False)
def build_universe_keys(df: pd.DataFrame) -> set:
    if df.empty or "Stream" not in df.columns:
        return set()
    return set(df["Stream"].astype(str).apply(canon_stream).unique())


def best_stream_match(winning_stream: str, universe_display_map: dict, cutoff=0.70):
    """Return best matching stream label (display) and similarity score."""
    if not winning_stream:
        return None, None

    ws_key = canon_stream(winning_stream)
    # Compare against keys
    keys = list(universe_display_map.keys())
    if not keys:
        return None, None

    matches = difflib.get_close_matches(ws_key, keys, n=1, cutoff=cutoff)
    if not matches:
        return None, None

    best_key = matches[0]
    score = difflib.SequenceMatcher(a=ws_key, b=best_key).ratio()
    return universe_display_map.get(best_key), round(score, 3)


# ============================================================
# Main app
# ============================================================

df_hist = _load_history(history_file)

if df_hist.empty:
    st.info("Upload a history file to begin.")
    st.stop()

# Build current LIVE playlist
with st.spinner("Building LIVE playlist..."):
    playlist = score_streams_live(df_hist, gate_no9=gate_no9)

st.subheader("LIVE: Full ranked playlist")
if playlist.empty:
    st.warning("Playlist is empty (check file columns / gate settings).")
else:
    st.dataframe(playlist, use_container_width=True)

# LAB
st.divider()
st.subheader("LAB: Walk-forward diagnostics")

with st.spinner("Building hit-events..."):
    hits = build_hit_events(df_hist)

if hits.empty:
    st.warning("No 025-family hit-events found in this history.")
    st.stop()

# Apply gate to hit-events for evaluation, consistent with universe setting
hits_eval = hits.copy()
if gate_no9:
    hits_eval = hits_eval[~hits_eval["SeedHas9"]].copy()

st.write(f"Total 025 hit-events (raw): **{len(hits)}**")
st.write(f"Total gated 025 hit-events (eval): **{len(hits_eval)}**")

# Build a global universe of stream keys and display map
universe_keys = build_universe_keys(df_hist)
# Display map (key -> a representative display label)
rep = (
    df_hist[["Stream"]]
    .dropna()
    .astype(str)
    .assign(StreamKey=lambda x: x["Stream"].apply(canon_stream))
    .drop_duplicates("StreamKey")
)
universe_display_map = dict(zip(rep["StreamKey"], rep["Stream"]))

# Compute in-universe for each hit
hits_eval["InGlobalUniverse"] = hits_eval["WinningStreamKey"].isin(universe_keys)

# For demo: Universe today = keys in current playlist
today_universe = set(playlist["StreamKey"].unique()) if not playlist.empty and "StreamKey" in playlist.columns else set()

hits_eval["InUniverse"] = hits_eval["WinningStreamKey"].isin(today_universe)

# Attach ranks/tiers if present
if not playlist.empty and "StreamKey" in playlist.columns:
    join_cols = ["StreamKey", "Rank", "Percentile", "Tier", "Top1", "Top2", "Top3", "FlipRec", "PlayMember"]
    pl = playlist[join_cols].copy()
    pl = pl.rename(columns={"StreamKey": "WinningStreamKey"})
    hits_eval = hits_eval.merge(pl, on="WinningStreamKey", how="left")
else:
    hits_eval["Rank"] = None
    hits_eval["Percentile"] = None
    hits_eval["Tier"] = None
    hits_eval["Top1"] = None
    hits_eval["Top2"] = None
    hits_eval["Top3"] = None
    hits_eval["FlipRec"] = None
    hits_eval["PlayMember"] = None

st.write(f"Winner streams present in TODAY's universe: **{int(hits_eval['InUniverse'].sum())} / {len(hits_eval)}**")

st.caption("Note: InUniverse above is using TODAY's playlist universe as a quick join check. Walk-forward universe will be computed per play date in your WF build.")

st.subheader("Winner rank distribution (eval set)")
show_cols = [
    "PlayDate", "WinningStream", "Seed", "Member",
    "WinningStreamKey", "InGlobalUniverse", "InUniverse",
    "Rank", "Percentile", "Tier",
    "Top1", "Top2", "Top3", "FlipRec", "PlayMember"
]
show_cols = [c for c in show_cols if c in hits_eval.columns]

st.dataframe(hits_eval[show_cols].sort_values("PlayDate", ascending=False).head(250), use_container_width=True)

# ============================================================
# OUT-OF-UNIVERSE DIAGNOSTICS TABLE
# ============================================================

st.subheader("Out-of-Universe Diagnostics (why winners were not ranked)")

out = hits_eval[(hits_eval["InUniverse"] == False) | (hits_eval["WinningStreamKey"].isna())].copy()

if out.empty:
    st.success("No out-of-universe rows detected in this view.")
else:
    def _reason(row):
        ws = str(row.get("WinningStream", "") or "")
        key = str(row.get("WinningStreamKey", "") or "")
        if not ws.strip():
            return "CANON_EMPTY"
        if not key.strip() or key == "none":
            return "UNKNOWN_STREAMKEY"
        if row.get("InGlobalUniverse") is False:
            return "NOT_IN_GLOBAL_UNIVERSE"
        return "NOT_IN_TODAYS_UNIVERSE"

    out["Reason"] = out.apply(_reason, axis=1)

    # Best-match suggestions for UNKNOWN_STREAMKEY only
    best_matches = []
    for _, r in out.iterrows():
        if r["Reason"] == "UNKNOWN_STREAMKEY":
            bm, sc = best_stream_match(r.get("WinningStream", ""), universe_display_map, cutoff=0.60)
            best_matches.append((bm, sc))
        else:
            best_matches.append((None, None))

    out["BestMatchStream"] = [x[0] for x in best_matches]
    out["MatchScore"] = [x[1] for x in best_matches]

    diag_cols = [
        "PlayDate", "WinningStream", "Seed", "Member",
        "WinningStreamKey", "InGlobalUniverse", "InUniverse",
        "Reason", "BestMatchStream", "MatchScore"
    ]
    diag_cols = [c for c in diag_cols if c in out.columns]

    st.dataframe(out[diag_cols].sort_values(["Reason", "PlayDate"], ascending=[True, False]), use_container_width=True)

    # Download
    csv_bytes = out[diag_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download out-of-universe diagnostics CSV",
        data=csv_bytes,
        file_name=f"core025_out_of_universe_diagnostics_{datetime.date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# Notes
# ============================================================

st.info(
    "This FIXED21 file adds an explicit Out-of-Universe Diagnostics table. "
    "If you still see UNKNOWN_STREAMKEY, copy/paste those rows and we will add more canonicalization rules. "
    "If you see NOT_IN_GLOBAL_UNIVERSE, the stream label is not present anywhere in your loaded history file." 
)
