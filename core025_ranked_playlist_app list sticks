
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Core 025 Ranked Playlist (0025/0225/0255)", layout="wide")

# -----------------------------
# Constants
# -----------------------------
MEMBERS = ["0025", "0225", "0255"]
MEMBER_DIGITS = {
    "0025": [0, 0, 2, 5],
    "0225": [0, 2, 2, 5],
    "0255": [0, 2, 5, 5],
}
PICK_MAP = {25: "0025", 225: "0225", 255: "0255"}

WORST_PAIRS_025 = set(["39", "55", "26", "29", "79"])
MIRROR_MAP = {0:5,5:0,1:6,6:1,2:7,7:2,3:8,8:3,4:9,9:4}
VTRAC_GROUP = {0:'A',5:'A', 1:'B',6:'B', 2:'C',7:'C', 3:'D',8:'D', 4:'E',9:'E'}
PRIMES = set([2,3,5,7])

# -----------------------------
# Parsing + feature helpers
# -----------------------------
def parse_digits_result(s: str):
    m = re.match(r"^\s*(\d)-(\d)-(\d)-(\d)\s*$", str(s))
    if not m:
        return None
    return [int(m.group(i)) for i in range(1, 5)]

def member_from_digits(d):
    sd = sorted(d)
    for m, md in MEMBER_DIGITS.items():
        if sd == sorted(md):
            return m
    return None

def struct_label(digs):
    digs = list(digs)
    counts = sorted([digs.count(i) for i in set(digs)], reverse=True)
    if counts == [2, 2]:
        return "double_double"
    if counts == [2, 1, 1]:
        return "double"
    if counts == [1, 1, 1, 1]:
        return "single"
    if counts == [3, 1]:
        return "triple"
    if counts == [4]:
        return "quad"
    return "other"

def digital_root(n: int) -> int:
    return 0 if n == 0 else 1 + (n - 1) % 9

def seed_has_worstpair_025(digs):
    digs = list(digs)
    for i in range(4):
        for j in range(i + 1, 4):
            p = "".join(sorted([str(digs[i]), str(digs[j])]))
            if p in WORST_PAIRS_025:
                return 1
    return 0

def basic_feats(digs, pref):
    digs = list(digs)
    s = sum(digs)
    sp = max(digs) - min(digs)
    absdiff = abs(digs[0] - digs[-1])
    even_ct = sum(d % 2 == 0 for d in digs)
    high_ct = sum(d >= 5 for d in digs)
    return {
        f"{pref}_sum": s,
        f"{pref}_spread": sp,
        f"{pref}_absdiff": absdiff,
        f"{pref}_sum_lastdigit": s % 10,
        f"{pref}_even_ct": even_ct,
        f"{pref}_high_ct": high_ct,
        f"{pref}_structure": struct_label(digs),
    }

def extra_feats(digs, pref):
    digs = list(digs)
    pairs = set()
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.add("".join(sorted([str(digs[i]), str(digs[j])])))

    s = sum(digs)
    seen = set(digs)

    mir_pairs = 0
    for d in set(digs):
        md = MIRROR_MAP[d]
        if md in seen and d < md:
            mir_pairs += 1

    vg = set(VTRAC_GROUP[d] for d in digs)

    feats = {
        f"{pref}_root": digital_root(s),
        f"{pref}_prime_ct": sum(d in PRIMES for d in digs),
        f"{pref}_mirrorpair_ct": mir_pairs,
        f"{pref}_vtrac_uniq": len(vg),
        f"{pref}_pair_unique_ct": len(pairs),
        f"{pref}_pair_has_06": int("06" in pairs),
        f"{pref}_pair_has_08": int("08" in pairs),
        f"{pref}_pair_has_07": int("07" in pairs),
        f"{pref}_pair_has_25": int("25" in pairs),
    }
    return feats, set(digs), pairs, vg

# -----------------------------
# Scoring helpers
# -----------------------------
def apply_op(x, op, val):
    if op == "==": return x == val
    if op == "!=": return x != val
    if op == "<":  return x < val
    if op == "<=": return x <= val
    if op == ">":  return x > val
    if op == ">=": return x >= val
    return np.zeros(len(x), dtype=bool)

def parse_val(v):
    v = str(v)
    if v in ("True", "False"):
        return (1 if v == "True" else 0), "bool"
    try:
        return int(float(v)), "num"
    except:
        return v, "str"

def tied(scores):
    mx = scores.max(axis=1)
    return (scores == mx[:, None]).sum(axis=1) >= 2

def top3(scores):
    order = np.argsort(-scores, axis=1)
    t1 = np.array([MEMBERS[i] for i in order[:, 0]])
    t2 = np.array([MEMBERS[i] for i in order[:, 1]])
    t3 = np.array([MEMBERS[i] for i in order[:, 2]])
    s1 = scores[np.arange(len(scores)), order[:, 0]]
    s2 = scores[np.arange(len(scores)), order[:, 1]]
    s3 = scores[np.arange(len(scores)), order[:, 2]]
    gap12 = s1 - s2
    return t1, t2, t3, s1, s2, s3, gap12

# -----------------------------
# Ranking + tier
# -----------------------------
def assign_tier(row, tierA_gap, tierB_gap_low, tierB_gap_high):
    gap = int(row["gap12"])
    top2rec = int(row["top2_recommended"])
    dead = int(row["dead_tie_preCR"])
    if dead == 1:
        return "Tier D (forced dead-tie)"
    if (gap >= tierA_gap) and (top2rec == 0):
        return "Tier A (strong)"
    if (gap >= tierB_gap_low) and (gap <= tierB_gap_high):
        return "Tier B (solid)"
    if top2rec == 1 and gap == 0:
        return "Tier C (fragile tie-pack break)"
    if top2rec == 1:
        return "Tier C (uncertain)"
    return "Tier B (solid)"

def reason_string(row):
    parts = [f"gap={int(row['gap12'])}"]
    if int(row["dead_tie_preCR"]) == 1:
        parts.append("dead_tie")
    if int(row["tie_fired"]) == 1:
        parts.append("tie_fired")
    if int(row["top2_recommended"]) == 1:
        parts.append("top2_rec")
    return ", ".join(parts)

# -----------------------------
# Per-member stack builder
# -----------------------------
def build_predicates(df, idx_top2rec):
    cand_cols = [
        "seed_sum","seed_spread","seed_absdiff","seed_high_ct","seed_even_ct",
        "abs_seed_minus_s2_sum","X_digits_overlap_seed_s2","X_pairs_overlap_seed_s2","X_vtrac_overlap_seed_s2",
        "seedX_pair_has_06","seedX_pair_has_08","seedX_pair_has_07","seedX_pair_has_25",
        "seed_has_worstpair_025","seedX_vtrac_uniq","seedX_mirrorpair_ct","seedX_prime_ct","seedX_root",
        "abs_s2_minus_s3_sum",
        "s2_sum","s3_sum","s2_spread","s3_spread","s2_absdiff","s3_absdiff"
    ]
    cand_cols = [c for c in cand_cols if c in df.columns]
    preds = []
    for col in cand_cols:
        s = df.loc[idx_top2rec, col].astype(int)
        if len(s) == 0:
            continue
        common = s.value_counts().head(6).index.tolist()
        for vv in common:
            preds.append((f"{col}=={int(vv)}", lambda D, c=col, v=int(vv): D[c].astype(int) == v))
        qs = [int(s.quantile(q)) for q in [0.2,0.4,0.5,0.6,0.8]]
        for vv in sorted(set(qs + [10,11,12,13,14,15,16,18,20,22,24,25,26,28,30,32,35])):
            preds.append((f"{col}<={int(vv)}", lambda D, c=col, v=int(vv): D[c].astype(int) <= v))
            preds.append((f"{col}>={int(vv)}", lambda D, c=col, v=int(vv): D[c].astype(int) >= v))
    seen = set()
    uniq = []
    for d, fn in preds:
        if d in seen:
            continue
        seen.add(d)
        uniq.append((d, fn))
    return uniq

def optimize_segment(df, preds, seg, base_pick, winner, max_rules=12, min_support=6):
    idx_base = (df["top2_recommended"] == 1) & (df["top1_postCR"] == seg)
    events = int(idx_base.sum())
    if events == 0:
        return [], base_pick.copy(), {"Segment Top1": seg, "Events": 0, "Baseline hits": 0, "Stack hits": 0, "Swaps": 0}

    cur_pick = base_pick.copy()
    idx_np = idx_base.to_numpy()

    def seg_hits(pick):
        return int((winner[idx_np] == pick[idx_np]).sum())

    base_hits = seg_hits(base_pick)
    cur_hits = seg_hits(cur_pick)
    rules = []

    for _ in range(max_rules):
        best = None
        best_pick = None

        for alt_choice in ["top2", "top3"]:
            alt = (df["top2"] if alt_choice == "top2" else df["top3"]).to_numpy()
            for desc, fn in preds:
                mask = (idx_base & fn(df)).fillna(False).to_numpy()
                support = int(mask.sum())
                if support < min_support:
                    continue
                new_pick = cur_pick.copy()
                new_pick[mask] = alt[mask]
                new_hits = seg_hits(new_pick)
                gain = new_hits - cur_hits
                if gain <= 0:
                    continue
                cand = {"alt": alt_choice, "desc": desc, "support": support, "gain": gain, "new_hits": new_hits}
                if (best is None) or (cand["gain"] > best["gain"]) or (cand["gain"] == best["gain"] and cand["new_hits"] > best["new_hits"]):
                    best = cand
                    best_pick = new_pick

        if best is None:
            break

        rules.append(best)
        cur_pick = best_pick
        cur_hits = best["new_hits"]

    swaps = int(((cur_pick != base_pick) & idx_np).sum())
    final_hits = seg_hits(cur_pick)
    met = {"Segment Top1": seg, "Events": events, "Baseline hits": base_hits, "Stack hits": final_hits, "Swaps": swaps}
    return rules, cur_pick, met

# -----------------------------
# UI
# -----------------------------
st.title("Core 025: Full Ranked List + Tier Labels (0025 / 0225 / 0255)")

with st.sidebar:
    st.header("Inputs")
    st.caption("Upload files, or leave blank to use default paths (if available).")
    default_hist_path = "/mnt/data/lotterypost_search_results_UPDATED_combined.txt"
    default_weights_path = "/mnt/data/025_rule_weights_rarityaware_detail_v1.csv"
    default_tiepack_path = "/mnt/data/025_tie_pack_v1.csv"

    hist_up = st.file_uploader("History (tab-delimited TXT/TSV)", type=["txt", "tsv"])
    weights_up = st.file_uploader("Rule Weights CSV", type=["csv"])
    tiepack_up = st.file_uploader("Tie-Pack CSV", type=["csv"])

    st.divider()
    st.subheader("Gate + Stack Settings")
    gate_no9 = st.checkbox("Gate: seed has NO digit 9", value=True)
    max_rules = st.slider("Max override rules per segment", 0, 20, 12, 1)
    min_support = st.slider("Min support per override rule", 1, 30, 6, 1)

    st.divider()
    st.subheader("Tier Label Settings")
    tierA_gap = st.number_input("Tier A: gap12 ≥", 0, 50, 3, 1)
    tierB_gap_low = st.number_input("Tier B: min gap", 0, 50, 1, 1)
    tierB_gap_high = st.number_input("Tier B: max gap", 0, 50, 2, 1)

    st.divider()
    run_btn = st.button("Build ranked playlist", type="primary", use_container_width=True)

def load_df(upload, default_path, read_fn):
    if upload is not None:
        return read_fn(upload), "uploaded"
    return read_fn(default_path), default_path

def read_history(src):
    return pd.read_csv(src, sep="\t", header=None, names=["date_str", "state", "game", "result_raw"], dtype=str)

def read_csv(src):
    return pd.read_csv(src, dtype=str)

if run_btn:
    with st.spinner("Running… computing from your real history + weights + tie-pack"):
        hist_raw, hist_src = load_df(hist_up, default_hist_path, read_history)
        w_raw, w_src = load_df(weights_up, default_weights_path, read_csv)
        tp_raw, tp_src = load_df(tiepack_up, default_tiepack_path, read_csv)

        # Parse digits + dates
        hist_raw["digits"] = hist_raw["result_raw"].apply(parse_digits_result)
        hist = hist_raw[hist_raw["digits"].notna()].copy()
        hist["date"] = pd.to_datetime(hist["date_str"], format="%a, %b %d, %Y", errors="coerce")
        hist = hist[hist["date"].notna()].copy()
        hist["stream"] = hist["state"].str.strip() + " | " + hist["game"].str.strip()
        hist = hist.sort_values(["stream", "date"]).reset_index(drop=True)
        hist["member"] = hist["digits"].apply(member_from_digits)

        for k in [1, 2, 3]:
            hist[f"lag{k}_digits"] = hist.groupby("stream")["digits"].shift(k)

        core = hist[hist["member"].isin(MEMBERS)].copy()
        core = core[core["lag3_digits"].notna()].copy().reset_index(drop=True)

        if gate_no9:
            core = core[core["lag1_digits"].apply(lambda x: 9 not in x)].copy().reset_index(drop=True)

        if len(core) == 0:
            st.error("No core hit-events found after applying your gate/settings.")
            st.stop()

        # Feature table
        rows = []
        for r in core.itertuples(index=False):
            seed_b = basic_feats(r.lag1_digits, "seed")
            s2_b = basic_feats(r.lag2_digits, "s2")
            s3_b = basic_feats(r.lag3_digits, "s3")

            seed_x, seed_set, seed_pairs, seed_vg = extra_feats(r.lag1_digits, "seedX")
            s2_x, s2_set, s2_pairs, s2_vg = extra_feats(r.lag2_digits, "s2X")
            s3_x, s3_set, s3_pairs, s3_vg = extra_feats(r.lag3_digits, "s3X")

            row = {
                "stream": r.stream,
                "date": r.date,
                "winner": r.member,
                "seed_digits": "".join(map(str, r.lag1_digits)),
                "s2_digits": "".join(map(str, r.lag2_digits)),
                "s3_digits": "".join(map(str, r.lag3_digits)),
                "seed_has_worstpair_025": seed_has_worstpair_025(r.lag1_digits),
                "X_digits_overlap_seed_s2": len(seed_set & s2_set),
                "X_pairs_overlap_seed_s2": len(seed_pairs & s2_pairs),
                "X_vtrac_overlap_seed_s2": len(seed_vg & s2_vg),
                "abs_seed_minus_s2_sum": abs(seed_b["seed_sum"] - s2_b["s2_sum"]),
                "abs_s2_minus_s3_sum": abs(s2_b["s2_sum"] - s3_b["s3_sum"]),
            }
            row.update(seed_b); row.update(s2_b); row.update(s3_b)
            row.update(seed_x); row.update(s2_x); row.update(s3_x)
            rows.append(row)

        df = pd.DataFrame(rows).reset_index(drop=True)
        N = len(df)

        # Weights
        w = w_raw.copy()
        w["new_weight"] = pd.to_numeric(w.get("new_weight"), errors="coerce").fillna(0).astype(int)
        w["pick"] = pd.to_numeric(w.get("pick"), errors="coerce").astype(int).map(PICK_MAP)
        w = w[(w["new_weight"] != 0) & w["pick"].isin(MEMBERS)].copy()

        # Tie-pack
        tp = tp_raw.copy()
        tp["weight"] = pd.to_numeric(tp.get("weight"), errors="coerce").fillna(0).astype(int)
        tp["pick"] = pd.to_numeric(tp.get("pick"), errors="coerce").astype(int).map(PICK_MAP)
        tp = tp[tp["pick"].isin(MEMBERS)].copy()

        pick_idx = {m:i for i,m in enumerate(MEMBERS)}

        # Weighted base scoring
        scores = np.zeros((N, len(MEMBERS)), dtype=int)
        for rr in w.itertuples(index=False):
            feat = str(getattr(rr, "feature"))
            if feat not in df.columns:
                continue
            op = str(getattr(rr, "op"))
            val_raw = str(getattr(rr, "value"))
            val, kind = parse_val(val_raw)
            if kind in ("num", "bool"):
                mask = apply_op(df[feat].astype(int).to_numpy(), op, int(val))
            else:
                mask = apply_op(df[feat].astype(str).to_numpy(), op, val)
            scores[mask, pick_idx[str(getattr(rr, "pick"))]] += int(getattr(rr, "new_weight"))

        base_top1, base_top2, base_top3, base_s1, base_s2, base_s3, base_gap = top3(scores)

        # Tie-pack for ties only
        scores_tp = scores.copy()
        tie_fired = np.zeros(N, dtype=int)
        tmask = tied(scores_tp)
        for rr in tp.itertuples(index=False):
            if not tmask.any():
                break
            feat = str(getattr(rr, "feature"))
            if feat not in df.columns:
                continue
            op = str(getattr(rr, "op"))
            val_raw = str(getattr(rr, "value"))
            val, kind = parse_val(val_raw)
            if kind in ("num", "bool"):
                m = apply_op(df[feat].astype(int).to_numpy(), op, int(val))
            else:
                m = apply_op(df[feat].astype(str).to_numpy(), op, val)
            apply = tmask & m
            if apply.any():
                tie_fired[apply] = 1
                scores_tp[apply, pick_idx[str(getattr(rr, "pick"))]] += int(getattr(rr, "weight"))
                tmask = tied(scores_tp)

        top1, top2, top3_, s1, s2, s3, gap12 = top3(scores_tp)

        df["base_gap"] = base_gap
        df["tie_fired"] = tie_fired
        df["dead_tie_preCR"] = (gap12 == 0).astype(int)
        df["top1"] = top1
        df["top2"] = top2
        df["top3"] = top3_
        df["score1"] = s1
        df["score2"] = s2
        df["score3"] = s3
        df["gap12"] = gap12

        # CR1/CR2
        df["top1_postCR"] = df["top1"].copy()
        dt = df["dead_tie_preCR"] == 1
        df.loc[dt & (df["seed_sum"].astype(int) <= 11), "top1_postCR"] = "0225"
        df.loc[dt & (df["seed_sum"].astype(int) > 11), "top1_postCR"] = "0025"

        # Top-2 recommended flag
        df["top2_recommended"] = ((df["base_gap"] == 0) & (df["tie_fired"] == 1) & (df["dead_tie_preCR"] == 0)).astype(int)

        winner = df["winner"].to_numpy()
        base_pick = df["top1_postCR"].to_numpy()

        # Build predicates + per-member stacks
        idx_top2rec = (df["top2_recommended"] == 1)
        preds = build_predicates(df, idx_top2rec)

        rules_by_seg = {}
        pick_by_seg = {}
        met_rows = []
        for seg in MEMBERS:
            rules, seg_pick, met = optimize_segment(df, preds, seg, base_pick, winner, max_rules=max_rules, min_support=min_support)
            rules_by_seg[seg] = rules
            pick_by_seg[seg] = seg_pick
            met_rows.append(met)

        final_pick = base_pick.copy()
        for seg in MEMBERS:
            idx = ((df["top2_recommended"] == 1) & (df["top1_postCR"] == seg)).to_numpy()
            final_pick[idx] = pick_by_seg[seg][idx]
        df["final_pick"] = final_pick

        # Tier + reason
        df["tier"] = df.apply(lambda r: assign_tier(r, tierA_gap, tierB_gap_low, tierB_gap_high), axis=1)
        df["why"] = df.apply(reason_string, axis=1)

        # Ranked playlist sort
        df["uncertainty_penalty"] = (df["dead_tie_preCR"] * 2 + df["top2_recommended"] * 1).astype(int)
        ranked = df.sort_values(by=["gap12","score1","uncertainty_penalty","date"], ascending=[False,False,True,False]).reset_index(drop=True)

        # Metrics
        hits = int((ranked["winner"] == ranked["final_pick"]).sum())
        acc = 100.0 * hits / len(ranked)
        t2n = int((ranked["top2_recommended"] == 1).sum())
        dtn = int((ranked["dead_tie_preCR"] == 1).sum())

        st.success(f"Done. Events={len(ranked):,} | Top-1 hits={hits:,} ({acc:.2f}%) | top2_rec={t2n:,} | dead_tie={dtn:,}")
        st.caption(f"History: {hist_src} | Weights: {w_src} | Tie-pack: {tp_src}")

        show_cols = ["stream","date","seed_digits","winner","final_pick","score1","score2","score3","gap12","top2_recommended","dead_tie_preCR","tie_fired","tier","why"]
        st.subheader("Ranked playlist (full list)")
        st.dataframe(ranked[show_cols], use_container_width=True, height=520)

        st.subheader("Per-member override stacks (impact)")
        met_df = pd.DataFrame(met_rows)
        st.dataframe(met_df, use_container_width=True)

        with st.expander("Show rule stacks"):
            for seg in MEMBERS:
                st.markdown(f"### Top1 segment = `{seg}` (applies only when Top-2 is recommended)")
                rules = rules_by_seg[seg]
                if not rules:
                    st.write("(no improving rules found)")
                    continue
                rtab = pd.DataFrame([{
                    "Step": i+1,
                    "Flip_to": "#2" if r["alt"] == "top2" else "#3",
                    "Condition": r["desc"],
                    "Support": r["support"],
                    "Gain (within segment)": r["gain"],
                } for i, r in enumerate(rules)])
                st.dataframe(rtab, use_container_width=True)

        st.subheader("Download")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes = ranked[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download ranked playlist (CSV)", csv_bytes, f"core025_ranked_playlist_{ts}.csv", "text/csv", use_container_width=True)

        # TXT: copy/paste friendly
        lines = []
        lines.append(f"Core025 ranked playlist — generated {ts}")
        lines.append(f"Gate seed_no9={gate_no9} | max_rules={max_rules} | min_support={min_support}")
        lines.append("")
        for i, r in ranked[show_cols].iterrows():
            lines.append(
                f"{i+1:04d} | {r['stream']} | {r['date'].date()} | seed={r['seed_digits']} | "
                f"pick={r['final_pick']} | winner={r['winner']} | gap={int(r['gap12'])} | {r['tier']} | {r['why']}"
            )
        txt_bytes = ("\n".join(lines)).encode("utf-8")
        st.download_button("Download ranked playlist (TXT)", txt_bytes, f"core025_ranked_playlist_{ts}.txt", "text/plain", use_container_width=True)

else:
    st.info("Click **Build ranked playlist** to generate your full ranked list (no cutoffs) with tier labels.")
