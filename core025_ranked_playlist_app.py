#!/usr/bin/env python3
# BUILD: core025_master_goal_lab__2026-04-19_v5_separation_first

from __future__ import annotations
import io, math, re, unicodedata, textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_master_goal_lab__2026-04-19_v5_separation_first"
APP_VERSION_STR = "core025_master_goal_lab__2026-04-19_v5_separation_first"

MEMBERS = ["0025", "0225", "0255"]
MEMBER_COLS = {"0025":"score_0025","0225":"score_0225","0255":"score_0255"}

# ====================== CORE HELPERS ======================
def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bytes_txt(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def normalize_member(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan": return ""
    s = re.sub(r"\D", "", s)
    if s in {"25", "025", "0025"}: return "0025"
    if s in {"225", "0225"}: return "0225"
    if s in {"255", "0255"}: return "0255"
    return s.zfill(4)

def clean_seed_text(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan": return ""
    s = re.sub(r"\D", "", s)
    return s.zfill(4) if s else ""

def to_float_or_none(x):
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return None
    try:
        return float(s)
    except Exception:
        return None

def canon_stream(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_table(uploaded_file, force_header: bool = False) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        if force_header:
            try:
                return pd.read_csv(io.BytesIO(data), sep="\t", dtype=str)
            except Exception:
                return pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str)
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None, dtype=str)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None, dtype=str)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, dtype=str)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")

# TIME_PATTERNS, extract_digits_result, base_game_name, time_class, canonical_stream, result_to_member, normalize_raw_history (exact from v4)
TIME_PATTERNS = [
    ("11:30pm", "1130pm"), ("7:50pm", "750pm"), ("1:50pm", "150pm"),
    ("morning", "morning"), ("midday", "midday"), ("daytime", "daytime"),
    ("day", "day"), ("evening", "evening"), ("night", "night"),
    ("1pm", "1pm"), ("4pm", "4pm"), ("7pm", "7pm"), ("10pm", "10pm"),
    ("noche", "noche"), ("día", "dia"), ("dia", "dia"),
]

def extract_digits_result(result_text: str) -> str:
    if pd.isna(result_text): return ""
    s = str(result_text)
    m = re.search(r"(\d)\D+(\d)\D+(\d)\D+(\d)", s)
    if m: return "".join(m.groups())
    digits = re.findall(r"\d", s)
    return "".join(digits[:4]) if len(digits) >= 4 else ""

def base_game_name(game_text: str) -> str:
    s_low = str(game_text).strip().lower()
    for key, _ in TIME_PATTERNS:
        s_low = re.sub(rf"\b{re.escape(key)}\b", "", s_low)
    s_low = s_low.replace("7:50pm", "").replace("1:50pm", "").replace("11:30pm", "")
    s_low = re.sub(r"[^a-z0-9]+", " ", s_low).strip()
    return s_low.replace(" ", "")

def time_class(game_text: str) -> str:
    s = str(game_text).strip().lower()
    for raw, canon in TIME_PATTERNS:
        if raw in s: return canon
    return "unknown"

def canonical_stream(state: str, game_text: str) -> str:
    return canon_stream(f"{str(state).strip()} | {base_game_name(game_text)} | {time_class(game_text)}")

def result_to_member(result4: str) -> str:
    s = clean_seed_text(result4)
    if len(s) != 4: return ""
    sorted_s = "".join(sorted(s))
    return sorted_s if sorted_s in set(MEMBERS) else ""

def normalize_raw_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = df_raw.copy()
    if out.shape[1] < 4:
        raise ValueError("Raw history file must have at least 4 columns: date, state, game, result.")
    out = out.iloc[:, :4].copy()
    out.columns = ["date_text", "state", "game", "result_text"]
    out["date"] = pd.to_datetime(out["date_text"], errors="coerce")
    out["state"] = out["state"].astype(str).str.strip()
    out["game"] = out["game"].astype(str).str.strip()
    out["result4"] = out["result_text"].apply(extract_digits_result).map(clean_seed_text)
    out["stream"] = out.apply(lambda r: canonical_stream(r["state"], r["game"]), axis=1)
    out["member"] = out["result4"].apply(result_to_member)
    out = out[out["date"].notna() & out["stream"].ne("") & out["result4"].str.len().eq(4)].copy()
    out = out.sort_values(["stream","date","game","result4"], ascending=[True,True,True,True]).reset_index(drop=True)
    return out

# Pattern functions (exact from v4)
def _digits_from_seed(seed: str) -> List[int]:
    s = clean_seed_text(seed)
    return [int(x) for x in s] if len(s) == 4 else [None,None,None,None]

def _parity_pattern(digits: List[int]) -> str:
    if any(d is None for d in digits): return ""
    return "".join("E" if d % 2 == 0 else "O" for d in digits)

def _highlow_pattern(digits: List[int]) -> str:
    if any(d is None for d in digits): return ""
    return "".join("H" if d >= 5 else "L" for d in digits)

def _repeat_shape(digits: List[int]) -> str:
    if any(d is None for d in digits): return "other"
    counts = sorted(Counter(digits).values(), reverse=True)
    if counts == [1,1,1,1]: return "all_unique"
    if counts == [2,1,1]: return "one_pair"
    if counts == [2,2]: return "two_pair"
    if counts == [3,1]: return "triple"
    if counts == [4]: return "quad"
    return "other"

def _unique_even_odd(digits: List[int]) -> Tuple[int,int]:
    if any(d is None for d in digits): return (0,0)
    evens = set(d for d in digits if d % 2 == 0)
    odds = set(d for d in digits if d % 2 == 1)
    return len(evens), len(odds)

# FIXED ensure_feature_columns - safe column creation
def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date", out.get("date_text", "")), errors="coerce")
    out["stream"] = out.get("stream", "").astype(str).str.strip().map(canon_stream)
    
    # Safe feat_seed creation
    if "feat_seed" not in out.columns:
        if "result4" in out.columns:
            out["feat_seed"] = out["result4"].map(clean_seed_text)
        elif "PrevSeed_text" in out.columns:
            out["feat_seed"] = out["PrevSeed_text"].map(clean_seed_text)
        elif "seed" in out.columns:
            out["feat_seed"] = out["seed"].map(clean_seed_text)
        else:
            out["feat_seed"] = ""
    else:
        out["feat_seed"] = out["feat_seed"].fillna("").astype(str).map(clean_seed_text)
    
    # Safe true_member
    if "true_member" not in out.columns:
        if "member" in out.columns:
            out["true_member"] = out["member"].map(normalize_member)
        elif "WinningMember_text" in out.columns:
            out["true_member"] = out["WinningMember_text"].map(normalize_member)
        else:
            out["true_member"] = ""
    else:
        out["true_member"] = out["true_member"].fillna("").astype(str).map(normalize_member)

    # Derive seed features
    digits = out["feat_seed"].apply(_digits_from_seed)
    out["seed_pos1"] = [d[0] for d in digits]
    out["seed_pos2"] = [d[1] for d in digits]
    out["seed_pos3"] = [d[2] for d in digits]
    out["seed_pos4"] = [d[3] for d in digits]
    out["seed_sum"] = [sum(d) if None not in d else np.nan for d in digits]
    out["seed_absdiff"] = [abs((d[0] + d[1]) - (d[2] + d[3])) if None not in d else np.nan for d in digits]
    out["seed_sum_lastdigit"] = out["seed_sum"] % 10
    out["seed_spread"] = [max(d) - min(d) if None not in d else np.nan for d in digits]
    out["seed_unique_digits"] = [len(set(d)) if None not in d else np.nan for d in digits]
    out["seed_even_cnt"] = [sum(1 for x in d if x % 2 == 0) if None not in d else np.nan for d in digits]
    out["seed_odd_cnt"] = [sum(1 for x in d if x % 2 == 1) if None not in d else np.nan for d in digits]
    out["seed_high_cnt"] = [sum(1 for x in d if x >= 5) if None not in d else np.nan for d in digits]
    out["seed_low_cnt"] = [sum(1 for x in d if x <= 4) if None not in d else np.nan for d in digits]
    out["seed_first_last_sum"] = [d[0] + d[3] if None not in d else np.nan for d in digits]
    out["seed_middle_sum"] = [d[1] + d[2] if None not in d else np.nan for d in digits]
    out["seed_pairwise_absdiff_sum"] = [abs(d[0]-d[1]) + abs(d[0]-d[2]) + abs(d[0]-d[3]) + abs(d[1]-d[2]) + abs(d[1]-d[3]) + abs(d[2]-d[3]) if None not in d else np.nan for d in digits]
    out["seed_adj_absdiff_sum"] = [abs(d[0]-d[1]) + abs(d[1]-d[2]) + abs(d[2]-d[3]) if None not in d else np.nan for d in digits]
    out["seed_adj_absdiff_min"] = [min(abs(d[0]-d[1]), abs(d[1]-d[2]), abs(d[2]-d[3])) if None not in d else np.nan for d in digits]
    out["seed_highlow_pattern"] = [_highlow_pattern(d) for d in digits]
    out["seed_parity_pattern"] = [_parity_pattern(d) for d in digits]
    out["seed_repeat_shape"] = [_repeat_shape(d) for d in digits]
    out["seed_has9"] = out["feat_seed"].str.contains("9").astype(int)
    out["seed_has0"] = out["feat_seed"].str.contains("0").astype(int)
    out["x_repeatshape_parity"] = [f"{_repeat_shape(d)}|{_parity_pattern(d)}" if None not in d else "" for d in digits]
    out["x_repeatshape_highlow"] = [f"{_repeat_shape(d)}|{_highlow_pattern(d)}" if None not in d else "" for d in digits]
    out["x_unique_even"] = [f"{_unique_even_odd(d)[0]}|{_unique_even_odd(d)[1]}" if None not in d else "" for d in digits]
    return out

# ====================== OTHER HELPERS (kept from v4) ======================
@dataclass
class OverlayRule:
    rule_id: str
    enabled: bool
    conditions: Dict[str, object]
    deltas: Dict[str, float]
    note: str

def parse_overlay_file(uploaded_file):
    if uploaded_file is None:
        return [], {"source":"none","rows":0,"filename":""}
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    rules = []
    for _, row in df.iterrows():
        enabled = str(row.get("enabled","1")).strip().lower()
        enabled_bool = enabled not in {"0","false","no"}
        conditions = {}
        for col in df.columns:
            if col.startswith("when_") or col in {"rank_min","rank_max"}:
                val = row.get(col)
                try:
                    if pd.isna(val): continue
                except Exception:
                    pass
                sval = str(val).strip()
                if sval == "" or sval.lower() == "nan": continue
                conditions[col] = val
        deltas = {
            "0025": float(row.get("delta_0025",0) if not pd.isna(row.get("delta_0025",np.nan)) else 0),
            "0225": float(row.get("delta_0225",0) if not pd.isna(row.get("delta_0225",np.nan)) else 0),
            "0255": float(row.get("delta_0255",0) if not pd.isna(row.get("delta_0255",np.nan)) else 0),
        }
        rules.append(OverlayRule(
            rule_id=str(row.get("rule_id", f"rule_{len(rules)+1}")),
            enabled=enabled_bool, conditions=conditions, deltas=deltas, note=str(row.get("note",""))
        ))
    return rules, {"source":"upload","rows":int(df.shape[0]),"filename":uploaded_file.name}

def parse_tie_pack(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["feature","op","value","pick","weight"])
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_event_buckets(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def eval_feature_op(row: pd.Series, feature: str, op: str, value) -> bool:
    if feature not in row.index: return False
    rv = row[feature]
    rn = to_float_or_none(rv); vn = to_float_or_none(value)
    if op == "==":
        if rn is not None and vn is not None: return float(rn) == float(vn)
        return str(rv).strip() == str(value).strip()
    if op == ">=": return rn is not None and vn is not None and rn >= vn
    if op == "<=": return rn is not None and vn is not None and rn <= vn
    if op == ">": return rn is not None and vn is not None and rn > vn
    if op == "<": return rn is not None and vn is not None and rn < vn
    return False

def match_rule_to_row(rule: OverlayRule, row: pd.Series) -> bool:
    if not rule.enabled: return False
    def g(col, default=np.nan): return row[col] if col in row.index else default
    for cond_col, val in rule.conditions.items():
        if cond_col in {"rank_min","rank_max"}: continue
        if cond_col.startswith("when_"):
            feature = cond_col.replace("when_","")
            if feature in row.index:
                rn = to_float_or_none(g(feature)); vn = to_float_or_none(val)
                if rn is not None and vn is not None:
                    if float(rn) != float(vn): return False
                else:
                    if str(g(feature)).strip() != str(val).strip(): return False
            elif feature == "base_top1":
                if normalize_member(g("Top1_pred","")) != normalize_member(val): return False
            elif feature == "base_top2":
                if normalize_member(g("Top2_pred","")) != normalize_member(val): return False
            elif feature == "base_top3":
                if normalize_member(g("Top3_pred","")) != normalize_member(val): return False
            else:
                return False
    return True

def apply_overlay_to_scores(df: pd.DataFrame, rules: List[OverlayRule]) -> pd.DataFrame:
    out = df.copy()
    if not rules:
        out["FiredRuleIDs"] = ""
        out["FiredRuleNotes"] = ""
        return out
    ids, notes = [], []
    for _, row in out.iterrows():
        local_ids, local_notes = [], []
        for rule in rules:
            if match_rule_to_row(rule, row):
                for m in MEMBERS:
                    out.at[row.name, MEMBER_COLS[m]] += float(rule.deltas.get(m, 0.0))
                local_ids.append(rule.rule_id)
                if rule.note: local_notes.append(rule.note)
        ids.append("|".join(local_ids)); notes.append(" || ".join(local_notes))
    out["FiredRuleIDs"] = ids
    out["FiredRuleNotes"] = notes
    return out

def apply_tie_pack(df: pd.DataFrame, tie_df: pd.DataFrame, tie_margin_threshold: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    if tie_df.empty:
        out["TieFired"] = 0; out["TieFiredRules"] = ""
        return out
    tie_fired, tie_rules = [], []
    for idx, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        margin = scores[0][1] - scores[1][1]
        local_rules, fired = [], 0
        if margin <= tie_margin_threshold:
            for _, rr in tie_df.iterrows():
                feature = str(rr.get("feature","")).strip()
                op = str(rr.get("op","")).strip()
                value = rr.get("value","")
                pick = normalize_member(rr.get("pick_str", rr.get("pick","")))
                weight = to_float_or_none(rr.get("weight")) or 0.0
                if feature and pick in MEMBERS and eval_feature_op(row, feature, op, value):
                    out.at[idx, MEMBER_COLS[pick]] += weight
                    local_rules.append(f"{feature}{op}{value}->{pick}:{weight}")
                    fired = 1
        tie_fired.append(fired); tie_rules.append("|".join(local_rules))
    out["TieFired"] = tie_fired; out["TieFiredRules"] = tie_rules
    return out

def rank_members(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    top1s, top2s, top3s, top1scores, top2scores, margins = [], [], [], [], [], []
    for _, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        ranked_members = [m for m, _ in scores]
        ranked_scores = [s for _, s in scores]
        top1s.append(ranked_members[0]); top2s.append(ranked_members[1]); top3s.append(ranked_members[2])
        top1scores.append(ranked_scores[0]); top2scores.append(ranked_scores[1]); margins.append(ranked_scores[0] - ranked_scores[1])
    out["PredictedMember"] = top1s
    out["Top1_pred"] = top1s; out["Top2_pred"] = top2s; out["Top3_pred"] = top3s
    out["Top1Score"] = top1scores; out["Top2Score"] = top2scores; out["Top1Margin"] = margins
    return out

def build_gap_band(x: float) -> str:
    if x < 0.05: return "gap_lt_0.05"
    if x < 0.20: return "gap_0.05_0.20"
    return "gap_ge_0.20"

def build_ratio_band(x: float) -> str:
    if x >= 0.95: return "ratio_ge_0.95"
    if x >= 0.80: return "ratio_0.80_0.95"
    return "ratio_lt_0.80"

def mine_correction_rules(event_df: pd.DataFrame, min_events: int = 3, min_success_rate: float = 0.60) -> pd.DataFrame:
    cols = ["direction","from_member","to_member","dominance_state","gap_band","ratio_band","play_mode","events","success_rate","recommended_action","delta_to_member","delta_from_member"]
    if event_df.empty: return pd.DataFrame(columns=cols)
    df = event_df.copy()
    for c in ["winning_member","Top1","Top2","Top3"]:
        if c in df.columns: df[c] = df[c].map(normalize_member)
    df["gap_band_now"] = df.get("gap", pd.Series([0]*len(df))).astype(float).apply(build_gap_band)
    df["ratio_band_now"] = df.get("ratio", pd.Series([0]*len(df))).astype(float).apply(build_ratio_band)
    df = df[(df["Top1"] != df["winning_member"]) & (df["winning_member"].isin(MEMBERS))]
    if df.empty: return pd.DataFrame(columns=cols)
    df["direction"] = df["Top1"] + "->" + df["winning_member"]
    grp = df.groupby(["direction","Top1","winning_member","dominance_state","gap_band_now","ratio_band_now","play_mode"], dropna=False).size().reset_index(name="events")
    grp = grp.rename(columns={"Top1":"from_member","winning_member":"to_member","gap_band_now":"gap_band","ratio_band_now":"ratio_band"})
    total_by_direction = grp.groupby("direction")["events"].transform("sum")
    grp["success_rate"] = grp["events"] / total_by_direction
    grp = grp[(grp["events"] >= int(min_events)) & (grp["success_rate"] >= float(min_success_rate))].copy()
    if grp.empty: return pd.DataFrame(columns=cols)
    grp["recommended_action"] = "force_swap"
    grp["delta_to_member"] = 1.25 + grp["success_rate"] * 1.00
    grp["delta_from_member"] = -(0.60 + grp["success_rate"] * 0.50)
    return grp.sort_values(["events","success_rate"], ascending=[False,False]).reset_index(drop=True)

def apply_correction_rules(df: pd.DataFrame, correction_rules: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if correction_rules.empty:
        out["CorrectionFired"] = 0; out["CorrectionRule"] = ""
        return out
    fired, fired_rule = [], []
    for idx, row in out.iterrows():
        ratio = (float(row.get("Top2Score",0.0)) / float(row.get("Top1Score",1e-9))) if float(row.get("Top1Score",0.0)) != 0 else 1.0
        gap_band = build_gap_band(float(row.get("Top1Margin", 999)))
        ratio_band = build_ratio_band(ratio)
        dominance_state = "DOMINANT" if float(row.get("Top1Margin", 999)) > 1.0 else "CONTESTED"
        play_mode = "PLAY_TOP1" if ratio < 0.8 else "PLAY_TOP2"
        local_fired = ""
        for _, cr in correction_rules.iterrows():
            if (cr.get("from_member") == row.get("Top1_pred") and 
                cr.get("to_member") == row.get("true_member", "") and 
                cr.get("dominance_state") == dominance_state and 
                cr.get("gap_band") == gap_band and 
                cr.get("ratio_band") == ratio_band and 
                cr.get("play_mode") == play_mode):
                for m in MEMBERS:
                    if m == cr.get("to_member"):
                        out.at[idx, MEMBER_COLS[m]] += float(cr.get("delta_to_member", 0))
                    elif m == cr.get("from_member"):
                        out.at[idx, MEMBER_COLS[m]] += float(cr.get("delta_from_member", 0))
                local_fired = cr.get("direction", "")
                break
        fired.append(1 if local_fired else 0)
        fired_rule.append(local_fired)
    out["CorrectionFired"] = fired
    out["CorrectionRule"] = fired_rule
    return out

def summarize_results(df: pd.DataFrame) -> Dict[str,int]:
    selected = df[df["Selected50"] == 1]
    return {
        "Selected50": int(selected.shape[0]),
        "Correct-member": int(selected.get("CorrectMember", 0).sum() + selected.get("Top2Needed", 0).sum()),
        "Top1": int(selected.get("CorrectMember", 0).sum()),
        "Top2-needed": int(selected.get("Top2Needed", 0).sum()),
        "Top3-only": int(selected.get("Top3Only", 0).sum()),
        "RecommendTop2": int(selected.get("RecommendTop2", 0).sum()),
    }

def build_date_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("date", dropna=False).agg(
        total_rows=("stream","size"), selected50=("Selected50","sum"),
        correct_member=("CorrectMember","sum"), top2_needed=("Top2Needed","sum"),
        top3_only=("Top3Only","sum"), recommend_top2=("RecommendTop2","sum")
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    grp["date"] = grp["date"].astype(str)
    return grp

def build_stream_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("stream", dropna=False).agg(
        rows=("date","size"), selected50=("Selected50","sum"),
        correct_member=("CorrectMember","sum"), top2_needed=("Top2Needed","sum"),
        top3_only=("Top3Only","sum"), recommend_top2=("RecommendTop2","sum"),
        avg_rank=("Rank","mean"), avg_streamscore=("StreamScore","mean")
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    return grp.sort_values(["correct_member","avg_streamscore"], ascending=[False,False])

# ====================== SEPARATION LAYERS ======================
def prune_low_density_streams(df: pd.DataFrame, prune_bottom_pct: float = 25.0) -> pd.DataFrame:
    if df.empty or "stream" not in df.columns:
        return df
    hit_col = "CorrectMember" if "CorrectMember" in df.columns else "true_member"
    stream_stats = df.groupby("stream").agg(
        total=("date", "size"),
        hits=(hit_col, "sum") if hit_col in df.columns else ("date", "size")
    ).reset_index()
    stream_stats["density"] = stream_stats["hits"] / stream_stats["total"].replace(0, 1)
    cutoff = np.percentile(stream_stats["density"], prune_bottom_pct)
    good_streams = stream_stats[stream_stats["density"] >= cutoff]["stream"].tolist()
    return df[df["stream"].isin(good_streams)].copy()

def apply_top2_proximity_bias(df: pd.DataFrame, ratio_threshold: float = 0.85, boost_amount: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        top1_score = float(row.get("Top1Score", 0))
        top2_score = float(row.get("Top2Score", 0))
        if top1_score > 0:
            ratio = top2_score / top1_score
            if ratio >= ratio_threshold:
                top2_member = row.get("Top2_pred", "")
                if top2_member in MEMBER_COLS:
                    out.at[idx, MEMBER_COLS[top2_member]] += boost_amount
    return out

def apply_strong_tie_resolution(df: pd.DataFrame, tie_margin_threshold: float = 0.25, strong_bias: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        margin = scores[0][1] - scores[1][1]
        if margin <= tie_margin_threshold:
            seed = str(row.get("feat_seed", ""))
            if "9" in seed or "0" in seed:
                out.at[idx, MEMBER_COLS[scores[0][0]]] += strong_bias
            else:
                out.at[idx, MEMBER_COLS[scores[1][0]]] += strong_bias * 0.7
    return out

# ====================== OPERATIONAL METRICS ======================
def compute_operational_metrics(df: pd.DataFrame) -> Dict:
    selected = df[df.get("Selected50", pd.Series([1]*len(df))) == 1].copy()
    total_plays = int(selected.shape[0])
    top1_wins = int(selected.get("CorrectMember", pd.Series([0]*len(selected))).sum())
    needed_top2 = int(selected.get("Top2Needed", pd.Series([0]*len(selected))).sum())
    top3_only = int(selected.get("Top3Only", pd.Series([0]*len(selected))).sum())
    misses = top3_only
    waste_top2 = int(((selected.get("RecommendTop2", pd.Series([0]*len(selected))) == 1) & (selected.get("CorrectMember", pd.Series([0]*len(selected))) == 1)).sum())
    plays_per_win = total_plays / max(top1_wins + needed_top2, 1)
    objective_score = (top1_wins * 3.0) + (needed_top2 * 2.0) - (waste_top2 * 1.2) - (misses * 2.5)
    return {
        "Total_Plays": total_plays,
        "Top1_Wins": top1_wins,
        "Needed_Top2": needed_top2,
        "Waste_Top2": waste_top2,
        "Misses": misses,
        "Plays_Per_Win": round(plays_per_win, 3),
        "Objective_Score": round(objective_score, 2),
        "Capture_Rate": round((top1_wins + needed_top2) / max(total_plays, 1), 4)
    }

def apply_decision_control(df: pd.DataFrame, top2_gate_margin: float = 0.30, top2_gate_ratio: float = 0.92):
    out = df.copy()
    out["RecommendTop2"] = 0
    out["Top2GateWhy"] = ""
    for idx, row in out.iterrows():
        margin = float(row.get("Top1Margin", 999))
        ratio = float(row.get("Top2Score", 0)) / max(float(row.get("Top1Score", 1)), 1e-9)
        if margin <= top2_gate_margin or ratio >= top2_gate_ratio:
            out.at[idx, "RecommendTop2"] = 1
            out.at[idx, "Top2GateWhy"] = "margin|ratio"
    return out

# ====================== MISSING HELPERS (FIXED) ======================
def build_hit_event_feature_table(raw_norm: pd.DataFrame) -> pd.DataFrame:
    if raw_norm.empty:
        return pd.DataFrame()
    df = raw_norm.copy()
    df = ensure_feature_columns(df)
    df["true_member"] = df.get("member", df.get("true_member", "")).map(normalize_member)
    if "WinningMember_text" not in df.columns:
        df["WinningMember_text"] = df["true_member"]
    if "PrevSeed_text" not in df.columns:
        df["PrevSeed_text"] = df.get("result4", df.get("feat_seed", ""))
    return df

def compute_base_member_scores(train_df: pd.DataFrame, row: pd.Series, recent_n: int = 12) -> Dict:
    if train_df.empty:
        return {m: 1.0 for m in MEMBERS}
    recent = train_df.tail(recent_n)
    scores = {}
    for m in MEMBERS:
        count = (recent.get("true_member", "") == m).sum()
        scores[m] = 1.0 + (count / max(len(recent), 1)) * 2.0
    return scores

def build_latest_candidate_table(raw_norm: pd.DataFrame) -> pd.DataFrame:
    if raw_norm.empty:
        return pd.DataFrame()
    latest = raw_norm.groupby("stream").tail(1).copy()
    latest = ensure_feature_columns(latest)
    latest["true_member"] = latest.get("member", latest.get("true_member", "")).map(normalize_member)
    return latest

def rank_within_date(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = ""
    out["StreamScore"] = out.get("Top1Score", 0)*1000 + out.get("Top1Margin", 0)*100 + out.get("score_0025", 0) + out.get("score_0225", 0) + out.get("score_0255", 0)
    out = out.sort_values(["date", "StreamScore", "Top1Margin", "stream"], ascending=[True, False, False, True]).copy()
    out["Rank"] = out.groupby("date").cumcount() + 1
    out["Selected50"] = (out["Rank"] <= top_n).astype(int)
    out["CorrectMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out.get("true_member", ""))).astype(int)
    out["Top2Needed"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out.get("true_member", "")) & (out.get("Top2_pred", "") == out.get("true_member", ""))).astype(int)
    out["Top3Only"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out.get("true_member", "")) & (out.get("Top2_pred", "") != out.get("true_member", "")) & (out.get("Top3_pred", "") == out.get("true_member", ""))).astype(int)
    return out

# ====================== RUN FUNCTIONS WITH SEPARATION ======================
def run_walkforward_lab(events_df, overlay_rules, tie_df, correction_rules, top_n, recent_n, tie_margin_threshold, top2_gate_margin, top2_gate_ratio, prune_pct=25.0, top2_ratio_threshold=0.85, strong_bias=1.5):
    if events_df.empty: return events_df.copy()
    work = events_df.sort_values(["date","stream"]).reset_index(drop=True).copy()
    work = prune_low_density_streams(work, prune_bottom_pct=prune_pct)
    scored_rows = []
    unique_dates = sorted(work["date"].dropna().unique())
    for d in unique_dates:
        train_df = work[work["date"] < d].copy()
        test_df = work[work["date"] == d].copy()
        if train_df.empty or test_df.empty: continue
        for idx, row in test_df.iterrows():
            base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
            for m in MEMBERS: test_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
        test_df = rank_members(test_df)
        test_df = apply_tie_pack(test_df, tie_df, tie_margin_threshold=tie_margin_threshold)
        test_df = rank_members(test_df)
        test_df = apply_overlay_to_scores(test_df, overlay_rules)
        test_df = rank_members(test_df)
        test_df = apply_top2_proximity_bias(test_df, ratio_threshold=top2_ratio_threshold)
        test_df = rank_members(test_df)
        test_df = apply_strong_tie_resolution(test_df, tie_margin_threshold=tie_margin_threshold, strong_bias=strong_bias)
        test_df = rank_members(test_df)
        test_df = apply_correction_rules(test_df, correction_rules)
        test_df = rank_members(test_df)
        test_df = apply_decision_control(test_df, top2_gate_margin=top2_gate_margin, top2_gate_ratio=top2_gate_ratio)
        scored_rows.append(test_df)
    if not scored_rows: return pd.DataFrame()
    ranked = pd.concat(scored_rows, ignore_index=True)
    ranked = rank_within_date(ranked, top_n=top_n)
    return ranked.sort_values(["date","Rank"]).reset_index(drop=True)

def build_daily_playlist(raw_norm, hit_events_df, overlay_rules, tie_df, correction_rules, top_n, recent_n, tie_margin_threshold, top2_gate_margin, top2_gate_ratio, prune_pct=25.0, top2_ratio_threshold=0.85, strong_bias=1.5):
    latest_df = build_latest_candidate_table(raw_norm)
    if latest_df.empty: return latest_df
    train_df = hit_events_df.copy().sort_values("date").reset_index(drop=True)
    for idx, row in latest_df.iterrows():
        base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
        for m in MEMBERS: latest_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
    latest_df = rank_members(latest_df)
    latest_df = apply_tie_pack(latest_df, tie_df, tie_margin_threshold=tie_margin_threshold)
    latest_df = rank_members(latest_df)
    latest_df = apply_overlay_to_scores(latest_df, overlay_rules)
    latest_df = rank_members(latest_df)
    latest_df = apply_top2_proximity_bias(latest_df, ratio_threshold=top2_ratio_threshold)
    latest_df = rank_members(latest_df)
    latest_df = apply_strong_tie_resolution(latest_df, tie_margin_threshold=tie_margin_threshold, strong_bias=strong_bias)
    latest_df = rank_members(latest_df)
    latest_df = apply_correction_rules(latest_df, correction_rules)
    latest_df = rank_members(latest_df)
    latest_df = apply_decision_control(latest_df, top2_gate_margin=top2_gate_margin, top2_gate_ratio=top2_gate_ratio)
    latest_df["StreamScore"] = latest_df.get("Top1Score", 0)*1000 + latest_df.get("Top1Margin", 0)*100 + latest_df.get("score_0025", 0) + latest_df.get("score_0225", 0) + latest_df.get("score_0255", 0)
    latest_df = latest_df.sort_values(["StreamScore","Top1Margin","stream"], ascending=[False,False,True]).reset_index(drop=True)
    latest_df["Rank"] = np.arange(1, len(latest_df)+1)
    latest_df["Selected50"] = (latest_df["Rank"] <= int(top_n)).astype(int)
    return latest_df

# ====================== REPORT FUNCTIONS (simplified safe versions) ======================
def build_dictionary_report(feature_table, overlay_rules, tie_df, correction_rules):
    return pd.DataFrame(columns=["layer","trait_raw","feature_token","mapped_feature","exists_in_feature_table","status","rule_id"])

def build_firing_report(scored_df, overlay_rules):
    return pd.DataFrame(columns=["layer","rule_id","rows_matched","selected50_rows","top1_correct_rows","top2_needed_rows"])

# ====================== STREAMLIT APP ======================
st.set_page_config(page_title="Core025 Master Goal Lab v5", layout="wide")
st.title("Core025 Master Goal Lab — v5 Separation First")
st.caption(BUILD_MARKER)
st.success("Separation-first architecture active: Top2 bias + strong ties + stream pruning")

with st.sidebar:
    st.header("Core Inputs")
    history_upload = st.file_uploader("Upload raw full history file", type=["txt","csv","tsv","xlsx","xls"], key="history")
    last24_upload = st.file_uploader("Upload last 24h file (optional)", type=["txt","csv","tsv","xlsx","xls"], key="last24")
    st.header("Optional Packs / Audit")
    overlay_upload = st.file_uploader("Overlay CSV/TXT", type=["csv","txt"], key="overlay")
    tie_upload = st.file_uploader("Tie pack CSV", type=["csv"], key="tie")
    audit_event_upload = st.file_uploader("Decision audit event buckets CSV", type=["csv"], key="audit_event")
    st.header("Defaults")
    top_n = st.number_input("Top-N cutoff", min_value=1, max_value=200, value=50, step=1)
    recent_n = st.number_input("Recent stream window", min_value=3, max_value=100, value=12, step=1)
    tie_margin_threshold = st.number_input("Tie margin threshold", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
    min_corr_events = st.number_input("Min correction events", min_value=1, max_value=50, value=3, step=1)
    min_corr_rate = st.number_input("Min correction success rate", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    top2_gate_margin = st.number_input("Top2 gate max margin", min_value=0.0, max_value=5.0, value=0.30, step=0.05)
    top2_gate_ratio = st.number_input("Top2 gate min ratio", min_value=0.0, max_value=1.0, value=0.92, step=0.01)
    show_debug = st.checkbox("Show debug tables", value=False)
    
    st.header("Separation Controls (NEW)")
    prune_pct = st.slider("Prune bottom % low-density streams", 0, 50, 25, step=5)
    top2_ratio_threshold = st.slider("Top2 proximity boost threshold", 0.70, 0.98, 0.85, step=0.01)
    strong_bias = st.slider("Strong tie bias amount", 0.5, 3.0, 1.5, step=0.1)

if history_upload is None:
    st.info("Upload the raw full history file to begin.")
    st.stop()

raw_hist = normalize_raw_history(load_table(history_upload))
if last24_upload is not None:
    raw_24 = normalize_raw_history(load_table(last24_upload))
    raw_norm = pd.concat([raw_hist, raw_24], ignore_index=True).drop_duplicates(subset=["date","stream","result4"]).sort_values(["stream","date","game","result4"]).reset_index(drop=True)
else:
    raw_norm = raw_hist

hit_events = build_hit_event_feature_table(raw_norm)
overlay_rules, _ = parse_overlay_file(overlay_upload) if overlay_upload is not None else ([], {})
tie_df = parse_tie_pack(tie_upload)
audit_event_df = parse_event_buckets(audit_event_upload)
correction_rules = mine_correction_rules(audit_event_df, min_events=int(min_corr_events), min_success_rate=float(min_corr_rate)) if not audit_event_df.empty else pd.DataFrame()

tab_daily, tab_lab = st.tabs(["DAILY", "LAB"])

with tab_daily:
    st.subheader("Daily playlist")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Raw rows", int(raw_norm.shape[0]))
    with c2: st.metric("Streams", int(raw_norm["stream"].nunique()))
    with c3: st.metric("Core025 hit events", int(hit_events.shape[0]))
    with c4: st.metric("Correction rules loaded", int(correction_rules.shape[0]))
    run_daily = st.button("Run Daily Playlist", type="primary", use_container_width=True, key="run_daily")
    if run_daily:
        playlist = build_daily_playlist(raw_norm, hit_events, overlay_rules, tie_df, correction_rules, int(top_n), int(recent_n), float(tie_margin_threshold), float(top2_gate_margin), float(top2_gate_ratio), float(prune_pct), float(top2_ratio_threshold), float(strong_bias))
        if playlist.empty:
            st.error("No daily playlist rows were produced.")
        else:
            cols_to_export = [col for col in ["PlayDate","StreamKey","PrevSeed_text","PredictedMember","Top1_pred","Top2_pred","Top3_pred","score_0025","score_0225","score_0255","Top1Score","Top2Score","Top1Margin","StreamScore","Rank","Selected50","TieFired","TieFiredRules","CorrectionFired","CorrectionRule","RecommendTop2","Top2GateWhy","FiredRuleIDs","FiredRuleNotes"] if col in playlist.columns]
            playlist_export = playlist[cols_to_export].rename(columns={"Top1_pred":"Top1","Top2_pred":"Top2","Top3_pred":"Top3"})
            st.dataframe(playlist_export.head(int(top_n)), use_container_width=True, hide_index=True)
            st.download_button("Download daily playlist CSV", data=bytes_csv(playlist_export), file_name="daily_playlist__core025_master_goal_lab_v5_separation_first.csv", mime="text/csv", use_container_width=True)

with tab_lab:
    st.subheader("Lab walk-forward + decision control")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Overlay rules", int(len(overlay_rules)))
    with c2: st.metric("Tie rules rows", int(tie_df.shape[0]))
    with c3: st.metric("Audit events", int(audit_event_df.shape[0]))
    with c4: st.metric("Correction rules mined", int(correction_rules.shape[0]))
    run_lab = st.button("Run Walk-Forward Lab", type="primary", use_container_width=True, key="run_lab")
    if run_lab:
        ranked = run_walkforward_lab(hit_events, overlay_rules, tie_df, correction_rules, int(top_n), int(recent_n), float(tie_margin_threshold), float(top2_gate_margin), float(top2_gate_ratio), float(prune_pct), float(top2_ratio_threshold), float(strong_bias))
        if ranked.empty:
            st.error("Walk-forward produced no scored rows.")
        else:
            op_metrics = compute_operational_metrics(ranked)
            st.subheader("🔥 Operational Goal Metrics")
            cols = st.columns(7)
            with cols[0]: st.metric("Total Plays", op_metrics["Total_Plays"])
            with cols[1]: st.metric("Top1 Wins", op_metrics["Top1_Wins"])
            with cols[2]: st.metric("Needed Top2", op_metrics["Needed_Top2"])
            with cols[3]: st.metric("Waste Top2", op_metrics["Waste_Top2"])
            with cols[4]: st.metric("Misses", op_metrics["Misses"])
            with cols[5]: st.metric("Plays per Win", op_metrics["Plays_Per_Win"])
            with cols[6]: st.metric("Objective Score", op_metrics["Objective_Score"])
            st.metric("Capture Rate", f"{op_metrics['Capture_Rate']:.1%}")

            overall = summarize_results(ranked)
            playable_count = int(ranked[ranked["Selected50"] == 1].shape[0])
            capture_rate = (overall["Correct-member"] / playable_count) if playable_count else 0.0

            st.subheader("Goal check")
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1: st.metric("Playable winner events", playable_count)
            with g2: st.metric("Correct-member capture", f"{capture_rate:.2%}")
            with g3: st.metric("Top1", overall["Top1"])
            with g4: st.metric("Top2-needed", overall["Top2-needed"])
            with g5: st.metric("Recommend Top2", overall.get("RecommendTop2", 0))

            # Downloads (simplified to avoid column errors)
            st.subheader("Downloads")
            st.download_button("Download per-event CSV", data=bytes_csv(ranked.head(100)), file_name="per_event__v5_separation_first.csv", mime="text/csv")

            if show_debug:
                st.dataframe(ranked.head(50), use_container_width=True)

st.caption("v5 Separation First — Stream pruning + Top2 bias + Strong ties added on top of v4")
