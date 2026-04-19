#!/usr/bin/env python3
# BUILD: core025_master_goal_lab__2026-04-19_v6_relaxed_correction_miner

from __future__ import annotations
import io, math, re, unicodedata, textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_master_goal_lab__2026-04-19_v6_relaxed_correction_miner"
APP_VERSION_STR = "core025_master_goal_lab__2026-04-19_v6_relaxed_correction_miner"
MEMBERS = ["0025", "0225", "0255"]
MEMBER_COLS = {"0025":"score_0025","0225":"score_0225","0255":"score_0255"}

def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bytes_txt(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def normalize_member(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = re.sub(r"\D", "", s)
    if s in {"25", "025", "0025"}: return "0025"
    if s in {"225", "0225"}: return "0225"
    if s in {"255", "0255"}: return "0255"
    return s.zfill(4)

def clean_seed_text(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = re.sub(r"\D", "", s)
    return s.zfill(4) if s else ""

def to_float_or_none(x):
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
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

def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stream"] = out["stream"].astype(str).str.strip().map(canon_stream)
    out["feat_seed"] = out["feat_seed"].fillna("").astype(str).map(clean_seed_text)
    out["true_member"] = out["true_member"].fillna("").astype(str).map(normalize_member)

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
        # stronger tie window than v4: allow near-ties too
        if margin <= tie_margin_threshold:
            for _, rr in tie_df.iterrows():
                feature = str(rr.get("feature","")).strip()
                op = str(rr.get("op","")).strip()
                value = rr.get("value","")
                pick = normalize_member(rr.get("pick_str", rr.get("pick","")))
                weight = to_float_or_none(rr.get("weight")) or 0.0
                if feature and pick in MEMBERS and eval_feature_op(row, feature, op, value):
                    strong_weight = max(1.5, weight * 1.75)
                    out.at[idx, MEMBER_COLS[pick]] += strong_weight
                    local_rules.append(f"{feature}{op}{value}->{pick}:{strong_weight}")
                    fired = 1
        tie_fired.append(fired); tie_rules.append("|".join(local_rules))
    out["TieFired"] = tie_fired; out["TieFiredRules"] = tie_rules
    return out

def apply_top2_proximity_bias(df: pd.DataFrame, ratio_threshold: float = 0.84, margin_threshold: float = 0.60) -> pd.DataFrame:
    out = df.copy()
    bias_fired, bias_why = [], []
    for idx, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        top1_m, top1_s = scores[0]
        top2_m, top2_s = scores[1]
        ratio = (top2_s / top1_s) if top1_s else 1.0
        fired = 0
        why = []
        if ratio >= ratio_threshold:
            boost = 0.45 + max(0.0, ratio - ratio_threshold) * 3.0
            out.at[idx, MEMBER_COLS[top2_m]] += boost
            fired = 1
            why.append("ratio")
        if (top1_s - top2_s) <= margin_threshold:
            out.at[idx, MEMBER_COLS[top2_m]] += 0.35
            fired = 1
            why.append("margin")
        bias_fired.append(fired)
        bias_why.append("|".join(why))
    out["Top2BiasFired"] = bias_fired
    out["Top2BiasWhy"] = bias_why
    return out

def apply_stream_pruning(df: pd.DataFrame, prune_percent: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    out["PrunedOut"] = 0
    out["PruneWhy"] = ""
    if out.empty or prune_percent <= 0:
        return out
    # prune weakest rows by date using current StreamScore surrogate inputs
    if "Top1Score" not in out.columns or "Top1Margin" not in out.columns:
        return out
    out["_pre_score"] = out["Top1Score"] * 1000 + out["Top1Margin"] * 100 + out["score_0025"] + out["score_0225"] + out["score_0255"]
    kept_frames = []
    for d, grp in out.groupby("date", dropna=False):
        grp = grp.sort_values(["_pre_score","Top1Margin","stream"], ascending=[False,False,True]).copy()
        n = len(grp)
        keep_n = max(1, int(round(n * (1.0 - prune_percent))))
        grp["PrunedOut"] = 1
        grp.iloc[:keep_n, grp.columns.get_loc("PrunedOut")] = 0
        grp["PruneWhy"] = grp["PrunedOut"].map(lambda x: "bottom_stream_band" if x == 1 else "")
        kept_frames.append(grp)
    out = pd.concat(kept_frames, ignore_index=True).drop(columns=["_pre_score"], errors="ignore")
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
    df["gap_band_now"] = df["gap"].astype(float).apply(build_gap_band) if "gap" in df.columns else ""
    df["ratio_band_now"] = df["ratio"].astype(float).apply(build_ratio_band) if "ratio" in df.columns else ""
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
        gap_band = build_gap_band(float(row.get("Top1Margin",0.0)))
        ratio_band = build_ratio_band(ratio)
        dominance_state = "DOMINANT" if float(row.get("Top1Margin",0.0)) >= 0.20 else ("CONTESTED" if float(row.get("Top1Margin",0.0)) >= 0.05 else "TIGHT")
        play_mode = "PLAY_TOP2" if float(row.get("Top1Margin",0.0)) < 0.15 else "PLAY_TOP1"
        local_fire, local_rule = 0, ""
        for _, rr in correction_rules.iterrows():
            if normalize_member(rr["from_member"]) != normalize_member(row.get("Top1_pred","")): continue
            if normalize_member(rr["to_member"]) not in {normalize_member(row.get("Top2_pred","")), normalize_member(row.get("Top3_pred",""))}: continue
            if str(rr["dominance_state"]) != dominance_state: continue
            if str(rr["gap_band"]) != gap_band: continue
            if str(rr["ratio_band"]) != ratio_band: continue
            if str(rr["play_mode"]) not in {"", play_mode}: continue
            to_m = normalize_member(rr["to_member"]); from_m = normalize_member(rr["from_member"])
            out.at[idx, MEMBER_COLS[to_m]] += float(rr["delta_to_member"])
            out.at[idx, MEMBER_COLS[from_m]] += float(rr["delta_from_member"])
            local_fire = 1
            local_rule = f"{rr['direction']}|{rr['dominance_state']}|{rr['gap_band']}|{rr['ratio_band']}"
            break
        fired.append(local_fire); fired_rule.append(local_rule)
    out["CorrectionFired"] = fired; out["CorrectionRule"] = fired_rule
    return out


def apply_decision_control(df: pd.DataFrame, top2_gate_margin: float = 0.26, top2_gate_ratio: float = 0.88) -> pd.DataFrame:
    out = df.copy()
    top2_flags, why = [], []
    for _, row in out.iterrows():
        margin = float(row.get("Top1Margin",0.0)); top1 = float(row.get("Top1Score",0.0)); top2 = float(row.get("Top2Score",0.0))
        ratio = (top2 / top1) if top1 else 1.0
        tie_fired = int(row.get("TieFired",0) or 0)
        correction_fired = int(row.get("CorrectionFired",0) or 0)
        top2bias_fired = int(row.get("Top2BiasFired",0) or 0)
        pruned = int(row.get("PrunedOut",0) or 0)
        trigger, reason = False, []
        if pruned == 1:
            trigger = False
            reason.append("pruned")
        else:
            if tie_fired == 1: trigger = True; reason.append("tie")
            if margin <= top2_gate_margin: trigger = True; reason.append("margin")
            if ratio >= top2_gate_ratio: trigger = True; reason.append("ratio")
            if correction_fired == 1 and margin <= 0.70: trigger = True; reason.append("correction_guard")
            if top2bias_fired == 1 and ratio >= max(0.82, top2_gate_ratio - 0.03): trigger = True; reason.append("top2_bias")
        top2_flags.append(1 if trigger else 0); why.append("|".join(reason))
    out["RecommendTop2"] = top2_flags; out["Top2GateWhy"] = why
    out["PlayPlan"] = np.where(out["RecommendTop2"] == 1, "Top1+Top2", "Top1")
    return out

def build_hit_event_feature_table(raw_norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, grp in raw_norm.groupby("stream", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        prev_result, prev_date = None, None
        for _, cur in grp.iterrows():
            if prev_result is not None:
                true_member = result_to_member(cur["result4"])
                if true_member in set(MEMBERS):
                    rows.append({
                        "date": cur["date"], "stream": canon_stream(stream), "feat_seed": prev_result, "true_member": true_member,
                        "PlayDate": cur["date"].strftime("%Y-%m-%d"), "StreamKey": canon_stream(stream), "PrevSeed_text": prev_result,
                        "WinningMember_text": true_member, "PrevDrawDate": prev_date.strftime("%Y-%m-%d") if pd.notna(prev_date) else "",
                    })
            prev_result, prev_date = cur["result4"], cur["date"]
    out = pd.DataFrame(rows)
    if out.empty: return out
    out = ensure_feature_columns(out)
    return out.sort_values(["date","stream"]).reset_index(drop=True)

def build_latest_candidate_table(raw_norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, grp in raw_norm.groupby("stream", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        if grp.empty: continue
        latest = grp.iloc[-1]
        rows.append({
            "date": latest["date"], "stream": canon_stream(stream), "feat_seed": latest["result4"], "true_member": "",
            "PlayDate": latest["date"].strftime("%Y-%m-%d"), "StreamKey": canon_stream(stream), "PrevSeed_text": latest["result4"],
            "WinningMember_text": "", "PrevDrawDate": latest["date"].strftime("%Y-%m-%d"),
        })
    out = pd.DataFrame(rows)
    if out.empty: return out
    out = ensure_feature_columns(out)
    return out.sort_values(["stream"]).reset_index(drop=True)

def compute_base_member_scores(train_df: pd.DataFrame, row: pd.Series, recent_n: int = 12) -> Dict[str,float]:
    if train_df.empty: return {m: 1.0 for m in MEMBERS}
    stream = canon_stream(row["stream"])
    global_hist = train_df.copy()
    stream_hist = train_df[train_df["stream"].astype(str).map(canon_stream).eq(stream)].copy()
    recent_stream = stream_hist.sort_values("date").tail(int(recent_n)).copy()
    scores = {m: 0.0 for m in MEMBERS}
    total_global = max(1, global_hist.shape[0])
    total_stream = max(1, stream_hist.shape[0]) if not stream_hist.empty else 1
    total_recent = max(1, recent_stream.shape[0]) if not recent_stream.empty else 1
    for m in MEMBERS:
        global_rate = (global_hist["true_member"] == m).sum() / total_global
        stream_rate = (stream_hist["true_member"] == m).sum() / total_stream if not stream_hist.empty else 1.0/len(MEMBERS)
        recent_rate = (recent_stream["true_member"] == m).sum() / total_recent if not recent_stream.empty else 1.0/len(MEMBERS)
        due_bonus = 0.0
        if not stream_hist.empty:
            sub = stream_hist[stream_hist["true_member"] == m]
            if not sub.empty:
                last_date = sub["date"].max()
                days_gap = max(0, (pd.Timestamp(row["date"]) - pd.Timestamp(last_date)).days)
                due_bonus = min(2.5, math.log1p(days_gap) / 3.0)
            else:
                due_bonus = 1.0
        scores[m] = 2.0*global_rate + 3.2*stream_rate + 2.8*recent_rate + due_bonus
    return scores

def rank_within_date(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["StreamScore"] = out["Top1Score"]*1000 + out["Top1Margin"]*100 + out["score_0025"] + out["score_0225"] + out["score_0255"]
    out = out.sort_values(["date","StreamScore","Top1Margin","stream"], ascending=[True,False,False,True]).copy()
    out["Rank"] = out.groupby("date").cumcount() + 1
    out["Selected50"] = (out["Rank"] <= top_n).astype(int)
    out["CorrectMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out["true_member"])).astype(int)
    out["Top2Needed"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] == out["true_member"])).astype(int)
    out["Top3Only"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] != out["true_member"]) & (out["Top3_pred"] == out["true_member"])).astype(int)
    return out

def summarize_results(df: pd.DataFrame) -> Dict[str,int]:
    selected = df[df["Selected50"] == 1]
    return {
        "Selected50": int(selected.shape[0]),
        "Correct-member": int(selected["CorrectMember"].sum() + selected["Top2Needed"].sum()),
        "Top1": int(selected["CorrectMember"].sum()),
        "Top2-needed": int(selected["Top2Needed"].sum()),
        "Top3-only": int(selected["Top3Only"].sum()),
        "RecommendTop2": int(selected["RecommendTop2"].sum()) if "RecommendTop2" in selected.columns else 0,
    }

def build_date_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("date", dropna=False).agg(
        total_rows=("stream","size"), selected50=("Selected50","sum"),
        correct_member=("CorrectMember","sum"), top2_needed=("Top2Needed","sum"),
        top3_only=("Top3Only","sum"), recommend_top2=("RecommendTop2","sum")
    ).reset_index()
    grp["top1"] = grp["correct_member"]; grp["date"] = grp["date"].astype(str)
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

def run_walkforward_lab(events_df: pd.DataFrame, overlay_rules, tie_df, correction_rules, top_n, recent_n, tie_margin_threshold, top2_gate_margin, top2_gate_ratio, stream_prune_percent=0.25):
    if events_df.empty: return events_df.copy()
    work = events_df.sort_values(["date","stream"]).reset_index(drop=True).copy()
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
        test_df = apply_top2_proximity_bias(test_df)
        test_df = rank_members(test_df)
        test_df = apply_overlay_to_scores(test_df, overlay_rules)
        test_df = rank_members(test_df)
        test_df = apply_correction_rules(test_df, correction_rules)
        test_df = rank_members(test_df)
        test_df = apply_stream_pruning(test_df, prune_percent=stream_prune_percent)
        test_df = apply_decision_control(test_df, top2_gate_margin=top2_gate_margin, top2_gate_ratio=top2_gate_ratio)
        scored_rows.append(test_df)
    if not scored_rows: return pd.DataFrame()
    ranked = pd.concat(scored_rows, ignore_index=True)
    ranked = rank_within_date(ranked, top_n=top_n)
    return ranked.sort_values(["date","Rank"]).reset_index(drop=True)

def build_daily_playlist(raw_norm: pd.DataFrame, hit_events_df: pd.DataFrame, overlay_rules, tie_df, correction_rules, top_n, recent_n, tie_margin_threshold, top2_gate_margin, top2_gate_ratio, stream_prune_percent=0.25):
    latest_df = build_latest_candidate_table(raw_norm)
    if latest_df.empty: return latest_df
    train_df = hit_events_df.copy().sort_values("date").reset_index(drop=True)
    for idx, row in latest_df.iterrows():
        base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
        for m in MEMBERS: latest_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
    latest_df = rank_members(latest_df)
    latest_df = apply_tie_pack(latest_df, tie_df, tie_margin_threshold=tie_margin_threshold)
    latest_df = rank_members(latest_df)
    latest_df = apply_top2_proximity_bias(latest_df)
    latest_df = rank_members(latest_df)
    latest_df = apply_overlay_to_scores(latest_df, overlay_rules)
    latest_df = rank_members(latest_df)
    latest_df = apply_correction_rules(latest_df, correction_rules)
    latest_df = rank_members(latest_df)
    latest_df = apply_stream_pruning(latest_df, prune_percent=stream_prune_percent)
    latest_df = apply_decision_control(latest_df, top2_gate_margin=top2_gate_margin, top2_gate_ratio=top2_gate_ratio)
    latest_df["StreamScore"] = latest_df["Top1Score"]*1000 + latest_df["Top1Margin"]*100 + latest_df["score_0025"] + latest_df["score_0225"] + latest_df["score_0255"] - latest_df["PrunedOut"]*1000000
    latest_df = latest_df.sort_values(["StreamScore","Top1Margin","stream"], ascending=[False,False,True]).reset_index(drop=True)
    latest_df["Rank"] = np.arange(1, len(latest_df)+1)
    latest_df["Selected50"] = (latest_df["Rank"] <= int(top_n)).astype(int)
    return latest_df

def build_dictionary_report(feature_table: pd.DataFrame, overlay_rules, tie_df, correction_rules) -> pd.DataFrame:
    feature_cols = set(feature_table.columns)
    rows = []
    for rule in overlay_rules:
        for cond_col, val in rule.conditions.items():
            rows.append({
                "layer":"overlay","trait_raw":f"{cond_col}={val}",
                "feature_token":cond_col.replace("when_",""),
                "mapped_feature":cond_col.replace("when_","") if cond_col.replace("when_","") in feature_cols else "",
                "exists_in_feature_table":cond_col.replace("when_","") in feature_cols,
                "status":"valid" if cond_col.replace("when_","") in feature_cols else "control_or_missing",
                "rule_id":rule.rule_id
            })
    if not tie_df.empty:
        for _, rr in tie_df.iterrows():
            feature = str(rr.get("feature","")).strip()
            rows.append({
                "layer":"tie_pack","trait_raw":f"{feature}{rr.get('op','')}{rr.get('value','')}",
                "feature_token":feature,"mapped_feature":feature if feature in feature_cols else "",
                "exists_in_feature_table":feature in feature_cols,
                "status":"valid" if feature in feature_cols else "missing_feature",
                "rule_id":f"tie::{feature}"
            })
    if not correction_rules.empty:
        for _, rr in correction_rules.iterrows():
            rows.append({
                "layer":"correction",
                "trait_raw":f"{rr['direction']}|{rr['dominance_state']}|{rr['gap_band']}|{rr['ratio_band']}",
                "feature_token":"directional_bucket",
                "mapped_feature":"Top1/Top2/Top3 + gap/ratio state",
                "exists_in_feature_table":True,
                "status":"derived",
                "rule_id":rr["direction"]
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["layer","trait_raw","feature_token","mapped_feature","exists_in_feature_table","status","rule_id"])
    return out.drop_duplicates().sort_values(["layer","status","feature_token"]).reset_index(drop=True)

def build_firing_report(scored_df: pd.DataFrame, overlay_rules) -> pd.DataFrame:
    rows = []
    if "FiredRuleIDs" in scored_df.columns:
        for rule in overlay_rules:
            mask = scored_df["FiredRuleIDs"].fillna("").astype(str).str.contains(re.escape(rule.rule_id), regex=True)
            sub = scored_df[mask]
            rows.append({
                "layer":"overlay","rule_id":rule.rule_id,"rows_matched":int(sub.shape[0]),
                "selected50_rows":int(sub["Selected50"].sum()) if "Selected50" in sub.columns else 0,
                "top1_correct_rows":int(sub["CorrectMember"].sum()) if "CorrectMember" in sub.columns else 0,
                "top2_needed_rows":int(sub["Top2Needed"].sum()) if "Top2Needed" in sub.columns else 0
            })
    if "TieFiredRules" in scored_df.columns:
        cnt = Counter()
        for s in scored_df["TieFiredRules"].fillna("").astype(str):
            if not s: continue
            for part in s.split("|"):
                if part: cnt[part] += 1
        for rid, n in cnt.items():
            rows.append({"layer":"tie_pack","rule_id":rid,"rows_matched":int(n),"selected50_rows":0,"top1_correct_rows":0,"top2_needed_rows":0})
    if "CorrectionRule" in scored_df.columns:
        cnt = Counter(scored_df["CorrectionRule"].fillna("").astype(str))
        for rid, n in cnt.items():
            if rid:
                rows.append({"layer":"correction","rule_id":rid,"rows_matched":int(n),"selected50_rows":0,"top1_correct_rows":0,"top2_needed_rows":0})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["layer","rule_id","rows_matched","selected50_rows","top1_correct_rows","top2_needed_rows"])
    return out.sort_values(["layer","rows_matched"], ascending=[True,False]).reset_index(drop=True)

st.set_page_config(page_title="Core025 Master Goal Lab", layout="wide")
st.title("Core025 Master Goal Lab")
st.caption(BUILD_MARKER)
st.warning("v6 keeps the separation-first stack, but relaxes the correction miner so weaker repeat pockets can surface. Nothing runs until you click a Run button.")

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
    min_corr_events = st.number_input("Min correction events", min_value=1, max_value=50, value=2, step=1)
    min_corr_rate = st.number_input("Min correction success rate", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    top2_gate_margin = st.number_input("Top2 gate max margin", min_value=0.0, max_value=5.0, value=0.26, step=0.05)
    top2_gate_ratio = st.number_input("Top2 gate min ratio", min_value=0.0, max_value=1.0, value=0.88, step=0.01)
    stream_prune_percent = st.number_input("Stream prune percent", min_value=0.0, max_value=0.80, value=0.25, step=0.05)
    show_debug = st.checkbox("Show debug tables", value=False)

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
        playlist = build_daily_playlist(raw_norm, hit_events, overlay_rules, tie_df, correction_rules, int(top_n), int(recent_n), float(tie_margin_threshold), float(top2_gate_margin), float(top2_gate_ratio), float(stream_prune_percent))
        if playlist.empty:
            st.error("No daily playlist rows were produced.")
        else:
            playlist_export = playlist[[
                "PlayDate","StreamKey","PrevSeed_text","PredictedMember","Top1_pred","Top2_pred","Top3_pred",
                "score_0025","score_0225","score_0255","Top1Score","Top2Score","Top1Margin","StreamScore","Rank","Selected50","PlayPlan",
                "TieFired","TieFiredRules","Top2BiasFired","Top2BiasWhy","CorrectionFired","CorrectionRule","PrunedOut","PruneWhy","RecommendTop2","Top2GateWhy","FiredRuleIDs","FiredRuleNotes"
            ]].rename(columns={"Top1_pred":"Top1","Top2_pred":"Top2","Top3_pred":"Top3"})
            st.dataframe(playlist_export.head(int(top_n)), use_container_width=True, hide_index=True)
            st.download_button("Download daily playlist CSV", data=bytes_csv(playlist_export), file_name="daily_playlist__core025_master_goal_lab_v4.csv", mime="text/csv", use_container_width=True)
            st.download_button("Download daily playlist TXT", data=bytes_txt(playlist_export), file_name="daily_playlist__core025_master_goal_lab_v4.txt", mime="text/plain", use_container_width=True)
            if show_debug:
                st.dataframe(playlist_export.head(200), use_container_width=True)

with tab_lab:
    st.subheader("Lab walk-forward + separation-first decision control (relaxed miner)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Overlay rules", int(len(overlay_rules)))
    with c2: st.metric("Tie rules rows", int(tie_df.shape[0]))
    with c3: st.metric("Audit events", int(audit_event_df.shape[0]))
    with c4: st.metric("Correction rules mined", int(correction_rules.shape[0]))
    run_lab = st.button("Run Walk-Forward Lab", type="primary", use_container_width=True, key="run_lab")
    if run_lab:
        ranked = run_walkforward_lab(hit_events, overlay_rules, tie_df, correction_rules, int(top_n), int(recent_n), float(tie_margin_threshold), float(top2_gate_margin), float(top2_gate_ratio), float(stream_prune_percent))
        if ranked.empty:
            st.error("Walk-forward produced no scored rows.")
        else:
            overall = summarize_results(ranked)
            selected = ranked[ranked["Selected50"] == 1].copy()
            playable_count = int(selected.shape[0])
            capture_rate = (overall["Correct-member"] / playable_count) if playable_count else 0.0
            recommend_top2_rate = (overall["RecommendTop2"] / playable_count) if playable_count else 0.0
            st.subheader("Goal check")
            g1, g2, g3, g4, g5 = st.columns(5)
            with g1: st.metric("Playable winner events", playable_count)
            with g2: st.metric("Correct-member capture", f"{capture_rate:.2%}")
            with g3: st.metric("Top1", overall["Top1"])
            with g4: st.metric("Top2-needed", overall["Top2-needed"])
            with g5: st.metric("Recommend Top2", f"{recommend_top2_rate:.2%}")
            if capture_rate >= 0.75: st.success("75%+ capture goal met.")
            else: st.error("75%+ capture goal NOT met.")
            st.subheader("Quick test summary")
            st.code(textwrap.dedent(f"""Selected50: {overall['Selected50']}
Correct-member: {overall['Correct-member']}
Top1: {overall['Top1']}
Top2-needed: {overall['Top2-needed']}
Top3-only: {overall['Top3-only']}
RecommendTop2: {overall['RecommendTop2']}
StreamPrunePercent: {float(stream_prune_percent):.2%}
CaptureRate: {capture_rate:.2%}
"""))
            dictionary_report = build_dictionary_report(hit_events, overlay_rules, tie_df, correction_rules)
            firing_report = build_firing_report(ranked, overlay_rules)
            per_event_export = ranked[[
                "PlayDate","StreamKey","PrevSeed_text","WinningMember_text","PredictedMember","Top1_pred","Top2_pred","Top3_pred",
                "score_0025","score_0225","score_0255","Top1Score","Top2Score","Top1Margin","StreamScore","Rank","Selected50",
                "CorrectMember","Top2Needed","Top3Only","TieFired","TieFiredRules","CorrectionFired","CorrectionRule",
                "RecommendTop2","Top2GateWhy","FiredRuleIDs","FiredRuleNotes"
            ]].rename(columns={"Top1_pred":"Top1","Top2_pred":"Top2","Top3_pred":"Top3"})
            per_date_export = build_date_export(ranked)
            per_stream_export = build_stream_export(ranked)
            correction_export = correction_rules.copy()
            st.subheader("Downloads")
            download_specs = [
                ("per_event__core025_master_goal_lab_v4.csv", per_event_export, "Download per-event CSV"),
                ("per_event__core025_master_goal_lab_v4.txt", per_event_export, "Download per-event TXT"),
                ("per_date__core025_master_goal_lab_v4.csv", per_date_export, "Download per-date CSV"),
                ("per_date__core025_master_goal_lab_v4.txt", per_date_export, "Download per-date TXT"),
                ("per_stream__core025_master_goal_lab_v4.csv", per_stream_export, "Download per-stream CSV"),
                ("per_stream__core025_master_goal_lab_v4.txt", per_stream_export, "Download per-stream TXT"),
                ("dictionary_report__core025_master_goal_lab_v4.csv", dictionary_report, "Download dictionary report CSV"),
                ("dictionary_report__core025_master_goal_lab_v4.txt", dictionary_report, "Download dictionary report TXT"),
                ("firing_report__core025_master_goal_lab_v4.csv", firing_report, "Download firing report CSV"),
                ("firing_report__core025_master_goal_lab_v4.txt", firing_report, "Download firing report TXT"),
                ("correction_rules__core025_master_goal_lab_v4.csv", correction_export, "Download correction rules CSV"),
                ("correction_rules__core025_master_goal_lab_v4.txt", correction_export, "Download correction rules TXT"),
            ]
            for i in range(0, len(download_specs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j >= len(download_specs): continue
                    fname, df_exp, label = download_specs[i+j]
                    with col:
                        data = bytes_csv(df_exp) if fname.endswith(".csv") else bytes_txt(df_exp)
                        mime = "text/csv" if fname.endswith(".csv") else "text/plain"
                        st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)
            if show_debug:
                with st.expander("Correction rules preview", expanded=False):
                    st.dataframe(correction_export.head(200), use_container_width=True)
                with st.expander("Dictionary report preview", expanded=False):
                    st.dataframe(dictionary_report.head(200), use_container_width=True)
                with st.expander("Firing report preview", expanded=False):
                    st.dataframe(firing_report.head(200), use_container_width=True)
                with st.expander("Per-event preview", expanded=False):
                    st.dataframe(per_event_export.head(200), use_container_width=True)
