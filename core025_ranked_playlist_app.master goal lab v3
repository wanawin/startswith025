#!/usr/bin/env python3
# BUILD: core025_master_goal_lab__2026-04-18_v3_tabs_run_correction

from __future__ import annotations
import io, math, re, unicodedata, textwrap, ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_master_goal_lab__2026-04-18_v3_tabs_run_correction"
APP_VERSION_STR = "core025_master_goal_lab__2026-04-18_v3_tabs_run_correction"
MEMBERS = ["0025", "0225", "0255"]
MEMBER_COLS = {"0025": "score_0025", "0225": "score_0225", "0255": "score_0255"}

# ============================================================
# General helpers
# ============================================================
def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bytes_txt(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def normalize_member(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = re.sub(r"\D", "", s)
    if s in {"25", "025", "0025"}:
        return "0025"
    if s in {"225", "0225"}:
        return "0225"
    if s in {"255", "0255"}:
        return "0255"
    return s.zfill(4)

def clean_seed_text(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = re.sub(r"\D", "", s)
    return s.zfill(4) if s else ""

def to_int_or_none(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def to_float_or_none(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None

def parse_digit_set(value) -> Optional[set]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.strip("[](){}")
    vals = []
    for p in re.split(r"[\s,|;/]+", s):
        if not p:
            continue
        try:
            vals.append(int(float(p)))
        except Exception:
            pass
    return set(vals) if vals else None

def parse_text_set(value) -> Optional[set]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.strip("[](){}")
    parts = [p.strip().upper() for p in re.split(r"[\s,|;/]+", s) if p.strip()]
    return set(parts) if parts else None

def canon_stream(s: str) -> str:
    if s is None:
        return ""
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

# ============================================================
# Raw history normalization
# ============================================================
TIME_PATTERNS = [
    ("11:30pm", "1130pm"), ("7:50pm", "750pm"), ("1:50pm", "150pm"),
    ("morning", "morning"), ("midday", "midday"), ("daytime", "daytime"),
    ("day", "day"), ("evening", "evening"), ("night", "night"),
    ("1pm", "1pm"), ("4pm", "4pm"), ("7pm", "7pm"), ("10pm", "10pm"),
    ("noche", "noche"), ("día", "dia"), ("dia", "dia"),
]

def extract_digits_result(result_text: str) -> str:
    if pd.isna(result_text):
        return ""
    s = str(result_text)
    m = re.search(r"(\d)\D+(\d)\D+(\d)\D+(\d)", s)
    if m:
        return "".join(m.groups())
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
        if raw in s:
            return canon
    return "unknown"

def canonical_stream(state: str, game_text: str) -> str:
    return canon_stream(f"{str(state).strip()} | {base_game_name(game_text)} | {time_class(game_text)}")

def result_to_member(result4: str) -> str:
    s = clean_seed_text(result4)
    if len(s) != 4:
        return ""
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
    out["time_class"] = out["game"].apply(time_class)
    out["base_game"] = out["game"].apply(base_game_name)
    out["member"] = out["result4"].apply(result_to_member)
    out = out[out["date"].notna() & out["stream"].ne("") & out["result4"].str.len().eq(4)].copy()
    out = out.sort_values(["stream", "date", "game", "result4"], ascending=[True, True, True, True]).reset_index(drop=True)
    return out

# ============================================================
# Feature engine
# ============================================================
def _digits_from_seed(seed: str) -> List[int]:
    s = clean_seed_text(seed)
    return [int(x) for x in s] if len(s) == 4 else [None, None, None, None]

def _parity_pattern(digits: List[int]) -> str:
    if any(d is None for d in digits):
        return ""
    return "".join("E" if d % 2 == 0 else "O" for d in digits)

def _highlow_pattern(digits: List[int]) -> str:
    if any(d is None for d in digits):
        return ""
    return "".join("H" if d >= 5 else "L" for d in digits)

def _repeat_shape(digits: List[int]) -> str:
    if any(d is None for d in digits):
        return "other"
    counts = sorted(Counter(digits).values(), reverse=True)
    if counts == [1, 1, 1, 1]:
        return "all_unique"
    if counts == [2, 1, 1]:
        return "one_pair"
    if counts == [2, 2]:
        return "two_pair"
    if counts == [3, 1]:
        return "triple"
    if counts == [4]:
        return "quad"
    return "other"

def _unique_even_odd(digits: List[int]) -> Tuple[int, int]:
    if any(d is None for d in digits):
        return (0, 0)
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
    out["seed_sum_mod3"] = out["seed_sum"] % 3
    out["seed_sum_mod4"] = out["seed_sum"] % 4
    out["seed_sum_mod5"] = out["seed_sum"] % 5
    out["seed_sum_mod6"] = out["seed_sum"] % 6
    out["seed_sum_mod9"] = out["seed_sum"] % 9
    out["seed_sum_mod10"] = out["seed_sum"] % 10
    out["seed_sum_mod11"] = out["seed_sum"] % 11
    out["seed_sum_mod12"] = out["seed_sum"] % 12
    out["seed_sum_mod13"] = out["seed_sum"] % 13
    out["seed_root"] = out["seed_sum"].apply(lambda x: int((x - 1) % 9 + 1) if pd.notna(x) and x > 0 else (0 if pd.notna(x) else np.nan))
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
    out["cnt_0_3"] = [sum(1 for x in d if 0 <= x <= 3) if None not in d else np.nan for d in digits]
    out["cnt_4_6"] = [sum(1 for x in d if 4 <= x <= 6) if None not in d else np.nan for d in digits]
    out["cnt_7_9"] = [sum(1 for x in d if 7 <= x <= 9) if None not in d else np.nan for d in digits]
    out["seed_has9"] = out["feat_seed"].str.contains("9").astype(int)
    out["seed_has0"] = out["feat_seed"].str.contains("0").astype(int)

    def vtrac_group(x: int) -> int:
        return x % 5
    out["seed_vtrac_groups"] = [len(set(vtrac_group(x) for x in d)) if None not in d else np.nan for d in digits]

    for digit in range(10):
        out[f"seed_has{digit}"] = [1 if (None not in d and digit in d) else 0 for d in digits]
        out[f"seed_cnt{digit}"] = [sum(1 for x in d if x == digit) if None not in d else np.nan for d in digits]

    out["x_repeatshape_parity"] = [f"{_repeat_shape(d)}|{_parity_pattern(d)}" if None not in d else "" for d in digits]
    out["x_repeatshape_highlow"] = [f"{_repeat_shape(d)}|{_highlow_pattern(d)}" if None not in d else "" for d in digits]
    out["x_unique_even"] = [f"{_unique_even_odd(d)[0]}|{_unique_even_odd(d)[1]}" if None not in d else "" for d in digits]

    def ordered_pairs(seed):
        s = clean_seed_text(seed)
        return [s[0:2], s[1:3], s[2:4]] if len(s) == 4 else []

    def unordered_pairs(seed):
        s = clean_seed_text(seed)
        if len(s) != 4:
            return []
        outp = []
        for i in range(4):
            for j in range(i + 1, 4):
                outp.append("".join(sorted([s[i], s[j]])))
        return outp

    opairs = out["feat_seed"].apply(ordered_pairs)
    upairs = out["feat_seed"].apply(unordered_pairs)
    out["seed_adj_pairs_ordered"] = opairs.apply(lambda xs: "|".join(xs))
    out["seed_pair_tokens"] = upairs.apply(lambda xs: "|".join(xs))
    for a in range(10):
        for b in range(10):
            tok = f"{a}{b}"
            out[f"adj_ord_has_{tok}"] = opairs.apply(lambda xs, t=tok: 1 if t in xs else 0)
    for a in range(10):
        for b in range(a, 10):
            tok = f"{a}{b}"
            out[f"pair_has_{tok}"] = upairs.apply(lambda xs, t=tok: 1 if t in xs else 0)
    return out

# ============================================================
# Packs
# ============================================================
@dataclass
class OverlayRule:
    rule_id: str
    enabled: bool
    conditions: Dict[str, object]
    deltas: Dict[str, float]
    note: str

def parse_overlay_file(uploaded_file):
    if uploaded_file is None:
        return [], {"source": "none", "rows": 0, "filename": ""}
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    rules = []
    for _, row in df.iterrows():
        enabled = str(row.get("enabled", "1")).strip().lower()
        enabled_bool = enabled not in {"0", "false", "no"}
        conditions = {}
        for col in df.columns:
            if col.startswith("when_") or col in {"rank_min", "rank_max"}:
                val = row.get(col)
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                sval = str(val).strip()
                if sval == "" or sval.lower() == "nan":
                    continue
                conditions[col] = val
        deltas = {
            "0025": float(row.get("delta_0025", 0) if not pd.isna(row.get("delta_0025", np.nan)) else 0),
            "0225": float(row.get("delta_0225", 0) if not pd.isna(row.get("delta_0225", np.nan)) else 0),
            "0255": float(row.get("delta_0255", 0) if not pd.isna(row.get("delta_0255", np.nan)) else 0),
        }
        rules.append(OverlayRule(
            rule_id=str(row.get("rule_id", f"rule_{len(rules)+1}")),
            enabled=enabled_bool,
            conditions=conditions,
            deltas=deltas,
            note=str(row.get("note", "")),
        ))
    return rules, {"source": "upload", "rows": int(df.shape[0]), "filename": uploaded_file.name}

def parse_tie_pack(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["feature", "op", "value", "pick", "weight"])
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ============================================================
# Rule matching and pack application
# ============================================================
def match_rule_to_row(rule: OverlayRule, row: pd.Series) -> bool:
    if not rule.enabled:
        return False
    def g(col, default=np.nan):
        return row[col] if col in row.index else default

    # support direct column references and legacy when_ refs
    for cond_col, val in rule.conditions.items():
        if cond_col in {"rank_min", "rank_max"}:
            continue
        if cond_col.startswith("when_"):
            feature = cond_col.replace("when_", "")
            # contains_any/all/none special
            if feature == "seed_contains_any":
                vals = parse_digit_set(val)
                seed_text = clean_seed_text(g("feat_seed", ""))
                if vals and not any(str(v) in seed_text for v in vals):
                    return False
                continue
            if feature == "seed_contains_all":
                vals = parse_digit_set(val)
                seed_text = clean_seed_text(g("feat_seed", ""))
                if vals and not all(str(v) in seed_text for v in vals):
                    return False
                continue
            if feature == "seed_contains_none":
                vals = parse_digit_set(val)
                seed_text = clean_seed_text(g("feat_seed", ""))
                if vals and any(str(v) in seed_text for v in vals):
                    return False
                continue
            # direct equality on existing feature if possible
            if feature in row.index:
                rv = row[feature]
                # numeric compare if possible
                rnum = to_float_or_none(rv); vnum = to_float_or_none(val)
                if rnum is not None and vnum is not None:
                    if float(rnum) != float(vnum):
                        return False
                else:
                    if str(rv).strip() != str(val).strip():
                        return False
            else:
                # allow base top fields
                if feature == "base_top1":
                    if normalize_member(g("Top1_pred", "")) != normalize_member(val):
                        return False
                elif feature == "base_top2":
                    if normalize_member(g("Top2_pred", "")) != normalize_member(val):
                        return False
                elif feature == "base_top3":
                    if normalize_member(g("Top3_pred", "")) != normalize_member(val):
                        return False
                else:
                    return False
    return True

def eval_feature_op(row: pd.Series, feature: str, op: str, value) -> bool:
    if feature not in row.index:
        return False
    rv = row[feature]
    if op == "==":
        rn = to_float_or_none(rv); vn = to_float_or_none(value)
        if rn is not None and vn is not None:
            return float(rn) == float(vn)
        return str(rv).strip() == str(value).strip()
    if op == ">=":
        rn = to_float_or_none(rv); vn = to_float_or_none(value)
        return rn is not None and vn is not None and rn >= vn
    if op == "<=":
        rn = to_float_or_none(rv); vn = to_float_or_none(value)
        return rn is not None and vn is not None and rn <= vn
    if op == ">":
        rn = to_float_or_none(rv); vn = to_float_or_none(value)
        return rn is not None and vn is not None and rn > vn
    if op == "<":
        rn = to_float_or_none(rv); vn = to_float_or_none(value)
        return rn is not None and vn is not None and rn < vn
    return False

def apply_overlay_to_scores(df: pd.DataFrame, rules: List[OverlayRule]) -> pd.DataFrame:
    out = df.copy()
    if not rules:
        out["FiredRuleIDs"] = ""
        out["FiredRuleNotes"] = ""
        return out
    rule_ids_per_row = []
    notes_per_row = []
    for _, row in out.iterrows():
        row_rule_ids = []
        row_notes = []
        for rule in rules:
            if match_rule_to_row(rule, row):
                for m in MEMBERS:
                    out.at[row.name, MEMBER_COLS[m]] += float(rule.deltas.get(m, 0.0))
                row_rule_ids.append(rule.rule_id)
                if rule.note:
                    row_notes.append(rule.note)
        rule_ids_per_row.append("|".join(row_rule_ids))
        notes_per_row.append(" || ".join(row_notes))
    out["FiredRuleIDs"] = rule_ids_per_row
    out["FiredRuleNotes"] = notes_per_row
    return out

def apply_tie_pack(df: pd.DataFrame, tie_df: pd.DataFrame, tie_margin_threshold: float = 0.25) -> pd.DataFrame:
    out = df.copy()
    if tie_df.empty:
        out["TieFired"] = 0
        out["TieFiredRules"] = ""
        return out
    tie_fired = []
    tie_rules = []
    for idx, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        margin = scores[0][1] - scores[1][1]
        local_rules = []
        fired = 0
        if margin <= tie_margin_threshold:
            for _, rr in tie_df.iterrows():
                feature = str(rr.get("feature", "")).strip()
                op = str(rr.get("op", "")).strip()
                value = rr.get("value", "")
                pick = normalize_member(rr.get("pick_str", rr.get("pick", "")))
                weight = to_float_or_none(rr.get("weight")) or 0.0
                if feature and pick in MEMBERS and eval_feature_op(row, feature, op, value):
                    out.at[idx, MEMBER_COLS[pick]] += weight
                    local_rules.append(f"{feature}{op}{value}->{pick}:{weight}")
                    fired = 1
        tie_fired.append(fired)
        tie_rules.append("|".join(local_rules))
    out["TieFired"] = tie_fired
    out["TieFiredRules"] = tie_rules
    return out

def rank_members(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    top1s, top2s, top3s, top1scores, margins = [], [], [], [], []
    for _, row in out.iterrows():
        scores = [(m, float(row[MEMBER_COLS[m]])) for m in MEMBERS]
        scores.sort(key=lambda x: (x[1], x[0]), reverse=True)
        ranked_members = [m for m, _ in scores]
        ranked_scores = [s for _, s in scores]
        top1s.append(ranked_members[0]); top2s.append(ranked_members[1]); top3s.append(ranked_members[2])
        top1scores.append(ranked_scores[0]); margins.append(ranked_scores[0] - ranked_scores[1])
    out["PredictedMember"] = top1s
    out["Top1_pred"] = top1s
    out["Top2_pred"] = top2s
    out["Top3_pred"] = top3s
    out["Top1Score"] = top1scores
    out["Top1Margin"] = margins
    return out

# ============================================================
# Audit-driven correction mining
# ============================================================
def parse_event_buckets(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def parse_failure_patterns(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_gap_band(x: float) -> str:
    if x < 0.05:
        return "gap_lt_0.05"
    if x < 0.20:
        return "gap_0.05_0.20"
    return "gap_ge_0.20"

def build_ratio_band(x: float) -> str:
    if x >= 0.95:
        return "ratio_ge_0.95"
    if x >= 0.80:
        return "ratio_0.80_0.95"
    return "ratio_lt_0.80"

def mine_correction_rules(event_df: pd.DataFrame, min_events: int = 3, min_success_rate: float = 0.60) -> pd.DataFrame:
    """
    Mine simple directional correction rules from event-bucket audit:
    group by current wrong direction + dominance/gap/ratio/play_mode
    """
    if event_df.empty:
        return pd.DataFrame(columns=[
            "direction", "from_member", "to_member", "dominance_state", "gap_band", "ratio_band", "play_mode",
            "events", "success_rate", "recommended_action", "delta_to_member", "delta_from_member"
        ])

    df = event_df.copy()
    # normalize members
    for c in ["winning_member", "Top1", "Top2", "Top3"]:
        if c in df.columns:
            df[c] = df[c].map(normalize_member)
    if "gap" in df.columns:
        df["gap_band_now"] = df["gap"].astype(float).apply(build_gap_band)
    else:
        df["gap_band_now"] = ""
    if "ratio" in df.columns:
        df["ratio_band_now"] = df["ratio"].astype(float).apply(build_ratio_band)
    else:
        df["ratio_band_now"] = ""

    # only wrong top1 rows where winner in top2 or top3
    df = df[(df["Top1"] != df["winning_member"]) & (df["winning_member"].isin(MEMBERS))]
    if df.empty:
        return pd.DataFrame(columns=[
            "direction", "from_member", "to_member", "dominance_state", "gap_band", "ratio_band", "play_mode",
            "events", "success_rate", "recommended_action", "delta_to_member", "delta_from_member"
        ])
    df["direction"] = df["Top1"] + "->" + df["winning_member"]
    grp = df.groupby(["direction", "Top1", "winning_member", "dominance_state", "gap_band_now", "ratio_band_now", "play_mode"], dropna=False).size().reset_index(name="events")
    grp = grp.rename(columns={
        "Top1": "from_member",
        "winning_member": "to_member",
        "gap_band_now": "gap_band",
        "ratio_band_now": "ratio_band",
    })
    total_by_direction = grp.groupby("direction")["events"].transform("sum")
    grp["success_rate"] = grp["events"] / total_by_direction
    grp = grp[(grp["events"] >= int(min_events)) & (grp["success_rate"] >= float(min_success_rate))].copy()
    if grp.empty:
        return pd.DataFrame(columns=[
            "direction", "from_member", "to_member", "dominance_state", "gap_band", "ratio_band", "play_mode",
            "events", "success_rate", "recommended_action", "delta_to_member", "delta_from_member"
        ])
    grp["recommended_action"] = "force_swap"
    grp["delta_to_member"] = 0.75 + grp["success_rate"] * 0.75
    grp["delta_from_member"] = -(0.35 + grp["success_rate"] * 0.35)
    return grp.sort_values(["events", "success_rate"], ascending=[False, False]).reset_index(drop=True)

def apply_correction_rules(df: pd.DataFrame, correction_rules: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if correction_rules.empty:
        out["CorrectionFired"] = 0
        out["CorrectionRule"] = ""
        return out
    fired = []
    fired_rule = []
    for idx, row in out.iterrows():
        local_fire = 0
        local_rule = ""
        direction_guess = normalize_member(row.get("Top1_pred", "")) + "->" + normalize_member(row.get("Top2_pred", ""))
        gap_band = build_gap_band(float(row.get("Top1Margin", 0.0)))
        ratio = (float(row.get("Top2Score", 0.0)) / float(row.get("Top1Score", 1e-9))) if float(row.get("Top1Score", 0.0)) != 0 else 1.0
        ratio_band = build_ratio_band(ratio)
        dominance_state = "DOMINANT" if float(row.get("Top1Margin", 0.0)) >= 0.20 else ("CONTESTED" if float(row.get("Top1Margin", 0.0)) >= 0.05 else "TIGHT")
        play_mode = "PLAY_TOP2" if float(row.get("Top1Margin", 0.0)) < 0.15 else "PLAY_TOP1"

        # match candidate correction where from_member equals current Top1 and to_member equals current Top2 or Top3
        for _, rr in correction_rules.iterrows():
            if normalize_member(rr["from_member"]) != normalize_member(row.get("Top1_pred", "")):
                continue
            if normalize_member(rr["to_member"]) not in {normalize_member(row.get("Top2_pred", "")), normalize_member(row.get("Top3_pred", ""))}:
                continue
            if str(rr["dominance_state"]) != dominance_state:
                continue
            if str(rr["gap_band"]) != gap_band:
                continue
            if str(rr["ratio_band"]) != ratio_band:
                continue
            # apply correction
            to_m = normalize_member(rr["to_member"])
            from_m = normalize_member(rr["from_member"])
            out.at[idx, MEMBER_COLS[to_m]] += float(rr["delta_to_member"])
            out.at[idx, MEMBER_COLS[from_m]] += float(rr["delta_from_member"])
            local_fire = 1
            local_rule = f"{rr['direction']}|{rr['dominance_state']}|{rr['gap_band']}|{rr['ratio_band']}"
            break
        fired.append(local_fire)
        fired_rule.append(local_rule)
    out["CorrectionFired"] = fired
    out["CorrectionRule"] = fired_rule
    return out

# ============================================================
# Event table builders
# ============================================================
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
                        "date": cur["date"],
                        "stream": canon_stream(stream),
                        "feat_seed": prev_result,
                        "true_member": true_member,
                        "PlayDate": cur["date"].strftime("%Y-%m-%d"),
                        "StreamKey": canon_stream(stream),
                        "PrevSeed_text": prev_result,
                        "WinningMember_text": true_member,
                        "PrevDrawDate": prev_date.strftime("%Y-%m-%d") if pd.notna(prev_date) else "",
                    })
            prev_result, prev_date = cur["result4"], cur["date"]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = ensure_feature_columns(out)
    return out.sort_values(["date", "stream"]).reset_index(drop=True)

def build_latest_candidate_table(raw_norm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, grp in raw_norm.groupby("stream", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        if grp.empty:
            continue
        latest = grp.iloc[-1]
        rows.append({
            "date": latest["date"],
            "stream": canon_stream(stream),
            "feat_seed": latest["result4"],
            "true_member": "",
            "PlayDate": latest["date"].strftime("%Y-%m-%d"),
            "StreamKey": canon_stream(stream),
            "PrevSeed_text": latest["result4"],
            "WinningMember_text": "",
            "PrevDrawDate": latest["date"].strftime("%Y-%m-%d"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = ensure_feature_columns(out)
    return out.sort_values(["stream"]).reset_index(drop=True)

# ============================================================
# Core scoring and walk-forward
# ============================================================
def compute_base_member_scores(train_df: pd.DataFrame, row: pd.Series, recent_n: int = 12) -> Dict[str, float]:
    if train_df.empty:
        return {m: 1.0 for m in MEMBERS}
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
        stream_rate = (stream_hist["true_member"] == m).sum() / total_stream if not stream_hist.empty else 1.0 / len(MEMBERS)
        recent_rate = (recent_stream["true_member"] == m).sum() / total_recent if not recent_stream.empty else 1.0 / len(MEMBERS)
        due_bonus = 0.0
        if not stream_hist.empty:
            sub = stream_hist[stream_hist["true_member"] == m]
            if not sub.empty:
                last_date = sub["date"].max()
                days_gap = max(0, (pd.Timestamp(row["date"]) - pd.Timestamp(last_date)).days)
                due_bonus = min(2.5, math.log1p(days_gap) / 3.0)
            else:
                due_bonus = 1.0
        scores[m] = 2.0 * global_rate + 3.2 * stream_rate + 2.8 * recent_rate + due_bonus
    return scores

def rank_within_date(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["StreamScore"] = out["Top1Score"] * 1000 + out["Top1Margin"] * 100 + out["score_0025"] + out["score_0225"] + out["score_0255"]
    out = out.sort_values(["date", "StreamScore", "Top1Margin", "stream"], ascending=[True, False, False, True]).copy()
    out["Rank"] = out.groupby("date").cumcount() + 1
    out["Selected50"] = (out["Rank"] <= top_n).astype(int)
    out["CorrectMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out["true_member"])).astype(int)
    out["Top2Needed"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] == out["true_member"])).astype(int)
    out["Top3Only"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] != out["true_member"]) & (out["Top3_pred"] == out["true_member"])).astype(int)
    out["PrevSeedHas9"] = out["feat_seed"].fillna("").astype(str).str.contains("9").astype(int)
    return out

def summarize_results(df: pd.DataFrame) -> Dict[str, int]:
    selected = df[df["Selected50"] == 1]
    return {
        "Selected50": int(selected.shape[0]),
        "Correct-member": int(selected["CorrectMember"].sum() + selected["Top2Needed"].sum()),
        "Top1": int(selected["CorrectMember"].sum()),
        "Top2-needed": int(selected["Top2Needed"].sum()),
        "Top3-only": int(selected["Top3Only"].sum()),
    }

def build_date_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("date", dropna=False).agg(
        total_rows=("stream", "size"),
        selected50=("Selected50", "sum"),
        correct_member=("CorrectMember", "sum"),
        top2_needed=("Top2Needed", "sum"),
        top3_only=("Top3Only", "sum"),
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    grp["date"] = grp["date"].astype(str)
    return grp

def build_stream_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("stream", dropna=False).agg(
        rows=("date", "size"),
        selected50=("Selected50", "sum"),
        correct_member=("CorrectMember", "sum"),
        top2_needed=("Top2Needed", "sum"),
        top3_only=("Top3Only", "sum"),
        avg_rank=("Rank", "mean"),
        avg_streamscore=("StreamScore", "mean"),
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    return grp.sort_values(["correct_member", "avg_streamscore"], ascending=[False, False])

def run_walkforward_lab(events_df: pd.DataFrame, overlay_rules: List[OverlayRule], tie_df: pd.DataFrame, correction_rules: pd.DataFrame, top_n: int, recent_n: int, tie_margin_threshold: float) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()
    work = events_df.sort_values(["date", "stream"]).reset_index(drop=True).copy()
    scored_rows = []
    unique_dates = sorted(work["date"].dropna().unique())

    for d in unique_dates:
        train_df = work[work["date"] < d].copy()
        test_df = work[work["date"] == d].copy()
        if train_df.empty or test_df.empty:
            continue

        for idx, row in test_df.iterrows():
            base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
            for m in MEMBERS:
                test_df.at[idx, MEMBER_COLS[m]] = base_scores[m]

        test_df = rank_members(test_df)
        test_df = apply_tie_pack(test_df, tie_df, tie_margin_threshold=tie_margin_threshold)
        test_df = rank_members(test_df)
        test_df = apply_overlay_to_scores(test_df, overlay_rules)
        test_df = rank_members(test_df)
        test_df = apply_correction_rules(test_df, correction_rules)
        test_df = rank_members(test_df)

        scored_rows.append(test_df)

    if not scored_rows:
        return pd.DataFrame()

    ranked = pd.concat(scored_rows, ignore_index=True)
    ranked = rank_within_date(ranked, top_n=top_n)
    return ranked.sort_values(["date", "Rank"]).reset_index(drop=True)

def build_daily_playlist(raw_norm: pd.DataFrame, hit_events_df: pd.DataFrame, overlay_rules: List[OverlayRule], tie_df: pd.DataFrame, correction_rules: pd.DataFrame, top_n: int, recent_n: int, tie_margin_threshold: float) -> pd.DataFrame:
    latest_df = build_latest_candidate_table(raw_norm)
    if latest_df.empty:
        return latest_df
    train_df = hit_events_df.copy().sort_values("date").reset_index(drop=True)
    for idx, row in latest_df.iterrows():
        base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
        for m in MEMBERS:
            latest_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
    latest_df = rank_members(latest_df)
    latest_df = apply_tie_pack(latest_df, tie_df, tie_margin_threshold=tie_margin_threshold)
    latest_df = rank_members(latest_df)
    latest_df = apply_overlay_to_scores(latest_df, overlay_rules)
    latest_df = rank_members(latest_df)
    latest_df = apply_correction_rules(latest_df, correction_rules)
    latest_df = rank_members(latest_df)

    latest_df["StreamScore"] = latest_df["Top1Score"] * 1000 + latest_df["Top1Margin"] * 100 + latest_df["score_0025"] + latest_df["score_0225"] + latest_df["score_0255"]
    latest_df = latest_df.sort_values(["StreamScore", "Top1Margin", "stream"], ascending=[False, False, True]).reset_index(drop=True)
    latest_df["Rank"] = np.arange(1, len(latest_df) + 1)
    latest_df["Selected50"] = (latest_df["Rank"] <= int(top_n)).astype(int)
    latest_df["PlayPlan"] = np.where(latest_df["Top1Margin"] < 0.75, "Top1+Top2", "Top1")
    return latest_df

# ============================================================
# Reports
# ============================================================
def build_dictionary_report(feature_table: pd.DataFrame, overlay_rules: List[OverlayRule], tie_df: pd.DataFrame, correction_rules: pd.DataFrame) -> pd.DataFrame:
    feature_cols = set(feature_table.columns)
    rows = []
    for rule in overlay_rules:
        for cond_col, val in rule.conditions.items():
            rows.append({
                "layer": "overlay",
                "trait_raw": f"{cond_col}={val}",
                "feature_token": cond_col.replace("when_", ""),
                "mapped_feature": cond_col.replace("when_", "") if cond_col.replace("when_", "") in feature_cols else "",
                "exists_in_feature_table": cond_col.replace("when_", "") in feature_cols,
                "status": "valid" if cond_col.replace("when_", "") in feature_cols else "control_or_missing",
                "rule_id": rule.rule_id,
            })
    if not tie_df.empty:
        for _, rr in tie_df.iterrows():
            feature = str(rr.get("feature", "")).strip()
            rows.append({
                "layer": "tie_pack",
                "trait_raw": f"{feature}{rr.get('op','')}{rr.get('value','')}",
                "feature_token": feature,
                "mapped_feature": feature if feature in feature_cols else "",
                "exists_in_feature_table": feature in feature_cols,
                "status": "valid" if feature in feature_cols else "missing_feature",
                "rule_id": f"tie::{feature}",
            })
    if not correction_rules.empty:
        for _, rr in correction_rules.iterrows():
            rows.append({
                "layer": "correction",
                "trait_raw": f"{rr['direction']}|{rr['dominance_state']}|{rr['gap_band']}|{rr['ratio_band']}",
                "feature_token": "directional_bucket",
                "mapped_feature": "Top1/Top2/Top3 + gap/ratio state",
                "exists_in_feature_table": True,
                "status": "derived",
                "rule_id": rr["direction"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["layer","trait_raw","feature_token","mapped_feature","exists_in_feature_table","status","rule_id"])
    return out.drop_duplicates().sort_values(["layer", "status", "feature_token"]).reset_index(drop=True)

def build_firing_report(scored_df: pd.DataFrame, overlay_rules: List[OverlayRule]) -> pd.DataFrame:
    rows = []
    if "FiredRuleIDs" in scored_df.columns:
        for rule in overlay_rules:
            mask = scored_df["FiredRuleIDs"].fillna("").astype(str).str.contains(re.escape(rule.rule_id), regex=True)
            sub = scored_df[mask]
            rows.append({
                "layer": "overlay",
                "rule_id": rule.rule_id,
                "rows_matched": int(sub.shape[0]),
                "selected50_rows": int(sub["Selected50"].sum()) if "Selected50" in sub.columns else 0,
                "top1_correct_rows": int(sub["CorrectMember"].sum()) if "CorrectMember" in sub.columns else 0,
                "top2_needed_rows": int(sub["Top2Needed"].sum()) if "Top2Needed" in sub.columns else 0,
            })
    if "TieFiredRules" in scored_df.columns:
        cnt = Counter()
        for s in scored_df["TieFiredRules"].fillna("").astype(str):
            if not s:
                continue
            for part in s.split("|"):
                if part:
                    cnt[part] += 1
        for rid, n in cnt.items():
            rows.append({"layer": "tie_pack", "rule_id": rid, "rows_matched": int(n), "selected50_rows": 0, "top1_correct_rows": 0, "top2_needed_rows": 0})
    if "CorrectionRule" in scored_df.columns:
        cnt = Counter(scored_df["CorrectionRule"].fillna("").astype(str))
        for rid, n in cnt.items():
            if rid:
                rows.append({"layer": "correction", "rule_id": rid, "rows_matched": int(n), "selected50_rows": 0, "top1_correct_rows": 0, "top2_needed_rows": 0})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["layer","rule_id","rows_matched","selected50_rows","top1_correct_rows","top2_needed_rows"])
    return out.sort_values(["layer", "rows_matched"], ascending=[True, False]).reset_index(drop=True)

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Core025 Master Goal Lab", layout="wide")
st.title("Core025 Master Goal Lab")
st.caption(BUILD_MARKER)
st.warning("Tabs separate DAILY and LAB. Use the Run buttons; nothing auto-runs.")

with st.sidebar:
    st.header("Core Inputs")
    history_upload = st.file_uploader("Upload raw full history file", type=["txt", "csv", "tsv", "xlsx", "xls"], key="history")
    last24_upload = st.file_uploader("Upload last 24h file (optional)", type=["txt", "csv", "tsv", "xlsx", "xls"], key="last24")
    top_n = st.number_input("Top-N cutoff", min_value=1, max_value=200, value=50, step=1)
    recent_n = st.number_input("Recent stream window", min_value=3, max_value=100, value=12, step=1)
    tie_margin_threshold = st.number_input("Tie margin threshold", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
    st.header("Optional Packs / Audit")
    overlay_upload = st.file_uploader("Overlay CSV/TXT", type=["csv", "txt"], key="overlay")
    tie_upload = st.file_uploader("Tie pack CSV", type=["csv"], key="tie")
    audit_event_upload = st.file_uploader("Decision audit event buckets CSV", type=["csv"], key="audit_event")
    audit_pattern_upload = st.file_uploader("Decision audit failure patterns CSV", type=["csv"], key="audit_pattern")
    min_corr_events = st.number_input("Min correction events", min_value=1, max_value=50, value=3, step=1)
    min_corr_rate = st.number_input("Min correction success rate", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    show_debug = st.checkbox("Show debug tables", value=False)

tab_daily, tab_lab = st.tabs(["DAILY", "LAB"])

if history_upload is None:
    st.info("Upload the raw full history file to begin.")
    st.stop()

# shared normalized input
raw_hist = normalize_raw_history(load_table(history_upload))
if last24_upload is not None:
    raw_24 = normalize_raw_history(load_table(last24_upload))
    raw_norm = pd.concat([raw_hist, raw_24], ignore_index=True).drop_duplicates(subset=["date","stream","result4"]).sort_values(["stream","date","game","result4"]).reset_index(drop=True)
else:
    raw_norm = raw_hist

hit_events = build_hit_event_feature_table(raw_norm)
overlay_rules, overlay_meta = parse_overlay_file(overlay_upload) if overlay_upload is not None else ([], {"source": "none", "rows": 0, "filename": ""})
tie_df = parse_tie_pack(tie_upload)
audit_event_df = parse_event_buckets(audit_event_upload)
audit_pattern_df = parse_failure_patterns(audit_pattern_upload)
correction_rules = mine_correction_rules(audit_event_df, min_events=int(min_corr_events), min_success_rate=float(min_corr_rate)) if not audit_event_df.empty else pd.DataFrame()

with tab_daily:
    st.subheader("Daily playlist")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Raw rows", int(raw_norm.shape[0]))
    with c2: st.metric("Streams", int(raw_norm["stream"].nunique()))
    with c3: st.metric("Core025 hit events", int(hit_events.shape[0]))
    with c4: st.metric("Correction rules loaded", int(correction_rules.shape[0]))

    run_daily = st.button("Run Daily Playlist", type="primary", use_container_width=True, key="run_daily")
    if run_daily:
        playlist = build_daily_playlist(
            raw_norm, hit_events, overlay_rules, tie_df, correction_rules,
            top_n=int(top_n), recent_n=int(recent_n), tie_margin_threshold=float(tie_margin_threshold)
        )
        if playlist.empty:
            st.error("No daily playlist rows were produced.")
        else:
            playlist_export = playlist[[
                "PlayDate", "StreamKey", "PrevSeed_text", "PredictedMember", "Top1_pred", "Top2_pred", "Top3_pred",
                "score_0025", "score_0225", "score_0255", "Top1Score", "Top1Margin", "StreamScore", "Rank", "Selected50", "PlayPlan",
                "TieFired", "TieFiredRules", "CorrectionFired", "CorrectionRule", "FiredRuleIDs", "FiredRuleNotes"
            ]].rename(columns={"Top1_pred": "Top1", "Top2_pred": "Top2", "Top3_pred": "Top3"})
            st.dataframe(playlist_export.head(int(top_n)), use_container_width=True, hide_index=True)

            st.download_button("Download daily playlist CSV", data=bytes_csv(playlist_export),
                               file_name="daily_playlist__core025_master_goal_lab_v3.csv", mime="text/csv", use_container_width=True)
            st.download_button("Download daily playlist TXT", data=bytes_txt(playlist_export),
                               file_name="daily_playlist__core025_master_goal_lab_v3.txt", mime="text/plain", use_container_width=True)

            if show_debug:
                st.dataframe(playlist_export.head(200), use_container_width=True)

with tab_lab:
    st.subheader("Lab walk-forward + correction mining")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Overlay rules", int(len(overlay_rules)))
    with c2: st.metric("Tie rules rows", int(tie_df.shape[0]))
    with c3: st.metric("Audit events", int(audit_event_df.shape[0]))
    with c4: st.metric("Correction rules mined", int(correction_rules.shape[0]))

    run_lab = st.button("Run Walk-Forward Lab", type="primary", use_container_width=True, key="run_lab")
    if run_lab:
        ranked = run_walkforward_lab(
            hit_events, overlay_rules, tie_df, correction_rules,
            top_n=int(top_n), recent_n=int(recent_n), tie_margin_threshold=float(tie_margin_threshold)
        )
        if ranked.empty:
            st.error("Walk-forward produced no scored rows.")
        else:
            overall = summarize_results(ranked)
            selected = ranked[ranked["Selected50"] == 1].copy()
            playable_count = int(selected.shape[0])
            capture_rate = (overall["Correct-member"] / playable_count) if playable_count else 0.0

            st.subheader("Goal check")
            g1, g2, g3, g4 = st.columns(4)
            with g1: st.metric("Playable winner events", playable_count)
            with g2: st.metric("Correct-member capture", f"{capture_rate:.2%}")
            with g3: st.metric("Top1", overall["Top1"])
            with g4: st.metric("Top2-needed", overall["Top2-needed"])
            if capture_rate >= 0.75:
                st.success("75%+ capture goal met.")
            else:
                st.error("75%+ capture goal NOT met.")

            st.subheader("Quick test summary")
            st.code(textwrap.dedent(f"""\
Selected50: {overall['Selected50']}
Correct-member: {overall['Correct-member']}
Top1: {overall['Top1']}
Top2-needed: {overall['Top2-needed']}
Top3-only: {overall['Top3-only']}
CaptureRate: {capture_rate:.2%}
"""))

            dictionary_report = build_dictionary_report(hit_events, overlay_rules, tie_df, correction_rules)
            firing_report = build_firing_report(ranked, overlay_rules)

            per_event_export = ranked[[
                "PlayDate", "StreamKey", "PrevSeed_text", "WinningMember_text", "PredictedMember", "Top1_pred", "Top2_pred", "Top3_pred",
                "score_0025", "score_0225", "score_0255", "Top1Score", "Top1Margin", "StreamScore", "Rank", "Selected50",
                "CorrectMember", "Top2Needed", "Top3Only", "TieFired", "TieFiredRules", "CorrectionFired", "CorrectionRule",
                "FiredRuleIDs", "FiredRuleNotes"
            ]].rename(columns={"Top1_pred": "Top1", "Top2_pred": "Top2", "Top3_pred": "Top3"})
            per_date_export = build_date_export(ranked)
            per_stream_export = build_stream_export(ranked)
            correction_export = correction_rules.copy()

            st.subheader("Downloads")
            download_specs = [
                ("per_event__core025_master_goal_lab_v3.csv", per_event_export, "Download per-event CSV"),
                ("per_event__core025_master_goal_lab_v3.txt", per_event_export, "Download per-event TXT"),
                ("per_date__core025_master_goal_lab_v3.csv", per_date_export, "Download per-date CSV"),
                ("per_date__core025_master_goal_lab_v3.txt", per_date_export, "Download per-date TXT"),
                ("per_stream__core025_master_goal_lab_v3.csv", per_stream_export, "Download per-stream CSV"),
                ("per_stream__core025_master_goal_lab_v3.txt", per_stream_export, "Download per-stream TXT"),
                ("dictionary_report__core025_master_goal_lab_v3.csv", dictionary_report, "Download dictionary report CSV"),
                ("dictionary_report__core025_master_goal_lab_v3.txt", dictionary_report, "Download dictionary report TXT"),
                ("firing_report__core025_master_goal_lab_v3.csv", firing_report, "Download firing report CSV"),
                ("firing_report__core025_master_goal_lab_v3.txt", firing_report, "Download firing report TXT"),
                ("correction_rules__core025_master_goal_lab_v3.csv", correction_export, "Download correction rules CSV"),
                ("correction_rules__core025_master_goal_lab_v3.txt", correction_export, "Download correction rules TXT"),
            ]
            for i in range(0, len(download_specs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j >= len(download_specs):
                        continue
                    fname, df_exp, label = download_specs[i + j]
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
