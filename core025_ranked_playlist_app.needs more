#!/usr/bin/env python3
# BUILD: core025_master_goal_lab__2026-04-17_v1

from __future__ import annotations
import io, math, re, textwrap
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_master_goal_lab__2026-04-17_v1"
APP_VERSION_STR = "core025_master_goal_lab__2026-04-17_v1"
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
    if s in {"25","025","0025"}: return "0025"
    if s in {"225","0225"}: return "0225"
    if s in {"255","0255"}: return "0255"
    return s.zfill(4)

def clean_seed_text(x) -> str:
    s = str(x).strip().replace("'", "")
    if s == "" or s.lower() == "nan":
        return ""
    s = re.sub(r"\D", "", s)
    return s.zfill(4) if s else ""

def to_int_or_none(x):
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def parse_digit_set(value) -> Optional[set]:
    if value is None: return None
    try:
        if pd.isna(value): return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan": return None
    s = s.strip("[](){}")
    vals = []
    for p in re.split(r"[\s,|;/]+", s):
        if not p: continue
        try: vals.append(int(float(p)))
        except Exception: pass
    return set(vals) if vals else None

def parse_text_set(value) -> Optional[set]:
    if value is None: return None
    try:
        if pd.isna(value): return None
    except Exception:
        pass
    s = str(value).strip()
    if s == "" or s.lower() == "nan": return None
    s = s.strip("[](){}")
    parts = [p.strip().upper() for p in re.split(r"[\s,|;/]+", s) if p.strip()]
    return set(parts) if parts else None

def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None, dtype=str)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None, dtype=str)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, dtype=str)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")

TIME_PATTERNS = [
    ("11:30pm","1130pm"),("7:50pm","750pm"),("1:50pm","150pm"),
    ("morning","morning"),("midday","midday"),("daytime","daytime"),
    ("day","day"),("evening","evening"),("night","night"),
    ("1pm","1pm"),("4pm","4pm"),("7pm","7pm"),("10pm","10pm"),
    ("noche","noche"),("día","dia"),("dia","dia"),
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
    s_low = s_low.replace("7:50pm","").replace("1:50pm","").replace("11:30pm","")
    s_low = re.sub(r"[^a-z0-9]+", " ", s_low).strip()
    return s_low.replace(" ", "")

def time_class(game_text: str) -> str:
    s = str(game_text).strip().lower()
    for raw, canon in TIME_PATTERNS:
        if raw in s: return canon
    return "unknown"

def canonical_stream(state: str, game_text: str) -> str:
    return f"{str(state).strip()} | {base_game_name(game_text)} | {time_class(game_text)}"

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
    out.columns = ["date_text","state","game","result_text"]
    out["date"] = pd.to_datetime(out["date_text"], errors="coerce")
    out["state"] = out["state"].astype(str).str.strip()
    out["game"] = out["game"].astype(str).str.strip()
    out["result4"] = out["result_text"].apply(extract_digits_result).map(clean_seed_text)
    out["stream"] = out.apply(lambda r: canonical_stream(r["state"], r["game"]), axis=1)
    out["time_class"] = out["game"].apply(time_class)
    out["base_game"] = out["game"].apply(base_game_name)
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
    out["stream"] = out["stream"].astype(str).str.strip().str.lower()
    out["feat_seed"] = out["feat_seed"].fillna("").astype(str).map(clean_seed_text)
    out["true_member"] = out["true_member"].fillna("").astype(str).map(normalize_member)

    digits = out["feat_seed"].apply(_digits_from_seed)
    out["seed_pos1"] = [d[0] for d in digits]
    out["seed_pos2"] = [d[1] for d in digits]
    out["seed_pos3"] = [d[2] for d in digits]
    out["seed_pos4"] = [d[3] for d in digits]
    out["seed_sum"] = [sum(d) if None not in d else np.nan for d in digits]
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
    out["seed_spread"] = [max(d)-min(d) if None not in d else np.nan for d in digits]
    out["seed_unique_digits"] = [len(set(d)) if None not in d else np.nan for d in digits]
    out["seed_even_cnt"] = [sum(1 for x in d if x % 2 == 0) if None not in d else np.nan for d in digits]
    out["seed_odd_cnt"] = [sum(1 for x in d if x % 2 == 1) if None not in d else np.nan for d in digits]
    out["seed_high_cnt"] = [sum(1 for x in d if x >= 5) if None not in d else np.nan for d in digits]
    out["seed_low_cnt"] = [sum(1 for x in d if x <= 4) if None not in d else np.nan for d in digits]
    out["seed_first_last_sum"] = [d[0]+d[3] if None not in d else np.nan for d in digits]
    out["seed_middle_sum"] = [d[1]+d[2] if None not in d else np.nan for d in digits]
    out["seed_pairwise_absdiff_sum"] = [abs(d[0]-d[1]) + abs(d[0]-d[2]) + abs(d[0]-d[3]) + abs(d[1]-d[2]) + abs(d[1]-d[3]) + abs(d[2]-d[3]) if None not in d else np.nan for d in digits]
    out["seed_adj_absdiff_sum"] = [abs(d[0]-d[1])+abs(d[1]-d[2])+abs(d[2]-d[3]) if None not in d else np.nan for d in digits]
    out["seed_adj_absdiff_min"] = [min(abs(d[0]-d[1]),abs(d[1]-d[2]),abs(d[2]-d[3])) if None not in d else np.nan for d in digits]
    out["seed_highlow_pattern"] = [_highlow_pattern(d) for d in digits]
    out["seed_parity_pattern"] = [_parity_pattern(d) for d in digits]
    out["seed_repeat_shape"] = [_repeat_shape(d) for d in digits]
    out["cnt_0_3"] = [sum(1 for x in d if 0 <= x <= 3) if None not in d else np.nan for d in digits]
    out["cnt_4_6"] = [sum(1 for x in d if 4 <= x <= 6) if None not in d else np.nan for d in digits]
    out["cnt_7_9"] = [sum(1 for x in d if 7 <= x <= 9) if None not in d else np.nan for d in digits]
    out["seed_has9"] = out["feat_seed"].str.contains("9").astype(int)
    out["seed_has0"] = out["feat_seed"].str.contains("0").astype(int)

    def vtrac_group(x:int) -> int: return x % 5
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
        if len(s) != 4: return []
        outp = []
        for i in range(4):
            for j in range(i+1, 4):
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
    if uploaded_file.name.lower().endswith(".txt"):
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    else:
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
            enabled=enabled_bool,
            conditions=conditions,
            deltas=deltas,
            note=str(row.get("note","")),
        ))
    return rules, {"source":"upload","rows":int(df.shape[0]),"filename":uploaded_file.name}

def match_rule_to_row(rule: OverlayRule, row: pd.Series) -> bool:
    if not rule.enabled: return False
    def g(col, default=np.nan): return row[col] if col in row.index else default

    if "when_base_top1" in rule.conditions and normalize_member(g("Top1_pred","")) != normalize_member(rule.conditions["when_base_top1"]):
        return False
    if "when_base_top2" in rule.conditions and normalize_member(g("Top2_pred","")) != normalize_member(rule.conditions["when_base_top2"]):
        return False
    if "when_base_top3" in rule.conditions and normalize_member(g("Top3_pred","")) != normalize_member(rule.conditions["when_base_top3"]):
        return False

    numeric_exact_map = {
        "when_seed_sum_in":"seed_sum", "when_seed_sum_lastdigit_in":"seed_sum_lastdigit",
        "when_seed_sum_lastdigit_not_in":"seed_sum_lastdigit", "when_seed_sum_mod3_in":"seed_sum_mod3",
        "when_seed_sum_mod5_in":"seed_sum_mod5", "when_seed_sum_mod6_in":"seed_sum_mod6",
        "when_seed_sum_mod9_in":"seed_sum_mod9", "when_seed_sum_mod10_in":"seed_sum_mod10",
        "when_seed_sum_mod11_in":"seed_sum_mod11", "when_seed_sum_mod12_in":"seed_sum_mod12",
        "when_seed_sum_mod13_in":"seed_sum_mod13", "when_seed_root_in":"seed_root",
        "when_seed_root_not_in":"seed_root", "when_seed_first_last_sum_in":"seed_first_last_sum",
        "when_seed_first_last_sum_not_in":"seed_first_last_sum", "when_seed_middle_sum_in":"seed_middle_sum",
        "when_seed_middle_sum_not_in":"seed_middle_sum", "when_seed_pairwise_absdiff_sum_in":"seed_pairwise_absdiff_sum",
        "when_seed_pairwise_absdiff_sum_not_in":"seed_pairwise_absdiff_sum",
        "when_seed_adj_absdiff_sum_in":"seed_adj_absdiff_sum", "when_seed_adj_absdiff_sum_not_in":"seed_adj_absdiff_sum",
        "when_seed_adj_absdiff_min_in":"seed_adj_absdiff_min", "when_seed_adj_absdiff_min_not_in":"seed_adj_absdiff_min",
        "when_seed_pos1_in":"seed_pos1", "when_seed_pos1_not_in":"seed_pos1",
        "when_seed_pos2_in":"seed_pos2", "when_seed_pos2_not_in":"seed_pos2",
        "when_seed_pos3_in":"seed_pos3", "when_seed_pos3_not_in":"seed_pos3",
        "when_seed_pos4_in":"seed_pos4", "when_seed_pos4_not_in":"seed_pos4",
    }
    for cond_col, feat_col in numeric_exact_map.items():
        if cond_col in rule.conditions:
            vals = parse_digit_set(rule.conditions[cond_col])
            rv = to_int_or_none(g(feat_col))
            if rv is None: return False
            if cond_col.endswith("_not_in"):
                if vals and rv in vals: return False
            else:
                if vals and rv not in vals: return False

    numeric_range_map = {
        "when_seed_sum_min":("seed_sum","min"), "when_seed_sum_max":("seed_sum","max"),
        "when_seed_spread_min":("seed_spread","min"), "when_seed_spread_max":("seed_spread","max"),
        "when_seed_high_min":("seed_high_cnt","min"), "when_seed_high_max":("seed_high_cnt","max"),
        "when_seed_low_min":("seed_low_cnt","min"), "when_seed_low_max":("seed_low_cnt","max"),
        "when_seed_vtrac_groups_min":("seed_vtrac_groups","min"), "when_seed_vtrac_groups_max":("seed_vtrac_groups","max"),
        "when_seed_count_digits_min":("seed_unique_digits","min"), "when_seed_count_digits_max":("seed_unique_digits","max"),
        "when_seed_first_last_sum_min":("seed_first_last_sum","min"), "when_seed_first_last_sum_max":("seed_first_last_sum","max"),
        "when_seed_middle_sum_min":("seed_middle_sum","min"), "when_seed_middle_sum_max":("seed_middle_sum","max"),
        "when_seed_pairwise_absdiff_sum_min":("seed_pairwise_absdiff_sum","min"), "when_seed_pairwise_absdiff_sum_max":("seed_pairwise_absdiff_sum","max"),
        "when_seed_adj_absdiff_sum_min":("seed_adj_absdiff_sum","min"), "when_seed_adj_absdiff_sum_max":("seed_adj_absdiff_sum","max"),
        "when_seed_adj_absdiff_min_min":("seed_adj_absdiff_min","min"), "when_seed_adj_absdiff_min_max":("seed_adj_absdiff_min","max"),
    }
    for cond_col, (feat_col, kind) in numeric_range_map.items():
        if cond_col in rule.conditions:
            rv = to_int_or_none(g(feat_col)); cv = to_int_or_none(rule.conditions[cond_col])
            if rv is None or cv is None: return False
            if kind == "min" and rv < cv: return False
            if kind == "max" and rv > cv: return False

    if "when_seed_count_digits_set" in rule.conditions:
        digit_set = parse_digit_set(rule.conditions["when_seed_count_digits_set"])
        if not digit_set: return False
        cnt = 0
        for d in digit_set:
            cnt += int(to_int_or_none(g(f"seed_cnt{d}")) or 0)
        mn = to_int_or_none(rule.conditions.get("when_seed_count_digits_min", None))
        mx = to_int_or_none(rule.conditions.get("when_seed_count_digits_max", None))
        if mn is not None and cnt < mn: return False
        if mx is not None and cnt > mx: return False

    seed_text = clean_seed_text(g("feat_seed",""))
    if "when_seed_contains_any" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_any"])
        if vals and not any(str(v) in seed_text for v in vals): return False
    if "when_seed_contains_all" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_all"])
        if vals and not all(str(v) in seed_text for v in vals): return False
    if "when_seed_contains_none" in rule.conditions:
        vals = parse_digit_set(rule.conditions["when_seed_contains_none"])
        if vals and any(str(v) in seed_text for v in vals): return False

    if "when_seed_highlow_pattern" in rule.conditions:
        allowed = parse_text_set(rule.conditions["when_seed_highlow_pattern"])
        if allowed and str(g("seed_highlow_pattern","")).upper() not in allowed:
            return False
    return True

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
                        "date": cur["date"], "stream": stream.lower(), "feat_seed": prev_result, "true_member": true_member,
                        "PlayDate": cur["date"].strftime("%Y-%m-%d"), "StreamKey": stream, "PrevSeed_text": prev_result,
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
            "date": latest["date"], "stream": stream.lower(), "feat_seed": latest["result4"], "true_member": "",
            "PlayDate": latest["date"].strftime("%Y-%m-%d"), "StreamKey": stream, "PrevSeed_text": latest["result4"],
            "WinningMember_text": "", "PrevDrawDate": latest["date"].strftime("%Y-%m-%d"),
        })
    out = pd.DataFrame(rows)
    if out.empty: return out
    out = ensure_feature_columns(out)
    return out.sort_values(["stream"]).reset_index(drop=True)

def compute_base_member_scores(train_df: pd.DataFrame, row: pd.Series, recent_n: int = 12) -> Dict[str, float]:
    if train_df.empty:
        return {m: 1.0 for m in MEMBERS}
    stream = str(row["stream"]).lower()
    global_hist = train_df.copy()
    stream_hist = train_df[train_df["stream"].astype(str).str.lower().eq(stream)].copy()
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
        scores[m] = 2.0*global_rate + 3.0*stream_rate + 2.5*recent_rate + due_bonus
    return scores

def apply_overlay_to_scores(df: pd.DataFrame, rules: List[OverlayRule]) -> pd.DataFrame:
    out = df.copy()
    if not rules:
        out["FiredRuleIDs"] = ""
        out["FiredRuleNotes"] = ""
        return out
    rule_ids_per_row, notes_per_row = [], []
    for _, row in out.iterrows():
        row_rule_ids, row_notes = [], []
        for rule in rules:
            if match_rule_to_row(rule, row):
                for m in MEMBERS:
                    out.at[row.name, MEMBER_COLS[m]] += float(rule.deltas.get(m, 0.0))
                row_rule_ids.append(rule.rule_id)
                if rule.note: row_notes.append(rule.note)
        rule_ids_per_row.append("|".join(row_rule_ids))
        notes_per_row.append(" || ".join(row_notes))
    out["FiredRuleIDs"] = rule_ids_per_row
    out["FiredRuleNotes"] = notes_per_row
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

def rank_within_date(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["StreamScore"] = out["Top1Score"]*1000 + out["Top1Margin"]*100 + out["score_0025"] + out["score_0225"] + out["score_0255"]
    out = out.sort_values(["date","StreamScore","Top1Margin","stream"], ascending=[True,False,False,True]).copy()
    out["Rank"] = out.groupby("date").cumcount() + 1
    out["Selected50"] = (out["Rank"] <= top_n).astype(int)
    out["CorrectMember"] = ((out["Selected50"] == 1) & (out["PredictedMember"] == out["true_member"])).astype(int)
    out["Top2Needed"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] == out["true_member"])).astype(int)
    out["Top3Only"] = ((out["Selected50"] == 1) & (out["PredictedMember"] != out["true_member"]) & (out["Top2_pred"] != out["true_member"]) & (out["Top3_pred"] == out["true_member"])).astype(int)
    out["PrevSeedHas9"] = out["feat_seed"].fillna("").astype(str).str.contains("9").astype(int)
    return out

def summarize_results(df: pd.DataFrame) -> Dict[str,int]:
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
        total_rows=("stream","size"), selected50=("Selected50","sum"),
        correct_member=("CorrectMember","sum"), top2_needed=("Top2Needed","sum"), top3_only=("Top3Only","sum")
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    grp["date"] = grp["date"].astype(str)
    return grp

def build_stream_export(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("stream", dropna=False).agg(
        rows=("date","size"), selected50=("Selected50","sum"),
        correct_member=("CorrectMember","sum"), top2_needed=("Top2Needed","sum"), top3_only=("Top3Only","sum"),
        avg_rank=("Rank","mean"), avg_streamscore=("StreamScore","mean")
    ).reset_index()
    grp["top1"] = grp["correct_member"]
    return grp.sort_values(["correct_member","avg_streamscore"], ascending=[False,False])

def run_walkforward_lab(events_df: pd.DataFrame, overlay_rules: List[OverlayRule], top_n: int, recent_n: int) -> pd.DataFrame:
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
            for m in MEMBERS:
                test_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
        test_df = rank_members(test_df)
        test_df = apply_overlay_to_scores(test_df, overlay_rules)
        test_df = rank_members(test_df)
        scored_rows.append(test_df)
    if not scored_rows: return pd.DataFrame()
    ranked = pd.concat(scored_rows, ignore_index=True)
    ranked = rank_within_date(ranked, top_n=top_n)
    return ranked.sort_values(["date","Rank"]).reset_index(drop=True)

def build_daily_playlist(raw_norm: pd.DataFrame, hit_events_df: pd.DataFrame, overlay_rules: List[OverlayRule], top_n: int, recent_n: int) -> pd.DataFrame:
    latest_df = build_latest_candidate_table(raw_norm)
    if latest_df.empty: return latest_df
    train_df = hit_events_df.copy().sort_values("date").reset_index(drop=True)
    for idx, row in latest_df.iterrows():
        base_scores = compute_base_member_scores(train_df, row, recent_n=recent_n)
        for m in MEMBERS:
            latest_df.at[idx, MEMBER_COLS[m]] = base_scores[m]
    latest_df = rank_members(latest_df)
    latest_df = apply_overlay_to_scores(latest_df, overlay_rules)
    latest_df = rank_members(latest_df)
    latest_df["StreamScore"] = latest_df["Top1Score"]*1000 + latest_df["Top1Margin"]*100 + latest_df["score_0025"] + latest_df["score_0225"] + latest_df["score_0255"]
    latest_df = latest_df.sort_values(["StreamScore","Top1Margin","stream"], ascending=[False,False,True]).reset_index(drop=True)
    latest_df["Rank"] = np.arange(1, len(latest_df)+1)
    latest_df["Selected50"] = (latest_df["Rank"] <= int(top_n)).astype(int)
    latest_df["PlayPlan"] = np.where(latest_df["Top1Margin"] < 0.75, "Top1+Top2", "Top1")
    return latest_df

st.set_page_config(page_title="Core025 Master Goal Lab", layout="wide")
st.title("Core025 Master Goal Lab")
st.caption(BUILD_MARKER)
st.warning("This app ingests the raw history file directly and normalizes it internally. Walk-forward is strict: only rows with date < test date are used to score each test date.")

with st.sidebar:
    st.header("Inputs")
    history_upload = st.file_uploader("Upload raw full history file", type=["txt","csv","tsv","xlsx","xls"], help="Use your raw history file exactly as you receive it.", key="master_history_upload")
    overlay_upload = st.file_uploader("Optional member score overlay CSV/TXT", type=["csv","txt"], help="Optional clean-room separator overlay. If omitted, baseline historical scoring still runs.", key="master_overlay_upload")
    top_n = st.number_input("Top-N cutoff per date", min_value=1, max_value=200, value=50, step=1)
    recent_n = st.number_input("Recent stream window for base scoring", min_value=3, max_value=100, value=12, step=1)
    show_debug = st.checkbox("Show debug tables", value=False)

if history_upload is None:
    st.info("Upload the raw full history file to begin.")
    st.stop()

raw_loaded = load_table(history_upload)
raw_norm = normalize_raw_history(raw_loaded)
hit_events = build_hit_event_feature_table(raw_norm)
overlay_rules, overlay_meta = parse_overlay_file(overlay_upload) if overlay_upload is not None else ([], {"source":"none","rows":0,"filename":""})

st.subheader("Source normalization summary")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Raw normalized rows", int(raw_norm.shape[0]))
with c2: st.metric("Streams", int(raw_norm["stream"].nunique()))
with c3: st.metric("Core025 hit events", int(hit_events.shape[0]))
with c4: st.metric("Overlay rules", int(len(overlay_rules)))

if hit_events.empty:
    st.error("No Core025 hit events were found after normalization. Check the source file.")
    st.stop()

ranked = run_walkforward_lab(hit_events, overlay_rules, top_n=int(top_n), recent_n=int(recent_n))
if ranked.empty:
    st.error("Walk-forward produced no scored rows. This usually means there is not enough history before the first hit events.")
    st.stop()

overall = summarize_results(ranked)
selected = ranked[ranked["Selected50"] == 1].copy()
playable_count = int(selected.shape[0])
capture_rate = (overall["Correct-member"] / playable_count) if playable_count else 0.0
goal_met = capture_rate >= 0.75

st.subheader("Goal check")
g1, g2, g3, g4 = st.columns(4)
with g1: st.metric("Playable winner events", playable_count)
with g2: st.metric("Correct-member capture", f"{capture_rate:.2%}")
with g3: st.metric("Top1", overall["Top1"])
with g4: st.metric("Top2-needed", overall["Top2-needed"])
if goal_met:
    st.success("75%+ capture goal met within the current playable winner-event universe.")
else:
    st.error("75%+ capture goal NOT met in the current walk-forward run.")

st.subheader("Quick test summary")
st.code(textwrap.dedent(f"""Selected50: {overall['Selected50']}
Correct-member: {overall['Correct-member']}
Top1: {overall['Top1']}
Top2-needed: {overall['Top2-needed']}
Top3-only: {overall['Top3-only']}
CaptureRate: {capture_rate:.2%}
"""))

playlist = build_daily_playlist(raw_norm, hit_events, overlay_rules, top_n=int(top_n), recent_n=int(recent_n))
st.subheader("Daily playlist")
playlist_export = playlist[[
    "PlayDate","StreamKey","PrevSeed_text","PredictedMember","Top1_pred","Top2_pred","Top3_pred",
    "score_0025","score_0225","score_0255","Top1Score","Top1Margin","StreamScore","Rank","Selected50","PlayPlan",
    "FiredRuleIDs","FiredRuleNotes"
]].rename(columns={"Top1_pred":"Top1","Top2_pred":"Top2","Top3_pred":"Top3"})
st.dataframe(playlist_export.head(int(top_n)), use_container_width=True, hide_index=True)

per_event_export = ranked[[
    "PlayDate","StreamKey","PrevSeed_text","WinningMember_text","PredictedMember","Top1_pred","Top2_pred","Top3_pred",
    "score_0025","score_0225","score_0255","Top1Score","Top1Margin","StreamScore","Rank","Selected50",
    "CorrectMember","Top2Needed","Top3Only","FiredRuleIDs","FiredRuleNotes"
]].rename(columns={"Top1_pred":"Top1","Top2_pred":"Top2","Top3_pred":"Top3"})
per_date_export = build_date_export(ranked)
per_stream_export = build_stream_export(ranked)
normalized_history_export = raw_norm.copy(); normalized_history_export["date"] = normalized_history_export["date"].astype(str)
feature_table_export = hit_events.copy(); feature_table_export["date"] = feature_table_export["date"].astype(str)

st.subheader("Downloads")
download_specs = [
    ("per_event__core025_master_goal_lab.csv", per_event_export, "Download per-event CSV"),
    ("per_event__core025_master_goal_lab.txt", per_event_export, "Download per-event TXT"),
    ("per_date__core025_master_goal_lab.csv", per_date_export, "Download per-date CSV"),
    ("per_date__core025_master_goal_lab.txt", per_date_export, "Download per-date TXT"),
    ("per_stream__core025_master_goal_lab.csv", per_stream_export, "Download per-stream CSV"),
    ("per_stream__core025_master_goal_lab.txt", per_stream_export, "Download per-stream TXT"),
    ("daily_playlist__core025_master_goal_lab.csv", playlist_export, "Download daily playlist CSV"),
    ("daily_playlist__core025_master_goal_lab.txt", playlist_export, "Download daily playlist TXT"),
    ("normalized_history__core025_master_goal_lab.csv", normalized_history_export, "Download normalized history CSV"),
    ("normalized_history__core025_master_goal_lab.txt", normalized_history_export, "Download normalized history TXT"),
    ("feature_table__core025_master_goal_lab.csv", feature_table_export, "Download feature table CSV"),
    ("feature_table__core025_master_goal_lab.txt", feature_table_export, "Download feature table TXT"),
]
for i in range(0, len(download_specs), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i+j >= len(download_specs): continue
        fname, df_exp, label = download_specs[i+j]
        with col:
            data = bytes_csv(df_exp) if fname.endswith(".csv") else bytes_txt(df_exp)
            mime = "text/csv" if fname.endswith(".csv") else "text/plain"
            st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)

if show_debug:
    with st.expander("Normalized raw history preview", expanded=False):
        st.dataframe(raw_norm.head(200), use_container_width=True)
    with st.expander("Hit-event feature table preview", expanded=False):
        st.dataframe(hit_events.head(200), use_container_width=True)
    with st.expander("Per-event preview", expanded=False):
        st.dataframe(per_event_export.head(200), use_container_width=True)
