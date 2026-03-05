import streamlit as st
import pandas as pd
import numpy as np
import ast
import hashlib
import re
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# =========================================================
# Core 025 Family App (Full)
# - Full ranked live list (no group-percentiles)
# - Top1/Top2/Top3 always shown with scores
# - Bold indicates PLAY MEMBERS (Top1-only or Top1+Top2), not merely rank
# - 1-miss Downranks (swap Top1↔Top2) + Top3→Top1 Rescues (LOCKED + optional mined)
# - Lab backtest with "percentile by row" (literally rank row 1..N)
# =========================================================

st.set_page_config(page_title="Core025 – Streams + Member Picker", layout="wide")

MEMBERS = [25, 225, 255]

# Digit-multiset signatures for the 3 members
MEMBER_SIGS = {
    25:  "0025",
    225: "0225",
    255: "0255",
}

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _dow_name(d: dt.date) -> str:
    return DOW_NAMES[d.weekday()]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_time_midday_strict(game: str) -> int:
    """Deterministic 0/1 classifier used by rulepacks."""
    if game is None:
        return 0
    g = str(game).strip().lower()
    g = g.replace("í", "i").replace("á", "a").replace("é", "e").replace("ó", "o").replace("ú", "u")
    if "evening" in g or "night" in g or "p.m" in g or "pm" in g:
        return 0
    if "midday" in g or "noon" in g or "matinee" in g:
        return 1
    if re.search(r"\bday\b", g) and ("pick" in g or "numbers" in g or "daily" in g or "p3" in g or "p4" in g):
        return 1
    if re.search(r"\bdia\b", g):
        return 1
    return 0


# -------------------------
# Safe expression evaluator
# -------------------------

_ALLOWED_BOOL_OPS = (ast.And, ast.Or)
_ALLOWED_UNARY_OPS = (ast.Not, ast.UAdd, ast.USub)
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
_ALLOWED_CMPS = (ast.Eq, ast.NotEq, ast.Gt, ast.GtE, ast.Lt, ast.LtE)


def safe_eval_expr(expr: str, ctx: Dict[str, object]) -> bool:
    """Safe eval for rulepacks. Missing/None features => False. Never raises."""
    if expr is None:
        return False
    e = str(expr).strip()
    if not e:
        return False

    # strip inline comments
    if "#" in e:
        e = e.split("#", 1)[0].strip()

    # normalize boolean ops
    e = re.sub(r"\bAND\b", "and", e, flags=re.I)
    e = re.sub(r"\bOR\b", "or", e, flags=re.I)
    e = re.sub(r"\bNOT\b", "not", e, flags=re.I)

    try:
        tree = ast.parse(e, mode="eval")
    except Exception:
        return False

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id not in ctx:
                raise KeyError(node.id)
            v = ctx.get(node.id)
            if v is None:
                raise KeyError(node.id)
            return v

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
            v = _eval(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(v)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v

        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN_OPS):
            a = _eval(node.left)
            b = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Mod):
                return a % b

        if isinstance(node, ast.BoolOp) and isinstance(node.op, _ALLOWED_BOOL_OPS):
            vals = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(bool(v) for v in vals)
            if isinstance(node.op, ast.Or):
                return any(bool(v) for v in vals)

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            ok = True
            for op, comp in zip(node.ops, node.comparators):
                if not isinstance(op, _ALLOWED_CMPS):
                    return False
                right = _eval(comp)
                if isinstance(op, ast.Eq):
                    ok = ok and (left == right)
                elif isinstance(op, ast.NotEq):
                    ok = ok and (left != right)
                elif isinstance(op, ast.Gt):
                    ok = ok and (left > right)
                elif isinstance(op, ast.GtE):
                    ok = ok and (left >= right)
                elif isinstance(op, ast.Lt):
                    ok = ok and (left < right)
                elif isinstance(op, ast.LtE):
                    ok = ok and (left <= right)
                left = right
            return ok

        return False

    try:
        return bool(_eval(tree))
    except Exception:
        return False


def extract_names(expr: str) -> List[str]:
    if not expr:
        return []
    try:
        t = ast.parse(expr, mode="eval")
    except Exception:
        return []
    names = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name):
            names.add(node.id)

    V().visit(t)
    return sorted(names)


# -------------------------
# Feature engineering
# -------------------------


def compute_features(seed4: str) -> Dict[str, object]:
    """Seed-only features. Additive only."""
    seed4 = "" if seed4 is None else str(seed4).strip()
    if not re.fullmatch(r"\d{4}", seed4):
        return {
            "seed_sum": 0,
            "seed_spread": 0,
            "seed_absdiff": 0,
            "seed_has_worstpair_025": 0,
            "seed_sum_lastdigit": 0,
            "seed_has_9": 0,
            "seed_has9": 0,
        }

    digs = [int(ch) for ch in seed4]
    cnt = {d: digs.count(d) for d in range(10)}
    sset = set(digs)

    seed_sum = sum(digs)
    seed_spread = max(digs) - min(digs)
    seed_absdiff = abs((digs[0] + digs[1]) - (digs[2] + digs[3]))

    # all unordered 2-digit pairs
    pairs = set()
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = digs[i], digs[j]
            pairs.add(f"{min(a, b)}{max(a, b)}")

    feats: Dict[str, object] = {
        "seed_sum": int(seed_sum),
        "seed_spread": int(seed_spread),
        "seed_absdiff": int(seed_absdiff),
        "seed_sum_lastdigit": int(seed_sum % 10),
        "seed_has_worstpair_025": int(bool(pairs.intersection({"39", "55", "26", "29", "79"}))),
        "seed_has_9": int(cnt[9] > 0),
        "seed_has9": int(cnt[9] > 0),
        "seed_cnt9": int(cnt[9]),
        "seed_unique": int(len(sset) == 4),  # boolean flag: 1 = no repeats
        "seed_has_pair": int(any(v >= 2 for v in cnt.values())),
        "seed_even_cnt": int(sum(1 for d in digs if d % 2 == 0)),
    }

    for d in range(10):
        feats[f"seed_has{d}"] = int(cnt[d] > 0)
        feats[f"seed_cnt{d}"] = int(cnt[d])

    # explicit worst pair flags (additive)
    feats["seed_pair_39"] = int(3 in sset and 9 in sset)
    feats["seed_pair_29"] = int(2 in sset and 9 in sset)
    feats["seed_pair_79"] = int(7 in sset and 9 in sset)
    feats["seed_pair_26"] = int(2 in sset and 6 in sset)
    feats["seed_pair_55"] = int(cnt[5] >= 2)

    consec = 0
    for x in range(0, 9):
        if x in sset and (x + 1) in sset:
            consec += 1
    feats["seed_consec_links"] = int(consec)
    feats["seed_consec_links_wrap"] = int(consec + (1 if (9 in sset and 0 in sset) else 0))

    mirror_pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
    feats["seed_mirrorpair_cnt"] = int(sum(1 for a, b in mirror_pairs if a in sset and b in sset))

    feats["seed_adj_absdiff_sum"] = int(abs(digs[0] - digs[1]) + abs(digs[1] - digs[2]) + abs(digs[2] - digs[3]))
    feats["seed_pairwise_absdiff_sum"] = int(sum(abs(digs[i] - digs[j]) for i in range(4) for j in range(i + 1, 4)))

    return feats


def normalize_result_4d(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if len(s) != 4:
        return None
    return s


def signature_4d(s4: str) -> str:
    return "".join(sorted(s4))


def winner_is_025_family(result4: str) -> bool:
    if not result4:
        return False
    sig = signature_4d(result4)
    return sig in {"".join(sorted(MEMBER_SIGS[m])) for m in MEMBERS}


def winner_member_id(result4: str) -> Optional[int]:
    if not result4:
        return None
    sig = signature_4d(result4)
    for m, s in MEMBER_SIGS.items():
        if sig == "".join(sorted(s)):
            return m
    return None


# -------------------------
# Rule packs
# -------------------------


@dataclass
class WeightRule:
    rule_id: str
    cond_expr: str
    w25: float
    w225: float
    w255: float
    notes: str = ""


@dataclass
class TieRule:
    rule_id: str
    cond_expr: str
    tie_bias_member: int
    tie_bias_add: float = 0.25
    notes: str = ""


def load_weights(path_or_upload) -> Tuple[List[WeightRule], Dict[str, object]]:
    """Loads weights CSV. Expected columns: rule_id, cond_expr (or expr), w25, w225, w255."""
    meta: Dict[str, object] = {"source": None, "sha256": None}

    if path_or_upload is None:
        return [], meta

    if hasattr(path_or_upload, "read"):
        b = path_or_upload.read()
        meta["source"] = getattr(path_or_upload, "name", "uploaded")
        meta["sha256"] = sha256_bytes(b)
        df = pd.read_csv(pd.io.common.BytesIO(b))
    else:
        meta["source"] = str(path_or_upload)
        try:
            meta["sha256"] = sha256_file(str(path_or_upload))
        except Exception:
            meta["sha256"] = None
        df = pd.read_csv(str(path_or_upload))

    df.columns = [c.strip() for c in df.columns]
    out: List[WeightRule] = []

    for _, r in df.iterrows():
        rid = str(r.get("rule_id", "")).strip()
        if not rid:
            continue
        expr = r.get("cond_expr", r.get("expr", ""))
        out.append(
            WeightRule(
                rule_id=rid,
                cond_expr=str(expr) if expr is not None else "",
                w25=float(r.get("w25", 0.0) if not pd.isna(r.get("w25", 0.0)) else 0.0),
                w225=float(r.get("w225", 0.0) if not pd.isna(r.get("w225", 0.0)) else 0.0),
                w255=float(r.get("w255", 0.0) if not pd.isna(r.get("w255", 0.0)) else 0.0),
                notes=str(r.get("notes", "")) if not pd.isna(r.get("notes", "")) else "",
            )
        )

    return out, meta


def load_tiepack(path_or_upload) -> Tuple[List[TieRule], Dict[str, object]]:
    """Loads tie-pack CSV. Expected columns: rule_id, cond_expr (or expr), tie_bias_member, tie_bias_add."""
    meta: Dict[str, object] = {"source": None, "sha256": None}

    if path_or_upload is None:
        return [], meta

    if hasattr(path_or_upload, "read"):
        b = path_or_upload.read()
        meta["source"] = getattr(path_or_upload, "name", "uploaded")
        meta["sha256"] = sha256_bytes(b)
        df = pd.read_csv(pd.io.common.BytesIO(b))
    else:
        meta["source"] = str(path_or_upload)
        try:
            meta["sha256"] = sha256_file(str(path_or_upload))
        except Exception:
            meta["sha256"] = None
        df = pd.read_csv(str(path_or_upload))

    df.columns = [c.strip() for c in df.columns]
    out: List[TieRule] = []

    for _, r in df.iterrows():
        rid = str(r.get("rule_id", "")).strip()
        if not rid:
            continue
        expr = r.get("cond_expr", r.get("expr", ""))
        bias_member = r.get("tie_bias_member", r.get("bias_member", None))
        try:
            bias_member = int(bias_member)
        except Exception:
            continue
        if bias_member not in MEMBERS:
            continue

        bias_add = r.get("tie_bias_add", r.get("bias_add", 0.25))
        try:
            bias_add = float(bias_add)
        except Exception:
            bias_add = 0.25

        out.append(
            TieRule(
                rule_id=rid,
                cond_expr=str(expr) if expr is not None else "",
                tie_bias_member=bias_member,
                tie_bias_add=bias_add,
                notes=str(r.get("notes", "")) if not pd.isna(r.get("notes", "")) else "",
            )
        )

    return out, meta


def _parse_needs(needs_val) -> List[str]:
    if needs_val is None or (isinstance(needs_val, float) and pd.isna(needs_val)):
        return []
    s = str(needs_val).strip()
    if not s:
        return []
    parts = re.split(r"[\|,]+", s)
    return [p.strip() for p in parts if p.strip()]


def translate_predicate_expr(expr: str) -> str:
    """Translate older shorthand rescue expressions to parser-safe cond_expr."""
    if expr is None:
        return ""
    s = str(expr).strip()
    if not s:
        return ""

    s = re.sub(r"\bAND\b", "and", s, flags=re.I)
    s = re.sub(r"\bOR\b", "or", s, flags=re.I)
    s = re.sub(r"\bNOT\b", "not", s, flags=re.I)

    s = re.sub(r"\bhas(\d)\b", r"(seed_has\1==1)", s)
    s = re.sub(r"\bno(\d)\b", r"(seed_has\1==0)", s)
    s = re.sub(r"\bcnt(\d)_ge(\d+)\b", r"(seed_cnt\1>=\2)", s)

    s = re.sub(r"\bspread_ge(\d+)\b", r"(seed_spread>=\1)", s)
    s = re.sub(r"\bspread_le(\d+)\b", r"(seed_spread<=\1)", s)

    s = re.sub(r"\bsum_ge(\d+)\b", r"(seed_sum>=\1)", s)
    s = re.sub(r"\bsum_le(\d+)\b", r"(seed_sum<=\1)", s)

    s = re.sub(r"\babsdiff_ge(\d+)\b", r"(seed_absdiff>=\1)", s)
    s = re.sub(r"\babsdiff_le(\d+)\b", r"(seed_absdiff<=\1)", s)

    s = re.sub(r"\bconsec_ge(\d+)\b", r"(seed_consec_links>=\1)", s)
    s = re.sub(r"\bmirrorpair_ge(\d+)\b", r"(seed_mirrorpair_cnt>=\1)", s)

    s = re.sub(r"\bgap_(\d+)_(\d+)\b", r"((gap_since_last025>=\1) and (gap_since_last025<=\2))", s)

    def _pair_repl(m):
        a, b = m.group(1), m.group(2)
        if a == b:
            return f"(seed_cnt{a}>=2)"
        return f"((seed_has{a}==1) and (seed_has{b}==1))"

    s = re.sub(r"\bpair_(\d)(\d)\b", _pair_repl, s)
    s = re.sub(r"\bworstpair_025\b", r"(seed_has_worstpair_025==1)", s)

    return s


def load_downranks(path_or_upload) -> List[Dict[str, object]]:
    """Downrank pack (XLSX/CSV). Expected: rule_id, target_member, cond_expr, suppression_factor, H, layers, needs_features."""
    if path_or_upload is None:
        return []

    if hasattr(path_or_upload, "read"):
        name = getattr(path_or_upload, "name", "").lower()
        b = path_or_upload.read()
        buf = pd.io.common.BytesIO(b)
        df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    else:
        p = str(path_or_upload)
        df = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    out = []

    for _, r in df.iterrows():
        rid = str(r.get("rule_id", "")).strip()
        if not rid:
            continue
        cond = r.get("cond_expr", r.get("condition_expr", r.get("expr", "")))
        tgt = r.get("target_member", r.get("target", None))
        try:
            tgt = int(tgt) if not pd.isna(tgt) else None
        except Exception:
            tgt = None

        out.append({
            "rule_id": rid,
            "rule_type": "DOWNRANK_SWAP",
            "target_member": tgt,
            "cond_expr": str(cond) if cond is not None else "",
            "suppression_factor": float(r.get("suppression_factor", 1.0)) if not pd.isna(r.get("suppression_factor", 1.0)) else 1.0,
            "H": float(r.get("H", 0.0)) if not pd.isna(r.get("H", 0.0)) else 0.0,
            "layers": int(r.get("layers", 0)) if not pd.isna(r.get("layers", 0)) else 0,
            "needs": _parse_needs(r.get("needs_features", r.get("needs", ""))),
        })

    return out


def load_rescues_v2(path_or_upload) -> List[Dict[str, object]]:
    """Rescue v2 (XLSX/CSV) using predicate_expr shorthand."""
    if path_or_upload is None:
        return []

    if hasattr(path_or_upload, "read"):
        name = getattr(path_or_upload, "name", "").lower()
        b = path_or_upload.read()
        buf = pd.io.common.BytesIO(b)
        df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    else:
        p = str(path_or_upload)
        df = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    out = []

    for _, r in df.iterrows():
        rid = str(r.get("rule_id", "")).strip()
        if not rid:
            continue
        when_top3 = r.get("when_top3", None)
        set_top1 = r.get("set_top1", None)
        try:
            when_top3 = int(when_top3) if not pd.isna(when_top3) else None
        except Exception:
            when_top3 = None
        try:
            set_top1 = int(set_top1) if not pd.isna(set_top1) else None
        except Exception:
            set_top1 = None

        pred = r.get("predicate_expr", r.get("cond_expr", ""))
        cond_expr = translate_predicate_expr(pred)

        out.append({
            "rule_id": rid,
            "rule_type": "RESCUE_TOP1",
            "when_top3": when_top3,
            "set_top1": set_top1,
            "cond_expr": cond_expr,
            "is_locked": bool(r.get("is_locked", r.get("LOCKED_AUTO", False))) or ("LOCKED" in rid.upper()),
            "precision": float(r.get("precision", 0.0)) if not pd.isna(r.get("precision", 0.0)) else 0.0,
            "gain": float(r.get("gain", 0.0)) if not pd.isna(r.get("gain", 0.0)) else 0.0,
            "fires": int(r.get("fires", 0)) if not pd.isna(r.get("fires", 0)) else 0,
            "needs": _parse_needs(r.get("needs_features", r.get("needs", ""))),
        })

    return out


def load_rescues_mined(path_or_upload) -> List[Dict[str, object]]:
    """Mined rescues (CSV/XLSX) with parser-safe cond_expr."""
    if path_or_upload is None:
        return []

    if hasattr(path_or_upload, "read"):
        name = getattr(path_or_upload, "name", "").lower()
        b = path_or_upload.read()
        buf = pd.io.common.BytesIO(b)
        df = pd.read_excel(buf) if name.endswith(".xlsx") else pd.read_csv(buf)
    else:
        p = str(path_or_upload)
        df = pd.read_excel(p) if p.lower().endswith(".xlsx") else pd.read_csv(p)

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    out = []

    for _, r in df.iterrows():
        rid = str(r.get("rule_id", "")).strip()
        if not rid:
            continue
        when_top3 = r.get("when_top3", None)
        set_top1 = r.get("set_top1", None)
        try:
            when_top3 = int(when_top3) if not pd.isna(when_top3) else None
        except Exception:
            when_top3 = None
        try:
            set_top1 = int(set_top1) if not pd.isna(set_top1) else None
        except Exception:
            set_top1 = None

        cond = r.get("cond_expr", "")

        out.append({
            "rule_id": rid,
            "rule_type": "RESCUE_TOP1",
            "when_top3": when_top3,
            "set_top1": set_top1,
            "cond_expr": str(cond) if cond is not None else "",
            "is_locked": bool(r.get("is_locked", False)) or ("LOCKED" in rid.upper()),
            "precision": float(r.get("precision", 0.0)) if not pd.isna(r.get("precision", 0.0)) else 0.0,
            "gain": float(r.get("gain", 0.0)) if not pd.isna(r.get("gain", 0.0)) else 0.0,
            "fires": int(r.get("fires", 0)) if not pd.isna(r.get("fires", 0)) else 0,
            "needs": _parse_needs(r.get("needs_features", r.get("needs", ""))),
        })

    return out


# -------------------------
# Member scoring + pipeline
# -------------------------


def infer_top3_member(top1: int, top2: int) -> int:
    rem = sorted(set(MEMBERS) - {int(top1), int(top2)})
    return int(rem[0]) if rem else int(top1)


def _needs_ok(rule: Dict[str, object], feats: Dict[str, object]) -> bool:
    for k in (rule.get("needs", []) or []):
        if k not in feats or feats.get(k) is None:
            return False
    return True


def apply_one_downrank_swap(top1: int, top2: int, feats: Dict[str, object], downranks: List[Dict[str, object]]) -> Tuple[int, int, str]:
    """At most one swap. Strongest = lowest suppression_factor, then higher H, then more layers."""
    cands = []
    for r in downranks:
        if r.get("rule_type") != "DOWNRANK_SWAP":
            continue
        if r.get("target_member") is None:
            continue
        if int(r["target_member"]) != int(top1):
            continue
        if not _needs_ok(r, feats):
            continue
        if safe_eval_expr(r.get("cond_expr", ""), feats):
            cands.append(r)

    if not cands:
        return int(top1), int(top2), ""

    cands.sort(key=lambda r: (r.get("suppression_factor", 1.0), -r.get("H", 0.0), -r.get("layers", 0), str(r.get("rule_id", ""))))
    best = cands[0]
    return int(top2), int(top1), str(best.get("rule_id", ""))


def apply_one_rescue(top1: int, top2: int, top3: int, feats: Dict[str, object], rescues: List[Dict[str, object]], locked_only: bool = False) -> Tuple[int, int, int, str]:
    """At most one Top3→Top1 rescue."""
    cands = []
    for r in rescues:
        if r.get("rule_type") != "RESCUE_TOP1":
            continue
        if locked_only and not r.get("is_locked", False):
            continue
        if r.get("when_top3") is None or r.get("set_top1") is None:
            continue
        if int(r["when_top3"]) != int(top3):
            continue
        if not _needs_ok(r, feats):
            continue
        if safe_eval_expr(r.get("cond_expr", ""), feats):
            cands.append(r)

    if not cands:
        return int(top1), int(top2), int(top3), ""

    cands.sort(key=lambda r: (
        0 if r.get("is_locked", False) else 1,
        -r.get("precision", 0.0),
        -r.get("gain", 0.0),
        -r.get("fires", 0),
        str(r.get("rule_id", "")),
    ))
    best = cands[0]
    new_top1 = int(best["set_top1"])

    order = [int(top1), int(top2), int(top3)]
    if new_top1 in order:
        order.remove(new_top1)
        order = [new_top1] + order
    else:
        order = [new_top1, int(top1), int(top2)]

    return order[0], order[1], order[2], str(best.get("rule_id", ""))


def score_seed(seed4: str,
              weights: List[WeightRule],
              tiepack: List[TieRule],
              extra_feats: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    feats = compute_features(seed4)
    if extra_feats:
        feats.update(extra_feats)

    scores = {25: 0.0, 225: 0.0, 255: 0.0}
    fired_rules = []

    # base weights
    for r in weights:
        if safe_eval_expr(r.cond_expr, feats):
            scores[25] += float(r.w25)
            scores[225] += float(r.w225)
            scores[255] += float(r.w255)
            fired_rules.append(r.rule_id)

    base_order = sorted(MEMBERS, key=lambda m: (-scores[m], m))
    base_top1, base_top2 = base_order[0], base_order[1]
    base_top3 = infer_top3_member(base_top1, base_top2)
    base_gap = float(scores[base_top1] - scores[base_top2])

    # tie-pack: only meant to break close ties
    tie_fired = 0
    tie_rule_fired = ""
    if abs(base_gap) <= 1e-12:  # exact tie
        for tr in tiepack:
            if safe_eval_expr(tr.cond_expr, feats):
                scores[int(tr.tie_bias_member)] += float(tr.tie_bias_add)
                tie_fired = 1
                tie_rule_fired = tr.rule_id
                break

    order_after_tie = sorted(MEMBERS, key=lambda m: (-scores[m], m))
    top1, top2 = order_after_tie[0], order_after_tie[1]
    top3 = infer_top3_member(top1, top2)

    dead_tie = int(abs(scores[top1] - scores[top2]) <= 1e-12)

    out = {
        "seed4": seed4,
        "feats": feats,
        "scores": scores,
        "fired_rules": fired_rules,
        "tie_fired": tie_fired,
        "tie_rule": tie_rule_fired,
        "dead_tie": dead_tie,
        "base_gap": float(scores[top1] - scores[top2]),
        "BaseTop1": int(base_top1),
        "BaseTop2": int(base_top2),
        "BaseTop3": int(base_top3),
        "Top1": int(top1),
        "Top2": int(top2),
        "Top3": int(top3),
    }

    # expose for trigger logic
    feats["tie_fired"] = int(tie_fired)
    feats["dead_tie"] = int(dead_tie)
    feats["base_gap0"] = int(abs(out["base_gap"]) <= 1e-12)

    return out


def apply_member_pipeline(scored: Dict[str, object],
                          downranks: List[Dict[str, object]],
                          locked_rescues: List[Dict[str, object]],
                          mined_rescues: List[Dict[str, object]],
                          micro_rescues: List[Dict[str, object]],
                          enable_downranks: bool,
                          enable_locked_rescues: bool,
                          enable_mined: bool,
                          enable_micro: bool,
                          selected_mined_ids: List[str]) -> Dict[str, object]:
    out = dict(scored)
    feats = out["feats"]

    # Start from current Top1/Top2 after tie-break
    top1, top2 = int(out["Top1"]), int(out["Top2"])

    fired_downrank = ""
    if enable_downranks and downranks:
        top1, top2, fired_downrank = apply_one_downrank_swap(top1, top2, feats, downranks)

    top3 = infer_top3_member(top1, top2)

    fired_lock = ""
    if enable_locked_rescues and locked_rescues:
        top1, top2, top3, fired_lock = apply_one_rescue(top1, top2, top3, feats, locked_rescues, locked_only=True)

    fired_mined = ""
    if enable_mined and mined_rescues and selected_mined_ids:
        mined_sel = [r for r in mined_rescues if str(r.get("rule_id", "")) in set(selected_mined_ids)]
        if mined_sel:
            top1, top2, top3, fired_mined = apply_one_rescue(top1, top2, top3, feats, mined_sel, locked_only=False)

    fired_micro = ""
    if enable_micro and micro_rescues:
        top1, top2, top3, fired_micro = apply_one_rescue(top1, top2, top3, feats, micro_rescues, locked_only=False)

    out["FinalTop1"] = int(top1)
    out["FinalTop2"] = int(top2)
    out["FinalTop3"] = int(top3)

    out["DownrankFired"] = fired_downrank
    out["LockedRescueFired"] = fired_lock
    out["MinedRescueFired"] = fired_mined
    out["MicroRescueFired"] = fired_micro

    return out


def determine_play_members(feats: Dict[str, object],
                           final_top1: int, final_top2: int, final_top3: int,
                           play_mode: str,
                           triggers: Dict[str, bool],
                           allow_play3: bool) -> List[int]:
    mode = (play_mode or "TOP1_ONLY").upper()

    def _on(name: str) -> bool:
        if not triggers.get(name, False):
            return False
        v = feats.get(name, 0)
        try:
            return bool(int(v))
        except Exception:
            return bool(v)

    if mode == "ALWAYS_TOP1_TOP2":
        return [int(final_top1), int(final_top2)]

    if mode == "CONDITIONAL_TOP2":
        fired = any(_on(k) for k in triggers.keys())
        if fired:
            plays = [int(final_top1), int(final_top2)]
            # intentionally hard to trigger
            if allow_play3 and _on("dead_tie") and _on("base_gap0") and _on("seed_has9"):
                plays.append(int(final_top3))
            return plays
        return [int(final_top1)]

    return [int(final_top1)]


# -------------------------
# History loading + cadence
# -------------------------


@st.cache_data(show_spinner=False)
def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Flexible column mapping
    col_date = None
    for c in df.columns:
        if c.lower() in {"date", "draw_date"}:
            col_date = c
            break
    if col_date is None:
        raise ValueError("History CSV must include a date column named 'date' or 'draw_date'.")

    col_state = None
    for c in df.columns:
        if c.lower() in {"state", "st"}:
            col_state = c
            break

    col_game = None
    for c in df.columns:
        if c.lower() in {"game", "draw", "draw_name"}:
            col_game = c
            break

    col_result = None
    for c in df.columns:
        if c.lower() in {"result", "winning", "winning_number", "number"}:
            col_result = c
            break
    if col_result is None:
        raise ValueError("History CSV must include a 4-digit result column named 'result' (or winning_number/number).")

    df = df.copy()
    df["date"] = pd.to_datetime(df[col_date]).dt.date
    df["state"] = df[col_state].astype(str) if col_state else "NA"
    df["game"] = df[col_game].astype(str) if col_game else "GAME"
    df["result4"] = df[col_result].apply(normalize_result_4d)

    df = df.dropna(subset=["result4"]).reset_index(drop=True)

    # stream_id = state + game
    df["stream_id"] = df["state"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()

    # Order per stream
    df = df.sort_values(["stream_id", "date"]).reset_index(drop=True)

    # seed4 = prior result in stream
    df["seed4"] = df.groupby("stream_id")["result4"].shift(1)

    # winner signals
    df["is_025"] = df["result4"].apply(winner_is_025_family).astype(int)
    df["winner_member"] = df["result4"].apply(winner_member_id)

    # cadence: draws since last 025
    def _cadence(s: pd.Series) -> pd.Series:
        out = []
        since = 999999
        for v in s.tolist():
            if int(v) == 1:
                since = 0
                out.append(since)
            else:
                since += 1
                out.append(since)
        return pd.Series(out, index=s.index)

    df["gap_since_last025"] = df.groupby("stream_id")["is_025"].transform(_cadence).astype(int)

    # rolling hits in last 30 draws (excluding current)
    df["rolling_025_30"] = (
        df.groupby("stream_id")["is_025"]
          .apply(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
          .reset_index(level=0, drop=True)
          .fillna(0)
          .astype(int)
    )

    return df


def latest_per_stream(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["stream_id", "date"]).groupby("stream_id", as_index=False).tail(1)


# -------------------------
# Styling
# -------------------------


def style_play_bold(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _style_row(row):
        plays = set(row.get("_PlayMembers", []))
        css = [""] * len(row)
        col_idx = {c: i for i, c in enumerate(row.index)}
        for c in ["Top1Member", "Top2Member", "Top3Member"]:
            if c in col_idx:
                try:
                    m = int(row[c])
                except Exception:
                    continue
                if m in plays:
                    css[col_idx[c]] = "font-weight: 900"
        return css

    return df.style.apply(_style_row, axis=1)


# =========================================================
# UI
# =========================================================

st.title("Core 025 – Stream Ranking + Member Picker")

with st.sidebar:
    st.header("Files")
    history_path = st.text_input("History CSV path", value="core025_history__2026-03-03.csv")

    weights_path = st.text_input("Weights CSV path", value="core025_rule_weights_v3_1__2026-03-03.csv")
    tie_path = st.text_input("Tie-pack CSV path", value="core025_tie_pack_v3_1__2026-03-03.csv")

    downranks_path = st.text_input("1-miss Downranks (XLSX/CSV) path", value="core025_1miss_downranks_addon_v1_1__2026-03-03.xlsx")
    rescues_path = st.text_input("LOCKED Rescues (XLSX/CSV) path", value="core025_rescue_rules_from_per_event_v2__2026-03-03.xlsx")

    mined_rescues_path = st.text_input("Mined Rescues (CSV/XLSX) path", value="core025_top3_rescue_mined_top3_recalc__2026-03-03.csv")
    micro_rescues_path = st.text_input("Micro Rescues (CSV/XLSX) path", value="core025_micro_rescues_10rules_recalc__2026-03-03.csv")

    st.divider()
    st.caption("If paths don’t work on Streamlit Cloud, upload instead.")

    history_up = st.file_uploader("Upload history CSV", type=["csv"], key="hist_up")
    weights_up = st.file_uploader("Upload weights CSV", type=["csv"], key="w_up")
    tie_up = st.file_uploader("Upload tie-pack CSV", type=["csv"], key="t_up")
    downranks_up = st.file_uploader("Upload downranks (xlsx/csv)", type=["xlsx", "csv"], key="d_up")
    rescues_up = st.file_uploader("Upload rescues (xlsx/csv)", type=["xlsx", "csv"], key="r_up")
    mined_up = st.file_uploader("Upload mined rescues (xlsx/csv)", type=["xlsx", "csv"], key="m_up")
    micro_up = st.file_uploader("Upload micro rescues (xlsx/csv)", type=["xlsx", "csv"], key="mi_up")

    st.divider()

    if st.button("Clear cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")


# Load history
try:
    if history_up is not None:
        b = history_up.read()
        hist = pd.read_csv(pd.io.common.BytesIO(b))
        # Save to temp-like df by reusing loader with a BytesIO is tricky; normalize via temp dataframe path logic.
        # We instead parse using the same transform logic by writing to a temporary buffer.
        # (Streamlit Cloud: this stays in memory.)
        tmp = pd.read_csv(pd.io.common.BytesIO(b))
        tmp.to_csv("__uploaded_history.csv", index=False)
        history_df = load_history_csv("__uploaded_history.csv")
        history_src = getattr(history_up, "name", "uploaded")
    else:
        history_df = load_history_csv(history_path)
        history_src = history_path
except Exception as e:
    st.error(f"History load failed: {e}")
    st.stop()

# Load weights/tie
try:
    weight_rules, w_meta = load_weights(weights_up if weights_up is not None else weights_path)
except Exception as e:
    st.error(f"Weights load failed: {e}")
    st.stop()

try:
    tie_rules, t_meta = load_tiepack(tie_up if tie_up is not None else tie_path)
except Exception as e:
    st.warning(f"Tie-pack load failed (will run without tie rules): {e}")
    tie_rules = []
    t_meta = {"source": None, "sha256": None}

# Add-on packs
try:
    downrank_rules = load_downranks(downranks_up if downranks_up is not None else downranks_path)
except Exception:
    downrank_rules = []

try:
    rescue_rules_all = load_rescues_v2(rescues_up if rescues_up is not None else rescues_path)
    locked_rescue_rules = [r for r in rescue_rules_all if r.get("is_locked", False)]
except Exception:
    locked_rescue_rules = []

try:
    mined_rescue_rules = load_rescues_mined(mined_up if mined_up is not None else mined_rescues_path)
except Exception:
    mined_rescue_rules = []

try:
    micro_rescue_rules = load_rescues_mined(micro_up if micro_up is not None else micro_rescues_path)
except Exception:
    micro_rescue_rules = []

st.caption(
    f"Loaded: history={history_src} | weights={w_meta.get('source')} | tie={t_meta.get('source')} | downranks={len(downrank_rules)} | locked rescues={len(locked_rescue_rules)} | mined rescues={len(mined_rescue_rules)} | micro rescues={len(micro_rescue_rules)}"
)

with st.expander("Rule file provenance (sha256)"):
    st.write({
        "history": history_src,
        "weights": w_meta,
        "tie": t_meta,
    })


tab_live, tab_lab, tab_debug = st.tabs(["LIVE", "LAB", "DEBUG"])

# =========================================================
# LIVE
# =========================================================

with tab_live:
    st.subheader("Full ranked stream list (most → least likely to produce a 025-family hit next)")

    target_date = st.date_input(
        "Target date for next draw (used for DOW-based rules)",
        value=dt.date.today(),
        key="live_target_date",
    )

    with st.expander("Play policy + add-ons", expanded=True):
        play_mode = st.radio(
            "Default play mode",
            options=["TOP1_ONLY", "CONDITIONAL_TOP2", "ALWAYS_TOP1_TOP2"],
            index=1,
            horizontal=True,
        )
        allow_play3 = st.checkbox("Allow extremely rare Play-3 recommendation", value=False)

        st.markdown("**Play-2 triggers (used only in CONDITIONAL_TOP2 mode):**")
        trig_seed_has9 = st.checkbox("Seed contains digit 9", value=True)
        trig_tie_fired = st.checkbox("TieFired", value=True)
        trig_dead_tie = st.checkbox("Dead-tie", value=True)
        trig_basegap0 = st.checkbox("BaseGap == 0", value=True)

        st.divider()

        enable_downranks = st.checkbox("Enable 1-miss downranks (Top1↔Top2 swap)", value=True)
        enable_locked_rescues = st.checkbox("Enable LOCKED rescues (Top3→Top1 overrides)", value=True)
        enable_mined = st.checkbox("Enable mined rescues (Top3→Top1)", value=True)
        enable_micro = st.checkbox("Enable micro rescues (tiny coverage)", value=False)

        mined_ids_all = [str(r.get("rule_id", "")) for r in mined_rescue_rules if str(r.get("rule_id", "")).strip()]
        default_ids = [rid for rid in ["RR-MINED-225-01", "RR-MINED-225-02"] if rid in mined_ids_all]
        selected_mined_ids = st.multiselect("Mined rescue IDs to enable", options=mined_ids_all, default=default_ids)

    st.divider()

    gate_no9 = st.checkbox(
        "Core gate (optional): hide streams whose seed contains digit 9",
        value=False,
        help="This is NOT required for the model to run. Use only if you intentionally want fewer streams.",
    )

    # Stream score knobs (kept simple + explicit)
    with st.expander("StreamScore knobs (ranking)", expanded=False):
        alpha_gap = st.slider("Gap weight (log1p(gap_since_last025))", 0.0, 5.0, 1.0, 0.25)
        beta_roll = st.slider("Rolling hits penalty (rolling_025_30)", 0.0, 5.0, 1.5, 0.25)
        gamma_top1 = st.slider("Top1Score weight", 0.0, 5.0, 1.0, 0.25)

    latest = latest_per_stream(history_df)

    rows = []
    for _, r in latest.iterrows():
        seed4 = r.get("seed4", None)
        if seed4 is None or not isinstance(seed4, str) or not re.fullmatch(r"\d{4}", seed4):
            continue

        feats_extra = {
            "seed_dow": _dow_name(r["date"]),
            "target_dow": _dow_name(target_date),
            "time_midday_strict": infer_time_midday_strict(r.get("game", "")),
            "gap_since_last025": int(r.get("gap_since_last025", 0)),
            "rolling_025_30": int(r.get("rolling_025_30", 0)),
        }

        if gate_no9 and ("9" in seed4):
            continue

        scored = score_seed(seed4, weight_rules, tie_rules, extra_feats=feats_extra)

        piped = apply_member_pipeline(
            scored,
            downranks=downrank_rules,
            locked_rescues=locked_rescue_rules,
            mined_rescues=mined_rescue_rules,
            micro_rescues=micro_rescue_rules,
            enable_downranks=enable_downranks,
            enable_locked_rescues=enable_locked_rescues,
            enable_mined=enable_mined,
            enable_micro=enable_micro,
            selected_mined_ids=selected_mined_ids,
        )

        triggers = {
            "seed_has9": trig_seed_has9,
            "tie_fired": trig_tie_fired,
            "dead_tie": trig_dead_tie,
            "base_gap0": trig_basegap0,
        }

        plays = determine_play_members(
            piped["feats"],
            piped["FinalTop1"],
            piped["FinalTop2"],
            piped["FinalTop3"],
            play_mode,
            triggers,
            allow_play3,
        )

        # StreamScore (simple + explicit)
        top1_score = float(piped["scores"].get(piped["FinalTop1"], 0.0))
        stream_score = (
            gamma_top1 * top1_score
            + alpha_gap * float(np.log1p(feats_extra["gap_since_last025"]))
            - beta_roll * float(feats_extra["rolling_025_30"])  # penalize streams that just hit a lot
        )

        rows.append({
            "Stream": r["stream_id"],
            "State": r["state"],
            "Game": r["game"],
            "SeedDate": r["date"],
            "Seed4": seed4,
            "GapSinceLast025": feats_extra["gap_since_last025"],
            "Rolling025_30": feats_extra["rolling_025_30"],
            "TargetDOW": feats_extra["target_dow"],
            "MiddayStrict": feats_extra["time_midday_strict"],

            "BaseTop1Member": piped["BaseTop1"],
            "BaseTop2Member": piped["BaseTop2"],
            "BaseTop3Member": piped["BaseTop3"],

            "Top1Member": piped["FinalTop1"],
            "Top2Member": piped["FinalTop2"],
            "Top3Member": piped["FinalTop3"],

            "Top1Score": float(piped["scores"].get(piped["FinalTop1"], 0.0)),
            "Top2Score": float(piped["scores"].get(piped["FinalTop2"], 0.0)),
            "Top3Score": float(piped["scores"].get(piped["FinalTop3"], 0.0)),

            "TieFired": int(piped.get("tie_fired", 0)),
            "DeadTie": int(piped.get("dead_tie", 0)),
            "BaseGap(#1-#2)": float(piped.get("base_gap", 0.0)),

            "DownrankFired": piped.get("DownrankFired", ""),
            "LockedRescueFired": piped.get("LockedRescueFired", ""),
            "MinedRescueFired": piped.get("MinedRescueFired", ""),
            "MicroRescueFired": piped.get("MicroRescueFired", ""),

            "PlayMembers": "+".join(str(x) for x in plays),
            "PlayCount": len(plays),
            "_PlayMembers": plays,

            "StreamScore": float(stream_score),
        })

    if not rows:
        st.warning("No streams to display (check history, seed availability, or gates).")
        st.stop()

    out = pd.DataFrame(rows)
    out = out.sort_values(["StreamScore", "GapSinceLast025"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    show_cols = [
        "Rank", "Stream", "State", "Game", "SeedDate", "Seed4",
        "GapSinceLast025", "Rolling025_30",
        "Top1Member", "Top1Score",
        "Top2Member", "Top2Score",
        "Top3Member", "Top3Score",
        "PlayMembers", "PlayCount",
        "DownrankFired", "LockedRescueFired", "MinedRescueFired", "MicroRescueFired",
        "StreamScore",
    ]

    st.dataframe(
        style_play_bold(out[show_cols + ["_PlayMembers"]].copy()).hide_columns(["_PlayMembers"]),
        use_container_width=True,
        height=650,
    )

    st.download_button(
        "Download LIVE playlist (CSV)",
        data=out.drop(columns=["_PlayMembers"]).to_csv(index=False).encode("utf-8"),
        file_name=f"core025_live_playlist_{target_date.isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# LAB
# =========================================================

with tab_lab:
    st.subheader("Lab Backtest (walk-forward) + Percentile-by-Row (rank row by row)")

    min_date = history_df["date"].min()
    max_date = history_df["date"].max()

    c1, c2, c3 = st.columns(3)
    with c1:
        bt_start = st.date_input("Backtest start", value=min_date, min_value=min_date, max_value=max_date, key="bt_start")
    with c2:
        bt_end = st.date_input("Backtest end", value=max_date, min_value=min_date, max_value=max_date, key="bt_end")
    with c3:
        bt_play_mode = st.selectbox("Evaluate play mode", options=["TOP1_ONLY", "TOP1_TOP2"], index=1)

    st.caption("Walk-forward: for each date, each stream predicts using its prior result as seed.")

    run = st.button("Run backtest", type="primary", use_container_width=True)

    if run:
        df = history_df.copy()
        df = df[(df["date"] >= bt_start) & (df["date"] <= bt_end)].copy()
        df = df.sort_values(["stream_id", "date"]).reset_index(drop=True)

        # Only rows where seed exists
        df = df.dropna(subset=["seed4"]).copy()

        # For row-by-row: build daily rankings using seed from that day’s row
        # (Prediction for that row’s date uses seed4 available on that row.)

        records = []

        unique_dates = sorted(df["date"].unique().tolist())
        prog = st.progress(0)

        for i, d in enumerate(unique_dates):
            day = df[df["date"] == d].copy()
            if day.empty:
                continue

            # Compute per-stream predictions for that date
            tmp_rows = []
            for _, r in day.iterrows():
                seed4 = r["seed4"]
                feats_extra = {
                    "seed_dow": _dow_name(r["date"]),
                    "target_dow": _dow_name(r["date"]),
                    "time_midday_strict": infer_time_midday_strict(r.get("game", "")),
                    "gap_since_last025": int(r.get("gap_since_last025", 0)),
                    "rolling_025_30": int(r.get("rolling_025_30", 0)),
                }

                scored = score_seed(seed4, weight_rules, tie_rules, extra_feats=feats_extra)
                piped = apply_member_pipeline(
                    scored,
                    downranks=downrank_rules,
                    locked_rescues=locked_rescue_rules,
                    mined_rescues=mined_rescue_rules,
                    micro_rescues=micro_rescue_rules,
                    enable_downranks=True,
                    enable_locked_rescues=True,
                    enable_mined=True,
                    enable_micro=False,
                    selected_mined_ids=["RR-MINED-225-01", "RR-MINED-225-02"],
                )

                top1_score = float(piped["scores"].get(piped["FinalTop1"], 0.0))
                stream_score = (
                    1.0 * top1_score
                    + 1.0 * float(np.log1p(feats_extra["gap_since_last025"]))
                    - 1.5 * float(feats_extra["rolling_025_30"])
                )

                tmp_rows.append({
                    "date": r["date"],
                    "stream_id": r["stream_id"],
                    "winner_is_025": int(r["is_025"]),
                    "winner_member": r["winner_member"],
                    "FinalTop1": piped["FinalTop1"],
                    "FinalTop2": piped["FinalTop2"],
                    "FinalTop3": piped["FinalTop3"],
                    "Top1Score": float(piped["scores"].get(piped["FinalTop1"], 0.0)),
                    "Top2Score": float(piped["scores"].get(piped["FinalTop2"], 0.0)),
                    "Top3Score": float(piped["scores"].get(piped["FinalTop3"], 0.0)),
                    "tie_fired": int(piped.get("tie_fired", 0)),
                    "dead_tie": int(piped.get("dead_tie", 0)),
                    "DownrankFired": piped.get("DownrankFired", ""),
                    "LockedRescueFired": piped.get("LockedRescueFired", ""),
                    "MinedRescueFired": piped.get("MinedRescueFired", ""),
                    "StreamScore": float(stream_score),
                })

            day_pred = pd.DataFrame(tmp_rows)
            if day_pred.empty:
                continue

            day_pred = day_pred.sort_values(["StreamScore"], ascending=False).reset_index(drop=True)
            day_pred["rank_row"] = np.arange(1, len(day_pred) + 1)

            # Evaluate hits only on streams where winner_is_025==1
            hits = day_pred[day_pred["winner_is_025"] == 1].copy()
            if not hits.empty:
                if bt_play_mode == "TOP1_ONLY":
                    hits["hit"] = (hits["winner_member"].astype(int) == hits["FinalTop1"].astype(int)).astype(int)
                else:
                    hits["hit"] = (
                        (hits["winner_member"].astype(int) == hits["FinalTop1"].astype(int))
                        | (hits["winner_member"].astype(int) == hits["FinalTop2"].astype(int))
                    ).astype(int)

                records.append(hits)

            prog.progress(int(((i + 1) / max(1, len(unique_dates))) * 100))

        prog.empty()

        if not records:
            st.warning("No 025-family hit events in the selected window.")
            st.stop()

        ev = pd.concat(records, ignore_index=True)

        total_events = len(ev)
        hits_total = int(ev["hit"].sum())
        hit_rate = hits_total / max(1, total_events)

        st.success(f"Events (025-family hits only): {total_events} | Captured: {hits_total} | Capture rate: {hit_rate:.1%} ({bt_play_mode})")

        # Percentile-by-row (literal rank row)
        row_table = (
            ev.groupby("rank_row")
              .agg(events=("hit", "size"), captured=("hit", "sum"))
              .reset_index()
        )
        row_table["capture_rate"] = row_table["captured"] / row_table["events"].clip(lower=1)
        st.markdown("### Percentile by Row (literal ranking row 1..N)")
        st.dataframe(row_table, use_container_width=True, height=420)

        # Miss breakdown by true member
        miss = ev[ev["hit"] == 0].copy()
        if not miss.empty:
            mb = miss.groupby("winner_member").size().reset_index(name="misses")
            st.markdown("### Miss breakdown (true winner member)")
            st.dataframe(mb, use_container_width=True)

        st.download_button(
            "Download LAB events (CSV)",
            data=ev.to_csv(index=False).encode("utf-8"),
            file_name=f"core025_lab_events_{bt_start.isoformat()}_{bt_end.isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# DEBUG
# =========================================================

with tab_debug:
    st.subheader("Debug / Feature Coverage")

    st.markdown("This panel helps ensure the app has the features required by the rulepacks.")

    sample = latest_per_stream(history_df).head(200)
    if sample.empty:
        st.info("No sample rows.")
        st.stop()

    # Build one merged feature dict for coverage checks
    r0 = sample.iloc[0]
    seed4 = r0.get("seed4")
    feats = compute_features(seed4)
    feats.update({
        "seed_dow": _dow_name(r0["date"]),
        "target_dow": _dow_name(dt.date.today()),
        "time_midday_strict": infer_time_midday_strict(r0.get("game", "")),
        "gap_since_last025": int(r0.get("gap_since_last025", 0)),
        "rolling_025_30": int(r0.get("rolling_025_30", 0)),
        "tie_fired": 0,
        "dead_tie": 0,
        "base_gap0": 0,
    })

    def pack_needs(pack_name: str, exprs: List[str]) -> pd.DataFrame:
        need = set()
        for e in exprs:
            need.update(extract_names(e))
        need = sorted(need)
        rows = []
        for k in need:
            rows.append({
                "feature": k,
                "available": int(k in feats),
                "sample_value": feats.get(k, None),
            })
        return pd.DataFrame(rows)

    w_exprs = [r.cond_expr for r in weight_rules]
    t_exprs = [r.cond_expr for r in tie_rules]
    d_exprs = [r.get("cond_expr", "") for r in downrank_rules]
    lock_exprs = [r.get("cond_expr", "") for r in locked_rescue_rules]
    mined_exprs = [r.get("cond_expr", "") for r in mined_rescue_rules]

    st.markdown("### Feature coverage by pack")
    st.write("If you see any rows with available=0, that pack needs a feature the app isn’t currently producing.")

    st.markdown("**Weights pack**")
    st.dataframe(pack_needs("weights", w_exprs), use_container_width=True, height=250)

    st.markdown("**Tie pack**")
    st.dataframe(pack_needs("tie", t_exprs), use_container_width=True, height=250)

    st.markdown("**Downranks**")
    st.dataframe(pack_needs("downranks", d_exprs), use_container_width=True, height=250)

    st.markdown("**Locked rescues**")
    st.dataframe(pack_needs("locked_rescues", lock_exprs), use_container_width=True, height=250)

    st.markdown("**Mined rescues**")
    st.dataframe(pack_needs("mined_rescues", mined_exprs), use_container_width=True, height=250)
