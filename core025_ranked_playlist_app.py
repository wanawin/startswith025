# CORE 025 APP PATCH — PAIRWISE CORRECTION + STRICT TOP2 GATE
# This is a safe additive module. Do NOT remove existing logic.

from collections import Counter

# =========================
# DERIVED TRAITS
# =========================

def derive_seed_traits(seed: str):
    s = str(seed).zfill(4)
    d = [int(x) for x in s]
    c = Counter(s)

    return {
        "sum": sum(d),
        "lead": d[0],
        "tail": d[-1],
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "has0": int("0" in s),
        "has9": int("9" in s),
        "even_count": sum(x % 2 == 0 for x in d),
        "high_count": sum(x >= 5 for x in d),
        "unique_count": len(c),
        "spread": max(d) - min(d),
        "tail_parity": d[-1] % 2,
    }

# =========================
# PAIRWISE CORRECTION ENGINE
# =========================

def apply_pairwise_correction(scores: dict, traits: dict, base_top_order: list):
    """
    scores: {"0025": float, "0225": float, "0255": float}
    base_top_order: [Top1, Top2, Top3] BEFORE correction
    """

    top1 = base_top_order[0]
    top2 = base_top_order[1]

    # Example: 0225 -> 0255 correction (only fires inside that pair)
    if top1 == "0225" and top2 == "0255":
        # Tight, data-driven condition placeholder (expand later)
        if traits["pos2"] in {1, 2} and traits["tail_parity"] == 1:
            scores["0255"] += 1.0
            scores["0225"] -= 0.7

    # Example: 0025 -> 0225 correction
    if top1 == "0025" and top2 == "0225":
        if traits["has0"] == 1 and traits["sum"] <= 15:
            scores["0225"] += 0.9
            scores["0025"] -= 0.6

    # Example: 0225 -> 0025 correction
    if top1 == "0225" and top2 == "0025":
        if traits["lead"] <= 3:
            scores["0025"] += 0.8
            scores["0225"] -= 0.5

    return scores

# =========================
# STRICT TOP2 GATE
# =========================

def evaluate_top2_gate(scores: dict, final_order: list, traits: dict):
    """
    Returns True ONLY when Top2 should be played
    """

    top1, top2 = final_order[0], final_order[1]

    gap = abs(scores[top1] - scores[top2])

    # HARD RULE: must be very small gap
    if gap > 0.35:
        return False

    # must be historically unstable pair (placeholder logic)
    unstable_pairs = {
        ("0225", "0255"),
        ("0025", "0225"),
    }

    if (top1, top2) not in unstable_pairs:
        return False

    # must not have strong correction signal
    if traits["sum"] < 10 or traits["unique_count"] <= 2:
        return False

    return True

# =========================
# MAIN HOOK (INTEGRATION POINT)
# =========================

def apply_enhanced_logic(seed, scores, base_top_order):
    """
    Call this AFTER existing scoring + overlays
    """

    traits = derive_seed_traits(seed)

    # STEP 1 — Pairwise correction
    scores = apply_pairwise_correction(scores, traits, base_top_order)

    # STEP 2 — Re-rank
    final_order = sorted(scores, key=scores.get, reverse=True)

    # STEP 3 — Top2 gate
    play_top2 = evaluate_top2_gate(scores, final_order, traits)

    return {
        "scores": scores,
        "final_order": final_order,
        "play_top2": play_top2,
        "traits": traits
    }

# =========================
# NOTES
# =========================
# - This does NOT remove any existing behavior
# - This should be inserted AFTER overlay scoring
# - Pairwise rules should be expanded using mined data
# - Top2 gate is intentionally strict to reduce cost
