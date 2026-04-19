# BUILD: core025_master_goal_lab__2026-04-19_v7_misranking_engine_FIXED_INPUT

import streamlit as st
import pandas as pd
import numpy as np
import io

BUILD = "core025_master_goal_lab__2026-04-19_v7_misranking_engine_FIXED_INPUT"

st.set_page_config(layout="wide")
st.title("Core025 Master Goal Lab - v7 Misranking Engine")
st.markdown(f"**BUILD:** {BUILD}")

# =========================
# FILE LOADER (FIXED)
# =========================
def load_file(upload):
    name = upload.name.lower()
    raw = upload.getvalue()

    # Try CSV first
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
        if df.shape[1] > 1:
            return df
    except:
        pass

    # Try tab-delimited (your history file)
    try:
        df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, dtype=str)
        return df
    except:
        pass

    # Fallback auto
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str)
    return df


# =========================
# NORMALIZATION
# =========================
def normalize_history(df):
    df = df.copy()

    # Handle no headers (txt case)
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["Date", "State", "Game", "Result"]

    df["Result"] = df["Result"].astype(str).str.replace("-", "")
    return df


# =========================
# SCORING (TEMP DIAGNOSTIC)
# =========================
def compute_scores(df):
    # DO NOT use your real model here yet
    df["score_0025"] = np.random.rand(len(df))
    df["score_0225"] = np.random.rand(len(df))
    df["score_0255"] = np.random.rand(len(df))
    return df


# =========================
# RANKING
# =========================
def rank_members(df):
    members = ["0025", "0225", "0255"]
    scores = df[["score_0025", "score_0225", "score_0255"]].values

    top1, top2, top3, gap12, ratio = [], [], [], [], []

    for row in scores:
        order = np.argsort(row)[::-1]
        m1, m2, m3 = [members[i] for i in order]

        s1, s2 = row[order[0]], row[order[1]]

        top1.append(m1)
        top2.append(m2)
        top3.append(m3)
        gap12.append(s1 - s2)
        ratio.append(s2 / (s1 + 1e-9))

    df["Top1"] = top1
    df["Top2"] = top2
    df["Top3"] = top3
    df["gap12"] = gap12
    df["ratio12"] = ratio

    return df


# =========================
# MISRANKING DETECTION
# =========================
def mine_misranking(df):
    return df[(df["Top1"] != df["Winner"]) & (df["Top2"] == df["Winner"])]


# =========================
# INSTABILITY ENGINE
# =========================
def compute_instability(df, mis_df):
    if len(mis_df) == 0:
        df["InstabilityScore"] = 0
        return df

    gap_mean = mis_df["gap12"].mean()
    ratio_mean = mis_df["ratio12"].mean()

    scores = []
    for _, row in df.iterrows():
        score = 0

        if row["gap12"] <= gap_mean:
            score += 1

        if row["ratio12"] >= ratio_mean:
            score += 1

        scores.append(score)

    df["InstabilityScore"] = scores
    return df


# =========================
# DECISION LAYER
# =========================
def apply_decision(df, threshold=1):
    df["RecommendTop2"] = df["InstabilityScore"] >= threshold
    return df


# =========================
# UI
# =========================
history_file = st.file_uploader(
    "Upload history file (.txt or .csv)", type=["txt", "csv"]
)

if history_file:
    df = load_file(history_file)
    df = normalize_history(df)

    # You must provide Winner column manually for now
    if "Winner" not in df.columns:
        st.error("⚠️ Add a 'Winner' column (0025 / 0225 / 0255) for testing")
        st.stop()

    df = compute_scores(df)
    df = rank_members(df)

    mis_df = mine_misranking(df)
    df = compute_instability(df, mis_df)
    df = apply_decision(df)

    correct = ((df["Top1"] == df["Winner"]) | (df["Top2"] == df["Winner"])).sum()
    total = len(df)
    capture = correct / total

    st.subheader("Results")
    st.write("Total rows:", total)
    st.write("Capture:", round(capture * 100, 2), "%")
    st.write("Top1 correct:", (df["Top1"] == df["Winner"]).sum())
    st.write(
        "Top2-needed:",
        ((df["Top1"] != df["Winner"]) & (df["Top2"] == df["Winner"])).sum(),
    )
    st.write("Misranking rows:", len(mis_df))

    st.subheader("Instability Score Distribution")
    st.write(df["InstabilityScore"].value_counts())

    st.subheader("Preview")
    st.dataframe(df.head(100))
