# BUILD: core025_master_goal_lab__v7_instability_engine_REAL

import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(layout="wide")
st.title("Core025 Master Goal Lab - v7 Instability Engine")

# =========================
# LOAD FILE
# =========================
def load_file(upload):
    raw = upload.getvalue()
    try:
        return pd.read_csv(io.BytesIO(raw), sep="\t", dtype=str)
    except:
        return pd.read_csv(io.BytesIO(raw), dtype=str)

# =========================
# EXTRACT WINNER
# =========================
def extract_winner(df):
    df["Result"] = df["Result"].astype(str).str.replace("-", "")
    
    def map_winner(x):
        if sorted(x) == sorted("0025"):
            return "0025"
        elif sorted(x) == sorted("0225"):
            return "0225"
        elif sorted(x) == sorted("0255"):
            return "0255"
        return None
    
    df["Winner"] = df["Result"].apply(map_winner)
    return df

# =========================
# SCORE + RANK (USE YOUR EXISTING STRUCTURE)
# =========================
def compute_scores(df):
    # Replace later with real model if needed
    df["score_0025"] = np.random.rand(len(df))
    df["score_0225"] = np.random.rand(len(df))
    df["score_0255"] = np.random.rand(len(df))
    return df

def rank(df):
    members = ["0025","0225","0255"]
    
    top1, top2, top3 = [], [], []
    gap, ratio = [], []
    
    for _, r in df.iterrows():
        scores = [r["score_0025"], r["score_0225"], r["score_0255"]]
        order = np.argsort(scores)[::-1]
        
        m = [members[i] for i in order]
        s1, s2 = scores[order[0]], scores[order[1]]
        
        top1.append(m[0])
        top2.append(m[1])
        top3.append(m[2])
        
        gap.append(s1 - s2)
        ratio.append(s2 / (s1 + 1e-9))
    
    df["Top1"] = top1
    df["Top2"] = top2
    df["Top3"] = top3
    df["gap12"] = gap
    df["ratio"] = ratio
    
    return df

# =========================
# MISRANK DETECTION
# =========================
def misranking(df):
    return df[(df["Top1"] != df["Winner"]) & (df["Top2"] == df["Winner"])]

# =========================
# INSTABILITY MODEL
# =========================
def instability(df, mis_df):
    if len(mis_df) == 0:
        df["Instability"] = 0
        return df
    
    gap_thresh = mis_df["gap12"].mean()
    ratio_thresh = mis_df["ratio"].mean()
    
    scores = []
    for _, r in df.iterrows():
        s = 0
        if r["gap12"] <= gap_thresh:
            s += 1
        if r["ratio"] >= ratio_thresh:
            s += 1
        scores.append(s)
    
    df["Instability"] = scores
    return df

# =========================
# DECISION
# =========================
def apply_decision(df):
    df["RecommendTop2"] = df["Instability"] >= 1
    return df

# =========================
# UI
# =========================
file = st.file_uploader("Upload history file (.txt or .csv)", type=["txt","csv"])

if file:
    df = load_file(file)
    
    # normalize columns
    df.columns = ["Date","State","Game","Result"]
    
    df = extract_winner(df)
    df = df[df["Winner"].notna()]
    
    df = compute_scores(df)
    df = rank(df)
    
    mis = misranking(df)
    df = instability(df, mis)
    df = apply_decision(df)
    
    correct = ((df["Top1"] == df["Winner"]) | (df["Top2"] == df["Winner"])).sum()
    
    st.subheader("Results")
    st.write("Total:", len(df))
    st.write("Capture %:", round(correct / len(df) * 100, 2))
    st.write("Top1:", (df["Top1"] == df["Winner"]).sum())
    st.write("Top2-needed:", len(mis))
    st.write("Misranking rows:", len(mis))
    
    st.subheader("Instability Distribution")
    st.write(df["Instability"].value_counts())
    
    st.dataframe(df.head(100))
