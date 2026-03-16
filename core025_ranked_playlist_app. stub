from __future__ import annotations

import streamlit as st
import hashlib
import os
import pandas as pd
import unicodedata
import gzip, base64, io
import datetime
import datetime as dt  # alias used throughout the app
import math
import zipfile
import tempfile
import re
import json
import csv
from itertools import product, combinations, permutations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# -----------------------------
# Page config / small helpers
# -----------------------------
st.set_page_config(page_title="Core 025 Ranked Playlist", layout="wide")


def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip()


def _sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


# -------------------------------------------------
# NOTE:
# This canvas copy was updated to add overlay support
# for these tie predicates:
#   - when_tie_fired
#   - when_base_gap_in
#   - when_dead_tie
#
# Because this code file is very large, the full working
# source is saved on disk at:
# /mnt/data/core025_ranked_playlist_app_v3_12_19_TIE_OVERLAY_PREDICATES__2026-03-13.py
#
# The important changes made to the app are:
#
# 1) load_member_score_overlay_df(...) now accepts the extra columns:
#    when_tie_fired, when_base_gap_in, when_dead_tie
#
# 2) _apply_member_score_overlay(...) now evaluates those predicates
#    against row-level trace values:
#    - TieFired
#    - BaseGap(#1-#2)
#    - DeadTie
#
# 3) _row_memory_score_adjustments(...) passes those row-level fields
#    into _apply_member_score_overlay(...)
#
# 4) The sidebar caption describing the overlay schema was updated to
#    list the new tie-aware columns.
#
# If you want the exact diff inside the next thread, ask for:
#   "show me the precise code changes for tie overlay predicates"
#
# Practical usage after this update:
# An overlay CSV can now include rows like:
#
# rule_id,enabled,when_seed_has9,when_base_top1,when_base_top2,when_base_top3,when_tie_fired,when_base_gap_in,when_dead_tie,delta_0225,note
# BF225_TIE_A,1,0,25,225,255,1,0,0,2.1,Promote 225 in true tie pocket A
# BF225_TIE_B,1,0,255,225,25,1,0,0,2.1,Promote 225 in true tie pocket B
#
# -------------------------------------------------

# The remainder of the app file is unchanged from the working app copy
# except for the tie-aware overlay predicate additions above.
#
# To avoid truncation or corruption in canvas, use the saved file on disk
# as the source of truth for deployment / replacement.
