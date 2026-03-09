from __future__ import annotations

import streamlit as st
import hashlib
import os
import pandas as pd
import unicodedata
import gzip, base64, io
import datetime
import datetime as dt  # alias used throughout app (dt.date.today(), etc.)
import difflib
import zipfile
import json

APP_VERSION_STR = "v3.12.15"


# ---- Calibration / Mining (no sklearn) ----
# Streamlit Cloud deployments often do not include scikit-learn by default.
# This app uses a small deterministic, NumPy-based L2-regularized logistic regression
# for StreamScore calibration, plus local AUC/AP helpers (no external ML deps).
import re
import numpy as np
import math
import ast
from typing import Dict, Any, List, Tuple, Optional

# Type alias used in annotations across Streamlit Cloud/Python versions
Rule = Dict[str, Any]
from dataclasses import dataclass

# --- Member ID normalization (supports legacy numeric IDs like 25/225/255 and canonical 4-digit IDs) ---
_CORE_MEMBER_KEYS = {"0025", "0225", "0255"}

def normalize_member_id(x: object) -> str:
    """Normalize member identifiers to canonical 4-digit strings: 0025 / 0225 / 0255.

    Accepts inputs like 25, '25', '0025', 225, '225', '0225', 25.0, etc.
    Returns the original string (stripped) if it cannot be normalized.
    """
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
    elif isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        if float(x).is_integer():
            s = str(int(x))
        else:
            s = str(x).strip()
    else:
        s = str(x).strip()
    if s == "":
        return ""
    # Prefer digit-only normalization when possible
    if re.fullmatch(r"\d+", s):
        if len(s) < 4:
            s2 = s.zfill(4)
            if s2 in _CORE_MEMBER_KEYS:
                return s2
        elif len(s) == 4 and s in _CORE_MEMBER_KEYS:
            return s
    # Already canonical?
    if s in _CORE_MEMBER_KEYS:
        return s
    # Some packs include 'pick_str' already canonicalized (e.g. 0225)
    if len(s) == 4 and s.isdigit() and s.zfill(4) in _CORE_MEMBER_KEYS:
        return s.zfill(4)
    return s

def member_to_legacy_id(x: object) -> object:
    """Return legacy numeric member IDs (25 / 225 / 255) for rule-expression contexts."""
    s = normalize_member_id(x)
    return {"0025": 25, "0225": 225, "0255": 255}.get(s, x)

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

def _logloss(y_true: np.ndarray, p: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=np.float64)
    if sample_weight is None:
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
    sw = np.asarray(sample_weight, dtype=np.float64)
    sw = np.where(np.isfinite(sw), sw, 1.0)
    sw_sum = float(sw.sum())
    if sw_sum <= 0:
        sw_sum = 1.0
    return float(-np.sum(sw * (y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))) / sw_sum)

def _auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if (j - i) > 1:
            avg_rank = ( (i + 1) + j ) / 2.0  # 1-indexed ranks
            ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)

def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return float("nan")
    cum_pos = np.cumsum(y_sorted)
    precision = cum_pos / (np.arange(len(y_sorted)) + 1)
    ap = float((precision * y_sorted).sum() / n_pos)
    return ap

def _dicts_to_design_matrix(dict_rows: List[Dict[str, Any]], feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    # We intentionally use pandas.get_dummies here (pandas is available on Streamlit Cloud)
    # to avoid scikit-learn DictVectorizer.
    import pandas as pd

    df = pd.DataFrame(dict_rows)
    if df.empty:
        if feature_columns is None:
            return np.zeros((0, 0), dtype=np.float64), []
        return np.zeros((0, len(feature_columns)), dtype=np.float64), list(feature_columns)

    # bool -> int
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=False)

    df = df.fillna(0.0)

    # force numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if feature_columns is None:
        feature_columns = list(df.columns)
    else:
        # add any missing columns with zeros, then reorder
        for c in feature_columns:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feature_columns]

    X = df.astype(np.float64).values
    return X, list(feature_columns)

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std

def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std

def _fit_logreg_l2(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    l2: float = 1.0,
    lr: float = 0.1,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    w = np.zeros(p, dtype=np.float64)
    b = 0.0

    if sample_weight is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        sw = np.where(np.isfinite(sw), sw, 1.0)

    sw_sum = float(sw.sum())
    if sw_sum <= 0:
        sw_sum = 1.0

    prev_loss = None
    for it in range(max_iter):
        p_hat = _sigmoid(X.dot(w) + b)
        err = (p_hat - y)

        grad_w = (X.T.dot(err * sw) / sw_sum) + l2 * w
        grad_b = float((err * sw).sum() / sw_sum)

        w -= lr * grad_w
        b -= lr * grad_b

        if it % 10 == 0 or it == (max_iter - 1):
            loss = _logloss(y, _sigmoid(X.dot(w) + b), sw) + 0.5 * l2 * float(np.dot(w, w))
            if prev_loss is not None and abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

    return w, float(b)

# Embedded rule files (gz+base64). This eliminates Streamlit Cloud path issues.
_EMBED_WEIGHTS_B64_GZ = """H4sIAHTHoWkC/42Q0W5bIQyG7/csaLINNnDRt9g9ojkkQTtLogNJXr/mJK3SdtEGkkE2/r/fbI5LMduS+1nP48lc8nwu5lQ3v81xntK11N2+m0Pa197SLnetpVFNu3oph9SXXLt5za3M9VDWipnK3HO6tPSeNodyfVdae1tfzEbB4/KD2LRSppRf21S3W/PyYpzRJI0AP1k0WKboEL04DxwCaYrgdr2Xgr4HGB3EH5L73NL1uLR+ynVJo6rivxYdkHgALA4AClIES8QBxA2mJfYRAzOBFYxRU6gvrEevUKdrdMMQecS101LyNBjhPsAwyu7z/vs4GInRWgkx2MhejX0fp53/pDm3PtWdfrpirBJWzhC9GUcSiRJuE6zGffDWgkXxjiCswvQvYRz+NVgVcD6QQJDIwoT+mX9ii1GiheiE4v/5hztmCHz5JX6CARRw5AN4F0ToEfMGUwA9YMsCAAA="""
_EMBED_TIEPACK_B64_GZ = """H4sIAHTHoWkC/3XNwQrCMBAE0LvfModmJcd+S4jJpi5WGrKJ/r5JBUGxpxmWx2zYCiOxr63nlvHwa2NkCTc8WZZr3bvTWhA6HeVEFsocnb9olJQwzzCgfjWYRkxk/xkC2bex30ZzYR8HsUcz2u5u9VqjLFKHPB/Kz5j5/fcC5Awqw+wAAAA="""

# Embedded mined packs (gz+base64) used as defaults (no uploads required)
_EMBED_1MISS_DOWNRANKS_V1_1_B64_GZ = """H4sIAAxorGkC/9WYXW/bNhSG7/crhNw0AZiB3x9AHCBAinYYkAJZt2FXgmoziZBYciV5Xf/9DilRlCw5TZukgBuUoUmKkZ9z9PI9qrYPNs1XqMltheoma7Y1Wtv1J1ulTVbd2gZlyyYvC7QqvxRVVtyna9vclSu0LItV7mf6Xmr/21ToIftqqxrdWNirsnVaWLuyK3SV1tvNpqwa9D69ydb5w9f0Lm9q5JoUYyq6Ho09IVBTZXmRvkeNrRv3y39sb6y93E8MBzbl8h769V1WWfQpq+1DXtjuo7sDuKXa3esNfK+y+uXymopTTBA5Xed1ndTb6jZfZg9J+LrJcV6ctnueoHd/Xlxfvr1MLz/8fXV9cfU7gnsNff8d0r/eXv+T/vHx+sPVO/TbTdKjSZpqa5OLq8vkY7khyWKRdPfcsk5Oz5P6S7ZpZ8/gE3Rocrwqk6JskqbcLu/cEDtBd1lN/UbQSb+UVd1ssryCv+0Ha/h+2Sq9tQod1wA+dcsXC3KSZMUq6YfihQ7zzny7x/lCnSCGjvpdUDJ/dTfeXnWEtDIGUYYAKUUE+5Yg+I3wr5gzrjSVWEsjpKAUxpjQRvHQGhghlBCiMaGcUiYx7WJEDzBGy6Ixqf1Mnhwbd8G3w+FWxXDQnQgoqhiijjjgNz4GOARAzBNnhmqhCG9b1hEXL0X8tWFzv9E6X62yr2ndVPmy8SNFmbosjbh5ZNvka5uOLtnhvi3yz1vrBoePAe9gt7MomW5zhDjXugsAQRoRhpSj/9382eFkPGQk7RMdMJn5jHfLpmr0tIwfCJCZZLwEVqYFrtq0DwkPU0QobbhiUvM9isOlJFJwwgRjmomOPz+I/Lf/2qJH70VZBNx+CtBFvj1zMQ2DckMniHbIw8URu4hdBVmOjfTqjgSS/qdDrtn43yxyymhId992yOVBIO/ymqcyQgcPcczPFoO8PFvIXep4Sp2MqbtFETPZSXRQ9o65Qhz+i6ArP8z8We5nRJ3+FKFpE9x08o7RoxnthWVnDEKA9+EeJLg5QpJrhag/RX3jlTwYGSWJia2XFcIJZJ4MrdMVQSRm3GgBVkbxwJseEG8QADEVdk8+CLP4tqL33Ad6PoCNR9yFwsYdn6bHTkf2ZY6zIZpJxbg0TCsdQPMXA/3aDqYVEti7tks4L2m0LHLXIbZroKq5r88XNKbycCLylJDHcPYhorwJMS6Hee9GtKZMUMMJUZIrsoeuBkfJDTZGMkhlGeiyw6A78uDhsER77fbwzIxso+EO81DpEAo4peeqIbOjOki6L1ENqLHps1UFlOIwUK7zqiorL8H2Mx4YDtPxHCzwCPETbIeZHIBi6vTGG4P5gHMtGGtnP1RAr6ShLLZsPg4ULPogECLEQR5GHOrtOqU07Wr+bhe40/Tj1tZQiQwWtLVOv2KxeNOteeOQxwk46DR2RAGBN3N9yfLjRMWz3qyMiIqf5i2w7k44hR71bXrGW6hHvIWKXQ2aLEFtqX9HoltZpo9ZCzqOgS9ZhBQaihuGBaOUBN7sxXi/rii3h9WOLPcvtjSaHnZ7dZrORSY6jemxOK1sICJCY+wiYlwNCT8iFjTPi4g8jIgU5fRt4viB4Gjv+0P8hEJn7oHhI1M4fmD4IDz73j06swjFDvWFv2nfOZqXCRs9IOGCXMbxNSNoP9EpHXicGI04SVE/iwQn2jlEoMe9GunecM84xBmGyhjoG6G5UgKbwJAflBiB7R6X87N++7sr+nlfPizyoa43qqPfmnT2WF0/5U8F2HNiJFaaM8JU4C8OhL9L30C/OwrYIHvPxxIflZ+NHDqeOHRK4KnvsX7jdckTsP4PUbzMRC0bAAA="""

_EMBED_RESCUES_V2_B64_GZ = """H4sIAAxorGkC/71YW2+cOBR+76/gsdV6sr5jpKZS2kTVaqNUbdJnRGa8KWrCzAJp//76CjY2mURqFwJhzMz5js/lO8ce5HZs9x3oH+9l3e7AMDbj4wDGpr+TY/0gH25lD35+k1097g8EHHq5a7fNKEFjf9ftRzmAK/DPfXuo1dttO+jh22aQ920ng6FOybtr2q7+1o7D9Alc1WOv/5tX9lENyWF0I+rp1eWnD39fnNdnX28+1dcXN2BxvAJfLq4/fL3ArL5EIPgywEz/bffdILf1nUTF2dV5cdccao5qwhn49EP2fbuTxc3+gIrNu+LQbr/rD6R4jdkbYJ4eHoexuJUFZsVr/b0/1Q0XTa9GMPsDM/bmBHByAgE8IRUXmM13PQQRE+V8LwFXX12qjTNqD1Lu6m/NUBmtx1b29fujGl+rHxXqR0VVNN2uGMZeNg/FvXpW421X3CgxxfsTgEqjMMMVRajktIRMCEzVUDxCGAV0qXDG0FpldckfKlK23VjjF5k6tXXG2N7WGBvVKazCE2nVy/gExGt+dnlZf1Aa/XV+dnNxnQYQmF6aaeTmgQTa6IkAgJBRgJPo1LaLVdJKGW/rSxuRqUspH8MpiYem7WsS+PlMo0CLoucFzM+hu8wbui5nVpHR+MzZyBhJf32DnpIdR+PvRFIg9c99P4wa5Lej+XlZRw+Kr5pd/faUzB7Q0pEVR51IvuJII6zbw6XWsyvJQq/qWeaeGUxJo8xyTbk49NjiAFpLgHzwELSi9xTub08TjlSAzg7qmxqWOanMScVrYZ3l3Q0SMJCpJTqBwtmFYPuwZl1mrbtVyck2TAsj3iKIYsGmuxEeD3mLaPbbGEcIm5cJWN9032uKNhymBnF4Rn4kHgvuACYHY218kgGwhJ4Rbr1FNPcGXGyCJx6qHBb2jsDH3JtxLgkckY8eFzQAVRbn6aj/ou32XpX/jwrldj9+VhjYFhu6OIzLYPTnY4tMebtmPs0Su/3j7b2cYa/bbitr1XFINVfVrxjswFc4dM6UGbn0e2j7ft/DJNKcNGJYWQTptaFeMM9ZyNLKu9PSSDSxhdGGGpGW0hLjKwAiYMXmuwj8PZFHYpsM3lzNZrxF3DK2HgDYwhi4DTruC22wkm0Q1NNzUUO4oBhBhjlRPZCo9PTcIy3ty5kFSJhBR4uz89WMFQEp6WZqERR0cVaFFRWvO47WfDW4DYPofI0OM8GYn0VMcwYzO78Fb6ahHSPn0mo9k/VVrhBsLqd6+cOnEuKOZrVBMYv9JVbdFZcyT+KzvLiqeDKjLylVQW/m0p1Hx3ptdJUcZLktE2whGTgslmYudZNxLjaN33Nc/Qul63bQS0vLr2sJciSwGCpnN8MnOp8YKwjZ3aOc4seCJgCqvVoqMjvHEJ1YJToXWzDlOYdmoisGRE586XxfrtS1JNufrjSIuNBDjIj5bgo3LHmFp3sZTM8rkKtCaiqwlv9mYsMtgVAS5XjBbGTFX5n2JkgiPJsumzWldwxbc8wibVYC4hgOPRoAlvxjrtSU5SnyOTPBbiJwzU6mVB/hYlfFF4tCDazsGp2uCyHPC+5cv7u+CkrxwwjPk2oCFbUnvxYrjIx3pySd2dyNBmZ6qmYt2uggiBNRE53MPeXHr2dfzi/Ok54vapGZyzaKozw2HWWY1ryyy33XYbn1Fk+AFt1Z1F86pU3cMCsORev+jLiw/Y4U52HvG+mumi3m10HGWfMyKBG9LN2aXKkt3bQUmEPBK8YZRqWNCveBVIhAveBiU4+M/equTJDm6nSEZFfrO0oTnM2JbY3HVnC1vf5fXG9Ztt5X2nKyqCYmnBflRFhEHpQz8gIDa0KOMSNEdTfBGI8xi+l3FqoXGvf3YcYhGy91XYEW8cnzXEZ9nTOrap5PvccHlcZ4uQ4hjjBsp0wQduJsFkzbDuUxZohaNr+QFW4lS31w2XUyy1pjfT82WhljauX5Lg/NHckq08wdhOYDv//NCBWClBhWBDqKZEyg+YXbAJs4EuNsuMZuDEzL11ZbBLLw4G6jjfjFOSapyYNJ4bSDdDsx8XY4rrIbMcRvkJBgayxvvnQhQbz14tNMKj6pQ5r3rzKBmRT0eFauxCxYK98ikbBIGFumrgoqcHbzxy3KqcAlExxDLiputxviMe8xn9/IN5f/AUzmDT2kGgAA"""




# ---- Embedded mined rulepacks (b64+gzip) ----
_EMBED_RULEPACK_V3_2_B64_GZ = """H4sIAO/WqWkC/72YW3PaOBTH3/kUGl4CU0F9wca0dWeSwLbZJKRD2O3uvngUrARNfFtJLsu33yNfwHYMyaTZzYUISZZ0fud/jg7haUA95mPJKMchDe8o9yThD1RispIsjvAqjnyWte4pkSmnwoso9amPOX1gIcUB2VIu8NwTaZLEXOKv3j0JWbD11kwKrF48TTOsomXsW5aF8728YufsgXqXWBNOseSERd5XLKmQ6k/2tpiYP6QGqh1CwmEFjmIY6fyC9UHIhEAi5Q9sRQLkx5uIk+gRw2mmN9/ni9P5pXc9uz6bLbLDer/PFn96t8vFzfwL7gkw2FsTYbiu3ken8ynadU1UF+7uZmC0G+niXRMbQw07uq3DH0O1VUNXjUnxRhtqI6v2nY/rWj6YTfry2+liOpvi7pLRwR2n5BHFUbBFvQwI0jBSHJDeH3Y71/+RzSKBff3P7rgBwtvEXMiEMK48204lf7Z80/pgF59eXWFT4RpPFBzDrOLStQovczR2DFtz7IllW0YxQW/h1cSlvzfM/rAzP4KojZFxiJF1QBf1Ls11tQoUqyKVfVOrqkaZbo+cscKgV62vYRjb+mT/ih01ZL6MAiil8+WYUNooWAcoaHuTC6mk4WdXd2rqScNPLmxb4aCVekjD3PnKjRZYoo7vlI63Sstzw6z6Vw7EeWWonOG/KI9zCEm8eqQS9ThNAoAhqd9X8TK7uriuxkqr1S1yqILQnoDQzTYQFW3smJhZ4rDNEkdp7qhoq99xgal8XyI4j6N7xkPqIzB8zYSM+XaILqJBYSsNWMgiAt0ZomHn68/wMI7GQsg4j3kW7KtI1gPCeBoQ9enN0IAEoL8GyJxuPgCLg/afPmd/E4BRBVDxfnEhQTy57skypcIn25N6CthP2Yt/pJvZtaAiwC41rVUMcSpir3maBDTyCR+QDdyaH1CxJWICzWd/LJHPyQZBxyC+H2wofRx2zn/K1MlR7dfyXr3LqHt+8lT59Xgwqq5XLh5NrCzbj6toxg3f15gdi4aKVFFe1UBevHg9GvqDRjt913oyHyvRWvrYKX1cnntSOe6kYU95/FtYbaBk+hFRIrZIxoiFSUBDGskh6p0GARpl2JDPHqAQQiAEFPv+sN/59a193XQslIqCrryARY/ioINLr1YnNwN7bOrHAyC75a3jeKIYrYp4AFcnqRTDzuVbRXYLlXp5NHpaM31y7f7x+z+f16RhjvPL36xIwq7Ybj8vFZA7S0OUr4/uSOQPO9Nns3yTBXT0JAQHpGUfsognJGcr2YACIm+5Eo2nOcDKJZItuAqIEL18vWL1fqkTWK+SCCqZIk+YKhsYE2d3MxpNrRy7CW6zHdF1tiMKyB0NijLBj6kACUmUEC4oUqcUkBRmr4L2qgrppQBb6qjneJlwqpJXE03bRXq0kuhcvRWTmmGvLyQP1U+m41hlHDktVk8OWA3lArrjMfEpR+rs1YKh5/M4EUiuKYjFgMvj75TxLBmDWBbweWExu9Xx1c35JSwEueQGKufFxXTmfbs4v/SWN/DzzcS9ZZyYHovuKXddw8rtfJIlAYEa6PW+syiifAqXubhl0YpeESHhYJ9duyR5YMYn17QthWy/H+qBFTJO9PfwYiBVavVbMjRGB5bs4tpXYWl3QcUqpYinAZQhmzWNEGypv1uqPSpCESAkP9MJMixVpqiDYRT/oJwzn2YPqStODUZC9UiQXfHwEOI2Ai3e8zhECeUDdclCOfeP+i/ER3UtclhI5ItsmFzHqUTE91n0gJKAbAWU/oWLjJ9wUXELZB3wYQOGT85OXog4z/3qqXaONYzNGgW9y55EZ2ijPt6sVHD6FFzzcQ8Q2KmDoJ46cukHU5E2LCXQpkJfaH8NwL7aMf5Xfe5qqrfQZs7EKBIHAszIQGoHkD5clwjuJkpC9EASNdHWB3BS3MYZ0ICu/gXHLCzOWBMAAA=="""
_EMBED_DOWNRANKS_V1_1_B64_GZ = """H4sIAO/WqWkC/62W3W7UMBCF73kKHsAge8Z/c4lUBBLSIhUE4soKu4ZGdHdLkoJ4e8aOnXTbkOWvq7qOt1p/PXPOTLvb6xjandjH/cfYiWY7tMeD2B4PuzbvrpsfsevFRrwUV+3QBynBlB3MO2PE0DXtIbwUQ+yH9CM/Dk33OQ7jWdnfHLdf4hD6q6aL4mPTx+v2EMtjf3tz08W+54vDJ0Y5do8uLsE8kUrkey9ev99cPtu8yhTh3fPLD+HN28vXmxfiqunh8bPNxWPehO/Hrh9umrbj38qHPX9qswufoxMovCMSgEIJBULJvPLn87N8KjVq58FKb8lYA8ryIRpPTteV+ESBUspLBRoArQRdMOG3MbeHgUL8qh7iOXAoIOEwG2VAWenMMg0SeOOUHlcsMGYJZubQ+ep9u9s1P0I/dO12yCeHY0jSMYnW3hcSJbxQzJYw/hgEz6nCYsAkBqPRkiqWK0YjixulqaLwW8o4T9qh9RqW0bS1yhqt0CB61Kaw6RWR4rd4mLCSJsE4AUJLstk0wgibXwXE4+nXIgggVH3yCgXEroAUJXSwMwpHAgRbpaA4ofnb1Pr8PUpJG5ywwL2CjWJQ8YtkEqs9a5M9m5dsl5opZxXdWXOmlFb8F9q6phIZZSVq8oZT5bTEigPncNg/5qF/EhgK4yQlE9PEBSdpWuIg5dE61JbQO+crh17imAM1Foc7Zx+37FpIqrAnhXI5N5QU0VOAvAc0QFopZ7X7JYvndqBJEllkYWxFwRWUk9ZSPcwwXgEz2AzjWZm5QBZ+dT2xW2jSw9Xrzcr1+7brjl12SPwq72SHUnbYf7WRpPS4iuAsAc4rLvMAt6Q7QKby2BWe/nYfAEKZBOMQCrvj9/D2Nvbc+1KMuJcnKs5SzvPU5v6eypQYmRMqsxAj6YtbU2+x1qkcI0hOTZ6B1RTBKWBudDy3eIIhSoMAqsLgEkx1zOjZe56Z5mkyi/FSJjBK/ZdfZm57/whmV8B4Fj2Y5aei6RxxbnCQJwONU53+ExucqyBnTc5jnJ2mfIAUNp40PiWfYXQupJ+6zlLyF5D4vxTek/HaOSO9q0z6fCG5+dwfFOQKy9iKcG1QPIQBwz1IkZXOa1Q4sZg1liRNJSmWwhw2CzPKmZn1Gyg/AWtjglfBCgAA"""
_EMBED_RESCUE_RULES_V2_B64_GZ = """H4sIAO/WqWkC/8WUXW/aMBSG7/crfLlJhyrOd7QrRFBXbYIK0Um7stzkNFgFO7UdWP/9TkI7PqRJRZqEAwecGPM+fl/bdmsUqgbnpe8cyMoro8FL26AXG9w8ooXdCrXwpo2gtVirSnoU+Lu1MANNoxqptFgp7+BprVpBYyrl+ln6h97S0/03dB5m+xtC4w6G3+27SlcWN6g9aEPjPi0Wo9vFdFr+GgUcfswn36elGD8s5zD/OV0s7sqpWM7vOZX+M4Iw6V+NbEXKRZQmbDwrWWW0w0o0yIfuSjoaGADnENxkkEBKVwLfVLMa/ZXMlkTJLLqqQ9aa6hk9+2yxXQ/QjinNHo1f0WTrLbovN8dCQ7h9GC9KUjqh/7srx8vpv+T2euntEGtBuopBoCMVsia92dDVJgEeQkxqkzw6bjFEwKGgem+c8mqLjNZ31C/nhfq/smfEljWdtDXWrNNrdI69mo7tpPZsYywy2TQ0X784J7TRxbRn9hC3aKWy750UeEBIRAsh4eXXxYs/nLpTE71C8aRol5xFEHhEcQtu0iKMguxQob9iIv5/OUw+7swAQJKtmAx6N8paYwW+BG8RjIHngylRftKKwaOM6hU9Si+O4BEqbulMw5fwjLvfciHxxjw9bhnRckiuy5tdaKw2exvfDvPa7MSyQ1fL14O7MUHRlst4HOaHSpx9Kvk1afPLY3y0885PF4prMaDGIQ+SMI14kuVFnEUJkYZXYf0DDdcQkHoHAAA="""

# Newer mined-layer bundles (from handoff pack 2026-03-05)
_EMBED_RULEPACK_V3_2_B64_GZ = """H4sIAPbmqWkC/61WW2/iOBR+51dYfQLJsLmQW7esxBS27U5hJNq9vkQmOS0WuTC2GcS/X+dWbBpo1U6rpo7t2Oe7nGOzbQIhjbGgwHAK6RJYKAh7BoFJJGie4SjPYlq2noCILQMeZgAxxJjBM00BJ2QPjON5yLebTc4Evg2fSEqTfbiiguPiERqG5dQt69ByHFztFdY7lx/oXXxFGGDBCM3CWyyAi+Jf+VpPrD4qBtQOLmSwHGe5HOn8js1+SjlHfMueaUQSFOe7jJFsjWU0k29/zxfj+ddwNp19mS7KYMO/pot/w4fHxbf5DeYSb7gi3ELj+QQ1bwG+eBnAh94LfJhgYd90zYGBLUs+ioZZNIL6xRgYQ0f7rcZNoxosJ938OV5MphN88Uihv2RA1ijPkj3qliQgA6MCOzJ7g4vO7Gfi7JavfCO3jH8beT0NfLjLGRcbQlmh42smqs/wifkXeHx/j23se0HBhWWr7JiGQo899HzLNXw3cFzHqieYLfQcs2P+Ytm9QWd+hpE2SqwTlDi69ApB8tUYjYzegQNHccOhaajGsLE79L0Cuqki1qB7rhkcntgvhuz3IZdm6Nyc80IbcqcduVFhNf2rUaXsNr0ayR0OeI1G821aCWthR0ZchOk3ojoNwgqAo/9UwP0Puv4L/g9YXoHd5NEaBOoy2CQStIC4V1h/en83U21/hE5XuWsaKlbTbsGqyPwC25aWcu0GcYNoWLeLP69monlvUF7n2RNlKcRIYltRLnK2H6C7rF/DgYSmNCOyu2Rh0Ln9BGTrpJVTyljOyiSNMqGb2nptan26bm+ZuOZHmJjD7lKScBL4+C3gx8gtBXktb31MyHQYjR63wGOy17P3MKHx89C0y6JdmNptbGoo8fuKfzVlSQJZTFif7OQ5donq/RDlaD795xHFjOyQ7OjnT/0dwHrQuf4MwqDNzlqZ0rssXeTgtbt1z1uqykM8DJyyIHsqHd6RzBpP5xx/CCtA1d1ClrG7j9BRLQQ/IHsxstZTqmpix/T8RtUm6kAJNjhC0wT/INfqF378FQHheyRyRNNNAilkYoC6182NCUUJkAziSwknzX+UYBOaAYrytJzb6/zxebVP6itvbhyiUO645id1bsRVJ+up7Nnmee+X57FznqcsR1GdCpKEzVbwQefr53P5iIlhU7fL28fVyD1/KlfTdLS2Vx3JtqK9q2Bz3/aEdDXdpqhaHS1JFg86kzcL9jHW4nIsU0BW2VjWh5ALRiOhqpsJKbjZltB6l1NpXy4WJYTzbrVWvXKvMYBcT0l0pRJURXCIrcB/Od2sYxOcK+oPVeyzcj+UkCUk9Wke58ClNwTaEMYBFTFymfTTj7D11oXlvUS1XGvO82LL3RtejiloO/vOnvqd+5+AXUP07vtb+53G9n2nSQi/BV1wAp08ydGS5SQGhooY1bO8G7N8w5FYgRTfktXx+5aysnxK8f8HXfr6IZMOAAA="""
_EMBED_DOWNRANKS_V1_1_B64_GZ = """H4sIAPbmqWkC/9WYbW/bNhDH3+9TCHnTBGAGPj8AdYACKdphQApk3Ya9IlSbaYTGkivJ6/rtd6REUbKUNG3TAk4QmiYpRvrd6e5/rPd3zhYb1BauRk2bt/sGbd32nattm9fvXYvydVtUJdpUn8o6Lz/YrWtvqw1aV+WmCDNDz7r/djW6yz+7ukE3DvaqXWNL5zZug65ss9/tqrpFr+1Nvi3uPtvbom2QbyzGVPQ9mnpCoLbOi9K+Rq1rWv8RvnY31l0eJsYDu2r9AfrNbV479C5v3F1Ruv6rvwO4pcbf6w08V1X/cnlNxTkmiJxvi6bJmn39vljnd1l83Oy0KM+7Pc/Qqz9fXF++vLSXb/6+un5x9TuCe4398Az2r5fX/9g/3l6/uXqFfrvJBjRZW+9d9uLqMntb7Ui2WmX9PXess/OLrPmU77rZ5/ANOjQ73VRZWbVZW+3Xt36InaHbvKFhI+jYT1XdtLu8qOF/h8EGni/f2PdOodMGwFu/fLUiZ1lebrJhKF3oMR/Md3tcrNQZYuhk2AVly1f3491VJ0grYxBlCJBSRHBoCYJPhH/FnHGlqcRaGiEFhSEmtFE8tgZGCCWEaEwop5RJTHsT0SM00bpsjXUfyaNN4y/4sjX8qmQNemAARRVD1AMH+iaYAEf+Ypk4M1QLRXjXsp64eCriPxo2Dxtti80m/2ybti7WbRgpK+udNOHmiW1bbJ2dXHLAfV8WH/fOD47fAt7D7mZRNt/mBHGudW8AgjQiDClP/6v5s+PxePBIOjg6YDLLHu+XzYPR4zx+FH/MzOMlsDIdcNW5fXR4mCJCacMVk5ov4+dSEik4YYIxzUSPnx+F+7t/XTmQDyFZRNphCsglvANyMbeC8kNniPbE48WJukhdBU6OjQyxHQkkw29PXLPJzyJxymh09tD2xOVREO+9mluZmIOAOOXPVyOvfL6Sh9DxHDqZQveLEmVy4OYQ13vkCnH4EzGqfCvy71I+E+j0p0SZzr1NH9sxetCfQ1Q5GAML4Ptoj9zbnCDJtUI0pNDQhDAeRYySxKTW4yacgN/J2PqgIojEjBstQMYoHnHTI8INb7+YB/UAPgZl8eVoPmAfxfIRazzBLhQ2PnWagTqdSJclzoZoJhXj0jCtdATNnwz0j1YvXRiBvRu3hlxJk1yRh+qwWwMFzYfmYkWTJ48nEk8Jbgx5DxEVBIjxLswHJaI1ZYIaToiSsGgZrgYxyQ02RjLwZBnhsuOAO5HfMVGie5X2OF8mtElrx3mocQgFmjJg1eDYKTZIep+fGojFZnBWFVGK40C5Leq6qkMAdh/xSGyYnudoQUCIHyE5zCz7ibnIm24MwgOyWtTUXnqoiF5JQ1lql81AQZyP7CCiGeRxmKHZby2lti/2+13gTu3bvWugBhkt6KqcYcVq9axf88wTTxOQ5TT2QAFB0HFDsfLNQMV3nahMgIqfpiuw7tObQg9KNr2gK9QDukKlroaALBUJuoL65OZjMn1AVtCZBQgXUmgoahgWjFIScbMnw/1jI3KXqA5i8nCepdE80d0bpOmSYZLKmKfEeUkDBhEaY28Q42tH+BWpkvkug8jjMEhZzc8Qp68DR/eeGuJHVDhLrwuf6MHp68JH1rnvxNHrRChzaKj3TXfSaJ7EavSIohZ4Mk6HixD3ibZ0JG+SMdIkRcMsEpxorw0BHg+hSA9Se64NFxAqY6BvhOZKCWwiQn5UkQj09rSKXxTaX13ILwvycW0P5bxRPfxOnbMHyvk5fipAlxMjsdKcEaYifnEk+L3zRvh9GmAj372YhvcU9dlEmuOZNKdE0kT14UOSR1D9H9OhVm8fGwAA"""
_EMBED_LOCKED_RESCUES_ONLY_B64_GZ = """H4sIAPbmqWkC/6WQUUvDMBDH3/0UeVS4jWZdO/Y4bFFRVilz4NORZbc22KUhuTk/vmkHKj4JcsflLvfP8bv4U0do9hBY8SmA0mx6C6x8Q4xHOu7Iw7kli9y7FJynvdGKCenDeViDjapGGYut4QCHzjiMGm3CMGVoso/dS0aBYX25QEtnGN9dSmO1pyNZBttH3VVdT+7qsixeJ4mEp+r2sSxw9bKpoNqWdf1QlLipnmUMw5nCLBu8UQ5ziWmeidW6ELq3gTQ2JMeyVSEKE5ASkukCMsijZXBvmnbyhSw2cUvhKegTCdfrN2Jx7cl149JBGCt2PbdxWPdO4Wb6E3T+Z9BAtMfIsxzB2BAeTPzYX9Qg00iYTPPlLE0W3xEGm8Psv+iftrn8/voBAAA="""

def _read_embedded_csv(b64_gz_text: str) -> pd.DataFrame:
    raw = gzip.decompress(base64.b64decode(b64_gz_text.encode("ascii")))
    return pd.read_csv(io.BytesIO(raw))


# ---- Mined action-rule packs (member eliminators, downranks, rescues) ----
import ast
import numpy as np

def _normalize_bool_expr(expr: str) -> str:
    if expr is None:
        return ""
    s = str(expr).strip()
    s = re.sub(r"\bAND\b", "and", s, flags=re.IGNORECASE)
    s = re.sub(r"\bOR\b", "or", s, flags=re.IGNORECASE)
    s = re.sub(r"\bNOT\b", "not", s, flags=re.IGNORECASE)
    return s

_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.Name, ast.Load, ast.Constant,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _inject_dynamic_name(ctx: dict, name: str) -> None:
    """Populate derived boolean/string aliases used in mined rule expressions."""
    # Weekday literals used in expressions like (target_dow == Tuesday)
    weekdays = {"Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"}
    if name in weekdays:
        ctx[name] = name
        return

    # Aliases for consolidated names
    if name == "seed_seed_has_worstpair_025":
        ctx[name] = int(ctx.get("seed_has_worstpair_025", 0))
        return
    if name == "has_worstpair025":
        ctx[name] = int(ctx.get("seed_has_worstpair_025", 0))
        return
    if name == "no_worstpair025":
        ctx[name] = 1 - int(ctx.get("seed_has_worstpair_025", 0))
        return
    if name == "midday_strict" and "time_midday_strict" in ctx:
        ctx[name] = int(ctx.get("time_midday_strict", 0))
        return
    if name == "time_midday_strict" and "midday_strict" in ctx:
        ctx[name] = int(ctx.get("midday_strict", 0))
        return
    if name == "consec":
        ctx[name] = int(ctx.get("consec_links", ctx.get("consec", 0)))
        return
    if name.startswith("consec_") and "consec_links" in ctx:
        # e.g., consec_ge2, consec_eq1, consec_le0
        m = re.fullmatch(r"consec_(ge|le|eq)(\d+)", name)
        if m:
            op, k = m.group(1), int(m.group(2))
            v = int(ctx.get("consec_links", 0))
            if op == "ge":
                ctx[name] = int(v >= k)
            elif op == "le":
                ctx[name] = int(v <= k)
            else:
                ctx[name] = int(v == k)
            return

    if name.startswith("mirrorpair_") and "mirrorpair_cnt" in ctx:
        m = re.fullmatch(r"mirrorpair_(ge|le|eq)(\d+)", name)
        if m:
            op, k = m.group(1), int(m.group(2))
            v = int(ctx.get("mirrorpair_cnt", 0))
            if op == "ge":
                ctx[name] = int(v >= k)
            elif op == "le":
                ctx[name] = int(v <= k)
            else:
                ctx[name] = int(v == k)
            return

    if name.startswith("even_") and "seed_even_cnt" in ctx:
        m = re.fullmatch(r"even_(ge|le|eq)(\d+)", name)
        if m:
            op, k = m.group(1), int(m.group(2))
            v = int(ctx.get("seed_even_cnt", 0))
            if op == "ge":
                ctx[name] = int(v >= k)
            elif op == "le":
                ctx[name] = int(v <= k)
            else:
                ctx[name] = int(v == k)
            return

    # Range helpers: sum_18_21, spread_4_6, gap_61_365, etc.
    m = re.fullmatch(r"(sum|spread)_(\d+)_(\d+)", name)
    if m:
        kind, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        base = "seed_sum" if kind == "sum" else "seed_spread"
        v = ctx.get(base, None)
        if v is None:
            ctx[name] = 0
        else:
            ctx[name] = int(a <= int(v) <= b)
        return

    # Fallback: default missing names to 0 (safe behavior)
    ctx[name] = 0

def _safe_eval_bool(expr: str, ctx: Dict[str, object]) -> bool:
    expr = _normalize_bool_expr(expr)
    if not expr:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            return False
        if isinstance(node, ast.Name) and node.id not in ctx:
            _inject_dynamic_name(ctx, node.id)
    try:
        return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ctx))
    except Exception:
        return False

def _stream_is_midday_strict(stream: str) -> int:
    s = (stream or "").lower()
    return 1 if ("midday" in s or "mid day" in s) else 0

@st.cache_data(show_spinner=False)
def load_mined_rule_dfs(
    rulepack_file: Optional[io.BytesIO] = None,
    downranks_file: Optional[io.BytesIO] = None,
    rescues_file: Optional[io.BytesIO] = None,
) -> Dict[str, pd.DataFrame]:
    if rulepack_file is None:
        df_rulepack = _read_embedded_csv(_EMBED_RULEPACK_V3_2_B64_GZ)
    else:
        df_rulepack = pd.read_csv(rulepack_file)

    if downranks_file is None:
        df_down = _read_embedded_csv(_EMBED_DOWNRANKS_V1_1_B64_GZ)
    else:
        df_down = pd.read_csv(downranks_file)

    if rescues_file is None:
        df_resc = _read_embedded_csv(_EMBED_RESCUE_RULES_V2_B64_GZ)
    else:
        df_resc = pd.read_csv(rescues_file)

    if "member_target" in df_rulepack.columns:
        df_rulepack["member_target"] = pd.to_numeric(df_rulepack["member_target"], errors="coerce")
    if "member_target" in df_down.columns:
        df_down["member_target"] = pd.to_numeric(df_down["member_target"], errors="coerce")
    if "when_top3" in df_resc.columns:
        df_resc["when_top3"] = pd.to_numeric(df_resc["when_top3"], errors="coerce")
    df_locked = _read_embedded_csv(_EMBED_LOCKED_RESCUES_ONLY_B64_GZ)
    return {"rulepack": df_rulepack, "downranks": df_down, "rescues": df_resc, "locked_rescues": df_locked}

def build_rule_context(
    *,
    seed4: str,
    stream: str,
    feats: Dict[str, object],
    tie_fired: int,
    dead_tie: int,
    base_gap: int,
    draws_since_last_025: Optional[int],
    rolling_025_30: Optional[int],
    rolling_025_60: Optional[int],
    target_dow: str,
    tier: Optional[str],
    top1: int,
    top2: int,
    top3: int,
) -> Dict[str, object]:
    ctx: Dict[str, object] = dict(feats)

    for d in range(10):
        ctx[f"has{d}"] = 1 if int(ctx.get(f"seed_has{d}", 0)) == 1 else 0
        ctx[f"no{d}"] = 1 if int(ctx.get(f"seed_has{d}", 0)) == 0 else 0
        ctx[f"cnt{d}"] = int(ctx.get(f"seed_cnt{d}", 0))
        for k in range(0, 5):
            ctx[f"cnt{d}_eq{k}"] = 1 if int(ctx.get(f"seed_cnt{d}", 0)) == k else 0

    ctx["has_pair"] = 1 if int(ctx.get("seed_has_pair", 0)) == 1 else 0
    ctx["no_pair"] = 1 if int(ctx.get("seed_has_pair", 0)) == 0 else 0

    consec = int(ctx.get("seed_consec_links", 0))
    for k in range(0, 4):
        ctx[f"consec_links_ge{k}"] = 1 if consec >= k else 0

    spread = int(ctx.get("seed_spread", 0))
    for k in range(0, 10):
        ctx[f"spread_ge{k}"] = 1 if spread >= k else 0
        ctx[f"spread_le{k}"] = 1 if spread <= k else 0

    mp = int(ctx.get("seed_mirrorpair_cnt", ctx.get("mirrorpair_cnt", 0)))
    for k in range(0, 4):
        ctx[f"mirrorpair_cnt_eq{k}"] = 1 if mp == k else 0
        ctx[f"mirrorpair_cnt_ge{k}"] = 1 if mp >= k else 0

    ctx["tie_fired"] = int(tie_fired)
    ctx["dead_tie"] = int(dead_tie)
    ctx["base_gap"] = int(base_gap)
    ctx["midday_strict"] = _stream_is_midday_strict(stream)
    ctx["time_midday_strict"] = ctx["midday_strict"]
    ctx["target_dow"] = str(target_dow)
    ctx["Tier"] = "" if tier is None else str(tier)

    gap = int(draws_since_last_025) if draws_since_last_025 is not None and str(draws_since_last_025) != "nan" else 9999
    ctx["WinnerDrawsSinceLast025"] = gap
    ctx["gap_0_60"] = 1 if 0 <= gap <= 60 else 0
    ctx["gap_61_365"] = 1 if 61 <= gap <= 365 else 0
    ctx["gap_ge366"] = 1 if gap >= 366 else 0

    ctx["rolling_025_30"] = int(rolling_025_30) if rolling_025_30 is not None and str(rolling_025_30) != "nan" else 0
    ctx["rolling_025_60"] = int(rolling_025_60) if rolling_025_60 is not None and str(rolling_025_60) != "nan" else 0

    ctx["Top1"] = int(top1)
    ctx["Top2"] = int(top2)
    ctx["Top3_infer"] = int(top3)

    if "seed_seed_has_worstpair_025" not in ctx:
        ctx["seed_seed_has_worstpair_025"] = int(ctx.get("seed_has_worstpair_025", 0))

    ctx["tierA"] = 1 if str(ctx.get("Tier", "")).upper() == "A" else 0
    ctx["tierB"] = 1 if str(ctx.get("Tier", "")).upper() == "B" else 0
    ctx["tierC"] = 1 if str(ctx.get("Tier", "")).upper() == "C" else 0

    return ctx

def apply_member_eliminators_from_rulepack(
    *,
    member_order: List[int],
    ctx: Dict[str, object],
    df_rulepack: pd.DataFrame,
) -> Tuple[List[int], List[str]]:
    fired: List[str] = []
    if df_rulepack is None or df_rulepack.empty:
        return member_order, fired
    eliminated = set()
    for _, r in df_rulepack.iterrows():
        action = str(r.get("action", ""))
        if not action.startswith("ELIM_MEMBER_"):
            continue
        if _safe_eval_bool(r.get("condition", ""), dict(ctx)):
            fired.append(str(r.get("rule_id", "")))
            mt = normalize_member_id(r.get("member_target"))
            if mt:
                eliminated.add(mt)
    if not eliminated:
        return member_order, fired
    keep = [m for m in member_order if m not in eliminated]
    drop = [m for m in member_order if m in eliminated]
    return keep + drop, fired

def apply_one_miss_downranks(
    *,
    member_order: List[int],
    ctx: Dict[str, object],
    df_down: pd.DataFrame,
) -> Tuple[List[int], Optional[str]]:
    if df_down is None or df_down.empty:
        return member_order, None
    top1 = normalize_member_id(member_order[0])
    top2 = normalize_member_id(member_order[1])
    candidates = []
    for _, r in df_down.iterrows():
        mt = normalize_member_id(r.get("member_target"))
        if not mt or mt != top1:
            continue
        if _safe_eval_bool(r.get("condition", ""), dict(ctx)):
            try:
                baseline = float(r.get("baseline_share", np.nan))
                pocket = float(r.get("pocket_share", np.nan))
                strength = (baseline - pocket) if (not np.isnan(baseline) and not np.isnan(pocket)) else float(r.get("suppression_factor", -0.25))
            except Exception:
                strength = float(r.get("suppression_factor", -0.25))
            candidates.append((strength, str(r.get("rule_id", ""))))
    if not candidates:
        return member_order, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_id = candidates[0][1]
    return [top2, top1, normalize_member_id(member_order[2])], best_id

def apply_rescues(
    *,
    member_order: List[int],
    ctx: Dict[str, object],
    df_rescues: pd.DataFrame,
    allow_guarded: bool = False,
) -> Tuple[List[int], Optional[str]]:
    if df_rescues is None or df_rescues.empty:
        return member_order, None
    top3 = normalize_member_id(member_order[2])
    cands = []
    for _, r in df_rescues.iterrows():
        status = str(r.get("status", ""))
        if (status != "LOCKED_AUTO") and (not allow_guarded):
            continue
        when_top3 = normalize_member_id(r.get("when_top3"))
        if not when_top3 or when_top3 != top3:
            continue
        if _safe_eval_bool(r.get("predicate_expr", ""), dict(ctx)):
            prio = 2 if status == "LOCKED_AUTO" else 1
            try:
                net = float(r.get("net_gain_hits", 0))
            except Exception:
                net = 0.0
            try:
                prec = float(r.get("flip_precision", 0))
            except Exception:
                prec = 0.0
            cands.append((prio, net, prec, str(r.get("rule_id", ""))))
    if not cands:
        return member_order, None
    cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    best_id = cands[0][3]
    return [member_order[2], member_order[0], member_order[1]], best_id

def apply_rulepack_overrides_to_top3(
    *,
    member_order: List[int],
    ctx: Dict[str, object],
    df_rulepack: pd.DataFrame,
) -> Tuple[List[int], Optional[str]]:
    if df_rulepack is None or df_rulepack.empty:
        return member_order, None
    top3 = member_order[2]
    cands = []
    for _, r in df_rulepack.iterrows():
        if str(r.get("action", "")) != "OVERRIDE_PICK_TO_TOP3":
            continue
        if _safe_eval_bool(r.get("condition", ""), dict(ctx)):
            cands.append(str(r.get("rule_id", "")))
    if not cands:
        return member_order, None
    cands.sort()
    best_id = cands[0]
    return [top3, member_order[0], member_order[1]], best_id

def apply_mined_member_layers(
    *,
    seed4: str,
    stream: str,
    s_base: Dict[str, object],
    feats: Dict[str, object],
    draws_since_last_025: Optional[int],
    rolling_025_30: Optional[int],
    rolling_025_60: Optional[int],
    target_dow: str,
    tier: Optional[str],
    mined_dfs: Dict[str, pd.DataFrame],
    enable_eliminators: bool,
    enable_downranks: bool,
    enable_locked_rescues: bool,
    allow_guarded_rescues: bool,
    enable_rulepack_top3_overrides: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    top1 = normalize_member_id(s_base["top1"])
    top2 = normalize_member_id(s_base["top2"])
    top3 = normalize_member_id(s_base["top3"])
    member_order = [top1, top2, top3]

    ctx = build_rule_context(
        seed4=seed4, stream=stream, feats=feats,
        tie_fired=int(s_base.get("tie_fired", 0)),
        dead_tie=int(s_base.get("dead_tie", 0)),
        base_gap=int(s_base.get("base_gap", 0)),
        draws_since_last_025=draws_since_last_025,
        rolling_025_30=rolling_025_30,
        rolling_025_60=rolling_025_60,
        target_dow=target_dow,
        tier=tier,
        top1=member_to_legacy_id(member_order[0]),
        top2=member_to_legacy_id(member_order[1]),
        top3=member_to_legacy_id(member_order[2]),
    )

    meta = {"elim_fired": [], "downrank_fired": None, "rescue_fired": None, "rulepack_top3_fired": None}

    df_rulepack = mined_dfs.get("rulepack")
    df_down = mined_dfs.get("downranks")
    df_resc = mined_dfs.get("rescues")

    if enable_eliminators:
        member_order, fired = apply_member_eliminators_from_rulepack(member_order=member_order, ctx=ctx, df_rulepack=df_rulepack)
        meta["elim_fired"] = fired

    ctx["Top1"], ctx["Top2"], ctx["Top3_infer"] = (
        member_to_legacy_id(member_order[0]),
        member_to_legacy_id(member_order[1]),
        member_to_legacy_id(member_order[2]),
    )

    if enable_downranks:
        member_order, did = apply_one_miss_downranks(member_order=member_order, ctx=ctx, df_down=df_down)
        meta["downrank_fired"] = did

    ctx["Top1"], ctx["Top2"], ctx["Top3_infer"] = (
        member_to_legacy_id(member_order[0]),
        member_to_legacy_id(member_order[1]),
        member_to_legacy_id(member_order[2]),
    )

    if enable_locked_rescues:
        member_order, rid = apply_rescues(member_order=member_order, ctx=ctx, df_rescues=df_resc, allow_guarded=allow_guarded_rescues)
        meta["rescue_fired"] = rid

    ctx["Top1"], ctx["Top2"], ctx["Top3_infer"] = (
        member_to_legacy_id(member_order[0]),
        member_to_legacy_id(member_order[1]),
        member_to_legacy_id(member_order[2]),
    )

    if enable_rulepack_top3_overrides:
        member_order, rid = apply_rulepack_overrides_to_top3(member_order=member_order, ctx=ctx, df_rulepack=df_rulepack)
        meta["rulepack_top3_fired"] = rid

    s_final = dict(s_base)
    s_final["top1_final"] = member_order[0]
    s_final["top2_final"] = member_order[1]
    s_final["top3_final"] = member_order[2]
    return s_final, meta
import numpy as np
import re
from dataclasses import dataclass

# --- Member ID normalization (supports legacy numeric IDs like 25/225/255 and canonical 4-digit IDs) ---
_CORE_MEMBER_KEYS = {"0025", "0225", "0255"}

def normalize_member_id(x: object) -> str:
    """Normalize member identifiers to canonical 4-digit strings: 0025 / 0225 / 0255.

    Accepts inputs like 25, '25', '0025', 225, '225', '0225', 25.0, etc.
    Returns the original string (stripped) if it cannot be normalized.
    """
    if x is None:
        return ""
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
    elif isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        if float(x).is_integer():
            s = str(int(x))
        else:
            s = str(x).strip()
    else:
        s = str(x).strip()
    if s == "":
        return ""
    # Prefer digit-only normalization when possible
    if re.fullmatch(r"\d+", s):
        if len(s) < 4:
            s2 = s.zfill(4)
            if s2 in _CORE_MEMBER_KEYS:
                return s2
        elif len(s) == 4 and s in _CORE_MEMBER_KEYS:
            return s
    # Already canonical?
    if s in _CORE_MEMBER_KEYS:
        return s
    # Some packs include 'pick_str' already canonicalized (e.g. 0225)
    if len(s) == 4 and s.isdigit() and s.zfill(4) in _CORE_MEMBER_KEYS:
        return s.zfill(4)
    return s

def member_to_legacy_id(x: object) -> object:
    """Return legacy numeric member IDs (25 / 225 / 255) for rule-expression contexts."""
    s = normalize_member_id(x)
    return {"0025": 25, "0225": 225, "0255": 255}.get(s, x)
from typing import Dict, List, Tuple, Optional

def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _embedded_raw_bytes(b64_gz: str) -> bytes:
    return gzip.decompress(base64.b64decode(b64_gz.encode("ascii")))


# -----------------------------
# Best-practice default settings
# -----------------------------
def apply_best_defaults(preset: str = "budget") -> None:
    """Apply a recommended preset without clearing uploaded files."""
    # NOTE: Many widgets have explicit keys; some rely on label-as-key (Streamlit default).
    # We set both where relevant.
    if preset == "coverage":
        cap = 80
        top2_support_thresh = 62
        allow_guarded_rescues = False
    else:  # "budget"
        cap = 50
        top2_support_thresh = 60
        allow_guarded_rescues = False

    # Explicit-key widgets
    st.session_state["wf_cap_streams"] = int(cap)
    st.session_state["downrank_strictness"] = "single_strongest_only"
    st.session_state["max_rules"] = 12
    st.session_state["min_support"] = 6
    st.session_state["play_top2_if_seed9"] = True
    st.session_state["play_top2_if_tie"] = True
    st.session_state["play_top2_if_low_support"] = True
    st.session_state["top2_support_threshold"] = float(top2_support_thresh)

    # Label-key widgets (fallback, if keys were not provided)
    label_defaults = {
        "Gate: seed has NO digit 9 (recommended)": True,
        "Enable member eliminators (0-miss separators)": True,
        "Enable 1-miss downranks (Top1 penalty swaps)": True,
        "Enable LOCKED rescues (RR-GREEDY-01/04 only)": True,
        "Allow GUARDED rescues (experimental, can hurt capture)": allow_guarded_rescues,
        "Enable stream-level skip suggestions (mark some streams as NO PLAY)":
            True,
        "If enabled, exclude 'NO PLAY' streams from the Top-N plays list":
            False,
        "Treat streams with 0 hits in the full history window as NO PLAY":
            False,
        "Prefer Top1-only plays (default), only add Top2 when needed":
            True,
        "Show seed_has9 regime split diagnostics (recommended)":
            True,
        "Show rule provenance (sha256 hashes)":
            False,
        "Show per-rule firing diagnostics in LIVE table (recommended for debugging)":
            False,
    }
    for k, v in label_defaults.items():
        st.session_state[k] = v



def _coerce_value(v: object):
    """Best-effort coercion for rule 'value' fields loaded from CSV.

    Goal: make uploaded rulepacks robust when pandas reads mixed types as strings.
    - '1' -> int(1)
    - '3.0' -> float(3.0)
    - 'True'/'False' -> bool
    Otherwise returns the original value.
    """
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return v
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    # integer?
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except Exception:
            return v
    # float?
    if re.fullmatch(r"-?\d+\.\d+", s):
        try:
            return float(s)
        except Exception:
            return v
    return v


@dataclass(frozen=True)
class Rule:
    """Lightweight rule container used by weights/tie-pack scoring."""
    feature: str
    op: str
    value: object
    pick: str
    weight: float = 0.0
    source: str = ""
    rule_id: str | None = None

    @staticmethod
    def from_row(row, source: str = "") -> "Rule":
        """Create a Rule from a pandas row/dict. Tries multiple column schemas."""
        # Accept dict-like / pandas Series
        def _get(key, default=None):
            try:
                return row.get(key, default)
            except Exception:
                try:
                    return row[key]
                except Exception:
                    return default

        # Common schema for weight/tie packs
        feature = _get("feature") or _get("feat") or _get("field") or ""
        op = _get("op") or _get("operator") or _get("cmp") or ""
        value = _get("value") if _get("value") is not None else _get("val")
        value = _coerce_value(value)
        pick = _get("pick_str") or _get("pick") or _get("member") or _get("target") or ""
        rid = _get("rule_id") or _get("id")
        # Weight schema support:
        # - tie-pack typically uses 'weight'
        # - mined/older packs may use 'new_weight' / 'old_weight'
        # - some packs use 'w'
        w_candidates: dict[str, float] = {}
        for k in ("weight", "new_weight", "old_weight", "w"):
            v = _get(k, None)
            if v is None:
                continue
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                continue
            try:
                w_candidates[k] = float(v)
            except Exception:
                continue

        if "weight" in w_candidates:
            w = w_candidates["weight"]
        else:
            # Prefer non-zero new_weight, otherwise non-zero old_weight, else fall back.
            w = 0.0
            if "new_weight" in w_candidates and w_candidates["new_weight"] != 0.0:
                w = w_candidates["new_weight"]
            elif "old_weight" in w_candidates and w_candidates["old_weight"] != 0.0:
                w = w_candidates["old_weight"]
            elif "new_weight" in w_candidates:
                w = w_candidates["new_weight"]
            elif "old_weight" in w_candidates:
                w = w_candidates["old_weight"]
            elif "w" in w_candidates:
                w = w_candidates["w"]


        return Rule(feature=str(feature).strip(), op=str(op).strip(), value=value, pick=normalize_member_id(pick), weight=w, source=source, rule_id=str(rid).strip() if rid is not None else None)




def _canon_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept Lottery Post exports that may use different header names.
    We try to map to required canonical columns:
      - Draw Date
      - State
      - Game
      - Results
    """
    colmap = {_canon_col(c): c for c in df.columns}

    # candidates for each required field (canonicalized)
    candidates = {
        "Draw Date": [
            "drawdate", "drawdateutc", "date", "drawdatedate", "drawingdate", "drawndate", "draw_date"
        ],
        "State": [
            "state", "jurisdiction", "province", "region", "location"
        ],
        "Game": [
            "game", "gamename", "lotterygame", "draw", "drawing", "game_name"
        ],
        "Results": [
            "results", "result", "winningnumbers", "winningnumber", "numbers", "winnumbers", "winning", "win"
        ],
    }

    resolved = {}
    for need, opts in candidates.items():
        found = None
        for o in opts:
            if o in colmap:
                found = colmap[o]
                break
        if found:
            resolved[need] = found

    # If already correct, nothing to do
    if all(k in df.columns for k in ["Draw Date", "State", "Game", "Results"]):
        return df

    # Attempt partial mapping, and keep original cols
    df2 = df.copy()
    for need, src in resolved.items():
        if need not in df2.columns and src in df2.columns:
            df2[need] = df2[src]

    return df2

# --- Stream key normalization (prevents "string mismatch" joins) ---
def canon_stream(s: str) -> str:
    """Canonicalize Stream labels like "State | Game" to a stable ASCII key."""
    if s is None:
        return ""
    s = str(s)
    # Normalize unicode (accents, smart quotes, etc.)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    s = s.replace('–', '-').replace('—', '-')
    # Normalize separators and whitespace
    s = s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    # Normalize the pipe separator if present
    if '|' in s:
        parts = [re.sub(r'\s+', ' ', p).strip() for p in s.split('|')]
        if len(parts) >= 2:
            left = parts[0].lower()
            right = ' | '.join(parts[1:]).lower()
            s = f"{left}|{right}"
        else:
            s = s.lower()
    else:
        s = s.lower()
    return s


# Auto-load best-practice defaults once per browser session.
# This prevents having to re-set a long list of toggles/sliders on each refresh.
# IMPORTANT: do NOT call st.rerun() before marking the flag, or Streamlit will loop forever.
if "_autoloaded_best_defaults" not in st.session_state:
    st.session_state["_autoloaded_best_defaults"] = True
    apply_best_defaults("budget")  # best-practice: budget-focused live defaults



# ------------------------------
# Download filename helpers
# ------------------------------

def _safe_fname_component(x) -> str:
    """Return a filesystem-safe, short-ish token."""
    s = "" if x is None else str(x)
    s = s.strip().replace(" ", "_").replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    s = re.sub(r"_+", "_", s)
    s = re.sub(r"-+", "-", s)
    return s[:80] if len(s) > 80 else s


def _wf_download_name(kind: str, payload: dict | None, rules_meta: dict | None, ext: str = "csv") -> str:
    """Build descriptive, stable filenames for walk-forward downloads."""
    payload = payload or {}
    rules_meta = rules_meta or {}

    def _short_sig(obj, n: int = 8) -> str:
        # Note: in this app, payload["sig"] is a tuple signature (not a string).
        # Hash repr(sig) so filenames remain stable + join-safe.
        if obj is None:
            return ""
        try:
            s = repr(obj)
        except Exception:
            s = str(obj)
        return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:n]

    ext = (ext or "csv").lstrip(".") or "csv"

    gate = payload.get("gate_mode") or payload.get("label") or "run"
    start = payload.get("start_date") or ""
    end = payload.get("end_date") or ""
    play = payload.get("play_date") or payload.get("target_date") or ""

    max_rules = payload.get("max_rules_used") or payload.get("max_rules") or ""
    min_sup = payload.get("min_support") or payload.get("support") or ""

    ranks = payload.get("selected_ranks")  # may be list/tuple OR pre-formatted string
    bins = payload.get("selected_bins")    # may be list/tuple OR pre-formatted string
    cap = payload.get("cap_streams")

    ranks_tag = ""
    if ranks:
        if isinstance(ranks, (list, tuple, set)):
            try:
                ranks_list = sorted(int(x) for x in ranks)
                if ranks_list:
                    ranks_tag = f"rows{ranks_list[0]}-{ranks_list[-1]}"
            except Exception:
                ranks_tag = f"rows{ranks}"
        else:
            ranks_tag = f"rows{ranks}"

    bins_tag = ""
    if bins:
        if isinstance(bins, (list, tuple, set)):
            try:
                bins_tag = f"bins{len(list(bins))}"
            except Exception:
                bins_tag = f"bins{bins}"
        else:
            bins_tag = f"bins{bins}"

    cap_tag = f"cap{cap}" if cap not in (None, "", 0) else ""

    rules_sha = (
        rules_meta.get("rules_sha256")
        or rules_meta.get("weights_sha256")
        or rules_meta.get("weights_sha")
        or ""
    )
    tie_sha = (
        rules_meta.get("tie_sha256")
        or rules_meta.get("tie_pack_sha256")
        or rules_meta.get("tie_pack_sha")
        or ""
    )

    run_id = _short_sig(payload.get("sig"), 8)

    parts = [
        _safe_fname_component("core025"),
        _safe_fname_component(kind),
        _safe_fname_component(gate),
        _safe_fname_component(f"train{start}_to_{end}"),
        _safe_fname_component(f"play{play}" if play else ""),
        _safe_fname_component(f"maxrules{max_rules}" if max_rules != "" else ""),
        _safe_fname_component(f"minsupp{min_sup}" if min_sup != "" else ""),
        _safe_fname_component(ranks_tag),
        _safe_fname_component(bins_tag),
        _safe_fname_component(cap_tag),
        _safe_fname_component(f"w{str(rules_sha)[:8]}" if rules_sha else ""),
        _safe_fname_component(f"t{str(tie_sha)[:8]}" if tie_sha else ""),
        _safe_fname_component(f"run{run_id}" if run_id else ""),
    ]
    parts = [str(p) for p in parts if p]
    return "_".join(parts) + f".{ext}"




def _wf_pct_bin_label(pct_0_1: float, bin_size_pct: int = 5) -> str:
    """Return a stable percentile-bin label like '00-05%' for pct in [0,1]."""
    try:
        pct = float(pct_0_1)
    except Exception:
        pct = 0.0
    pct = 0.0 if pct != pct else max(0.0, min(1.0, pct))  # NaN-safe clamp
    bs = int(bin_size_pct) if int(bin_size_pct) > 0 else 5
    lo = int((pct * 100.0) // bs) * bs
    hi = min(100, lo + bs)
    return f"{lo:02d}-{hi:02d}%"


def _wf_suggest_bins_greedy(per_event: pd.DataFrame,
                            per_stream: pd.DataFrame,
                            cap_streams: int = 50,
                            bin_size_pct: int = 5) -> list[str]:
    """Greedy pick bins by winner frequency until average daily candidate streams reaches ~cap."""
    if per_event is None or per_event.empty or per_stream is None or per_stream.empty:
        return []

    pe = per_event.copy()
    ps = per_stream.copy()

    if "Percentile" not in pe.columns or "Percentile" not in ps.columns:
        return []

    pe["PctBin"] = pe["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    ps["PctBin"] = ps["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))

    pe_in = pe[pe.get("InUniverse", 0).astype(int) == 1].copy()
    if pe_in.empty:
        return []

    win_counts = pe_in["PctBin"].value_counts().to_dict()

    # Candidate count per day per bin
    if "PlayDate" not in ps.columns:
        return []
    daily_bin_counts = ps.groupby(["PlayDate", "PctBin"]).size().reset_index(name="n")
    avg_bin = daily_bin_counts.groupby("PctBin")["n"].mean().to_dict()

    bins_sorted = sorted(win_counts.keys(), key=lambda b: (-win_counts.get(b, 0), b))
    selected = []
    cur_avg = 0.0
    target = float(max(1, int(cap_streams)))

    for b in bins_sorted:
        selected.append(b)
        cur_avg = sum(avg_bin.get(x, 0.0) for x in selected)
        if cur_avg >= target:
            break

    return selected



def _wf_select_cap50_by_bins(per_stream: pd.DataFrame,
                             selected_bins: list[str],
                             cap_streams: int = 50,
                             bin_size_pct: int = 5,
                             fill_to_cap: bool = True) -> pd.DataFrame:
    """Mark Selected50=1 for each day using selected percentile bins, then trim/fill to exact cap.

    Tie-breakers are deterministic and aligned to the ranked playlist:
      - Prefer lower Rank (Rank=1 is best)
      - Then stable StreamKey
    """
    if per_stream is None or per_stream.empty:
        return per_stream.copy() if per_stream is not None else pd.DataFrame()

    df = per_stream.copy()
    if "PlayDate" not in df.columns:
        return df

    # Bin label for each stream/day row
    if "Percentile" in df.columns:
        df["PctBin"] = df["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    else:
        df["PctBin"] = ""

    df["Selected50"] = 0

    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers (Rank ascending is best)
    sort_by = []
    sort_asc = []
    if "Rank" in df.columns:
        sort_by.append("Rank")
        sort_asc.append(True)
    elif "StreamScore" in df.columns:
        sort_by.append("StreamScore")
        sort_asc.append(False)

    if "StreamKey" in df.columns:
        sort_by.append("StreamKey")
        sort_asc.append(True)
    else:
        sort_by.append(df.columns[0])
        sort_asc.append(True)

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        if "skip_exclude_from_topn" in g.columns:
            g = g[g["skip_exclude_from_topn"] != True].copy()
        cap_d = min(cap, len(g))

        cand = g[g["PctBin"].isin(selected_bins)] if selected_bins else g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_by, ascending=sort_asc).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_by, ascending=sort_asc).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df

    df["PctBin"] = df["Percentile"].apply(lambda x: _wf_pct_bin_label(x, bin_size_pct))
    df["Selected50"] = 0

    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers
    sort_cols = []
    if "StreamScore" in df.columns:
        sort_cols.append("StreamScore")
    if "Rank" in df.columns:
        sort_cols.append("Rank")
    sort_cols.append("StreamKey" if "StreamKey" in df.columns else df.columns[0])

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        if "skip_exclude_from_topn" in g.columns:
            g = g[g["skip_exclude_from_topn"] != True].copy()
        cap_d = min(cap, len(g))

        cand = g[g["PctBin"].isin(selected_bins)] if selected_bins else g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_cols, ascending=[False]*len(sort_cols[:-1]) + [True]).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_cols, ascending=[False]*len(sort_cols[:-1]) + [True]).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df



# ------------------------------
# Walk-forward play-method helpers (50 streams/day by historically best winner-producing rank rows)
# ------------------------------

def _wf_suggest_rank_rows_topk(per_event: pd.DataFrame,
                               cap_streams: int = 50) -> list[int]:
    """Pick the top-K rank rows (non-contiguous) with the most in-universe winners."""
    if per_event is None or per_event.empty:
        return []

    pe = per_event.copy()
    if "Rank" not in pe.columns:
        return []

    # Use only in-universe events (where the winning stream was in that day's ranked universe)
    try:
        pe_in = pe[pe.get("InUniverse", 0).astype(int) == 1].copy()
    except Exception:
        pe_in = pe[pe.get("InUniverse", 0) == 1].copy()

    pe_in["Rank"] = pd.to_numeric(pe_in["Rank"], errors="coerce")
    pe_in = pe_in.dropna(subset=["Rank"])
    if pe_in.empty:
        return []

    counts = pe_in["Rank"].astype(int).value_counts()
    # sort by wins desc, then rank asc
    ordered = sorted(counts.index.tolist(), key=lambda r: (-int(counts.get(r, 0)), int(r)))
    k = int(cap_streams) if int(cap_streams) > 0 else 50
    selected = ordered[:k]
    return [int(x) for x in selected]


def _wf_select_cap50_by_ranks(per_stream: pd.DataFrame,
                              selected_ranks: list[int],
                              cap_streams: int = 50,
                              fill_to_cap: bool = True) -> pd.DataFrame:
    """Mark Selected50=1 for each day using selected Rank rows, then trim/fill to exact cap."""
    if per_stream is None or per_stream.empty:
        return per_stream.copy() if per_stream is not None else pd.DataFrame()

    df = per_stream.copy()
    if "PlayDate" not in df.columns:
        return df

    df["Selected50"] = 0
    cap = int(cap_streams)
    if cap < 1:
        cap = 1

    # Deterministic tie-breakers (Rank ascending is best)
    sort_by = []
    sort_asc = []
    if "Rank" in df.columns:
        sort_by.append("Rank")
        sort_asc.append(True)
    elif "StreamScore" in df.columns:
        sort_by.append("StreamScore")
        sort_asc.append(False)

    if "StreamKey" in df.columns:
        sort_by.append("StreamKey")
        sort_asc.append(True)
    else:
        sort_by.append(df.columns[0])
        sort_asc.append(True)

    # Normalize selected_ranks
    sel = set(int(x) for x in selected_ranks) if selected_ranks else set()

    for d, g in df.groupby("PlayDate", sort=False):
        g = g.copy()
        if "skip_exclude_from_topn" in g.columns:
            g = g[g["skip_exclude_from_topn"] != True].copy()
        cap_d = min(cap, len(g))

        if sel and "Rank" in g.columns:
            cand = g[pd.to_numeric(g["Rank"], errors="coerce").fillna(-1).astype(int).isin(sel)]
        else:
            cand = g.iloc[0:0]

        if len(cand) > cap_d:
            cand = cand.sort_values(sort_by, ascending=sort_asc).head(cap_d)
        elif len(cand) < cap_d and fill_to_cap:
            needed = cap_d - len(cand)
            rest = g[~g.index.isin(cand.index)].copy()
            rest = rest.sort_values(sort_by, ascending=sort_asc).head(needed)
            cand = pd.concat([cand, rest], ignore_index=False)

        if len(cand) > 0:
            df.loc[cand.index, "Selected50"] = 1

    return df


def _wf_augment_event_date_with_play50(per_event: pd.DataFrame,
                                      per_date: pd.DataFrame,
                                      per_stream_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add Selected50 + CapturedMember fields to per_event, and daily selection/capture stats to per_date."""
    pe = per_event.copy() if per_event is not None else pd.DataFrame()
    pdx = per_date.copy() if per_date is not None else pd.DataFrame()

    if pe.empty or per_stream_selected is None or per_stream_selected.empty:
        return pe, pdx

    # Merge Selected50 and stream-level play members back onto per_event
    key_cols = ["PlayDate", "StreamKey"] if "StreamKey" in pe.columns else ["PlayDate", "WinningStreamKey"]
    stream_key_col = "StreamKey" if "StreamKey" in per_stream_selected.columns else "WinningStreamKey"

    ps = per_stream_selected.copy()
    if "StreamKey" not in ps.columns and "WinningStreamKey" in ps.columns:
        ps = ps.rename(columns={"WinningStreamKey": "StreamKey"})
    if "StreamKey" in ps.columns:
        merge_candidates = [
            "Selected50", "PredictedMember", "Top1Base", "Top1", "Top2Base", "Top2", "Top3Base", "Top3",
            "Top1Score", "BaseGap(#1-#2)", "TieFired", "DeadTie", "ForcedPick", "ElimFired",
            "DownrankFired", "RescueFired", "RulepackTop3Fired", "SeedDate", "Seed",
            "CadenceRolling025_30", "CadenceDrawsSinceLast025", "SinceLast025(draws)", "Rolling025_30",
            "SeedSum", "SeedSpread", "SeedAbsDiff", "WorstPair025",
            "PlayPlan", "PlayMembers", "PlayCount", "PlayMember", "FlipRec"
        ]
        keep_cols = ["PlayDate", "StreamKey"] + [c for c in merge_candidates if c in ps.columns]
        ps_small = ps[keep_cols].copy()
    else:
        ps_small = ps[["PlayDate", stream_key_col, "Selected50"]].copy()

    # Normalize per_event stream key column
    if "WinningStreamKey" in pe.columns and "StreamKey" not in pe.columns:
        pe = pe.rename(columns={"WinningStreamKey": "StreamKey"})
    if "StreamKey" not in pe.columns:
        return pe, pdx

    pe = pe.merge(ps_small, how="left", on=["PlayDate", "StreamKey"], suffixes=("", "_stream"))

    for c in [col[:-7] for col in pe.columns if col.endswith("_stream")]:
        sc = f"{c}_stream"
        if c not in pe.columns:
            pe[c] = pe[sc]
        else:
            try:
                pe[c] = pe[c].where(~pe[c].isna(), pe[sc])
            except Exception:
                pass
        pe.drop(columns=[sc], inplace=True, errors="ignore")

    pe["Selected50"] = pe["Selected50"].fillna(0).astype(int)

    def _member_set(row):
        s = row.get("PlayMembers", None)
        if pd.notna(s):
            parts = [normalize_member_id(p.strip()) for p in str(s).split("+") if str(p).strip()]
            parts = [p for p in parts if p]
            if parts:
                return set(parts)
        # fallback to PlayMember (Top1)
        pm = normalize_member_id(row.get("PlayMember", None))
        if pm:
            return {pm}
        t1 = normalize_member_id(row.get("Top1", None))
        if t1:
            return {t1}
        return set()

    pe["_PlaySet"] = pe.apply(_member_set, axis=1)
    pe["CapturedMember"] = 0
    if "WinningMember" in pe.columns:
        pe["CapturedMember"] = pe.apply(
            lambda r: 1
            if (
                int(r.get("InUniverse", 0)) == 1
                and int(r.get("Selected50", 0)) == 1
                and normalize_member_id(r.get("WinningMember")) in r.get("_PlaySet", set())
            )
            else 0,
            axis=1,
        )

    pe["CapturedBy"] = ""
    if "WinningMember" in pe.columns:
        def _captured_by(r):
            if int(r.get("CapturedMember", 0)) != 1:
                return ""
            winner = normalize_member_id(r.get("WinningMember"))
            if winner == normalize_member_id(r.get("Top1")):
                return "TOP1"
            if winner == normalize_member_id(r.get("Top2")):
                return "TOP2"
            return "PLAYSET"
        pe["CapturedBy"] = pe.apply(_captured_by, axis=1)

    pe.drop(columns=["_PlaySet"], inplace=True, errors="ignore")

    # Daily selection stats
    if not pdx.empty and "PlayDate" in pdx.columns and "PlayDate" in ps.columns:
        sel_counts = ps.groupby("PlayDate")["Selected50"].sum().reset_index(name="SelectedStreams")
        pdx = pdx.merge(sel_counts, how="left", on="PlayDate")
        pdx["SelectedStreams"] = pdx["SelectedStreams"].fillna(0).astype(int)

        # Daily capture stats (for hit-events)
        if "HitEvents" in pdx.columns:
            day_hits = pe.groupby("PlayDate").agg(
                HitEvents=("PlayDate", "size"),
                HitStreams_Selected50=("Selected50", "sum"),
                HitMembers_Captured=("CapturedMember", "sum"),
            ).reset_index()
            pdx = pdx.drop(columns=["HitEvents"], errors="ignore").merge(day_hits, how="left", on="PlayDate")
            pdx["HitEvents"] = pdx["HitEvents"].fillna(0).astype(int)
            pdx["HitStreams_Selected50"] = pdx["HitStreams_Selected50"].fillna(0).astype(int)
            pdx["HitMembers_Captured"] = pdx["HitMembers_Captured"].fillna(0).astype(int)

    return pe, pdx

st.set_page_config(page_title="Core 025 — Live Ranked Streams + Lab Backtest", layout="wide")

# ----------------------------
# Robust parsing for Lottery Post "Results"
# ----------------------------
_DIG4_RE = re.compile(r"(\d)[^\d]+(\d)[^\d]+(\d)[^\d]+(\d)")

def extract_4digits(val) -> Optional[str]:
    """Extract a clean 4-digit result string (preserving leading zeros).

    Strict on purpose: we do NOT "borrow" digits from Fireball/Wildball/Bonus/etc.
    If we can't confidently parse a 4-digit result, return None and exclude the row.
    """
    if val is None or pd.isna(val):
        return None

    s = str(val).strip()
    if not s:
        return None

    # Strip anything after comma and after common add-on keywords to avoid capturing extra digits.
    s0 = s.split(",", 1)[0].strip()
    s0 = re.split(r"\b(?:wild\s*ball|fireball|bonus|multiplier)\b", s0, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    m = _DIG4_RE.search(s0)
    if m:
        out = "".join(m.groups())
        return out if len(out) == 4 else None

    # Accept a standalone 4-digit chunk if present.
    m2 = re.search(r"\b(\d{4})\b", s0)
    if m2:
        return m2.group(1)

    return None

def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    # Try to map variant headers into canonical names
    df = _map_columns(df)

    # Required columns (Lottery Post export style)
    needed = ["Draw Date", "State", "Game", "Results"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        # Helpful debug: show available columns
        raise ValueError(
            f"History file missing columns: {missing}. Need: {needed}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out["Draw Date"] = pd.to_datetime(out["Draw Date"], errors="coerce")
    out["result_raw"] = out["Results"].astype(str)
    out["result"] = out["result_raw"].apply(extract_4digits)

    # diagnostics
    parse_fail = int(out["result"].isna().sum())
    out.attrs["parse_fail_count"] = parse_fail
    if parse_fail:
        bad = out.loc[out["result"].isna(), ["Draw Date","State","Game","Results"]].head(10)
        out.attrs["parse_fail_examples"] = bad.to_dict(orient="records")

    # drop bad rows
    out = out.dropna(subset=["Draw Date", "State", "Game", "result"]).copy()

    out["stream"] = out["State"].astype(str).str.strip() + " | " + out["Game"].astype(str).str.strip()
    out = out.sort_values(["stream", "Draw Date"]).reset_index(drop=True)

    # within-stream order index
    out["stream_idx"] = out.groupby("stream").cumcount()
    return out

# ----------------------------
# Core 025 helpers

# ----------------------------
TARGETS = {"0025": "0025", "0225": "0225", "0255": "0255"}
TARGET_SET = set(TARGETS.keys())

def as_member(result4: str) -> Optional[str]:
    """Map a 4-digit result to its 025-family member if it is a permutation of {0,0,2,5}, {0,2,2,5}, {0,2,5,5}."""
    # Count digits
    if not result4 or len(result4) != 4 or not result4.isdigit():
        return None
    from collections import Counter
    c = Counter(result4)
    # canonical patterns
    patt = tuple(sorted(c.items()))
    # 0025 => 0x2,2x1,5x1
    if c.get("0",0)==2 and c.get("2",0)==1 and c.get("5",0)==1:
        return "0025"
    # 0225 => 0x1,2x2,5x1
    if c.get("0",0)==1 and c.get("2",0)==2 and c.get("5",0)==1:
        return "0225"
    # 0255 => 0x1,2x1,5x2
    if c.get("0",0)==1 and c.get("2",0)==1 and c.get("5",0)==2:
        return "0255"
    return None

def seed_digits(seed4: str) -> List[int]:
    return [int(ch) for ch in seed4]

def seed_sum(seed4: str) -> int:
    return sum(seed_digits(seed4))

def seed_spread(seed4: str) -> int:
    d = seed_digits(seed4)
    return max(d) - min(d)

def seed_absdiff(seed4: str) -> int:
    d = seed_digits(seed4)
    # definition used in your weight table: abs( (d0+d1) - (d2+d3) )
    return abs((d[0]+d[1]) - (d[2]+d[3]))

WORST_PAIRS_025 = {("3","9"), ("5","5"), ("2","6"), ("2","9"), ("7","9")}
def seed_has_worstpair_025(seed4: str) -> bool:
    # unordered digit pairs from 4 digits (6 combos)
    digs = list(seed4)
    pairs = []
    for i in range(4):
        for j in range(i+1,4):
            a,b = digs[i], digs[j]
            pairs.append(tuple(sorted((a,b))))
    return any(p in WORST_PAIRS_025 for p in pairs)

def seed_sum_lastdigit(seed4: str) -> int:
    return seed_sum(seed4) % 10


_MIRROR_PAIRS = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

def seed_even_cnt(seed4: str) -> int:
    return sum(1 for d in seed_digits(seed4) if d % 2 == 0)

def seed_is_unique(seed4: str) -> int:
    ds = seed_digits(seed4)
    return 1 if len(set(ds)) == 4 else 0

def seed_has_pair(seed4: str) -> int:
    return 0 if seed_is_unique(seed4) == 1 else 1

def seed_consec_links(seed4: str) -> int:
    # Count adjacent consecutive links in the UNIQUE sorted digit set (e.g., {1,2,4,7} has 1 link (1-2))
    s = sorted(set(seed_digits(seed4)))
    return sum(1 for a, b in zip(s, s[1:]) if b - a == 1)

def seed_mirrorpair_cnt(seed4: str) -> int:
    s = set(seed_digits(seed4))
    return sum(1 for a, b in _MIRROR_PAIRS if a in s and b in s)

def seed_adj_absdiff_sum(seed4: str) -> int:
    ds = seed_digits(seed4)
    return sum(abs(ds[i] - ds[i + 1]) for i in range(len(ds) - 1))

def seed_pairwise_absdiff_sum(seed4: str) -> int:
    ds = seed_digits(seed4)
    tot = 0
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            tot += abs(ds[i] - ds[j])
    return tot

def seed_digit_counts(seed4: str) -> Dict[int, int]:
    ds = seed_digits(seed4)
    return {d: ds.count(d) for d in range(10)}
# Feature compute

def compute_features(seed4: str) -> Dict[str, object]:
    # NOTE: Keep legacy keys (seed_has_9, seed_spread, etc.) so older rulepacks keep working.
    counts = seed_digit_counts(seed4)
    ds = seed_digits(seed4)
    dset = set(ds)

    feats: Dict[str, object] = {
        # legacy keys used by embedded weights/tie-pack
        "seed_sum": seed_sum(seed4),
        "seed_spread": seed_spread(seed4),
        "seed_has_0": 1 if 0 in dset else 0,
        "seed_has_2": 1 if 2 in dset else 0,
        "seed_has_5": 1 if 5 in dset else 0,
        "seed_has_9": 1 if 9 in dset else 0,
        "seed_has_worstpair_025": seed_has_worstpair_025(seed4),
        "seed_sum_lastdigit": seed_sum_lastdigit(seed4),
    }

    # expanded digit presence / counts
    for d in range(10):
        feats[f"seed_has{d}"] = 1 if d in dset else 0
        feats[f"seed_cnt{d}"] = int(counts[d])

    # expanded structural traits
    feats["seed_unique"] = seed_is_unique(seed4)
    feats["seed_has_pair"] = seed_has_pair(seed4)
    feats["seed_even_cnt"] = seed_even_cnt(seed4)
    feats["seed_consec_links"] = seed_consec_links(seed4)
    feats["seed_mirrorpair_cnt"] = seed_mirrorpair_cnt(seed4)
    feats["seed_adj_absdiff_sum"] = seed_adj_absdiff_sum(seed4)
    feats["seed_pairwise_absdiff_sum"] = seed_pairwise_absdiff_sum(seed4)

    # used by embedded rarity-aware weights
    feats["seed_absdiff"] = seed_absdiff(seed4)

    # aliases used by some mined packs
    feats["mirrorpair_cnt"] = feats["seed_mirrorpair_cnt"]
    feats["seed_has9"] = feats["seed_has9"]
    feats["seed_has2"] = feats["seed_has2"]
    feats["seed_has5"] = feats["seed_has5"]
    feats["seed_has0"] = feats["seed_has0"]

    
    # Unordered pair flags (00..99 where a<=b). These are used by mined pockets like pair_01, pair_55, etc.
    for a in range(10):
        for b in range(a, 10):
            key = f"pair_{a}{b}"
            if a == b:
                feats[key] = 1 if counts.get(a, 0) >= 2 else 0
            else:
                feats[key] = 1 if (counts.get(a, 0) >= 1 and counts.get(b, 0) >= 1) else 0
    feats["no_pair"] = 1 if feats.get("seed_has_pair", 0) == 0 else 0
    return feats


def _build_transitions_from_history(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build same-stream transitions from the raw history table.
    Expected columns: Draw Date, stream, Result.
    Output rows correspond to TARGET draws; seed_result is prior draw in same stream.
    """
    df = hist_df.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    if "stream" not in df.columns:
        raise ValueError("History table missing 'stream' column.")
    if "Draw Date" not in df.columns:
        raise ValueError("History table missing 'Draw Date' column.")
    if "Result" not in df.columns:
        raise ValueError("History table missing 'Result' column.")

    df["_dt"] = pd.to_datetime(df["Draw Date"], errors="coerce")
    df = df.dropna(subset=["_dt", "stream", "Result"]).copy()
    df["Result"] = df["Result"].astype(str).str.strip()

    df = df.sort_values(["stream", "_dt"]).reset_index(drop=True)
    df["seed_result"] = df.groupby("stream")["Result"].shift(1)
    df = df.dropna(subset=["seed_result"]).copy()

    df["member"] = df["Result"].apply(as_member)
    df["is_025"] = df["member"].isin(["0025", "0225", "0255"]).astype(int)

    return df[["stream", "Draw Date", "_dt", "seed_result", "Result", "member", "is_025"]].reset_index(drop=True)


def _add_stream_history_features(dfm: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-respecting (no-lookahead) cadence-like features for each transition row.
    Computed from prior TARGET outcomes within the same stream (shifted by 1),
    which corresponds to information available up through the seed time.
    """
    dfm = dfm.sort_values(["stream", "_dt"]).copy()
    g = dfm.groupby("stream", sort=False)

    prior_hits = g["is_025"].apply(lambda s: s.shift(1).fillna(0).astype(int)).reset_index(level=0, drop=True)
    prior_n = g.cumcount()

    # Smoothed base rate
    alpha = 1.0
    dfm["stream_base_rate"] = (prior_hits.groupby(dfm["stream"]).cumsum() + alpha) / (prior_n + 2 * alpha)

    # Rolling windows on prior hits
    dfm["rolling_025_30"] = g["is_025"].apply(lambda s: s.shift(1).rolling(30, min_periods=1).sum()).reset_index(level=0, drop=True)
    dfm["rolling_025_60"] = g["is_025"].apply(lambda s: s.shift(1).rolling(60, min_periods=1).sum()).reset_index(level=0, drop=True)

    # Draws since last hit (prior hits)
    def _draws_since_last_hit(s: pd.Series) -> pd.Series:
        prior = s.shift(1).fillna(0).astype(int)
        idx = np.arange(len(prior))
        last_hit = np.where(prior.values == 1, idx, np.nan)
        last_hit = pd.Series(last_hit).ffill()
        out = idx - last_hit
        out = pd.Series(out).fillna(idx + 1)
        return out

    dfm["draws_since_last_025"] = g["is_025"].apply(_draws_since_last_hit).reset_index(level=0, drop=True)
    dfm["never_hit_before"] = (prior_hits.groupby(dfm["stream"]).cumsum() == 0).astype(int)

    for c in ["stream_base_rate", "rolling_025_30", "rolling_025_60", "draws_since_last_025"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce").fillna(0.0)

    return dfm


def _make_streamscore_feature_dict(row: dict) -> dict:
    seed4 = str(row["seed_result"]).zfill(4)
    feats = compute_features(seed4, target_dow=row.get("target_dow", None), time_class=row.get("time_class", None))

    d = {
        "stream": row["stream"],
        # seed primitives
        "seed_sum": feats.get("seed_sum", 0),
        "seed_spread": feats.get("seed_spread", 0),
        "seed_absdiff_sum": feats.get("seed_absdiff_sum", 0),
        "seed_pairwise_absdiff_sum": feats.get("seed_pairwise_absdiff_sum", 0),
        "seed_even_cnt": feats.get("seed_even_cnt", 0),
        "seed_consec_links": feats.get("seed_consec_links", 0),
        "seed_unique": feats.get("seed_unique", 0),
        "seed_mirrorpair_cnt": feats.get("seed_mirrorpair_cnt", 0),
        "seed_has_worstpair_025": feats.get("seed_has_worstpair_025", 0),
        "seed_has9": feats.get("seed_has9", 0),
        "seed_has0": feats.get("seed_has0", 0),
        "seed_has2": feats.get("seed_has2", 0),
        "seed_has5": feats.get("seed_has5", 0),
        "seed_cnt0": feats.get("seed_cnt0", 0),
        "seed_cnt2": feats.get("seed_cnt2", 0),
        "seed_cnt5": feats.get("seed_cnt5", 0),
        "seed_cnt9": feats.get("seed_cnt9", 0),
        # stream history (no-leak)
        "draws_since_last_025": float(row.get("draws_since_last_025", 0.0)),
        "rolling_025_30": float(row.get("rolling_025_30", 0.0)),
        "rolling_025_60": float(row.get("rolling_025_60", 0.0)),
        "stream_base_rate": float(row.get("stream_base_rate", 0.0)),
        "never_hit_before": int(row.get("never_hit_before", 0)),
    }
    return d


def _time_split_mask(dfm: pd.DataFrame, frac_train: float = 0.8) -> tuple[pd.Series, pd.Series]:
    if dfm.empty:
        return pd.Series([], dtype=bool), pd.Series([], dtype=bool)
    dfm = dfm.sort_values("_dt").copy()
    cut = dfm["_dt"].quantile(frac_train)
    train_mask = dfm["_dt"] <= cut
    test_mask = dfm["_dt"] > cut
    return train_mask, test_mask


def _calibrate_streamscore_logreg(
    trans_df: pd.DataFrame,
    feature_dicts: list,
    y: np.ndarray,
    frac_train: float = 0.8,
    C: float = 1.0,
    max_iter: int = 400,
    lr: float = 0.1,
) -> dict:
    """Time-respecting StreamScore calibration (no sklearn).

    We fit an L2-regularized logistic regression using NumPy on TRAIN only,
    then evaluate on TEST. Columns (including stream one-hots) are derived
    from the full feature dict list (safe; no outcome leakage), but scaling
    and fitting are done on train only.
    """
    if len(trans_df) != len(feature_dicts):
        raise ValueError("feature_dicts length must match trans_df length")

    # time split (seed_time is sortable string or datetime)
    times = trans_df["seed_time"].astype(str).values
    order = np.argsort(times, kind="mergesort")
    n = len(order)
    cut = int(max(1, min(n - 1, round(n * float(frac_train)))))
    train_idx = order[:cut]
    test_idx = order[cut:]

    y = np.asarray(y, dtype=np.int32)

    # Build full design matrix (safe feature space)
    X_full, feat_cols = _dicts_to_design_matrix(feature_dicts, feature_columns=None)

    # Standardize using TRAIN only
    mean, std = _standardize_fit(X_full[train_idx])
    X_train = _standardize_apply(X_full[train_idx], mean, std)
    X_test = _standardize_apply(X_full[test_idx], mean, std)

    y_train = y[train_idx]
    y_test = y[test_idx]

    # balanced weights
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = (n_neg / max(1, n_pos)) if (n_pos > 0) else 1.0
    sw_train = np.where(y_train == 1, pos_weight, 1.0).astype(np.float64)

    # sklearn's C is inverse regularization strength; map to l2.
    l2 = 1.0 / max(1e-9, float(C))

    w, b = _fit_logreg_l2(
        X_train,
        y_train.astype(np.float64),
        sample_weight=sw_train,
        l2=l2,
        lr=float(lr),
        max_iter=int(max_iter),
    )

    p_train = _sigmoid(X_train.dot(w) + b)
    p_test = _sigmoid(X_test.dot(w) + b)

    out = {
        "model_type": "numpy_logreg_l2",
        "frac_train": float(frac_train),
        "C": float(C),
        "l2": float(l2),
        "lr": float(lr),
        "max_iter": int(max_iter),
        "feature_columns": feat_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": w.tolist(),
        "intercept": float(b),
        "train_logloss": _logloss(y_train, p_train, sw_train),
        "test_logloss": _logloss(y_test, p_test, None),
        "train_auc": _auc_score(y_train, p_train),
        "test_auc": _auc_score(y_test, p_test),
        "train_ap": _average_precision(y_train, p_train),
        "test_ap": _average_precision(y_test, p_test),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "train_pos": int(n_pos),
        "train_neg": int(n_neg),
        "pos_weight": float(pos_weight),
    }

    # weights table for export/audit
    w_df = pd.DataFrame({"feature": feat_cols, "weight": w})
    w_df = w_df.sort_values("weight", ascending=False).reset_index(drop=True)
    out["weights_df"] = w_df

    return out

def _default_streamskip_candidate_pockets() -> dict:
    """
    Deterministic, broad candidate pocket set for OOS mining.
    These are intentionally simple boolean/threshold predicates on seed-only features.
    """
    pockets = {}

    # Digit presence
    for d in range(10):
        pockets[f"seed_has{d} == 1"] = (lambda f, d=d: bool(f.get(f"seed_has{d}", 0) == 1))

    # Key structural primitives
    pockets["seed_unique == 1"] = lambda f: bool(f.get("seed_unique", 0) == 1)
    pockets["seed_has_worstpair_025 == 1"] = lambda f: bool(f.get("seed_has_worstpair_025", 0) == 1)

    # Even count buckets
    for k in [0, 1, 2, 3, 4]:
        pockets[f"seed_even_cnt == {k}"] = (lambda f, k=k: bool(f.get("seed_even_cnt", -1) == k))

    # Consecutive links
    for k in [0, 1, 2, 3]:
        pockets[f"seed_consec_links == {k}"] = (lambda f, k=k: bool(f.get("seed_consec_links", -1) == k))

    # Mirror pair count
    for k in [0, 1, 2]:
        pockets[f"seed_mirrorpair_cnt == {k}"] = (lambda f, k=k: bool(f.get("seed_mirrorpair_cnt", -1) == k))

    # Spread thresholds
    for thr in [1, 2, 3, 4, 5, 6]:
        pockets[f"seed_spread <= {thr}"] = (lambda f, thr=thr: bool(f.get("seed_spread", 0) <= thr))
    for thr in [7, 8, 9]:
        pockets[f"seed_spread >= {thr}"] = (lambda f, thr=thr: bool(f.get("seed_spread", 0) >= thr))

    # Sum thresholds
    for thr in [4, 6, 8, 10, 12, 14, 16]:
        pockets[f"seed_sum <= {thr}"] = (lambda f, thr=thr: bool(f.get("seed_sum", 0) <= thr))
    for thr in [18, 20, 22, 24, 26, 28, 30]:
        pockets[f"seed_sum >= {thr}"] = (lambda f, thr=thr: bool(f.get("seed_sum", 0) >= thr))

    return pockets


def _mine_streamskip_pockets_oos(
    dfm: pd.DataFrame,
    pockets: dict,
    frac_train: float = 0.8,
    min_support_global: int = 3300,
    min_support_stream: int = 200,
) -> pd.DataFrame:
    dfm = _add_stream_history_features(dfm)
    train_mask, test_mask = _time_split_mask(dfm, frac_train=frac_train)

    dfm = dfm.copy()
    dfm["seed4"] = dfm["seed_result"].astype(str).str.zfill(4)
    dfm["feats"] = dfm["seed4"].apply(lambda s: compute_features(s))

    rows = []
    for pname, fn in pockets.items():
        fired = dfm["feats"].apply(fn).astype(bool)

        Ntr = int((fired & train_mask).sum())
        Htr = int(((fired & train_mask) & (dfm["is_025"] == 1)).sum())
        Nte = int((fired & test_mask).sum())
        Hte = int(((fired & test_mask) & (dfm["is_025"] == 1)).sum())

        keep_global = (Htr == 0 and Hte == 0 and Ntr >= min_support_global and Nte >= max(1, int(min_support_global * 0.15)))
        rows.append({
            "scope": "GLOBAL",
            "stream": "(all)",
            "pocket": pname,
            "N_train": Ntr, "H_train": Htr,
            "N_test": Nte, "H_test": Hte,
            "promote_global_ready": bool(keep_global),
            "suggested_tier": ("LOCKED" if keep_global else "GUARDED"),
        })

        for stream, sdf in dfm.groupby("stream", sort=False):
            f = fired.loc[sdf.index]
            tr = train_mask.loc[sdf.index]
            te = test_mask.loc[sdf.index]

            Ntr_s = int((f & tr).sum())
            Htr_s = int(((f & tr) & (sdf["is_025"] == 1)).sum())
            Nte_s = int((f & te).sum())
            Hte_s = int(((f & te) & (sdf["is_025"] == 1)).sum())

            keep_s = (Htr_s == 0 and Hte_s == 0 and Ntr_s >= min_support_stream and Nte_s >= max(1, int(min_support_stream * 0.15)))
            if Ntr_s >= min_support_stream or Nte_s >= min_support_stream:
                rows.append({
                    "scope": "STREAM",
                    "stream": stream,
                    "pocket": pname,
                    "N_train": Ntr_s, "H_train": Htr_s,
                    "N_test": Nte_s, "H_test": Hte_s,
                    "promote_stream_candidate": bool(keep_s),
                    "suggested_tier": ("LOCKED" if keep_s else "GUARDED"),
                })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["candidate"] = res.get("promote_global_ready", False).fillna(False) | res.get("promote_stream_candidate", False).fillna(False)
    res = res.sort_values(["scope", "candidate", "N_train"], ascending=[True, False, False]).reset_index(drop=True)
    return res



def op_match(actual, op: str, value) -> bool:
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    if op == "<=":
        return actual <= value
    if op == "<":
        return actual < value
    if op == ">=":
        return actual >= value
    if op == ">":
        return actual > value
    # fallback string compare
    return str(actual) == str(value)

def load_rules(weights_csv: str, tie_csv: str, max_rules: int, min_support: int,
               weights_file_up=None, tie_up=None) -> Tuple[List['Rule'], List['Rule'], Dict]:
    """
    Load rule-weights + tie-pack, with clear provenance so audits can confirm what file was used.

    Priority order (both files):
      1) CSV at provided path
      2) Uploaded file in the UI
      3) Embedded fallback bundled in the app
    """
    import hashlib

    def _sha256(b: bytes) -> str:
        try:
            return hashlib.sha256(b).hexdigest()
        except Exception:
            return ""

    def _read_csv_bytes_from_path(p: str) -> Optional[bytes]:
        if not p:
            return None
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read()
        except Exception:
            return None
        return None

    def _read_csv_bytes_from_upload(up) -> Optional[bytes]:
        if up is None:
            return None
        # Allow passing raw bytes (e.g., cached in st.session_state)
        if isinstance(up, (bytes, bytearray)):
            return bytes(up)
        try:
            # Streamlit UploadedFile supports getvalue()
            return up.getvalue()
        except Exception:
            try:
                # Fallback: read()
                return up.read()
            except Exception:
                return None

    def _read_csv_bytes_from_embedded(b64_gz: str) -> bytes:
        return gzip.decompress(base64.b64decode(b64_gz.encode("ascii")))

    meta: Dict = {
        "weights": {"source": None, "rows": 0, "sha256": "", "filename": ""},
        "tie_pack": {"source": None, "rows": 0, "sha256": "", "filename": ""},
    }

    # ---------------- Weights ----------------
    w_bytes = _read_csv_bytes_from_path(weights_csv)
    w_source = "path" if w_bytes is not None else None

    if w_bytes is None:
        w_bytes = _read_csv_bytes_from_upload(weights_file_up)
        w_source = "upload" if w_bytes is not None else None

    if w_bytes is None:
        w_bytes = _read_csv_bytes_from_embedded(_EMBED_WEIGHTS_B64_GZ)
        w_source = "embedded"

    meta["weights"]["source"] = w_source

    # Best-effort filename for audits/UI
    if w_source == "path":
        meta["weights"]["filename"] = str(weights_csv)
    elif w_source == "upload":
        meta["weights"]["filename"] = getattr(weights_file_up, "name", "") or str(st.session_state.get("weights_override_name", "uploaded_weights.csv"))
    else:
        meta["weights"]["filename"] = "embedded:weights"
    meta["weights"]["sha256"] = _sha256(w_bytes)

    w = pd.read_csv(io.BytesIO(w_bytes)).copy()
    if "support" in w.columns:
        w = w[w["support"].fillna(0).astype(int) >= int(min_support)].copy()

    if "priority" in w.columns:
        w = w.sort_values(["priority"], ascending=True)

    base_rules: List['Rule'] = [Rule.from_row(r) for _, r in w.head(int(max_rules)).iterrows()]
    meta["weights"]["rows"] = int(len(w))

    # ---------------- Tie-pack ----------------
    t_bytes = _read_csv_bytes_from_path(tie_csv)
    t_source = "path" if t_bytes is not None else None

    if t_bytes is None:
        t_bytes = _read_csv_bytes_from_upload(tie_up)
        t_source = "upload" if t_bytes is not None else None

    if t_bytes is None:
        t_bytes = _read_csv_bytes_from_embedded(_EMBED_TIEPACK_B64_GZ)
        t_source = "embedded"

    meta["tie_pack"]["source"] = t_source

    if t_source == "path":
        meta["tie_pack"]["filename"] = str(tie_csv)
    elif t_source == "upload":
        meta["tie_pack"]["filename"] = getattr(tie_up, "name", "") or str(st.session_state.get("tie_override_name", "uploaded_tiepack.csv"))
    else:
        meta["tie_pack"]["filename"] = "embedded:tie_pack"
    meta["tie_pack"]["sha256"] = _sha256(t_bytes)

    t = pd.read_csv(io.BytesIO(t_bytes)).copy()
    if "priority" in t.columns:
        t = t.sort_values(["priority"], ascending=True)

    tie_rules: List['Rule'] = [Rule.from_row(r) for _, r in t.iterrows()]
    meta["tie_pack"]["rows"] = int(len(t))

    return base_rules, tie_rules, meta
def score_seed(seed4: str, base_rules: List['Rule'], tie_rules: List['Rule']) -> Dict[str, object]:
    feats = compute_features(seed4)

    # base scoring
    scores = {m: 0.0 for m in TARGET_SET}
    fired = []
    for rule in base_rules:
        if rule.feature not in feats:
            continue
        if op_match(feats[rule.feature], rule.op, rule.value):
            if rule.pick in scores:

                scores[rule.pick] += rule.weight

            else:

                # Unknown pick (likely a legacy/uncanonical value). Skip safely.

                continue
            fired.append((rule.source, rule.feature, rule.op, rule.value, rule.pick, rule.weight))

    # rank
    ordered = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top1, top1s = ordered[0]
    top2, top2s = ordered[1]
    top3, top3s = ordered[2]
    base_gap = top1s - top2s

    tie_fired = 0
    tie_break_fired = []

    # if tie or tiny gap, apply tie-pack to try to separate
    if base_gap == 0:
        tie_scores = {m: 0.0 for m in TARGET_SET}
        for rule in tie_rules:
            if rule.feature not in feats:
                continue
            if op_match(feats[rule.feature], rule.op, rule.value):
                if rule.pick in tie_scores:

                    tie_scores[rule.pick] += rule.weight

                else:

                    # Unknown pick (likely a legacy/uncanonical value). Skip safely.

                    continue
                tie_break_fired.append((rule.source, rule.feature, rule.op, rule.value, rule.pick, rule.weight))
        # apply tie_scores as secondary key
        ordered2 = sorted(scores.items(), key=lambda kv: (kv[1], tie_scores[kv[0]], kv[0]), reverse=True)
        tie_fired = 1 if any(v>0 for v in tie_scores.values()) else 0
        top1, top1s = ordered2[0]
        top2, top2s = ordered2[1]
        top3, top3s = ordered2[2]

        # if still tied even after tie_scores, that's a dead tie
        dead_tie = 1 if (top1s == top2s and tie_scores[top1] == tie_scores[top2]) else 0
    else:
        dead_tie = 0

    # coverage rules CR1/CR2
    forced = None
    if dead_tie == 1:
        ssum = feats["seed_sum"]
        forced = "0225" if ssum <= 11 else "0025"
        top1 = forced

        # IMPORTANT: ensure Top2 is always distinct from Top1.
        # (Previously, forcing Top1 could make Top1==Top2, which collapses "play 2 members" into 1.)
        try:
            ordered_list = [m for (m, _) in ordered2]  # base_gap==0 branch
        except Exception:
            ordered_list = [m for (m, _) in ordered]

        rest = [m for m in ordered_list if m != top1]
        if len(rest) >= 2:
            top2, top3 = rest[0], rest[1]
        else:
            rest2 = [m for m in TARGET_SET if m != top1]
            top2, top3 = rest2[0], rest2[1]

        # update gap for display/diagnostics
        base_gap = float(scores.get(top1, 0.0) - scores.get(top2, 0.0))

    return {
        "seed": seed4,
        "features": feats,
        "scores": scores,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "base_gap": base_gap,
        "tie_fired": tie_fired,
        "dead_tie": dead_tie,
        "forced_pick": forced,
        "base_rules_fired": fired,
        "tie_rules_fired": tie_break_fired,
    }

# ----------------------------
# Stream cadence features
# ----------------------------
def compute_cadence_metrics(hist: pd.DataFrame) -> pd.DataFrame:
    """
    For each stream and each row, compute:
      - is_025: whether this result is 025-family
      - draws_since_last_025: within stream, draws since last 025 (excluding current row)
      - rolling_025_30: count of 025 hits in prior 30 draws (excluding current)
    """
    df = hist.copy()
    df["member"] = df["result"].apply(as_member)
    df["is_025"] = df["member"].notna().astype(int)

    # draws since last 025
    def _since_last(s: pd.Series) -> pd.Series:
        last = -1
        out = []
        for i, v in enumerate(s.values):
            out.append(i - last if last != -1 else 9999)
            if v == 1:
                last = i
        return pd.Series(out, index=s.index)

    df["draws_since_last_025"] = df.groupby("stream")["is_025"].apply(_since_last).reset_index(level=0, drop=True)

    # rolling prior-30 025 count
    df["rolling_025_30"] = (
        df.groupby("stream")["is_025"]
          .apply(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
          .reset_index(level=0, drop=True)
          .fillna(0)
          .astype(int)
    )
    return df

def stream_score_row(seed_score: Dict[str, object], cadence_row: Dict[str, object],
                     w_match=1.0, w_gap=0.6, w_roll=0.25, w_since=0.15,
                     since_mode: str = "due",
                     cooldown_k=2, cooldown_penalty=1.0) -> float:
    """
    StreamScore for ranking all streams (full list):
      - match_strength: total base points (sum of top member scores)
      - gap: top1 - top2 (after tie-pack/coverage)
      - rolling_025_30: prior 30 hit count
      - draws_since_last_025: larger means "more due" (weak effect)
      - cooldown: if last 025 was within last k draws, downrate.
    """
    scores = seed_score["scores"]
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1, top1s = ordered[0]
    top2, top2s = ordered[1]
    match_strength = float(top1s)

    gap = float(top1s - top2s)

    roll = float(cadence_row.get("rolling_025_30", 0))
    since = float(cadence_row.get("draws_since_last_025", 9999))
    since_norm = min(since, 50.0) / 50.0  # 0..1 where 1 is most 'due'

    if since_mode == "recent":

        since_component = 1.0 - since_norm

    elif since_mode == "blend":

        # U-shape: reward both extremes (very recent OR very due)

        since_component = abs(since_norm - 0.5) * 2.0

    else:

        # legacy

        since_component = since_norm

    score = (w_match * match_strength) + (w_gap * gap) + (w_roll * roll) + (w_since * since_component)

    # cooldown
    if since <= cooldown_k:
        score -= cooldown_penalty

    # tie risk downrate
    if seed_score["dead_tie"] == 1:
        score -= 0.5
    elif seed_score["tie_fired"] == 1 and seed_score["base_gap"] == 0:
        score -= 0.2

    return score

# ----------------------------
# UI
# ----------------------------
st.title("Core 025 — Full Ranked Stream Playlist (Live) + Lab Backtest")
st.caption(f"App version: {APP_VERSION_STR}")
with st.sidebar:
    st.header("Inputs")
    st.caption(f"App version: {APP_VERSION_STR}")

    st.subheader("Quick defaults")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load BEST (Budget 50)", use_container_width=True):
            apply_best_defaults("budget")
    with c2:
        if st.button("Load BEST (Coverage 80)", use_container_width=True):
            apply_best_defaults("coverage")
    st.caption("Tip: Streamlit keeps settings during reruns. Use these buttons if you changed a bunch of sliders/toggles and want to snap back instantly.")

    hist_file = st.file_uploader("Upload Lottery Post export (CSV or TXT)", type=["csv","txt"])
    hist_update_file = st.file_uploader("Optional incremental update (last 24h+) — CSV or TXT", type=["csv","txt"])
    
    st.caption("Required columns (CSV or TXT): Draw Date, State, Game, Results")

    st.divider()
    st.header("Walk-forward (no-lookahead) dates")
    st.caption("Predictions NEVER use results after End date.")
    # These date inputs are rendered AFTER the history file is parsed, so we can safely
    # compute sensible defaults from the uploaded history without triggering Streamlit
    # session_state widget-key mutation errors.
    wf_dates_box = st.container()

    st.divider()
    st.header("Rule Settings (member engine)")
    gate_no9 = st.checkbox("Gate: seed has NO digit 9 (recommended)", value=True)
    max_rules = st.slider("Max base rules used", 1, 20, 12)
    min_support = st.slider("Min support (n_hits_gate) per base rule", 1, 50, 6)
    st.caption("Tie-pack rules always used as tie-breakers when base scores tie.")

    st.divider()
    st.header("Stream Rank Settings (live)")
    w_match = st.slider("Weight: trait match strength", 0.0, 3.0, 1.0, 0.05)
    w_gap   = st.slider("Weight: #1–#2 gap", 0.0, 3.0, 0.6, 0.05)
    w_roll  = st.slider("Weight: rolling 30-draw 025 hits", 0.0, 2.0, 0.25, 0.05)
    w_since = st.slider("Weight: draws since last 025 (due)", 0.0, 2.0, 0.15, 0.05)
    since_mode = st.selectbox(
        "Draws since last 025 mode",
        options=["due", "recent", "blend"],
        index=0,
        format_func=lambda x: {
            "due": "Due (higher = better) — legacy",
            "recent": "Recent (lower = better) — momentum",
            "blend": "Blend extremes (recent + very due) — U-shape",
        }[x],
        help="Only changes how the draws-since component is scored. Default keeps legacy behavior.",
    )
    cooldown_k = st.slider("Cooldown (draws) after a 025 hit", 0, 10, 2)
    cooldown_penalty = st.slider("Cooldown penalty", 0.0, 3.0, 1.0, 0.05)

    st.divider()

    st.divider()
    st.subheader("Calibration (session)")
    _has_calib = ("calib_streamscore" in st.session_state) and (st.session_state.get("calib_streamscore") is not None)
    use_calibrated_streamscore = st.checkbox(
        "Use calibrated StreamScore (from LAB run)",
        value=False,
        disabled=not _has_calib,
        help="If you ran StreamScore calibration in LAB, this will use the fitted model to rank streams (instead of the legacy hand-weighted StreamScore).",
    )
    if not _has_calib:
        st.caption("Run LAB → Calibration to fit a StreamScore model, then come back here to enable it.")

    st.subheader("Stream Skip Suggestions (experimental)")
    enable_stream_skip = st.checkbox(
        "Enable stream-level skip suggestions (mark some streams as NO PLAY)",
        value=False,
        help="These are mined from your uploaded history. They never remove rows; they only tag/penalize streams and can optionally exclude them from the Top-N play list.",
    )
    skip_exclude_from_topn = st.checkbox(
        "If enabled, exclude 'NO PLAY' streams from the Top-N plays list",
        value=False,
        help="Keeps the ranked table intact, but the Top-N play list will skip any stream tagged NO PLAY.",
    )
    skip_penalty = st.slider(
        "Score penalty for suggested NO PLAY streams",
        0.0, 10.0, 3.0, 0.25,
        help="If >0, tagged streams are pushed down the ranking by subtracting this from StreamScore.",
    )
    treat_neverhit_as_skip = st.checkbox(
        "Treat streams with 0 hits in the full history window as NO PLAY",
        value=False,
        help="Only applies to the streams that never produced a 025-family hit anywhere in the uploaded history window.",
    )

    st.divider()

    st.subheader("Actions")
    c1, c2 = st.columns(2)
    if c1.button("Clear cache", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        # Clear internal cached artifacts (do NOT remove widget keys after widgets are created)
        for _k in ["_cached_live_rows", "_cached_playlist", "_inputs_sig"]:
            if _k in st.session_state:
                del st.session_state[_k]
        # Force a rebuild on next run
        st.session_state["playlist_force_rebuild"] = st.session_state.get("playlist_force_rebuild", 0) + 1
        st.success("Cache cleared.")
        st.rerun()

    if c2.button("Rebuild playlist", use_container_width=True):
        # Bump a counter that participates in the cache signature
        st.session_state["playlist_force_rebuild"] = st.session_state.get("playlist_force_rebuild", 0) + 1
        for _k in ["_cached_playlist"]:
            if _k in st.session_state:
                del st.session_state[_k]
        st.success("Rebuilding playlist…")
        st.rerun()

    st.header("Files")
    st.caption("Optional. Leave these blank to use the **embedded packs shipped inside this app file** (recommended for Streamlit Cloud). "
               "If you provide an upload or a valid repo path, that external file will be used instead.")

    weights_csv = st.text_input("Weights CSV path (optional)", value="")
    tie_csv = st.text_input("Tie-pack CSV path (optional)", value="")

    weights_file_up = st.file_uploader("Upload weights CSV (optional)", type=["csv"], key="weights_file_upload")
    tiepack_file_up = st.file_uploader("Upload tie-pack CSV (optional)", type=["csv"], key="tie_upload")

# Sticky upload overrides: persist uploaded CSV bytes in session_state so changing sliders
# (e.g., max base weights) does NOT silently fall back to embedded packs.
if weights_file_up is not None:
    try:
        st.session_state["weights_override_bytes"] = weights_file_up.getvalue()
        st.session_state["weights_override_name"] = getattr(weights_file_up, "name", "uploaded_weights.csv")
    except Exception:
        pass
if tiepack_file_up is not None:
    try:
        st.session_state["tie_override_bytes"] = tiepack_file_up.getvalue()
        st.session_state["tie_override_name"] = getattr(tiepack_file_up, "name", "uploaded_tiepack.csv")
    except Exception:
        pass


tab_live, tab_lab = st.tabs(["LIVE: Full ranked stream list", "LAB: Historical 025 hit-events"])

if hist_file is None:
    st.info("Upload your Lottery Post export CSV in the sidebar to begin.")
    st.stop()

# Load history

def _peek_first_line(uploaded) -> str:
    try:
        b = uploaded.getvalue()
        s = b[:5000].decode("utf-8", errors="ignore")
        return s.splitlines()[0] if s.splitlines() else ""
    except Exception:
        return ""

def _read_txt(f):
    """
    Robust reader for LotteryPost tab-delimited exports that may contain:
      - headerless 4-column rows: date, state, game, result
      - occasional stray/continuation lines (e.g., 'Fireball: 9') or broken records
    We prefer tab-delimited parsing and skip malformed lines.
    """
    first = _peek_first_line(f)
    headerish = bool(
        re.search(r"date", first, re.I)
        and re.search(r"state", first, re.I)
        and re.search(r"game", first, re.I)
        and re.search(r"result", first, re.I)
    )

    f.seek(0)
    if headerish:
        try:
            return pd.read_csv(f, sep="	", engine="python", dtype=str, on_bad_lines="skip")
        except Exception:
            f.seek(0)
            return pd.read_csv(f, sep=None, engine="python", dtype=str, on_bad_lines="skip")
    else:
        try:
            return pd.read_csv(
                f,
                sep="	",
                engine="python",
                header=None,
                names=["Draw Date", "State", "Game", "Results"],
                dtype=str,
                on_bad_lines="skip",
            )
        except Exception:
            f.seek(0)
            df0 = pd.read_csv(f, sep=None, engine="python", header=None, dtype=str, on_bad_lines="skip")
            if df0.shape[1] >= 4:
                df0 = df0.iloc[:, :4].copy()
                df0.columns = ["Draw Date", "State", "Game", "Results"]
            return df0

try:
    filename = getattr(hist_file, "name", "").lower()

    if filename.endswith(".txt") or filename.endswith(".tsv"):
        raw = _read_txt(hist_file)
    else:
        raw = pd.read_csv(hist_file)

    hist = normalize_history(raw)
    if hist_update_file is not None:
        try:
            raw_u = _read_any(hist_update_file)
            hist_u = normalize_history(raw_u)
            hist = pd.concat([hist_u, hist], ignore_index=True)
            # best-effort dedupe on core export columns
            dedupe_cols = [c for c in ["Draw Date", "State", "Game", "Results"] if c in hist.columns]
            if dedupe_cols:
                hist = hist.drop_duplicates(subset=dedupe_cols, keep="first")
        except Exception as e:
            st.warning(f"Could not apply incremental update file: {e}")


    if len(hist) == 0:
        st.error(
            "No valid history rows were parsed after normalization. This usually means the TXT/TSV has broken/extra lines or the date/result formats are not being recognized."
        )
        st.caption(
            "Tip: Your file should be tab-delimited with 4 columns per row: date, state, game, result (e.g., 'Thu, Mar 14, 2024	Missouri	Pick 4 Midday	9-3-1-9')."
        )
        with st.expander("Show first 50 raw rows (as parsed)"):
            st.dataframe(raw.head(50), use_container_width=True)
        st.stop()
except Exception as e:
    err = str(e)
    hint = ""
    if "History file missing columns" in err:
        hint = " (Common cause: you uploaded a WEIGHTS or TIE-PACK CSV into the History uploader. Upload the Lottery Post history export instead.)"
    st.error(f"Failed to read/normalize history: {err}{hint}")
    st.stop()
# Diagnostics
st.write(f"Loaded history rows (parsed): {len(hist):,}")
dmin, dmax = hist["Draw Date"].min(), hist["Draw Date"].max()

st.write(f"History date span: {dmin.date()} → {dmax.date()}")

# ---- Walk-forward date defaults + validation ----
_default_start = dmin.date()
_default_end = dmax.date()
_default_play = (_default_end + datetime.timedelta(days=1))

# Render the date inputs AFTER history is parsed so defaults are based on this upload,
# without mutating widget-backed session_state keys.
with wf_dates_box:
    wf_start = st.date_input("Start date (training ends at or before this)", value=st.session_state.get("wf_start", _default_start), key="wf_start")
    wf_end = st.date_input("End date (last training day)", value=st.session_state.get("wf_end", _default_end), key="wf_end")
    wf_play = st.date_input("Play date (prediction target day)", value=st.session_state.get("wf_play", _default_play), key="wf_play")

# No-lookahead guard: End date must be BEFORE Play date
if wf_end >= wf_play:
    st.warning("No-lookahead guard: End date should be BEFORE Play date. Adjusting End date to Play date - 1 day.")
    wf_end = (pd.Timestamp(wf_play) - pd.Timedelta(days=1)).date()

# Ensure start <= end and clamp to file span
if wf_start > wf_end:
    st.warning("Start date was after End date. Adjusting Start date to End date.")
    wf_start = wf_end

if wf_start < _default_start:
    wf_start = _default_start
if wf_end > _default_end:
    wf_end = _default_end

st.caption(f"Walk-forward window in effect: {wf_start} → {wf_end} (Play date: {wf_play}). Model cannot see results after {wf_end}.")

# If user requested a rebuild, clear cached computations so the playlist is rebuilt deterministically
if st.session_state.get("playlist_force_rebuild", 0) > 0:
    try:
        st.cache_data.clear()
    except Exception:
        pass
pf = getattr(hist, "attrs", {}).get("parse_fail_count", 0)
if pf:
    st.warning(f"Result-parse failures: {pf:,} rows could not be parsed into 4 digits and were ignored.")
    ex = getattr(hist, "attrs", {}).get("parse_fail_examples", [])
    if ex:
        with st.expander("Show parse-failure examples"):
            st.dataframe(pd.DataFrame(ex), use_container_width=True)

# Load rules (from local files next to app, or user can type full path)
try:
    base_rules, tie_rules, rules_meta = load_rules(weights_csv, tie_csv, max_rules=max_rules, min_support=min_support, weights_file_up=(weights_file_up if weights_file_up is not None else st.session_state.get('weights_override_bytes')), tie_up=(tiepack_file_up if tiepack_file_up is not None else st.session_state.get('tie_override_bytes')))
except Exception as e:
    st.error(f"Failed to load rules from '{weights_csv}' and '{tie_csv}': {e}")
    st.stop()

st.caption(
    f"Rule files in effect — Weights: {rules_meta['weights']['source']} (rows={rules_meta['weights']['rows']}) | "
    f"Tie-pack: {rules_meta['tie_pack']['source']} (rows={rules_meta['tie_pack']['rows']})"
)

with st.expander("Sanity checks (must PASS before trusting rankings)", expanded=True):
    nz_base = sum(1 for r in base_rules if getattr(r, 'weight', 0.0) != 0.0)
    nz_tie = sum(1 for r in tie_rules if getattr(r, 'weight', 0.0) != 0.0)
    st.write(f"Base rules loaded: {len(base_rules)} (non-zero weights: {nz_base})")
    st.write(f"Tie-pack rules loaded: {len(tie_rules)} (non-zero weights: {nz_tie})")

    if len(base_rules) == 0:
        st.error("FAIL: No base rules loaded — rankings will be meaningless.")
    elif nz_base == 0:
        st.error("FAIL: All base-rule weights are 0 — scores will be 0 and ranking will be arbitrary.")
    else:
        st.success("PASS: Base rules have non-zero weights.")

    if len(tie_rules) == 0:
        st.warning("WARN: No tie-pack rules loaded — dead ties may default to PLAY 1 unless handled elsewhere.")
    elif nz_tie == 0:
        st.warning("WARN: Tie-pack weights are all 0 — tie-breaks may not trigger.")
    else:
        st.success("PASS: Tie-pack has non-zero weights.")

    st.caption("If any FAIL appears above, stop and fix before trusting outputs — clearing runtime errors alone does NOT mean the scoring is working.")


# ---- Mined member-layer packs controls ----
st.sidebar.markdown("---")
with st.sidebar.expander("Mined member-layer packs (elims / 1-miss downranks / rescues)", expanded=False):
    st.caption("Optional uploads override the embedded packs from this thread.")
    rp_up = st.file_uploader("Rulepack v3.1 (CSV) — member eliminators + optional overrides", type=["csv"], key="mined_rulepack")
    dr_up = st.file_uploader("1-miss downranks addon (CSV)", type=["csv"], key="mined_downranks")
    rs_up = st.file_uploader("Rescue rules (CSV) — per-event mined", type=["csv"], key="mined_rescues")

    enable_elims = st.checkbox("Enable member eliminators (RULEPACK v3.1)", value=True)
    enable_1miss = st.checkbox("Enable 1-miss downranks (Top1↔Top2 swap)", value=True)
    enable_rescues = st.checkbox("Enable LOCKED rescues (Top3→Top1 promotion)", value=True)
    allow_guarded = st.checkbox("Also allow GUARDED rescues (experimental)", value=False)
    enable_rulepack_top3 = st.checkbox("Enable RULEPACK OVERRIDE_PICK_TO_TOP3 (experimental)", value=False)

mined_dfs = load_mined_rule_dfs(
    rulepack_file=rp_up if rp_up is not None else None,
    downranks_file=dr_up if dr_up is not None else None,
    rescues_file=rs_up if rs_up is not None else None,
)
with st.expander("Rule file provenance (sha256 hashes)"):
    st.code(
        f"Weights source: {rules_meta['weights']['source']}\n"
        f"Weights sha256: {rules_meta['weights']['sha256']}\n"
        f"Tie-pack source: {rules_meta['tie_pack']['source']}\n"
        f"Tie-pack sha256: {rules_meta['tie_pack']['sha256']}\n"
    )


with st.expander("Loaded Files (source / filename / sha256 / row count)"):
    try:
        loaded_rows = []

        # Weights / Tie-pack (meta comes from load_rules)
        w_meta = rules_meta.get("weights", {})
        t_meta = rules_meta.get("tie_pack", {})

        def _fname_from_meta(meta_obj, path_str, up_obj, embedded_name):
            meta_obj = meta_obj or {}
            # Prefer explicit filename captured during load_rules (supports cached uploads)
            fn = meta_obj.get("filename", "")
            if fn:
                return fn
            src = meta_obj.get("source")
            if src == "upload":
                if up_obj is not None and hasattr(up_obj, "name"):
                    return up_obj.name
                # fallback to session_state if available
                return str(st.session_state.get("weights_override_name") or st.session_state.get("tie_override_name") or "uploaded.csv")
            if src == "path" and path_str:
                return str(path_str)
            return embedded_name


        loaded_rows.append({
            "pack": "weights",
            "source": w_meta.get("source", ""),
            "filename": _fname_from_meta(w_meta, weights_csv, weights_file_up, "embedded:weights"),
            "sha256": w_meta.get("sha256", ""),
            "rows": int(w_meta.get("rows", 0)),
        })
        loaded_rows.append({
            "pack": "tie_pack",
            "source": t_meta.get("source", ""),
            "filename": _fname_from_meta(t_meta, tie_csv, tiepack_file_up, "embedded:tie_pack"),
            "sha256": t_meta.get("sha256", ""),
            "rows": int(t_meta.get("rows", 0)),
        })

        # Member-layer mined packs
        def _pack_meta_from_upload_or_embedded(up_obj, embedded_b64_gz, pack_label):
            if up_obj is not None:
                try:
                    b = up_obj.getvalue() if hasattr(up_obj, "getvalue") else up_obj.read()
                    return {"source": "upload", "filename": getattr(up_obj, "name", "(upload)"), "sha256": _sha256_hex(b)}
                except Exception:
                    return {"source": "upload", "filename": getattr(up_obj, "name", "(upload)"), "sha256": ""}
            b = _embedded_raw_bytes(embedded_b64_gz)
            return {"source": "embedded", "filename": f"embedded:{pack_label}", "sha256": _sha256_hex(b)}

        rp_meta = _pack_meta_from_upload_or_embedded(rp_up, _EMBED_RULEPACK_V3_2_B64_GZ, "rulepack_v3_2")
        dr_meta = _pack_meta_from_upload_or_embedded(dr_up, _EMBED_1MISS_DOWNRANKS_V1_1_B64_GZ, "1miss_downranks_v1_1")
        rr_meta = _pack_meta_from_upload_or_embedded(rs_up, _EMBED_RESCUES_V2_B64_GZ, "rescues_v2")

        loaded_rows.append({"pack": "rulepack", **rp_meta, "rows": int(len(mined_dfs.get("rulepack", [])))})
        loaded_rows.append({"pack": "downranks", **dr_meta, "rows": int(len(mined_dfs.get("downranks", [])))})
        loaded_rows.append({"pack": "rescues", **rr_meta, "rows": int(len(mined_dfs.get("rescues", [])))})

        st.dataframe(pd.DataFrame(loaded_rows), use_container_width=True)
    except Exception as e:
        st.error(f"Loaded Files panel error: {e}")


with st.expander("Active base rules (after support + max_rules filters)"):
    if base_rules:
        st.dataframe(pd.DataFrame([{
            "feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": r.weight
        } for r in base_rules]), use_container_width=True)
    else:
        st.warning("No base rules active under the current settings. Lower min_support or increase max_rules.")

with st.expander("Tie-pack rules"):
    st.dataframe(pd.DataFrame([{
        "feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": r.weight
    } for r in tie_rules]), use_container_width=True)

# Cadence metrics (for live)

# Apply walk-forward window to history (NO LOOKAHEAD)
hist_wf = hist[(hist["Draw Date"].dt.date >= wf_start) & (hist["Draw Date"].dt.date <= wf_end)].copy()

# Re-sort and re-index per stream after filtering
hist_wf = hist_wf.sort_values(["stream", "Draw Date"]).reset_index(drop=True)
hist_wf["stream_idx"] = hist_wf.groupby("stream").cumcount()

# Cadence metrics computed ONLY on visible history
hist_cad = compute_cadence_metrics(hist_wf)

# ----------------------------
# LIVE MODE
# ----------------------------
with tab_live:
    st.subheader("Full ranked stream list (most → least likely to produce a 025-family hit next)")

    # Latest seed per stream
    latest = hist_cad.sort_values(["stream", "Draw Date"]).groupby("stream").tail(1).copy()
    latest = latest.rename(columns={"result": "seed_result", "Draw Date": "seed_date"})

    # Gate no9 (locked core gate)
    pre_gate_latest = latest.copy()
    gated_out_live = pre_gate_latest.iloc[0:0].copy()
    if gate_no9:
        _mask9 = pre_gate_latest["seed_result"].astype(str).str.contains("9")
        gated_out_live = pre_gate_latest[_mask9].copy()
        latest = pre_gate_latest[~_mask9].copy()
    else:
        latest = pre_gate_latest

    # Visibility: which streams were excluded by the NO9 gate today
    if gate_no9:
        with st.expander(f"Show streams gated out by NO9 today ({len(gated_out_live):,})", expanded=False):
            if gated_out_live.empty:
                st.write("None.")
            else:
                g = gated_out_live.rename(columns={"stream": "Stream", "seed_date": "SeedDate", "seed_result": "Seed"}).copy()
                g["StreamKey"] = g["Stream"].apply(canon_stream)
                g["SeedDate"] = g["SeedDate"].apply(lambda x: x.date() if pd.notna(x) else None)
                show_cols = [c for c in ["Stream", "StreamKey", "SeedDate", "Seed"] if c in g.columns]
                st.dataframe(g[show_cols].sort_values(["Stream"]), use_container_width=True)

    target_dow_live = dt.date.today().strftime('%A')
    rows = []

    # Streams with 0 hits in the entire (uploaded) history window
    _tmp_hits = hist_wf.copy()
    _tmp_hits["member"] = _tmp_hits["result"].apply(as_member)
    _tmp_hits["is_025"] = (_tmp_hits["member"].isin(["0025", "0225", "0255"])).astype(int)
    _stream_hit = _tmp_hits.groupby("stream")["is_025"].sum()
    _stream_total = _tmp_hits.groupby("stream")["is_025"].size()
    stream_base_rate_map = (_stream_hit / _stream_total.replace(0, np.nan)).fillna(0.0).to_dict()
    zero_hit_streams = set(_stream_hit[_stream_hit == 0].index.tolist())

    # A small, history-mined starter set of "true stream-skip" pockets (can be extended in Lab)
    # (These rules are advisory unless you enable the Stream Skip toggle.)
    _pocket_skip_rules = [
        {"id": "SSKIP-IA-01", "stream": "Iowa | Pick 4 Evening", "when": lambda feats: feats.get("seed_mirrorpair_cnt", 0) == 0, "reason": "seed_mirrorpair_cnt==0"},
        {"id": "SSKIP-DC-01", "stream": "Washington, D.C. | DC-4 7:50pm", "when": lambda feats: feats.get("seed_mirrorpair_cnt", 0) == 0, "reason": "seed_mirrorpair_cnt==0"},
        {"id": "SSKIP-LA-01", "stream": "Louisiana | Pick 4", "when": lambda feats: feats.get("seed_has_pair", 0) == 0, "reason": "seed_has_pair==0"},
    ]

    for _, r in latest.iterrows():
        stream = r["stream"]
        seed4 = r["seed_result"]
        s = score_seed(seed4, base_rules, tie_rules)
        feats = s["features"]

        # Apply mined member-layer actions (elims / downranks / rescues)
        s_mined, meta_mined = apply_mined_member_layers(
            seed4=seed4,
            stream=stream,
            s_base=s,
            feats=feats,
            draws_since_last_025=r.get("draws_since_last_025", None),
            rolling_025_30=r.get("rolling_025_30", None),
            rolling_025_60=r.get("rolling_025_60", None),
            target_dow=target_dow_live,
            tier=None,
            mined_dfs=mined_dfs,
            enable_eliminators=enable_elims,
            enable_downranks=enable_1miss,
            enable_locked_rescues=enable_rescues,
            allow_guarded_rescues=allow_guarded,
            enable_rulepack_top3_overrides=enable_rulepack_top3,
        )


        cadence_row = {
            "draws_since_last_025": int(r.get("draws_since_last_025", 9999)),
            "rolling_025_30": int(r.get("rolling_025_30", 0)),
        }
        ss_legacy = stream_score_row(
            s, cadence_row,
            w_match=w_match, w_gap=w_gap, w_roll=w_roll, w_since=w_since,
            since_mode=since_mode,
            cooldown_k=cooldown_k, cooldown_penalty=cooldown_penalty
        )
        ss = ss_legacy
        streamprob = None

        if use_calibrated_streamscore and st.session_state.get("calib_streamscore") is not None:
            calib = st.session_state["calib_streamscore"]
            row_for_model = {
                "stream": stream,
                "seed_result": seed4,
                "draws_since_last_025": int(r.get("draws_since_last_025", 0)),
                "rolling_025_30": float(r.get("rolling_025_30", 0)),
                "rolling_025_60": float(r.get("rolling_025_60", 0)),
                "stream_base_rate": float(stream_base_rate_map.get(stream, 0.0)) if "stream_base_rate_map" in globals() else 0.0,
                "never_hit_before": 1 if ("zero_hit_streams" in globals() and stream in zero_hit_streams) else 0,
            }
            Xd = [_make_streamscore_feature_dict(row_for_model)]
            try:
                Xmat, _cols = _dicts_to_design_matrix(Xd, feature_columns=calib.get("feature_columns"))
                mean = np.asarray(calib.get("mean", []), dtype=np.float64)
                std = np.asarray(calib.get("std", []), dtype=np.float64)
                if mean.size and std.size:
                    Xmat = _standardize_apply(Xmat, mean, std)
                w = np.asarray(calib.get("coef", []), dtype=np.float64)
                b = float(calib.get("intercept", 0.0))
                if w.size == 0 or Xmat.shape[1] != w.size:
                    streamprob = None
                    ss = None
                else:
                    streamprob = float(_sigmoid(Xmat.dot(w) + b)[0])
                    ss = streamprob * 1000.0
            except Exception:
                streamprob = None
                ss = None

        # feats already computed above

        # Stream-level NO-PLAY suggestions (true stream-skip candidates)
        skip_reasons = []
        if treat_neverhit_as_skip and r["stream"] in zero_hit_streams:
            skip_reasons.append("zero_hits_in_history")
        for rule in _pocket_skip_rules:
            if r["stream"] == rule["stream"] and rule["when"](feats):
                skip_reasons.append(f'{rule["id"]}:{rule["reason"]}')

        skip_suggested = 1 if skip_reasons else 0
        skip_reason = " | ".join(skip_reasons) if skip_reasons else ""

        ss_adj = ss
        if enable_stream_skip and skip_suggested and skip_penalty > 0:
            ss_adj = ss - float(skip_penalty)

        rows.append({
            "Stream": r["stream"],
            "SeedDate": r["seed_date"].date() if pd.notna(r["seed_date"]) else None,
            "Seed": seed4,
            "PredictedMember": s_mined["top1_final"],
            "Top1Base": s["top1"],
            "Top2": s_mined["top2_final"],
            "Top2Base": s["top2"],
            "Top3": s_mined["top3_final"],
            "Top3Base": s["top3"],
            "Top1Score": max(s["scores"].values()),
            "BaseGap(#1-#2)": float(s["base_gap"]),
            "TieFired": int(s["tie_fired"]),
            "DeadTie": int(s["dead_tie"]),
                                "CadenceRolling025_30": int(cadence_row.get("rolling_025_30", 0)),
                                "CadenceDrawsSinceLast025": int(cadence_row.get("draws_since_last_025", 9999)),
            "ForcedPick": s["forced_pick"] if s["forced_pick"] else "",
            "ElimFired": ",".join(meta_mined.get("elim_fired", [])),
            "DownrankFired": meta_mined.get("downrank_fired") or "",
            "RescueFired": meta_mined.get("rescue_fired") or "",
            "RulepackTop3Fired": meta_mined.get("rulepack_top3_fired") or "",
            "SinceLast025(draws)": cadence_row["draws_since_last_025"],
            "Rolling025_30": cadence_row["rolling_025_30"],
            "SeedSum": feats["seed_sum"],
            "SeedSpread": feats["seed_spread"],
            "SeedAbsDiff": feats.get("seed_absdiff_sum", feats.get("seed_pairwise_absdiff_sum", 0)),
            "WorstPair025": feats["seed_has_worstpair_025"],
            "SkipSuggested": int(skip_suggested),
            "SkipReason": skip_reason,
            "StreamProb": (float(streamprob) if streamprob is not None else np.nan),
            "StreamScore": float(ss_adj),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["StreamKey"] = out["Stream"].apply(canon_stream)
    if out.empty:
        st.warning("No streams left after gating/filtering. Try turning off gate_no9.")
    else:
        out = out.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=False).reset_index(drop=True)
        out.insert(0, "Rank", np.arange(1, len(out)+1))

        # Tier labels (no cutoff): A strongest, then B, then C
        # thresholds derived from relative distribution each run
        qA = out["StreamScore"].quantile(0.20)
        qB = out["StreamScore"].quantile(0.50)

        def tier(s):
            if s >= qA:
                return "A"
            if s >= qB:
                return "B"
            return "C"
        out["Tier"] = out["StreamScore"].apply(tier)

        # --- Flip recommendation (Top-2 trigger) ---
        # This is the same "Top-2 recommended" segment we used in the earlier analysis:
        # base_gap == 0 AND tie_fired == 1 AND dead_tie == 0
        out["FlipRec"] = ((out["BaseGap(#1-#2)"] == 0) & (out["TieFired"] == 1) & (out["DeadTie"] == 0)).astype(int)
        # Legacy single-member suggestion (kept for continuity): FlipRec means "prefer TOP2"
# NEW default play policy (requested): when TieFired (FlipRec=1), play TOP1 + TOP2 (2 members)
        # --- PlayCount policy controls (LIVE) ---
        auto_top2_seed9 = st.checkbox("Auto Top2 when seed_has_9=1 (volume control)", value=True)

        # Identify seed-has-9 flag column name if present
        _seed9_col = None
        if "seed_has_9" in out.columns:
            _seed9_col = "seed_has_9"
        elif "seed_has9" in out.columns:
            _seed9_col = "seed_has9"
        seed9_mask = (out[_seed9_col] == 1) if _seed9_col else False

        # Identify stream-skip flag column if present
        _skip_col = None
        if "skip_exclude_from_topn" in out.columns:
            _skip_col = "skip_exclude_from_topn"
        elif "SkipExcludeTopN" in out.columns:
            _skip_col = "SkipExcludeTopN"
        skip_mask = (out[_skip_col] == True) if _skip_col else False

        need_top2_mask = (out["FlipRec"] == 1) | ((seed9_mask) if auto_top2_seed9 else False)

        out["PlayReason"] = np.select(
            [
                skip_mask,
                (out["FlipRec"] == 1) & (seed9_mask if _seed9_col else False),
                (out["FlipRec"] == 1),
                (seed9_mask if _seed9_col else False) & auto_top2_seed9,
            ],
            [
                "stream-skip",
                "tie+seed9",
                "tie",
                "seed9",
            ],
            default=""
        )

        out["PlayPlan"] = np.select(
            [skip_mask, need_top2_mask],
            ["NO PLAY", "PLAY 2"],
            default="PLAY 1",
        )

        out["PlayMembers"] = np.select(
            [skip_mask, need_top2_mask],
            ["", out["PredictedMember"].astype(str) + " + " + out["Top2"].astype(str)],
            default=out["PredictedMember"].astype(str),
        )

        out["PlayPick"] = np.select(
            [skip_mask, need_top2_mask],
            ["", "Top1+Top2"],
            default="Top1",
        )

        out["PlayMember"] = np.select(
            [skip_mask, need_top2_mask],
            ["", out["Top2"].astype(str)],
            default=out["PredictedMember"].astype(str),
        )

        out["PlayCount"] = np.select(
            [skip_mask, need_top2_mask],
            [0, 2],
            default=1,
        ).astype(int)


        # Put the play columns right next to the prediction columns for visibility
        _cols = list(out.columns)
        for c in ["PlayPlan", "PlayMembers", "PlayCount", "PlayPick", "PlayMember", "FlipRec"]:
            if c in _cols:
                _cols.remove(c)
        # Insert after PredictedMember (or after Seed if that col is missing)
        if 'PredictedMember' in _cols:
            _i = _cols.index('PredictedMember') + 1
        elif 'Seed' in _cols:
            _i = _cols.index('Seed') + 1
        else:
            _i = min(5, len(_cols))
        _cols[_i:_i] = ["PlayPlan", "PlayMembers", "PlayCount", "PlayPick", "PlayMember", "FlipRec"]
        out = out[_cols]

        # Row-by-row percentile (literal row position after ranking)
        if "Rank" in out.columns:
            _n = len(out)
            out["RowRank"] = out["Rank"].astype(int)
            if _n > 1:
                out["RowPercentile"] = 100.0 * (1 - (out["RowRank"] - 1) / (_n - 1))
            else:
                out["RowPercentile"] = 100.0

        st.caption("Full list ranked. Tier A ≈ top 20% by StreamScore; Tier B ≈ next 30%; Tier C ≈ bottom 50% (no cutoff). "
           "FlipRec=1 means the app recommends playing TOP1 + TOP2 for that stream (PlayMembers; PlayCount=2).")
        def _style_play(df):
            # Bold exactly what to play:
            # PlayCount=1 -> bold Top1 (and play columns)
            # PlayCount=2 -> bold Top1 + Top2 (and play columns)
            # PlayCount=3 -> bold Top1 + Top2 + Top3 (and play columns)
            def _row_style(row):
                pc = int(row.get("PlayCount", 0) or 0)
                bold_cols = {"PlayPlan", "PlayMembers", "PlayCount", "PlayMember", "PlayPick", "PlayReason"}

                # "Top1" in this app is PredictedMember (but we bold both if present)
                if "PredictedMember" in df.columns:
                    bold_cols.add("PredictedMember")
                if "Top1" in df.columns:
                    bold_cols.add("Top1")

                if pc >= 2 and "Top2" in df.columns:
                    bold_cols.add("Top2")
                if pc >= 3 and "Top3" in df.columns:
                    bold_cols.add("Top3")

                # If explicitly NO PLAY (PlayCount==0), still emphasize PlayPlan/PlayMembers
                return ["font-weight:700;" if col in bold_cols and (pc >= 1 or col in {"PlayPlan", "PlayMembers", "PlayCount"}) else "" for col in df.columns]

            return df.style.apply(_row_style, axis=1)

        split_has9 = st.checkbox("Split view: seed_has_9 streams (1) vs (0)", value=True)
        if split_has9 and "seed_has_9" in out.columns:
            out_has9 = out[out["seed_has_9"] == 1].copy()
            out_no9 = out[out["seed_has_9"] == 0].copy()

            st.subheader("Seed has 9 = 1")
            st.dataframe(_style_play(out_has9), use_container_width=True, height=420)

            st.subheader("Seed has 9 = 0")
            st.dataframe(_style_play(out_no9), use_container_width=True, height=420)
        else:
            st.dataframe(_style_play(out), use_container_width=True, height=650)

        # Export
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download LIVE ranked streams (CSV)", data=csv_bytes, file_name="core025_live_ranked_streams.csv", mime="text/csv")


# ----------------------------
# LAB MODE (historical 025 hit-events)
# ----------------------------
with tab_lab:
    st.subheader("Historical 025 hit-events (seed → next winner member)")

    st.markdown("### Daily reality check (includes 0-hit days)")
    _daily_df = hist_wf.copy()
    _daily_df["member"] = _daily_df["result"].apply(as_member)
    _daily_df["is_025"] = (_daily_df["member"].isin(["0025", "0225", "0255"])).astype(int)

    _daily = (
        _daily_df.groupby("Draw Date")["is_025"]
        .sum()
        .rename("hit_events")
        .reset_index()
        .sort_values("Draw Date")
    )
    _total_days = len(_daily)
    _hit_days = int((_daily["hit_events"] > 0).sum())
    _zero_days = int((_daily["hit_events"] == 0).sum())

    if _total_days:
        st.write(
            f"Dates in window: **{_total_days}**  |  Hit-days: **{_hit_days}** ({_hit_days/_total_days:.1%})  |  0-hit days: **{_zero_days}** ({_zero_days/_total_days:.1%})"
        )

    with st.expander("Show daily hit-event table (0-hit days included)", expanded=False):
        st.dataframe(_daily.sort_values("Draw Date", ascending=False), use_container_width=True)

    with st.expander("Stream-skip miner: find stream-level 'disqualify all members' pockets", expanded=False):
        st.caption(
            "This mines for stream+condition pockets that produced **0** 025-family hits in the full history window (potential true stream-skips). "
            "These are *advisory* and should be verified on updated windows."
        )
        ss_min_support = st.slider("Min pocket size (support)", 50, 2500, 500, 50, key="ss_min_support")
        ss_min_total_hits = st.slider("Min total hits in stream (overall) to consider", 0, 50, 5, 1, key="ss_min_total_hits")
        run_ss = st.button("Run stream-skip miner", key="run_ss_miner")

        if run_ss:
            dfm = _daily_df.copy().sort_values(["stream", "Draw Date"]).reset_index(drop=True)
            dfm["seed"] = dfm.groupby("stream")["result"].shift(1)
            dfm = dfm.dropna(subset=["seed"]).copy()

            feats_list = dfm["seed"].astype(str).apply(compute_features)
            feats_df = pd.DataFrame(list(feats_list))
            dfm = pd.concat([dfm.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)

            pockets = {
                "seed_mirrorpair_cnt==0": (dfm["seed_mirrorpair_cnt"] == 0),
                "seed_has_pair==0": (dfm["seed_has_pair"] == 0),
                "seed_has_pair==1": (dfm["seed_has_pair"] == 1),
                "seed_spread<=3": (dfm["seed_spread"] <= 3),
                "seed_spread_4_6": ((dfm["seed_spread"] >= 4) & (dfm["seed_spread"] <= 6)),
                "seed_spread>=7": (dfm["seed_spread"] >= 7),
                "even_eq1": (dfm["seed_even_cnt"] == 1),
                "even_eq2": (dfm["seed_even_cnt"] == 2),
                "even_eq3": (dfm["seed_even_cnt"] == 3),
                "consec_ge2": (dfm["seed_consec_links"] >= 2),
                "no_pair": (dfm["no_pair"] == 1),
                "pair_01": (dfm.get("pair_01", 0) == 1),
                "pair_03": (dfm.get("pair_03", 0) == 1),
                "pair_06": (dfm.get("pair_06", 0) == 1),
                "pair_12": (dfm.get("pair_12", 0) == 1),
                "pair_17": (dfm.get("pair_17", 0) == 1),
                "pair_55": (dfm.get("pair_55", 0) == 1),
            }

            total_hits = dfm.groupby("stream")["is_025"].sum().to_dict()
            rows_out = []
            for cname, mask in pockets.items():
                g = (
                    dfm[mask]
                    .groupby("stream")["is_025"]
                    .agg(n="size", hits="sum")
                    .reset_index()
                )
                g["pocket"] = cname
                g["hits_total"] = g["stream"].map(total_hits).fillna(0).astype(int)
                rows_out.append(g)

            res = pd.concat(rows_out, ignore_index=True)
            res = res[
                (res["n"] >= ss_min_support)
                & (res["hits"] == 0)
                & (res["hits_total"] >= ss_min_total_hits)
            ].sort_values(["hits_total", "n"], ascending=[False, False])

            st.write(f"Candidates found: **{len(res)}**")
            st.dataframe(res, use_container_width=True)


    with st.expander("Calibration & OOS Mining (StreamScore + StreamSkip candidates)", expanded=False):
        st.write(
            "Uses ONLY the uploaded history to (1) fit a time-respecting StreamScore model "
            "to predict *family-hit-next* (`is_025`) and (2) mine **out-of-sample** stream-skip pockets "
            "(conditions where the next draw was never a family hit on TRAIN *and* TEST)."
        )

        try:
            dfm_all = _build_transitions_from_history(hist_wf)
        except Exception as e:
            st.error(f"Could not build transitions from history: {e}")
            dfm_all = None

        if dfm_all is not None and not dfm_all.empty:
            st.caption(
                f"Transitions (seed→next): **{len(dfm_all):,}** | "
                f"Streams: **{dfm_all['stream'].nunique():,}** | "
                f"Family-hit-next events: **{int(dfm_all['is_025'].sum()):,}**"
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                frac_train = st.slider("Train fraction (time split)", 0.50, 0.90, 0.80, 0.05)
            with c2:
                C = st.number_input("LogReg C (regularization)", min_value=0.01, max_value=50.0, value=1.0, step=0.1)
            with c3:
                use_balanced = st.checkbox("Use balanced class weights", value=True)

            st.divider()
            st.subheader("A) StreamScore calibration")

            if st.button("Run StreamScore calibration now", key="run_streamscore_calib"):
                with st.spinner("Training StreamScore model (time-split)…"):
                    cw = "balanced" if use_balanced else None
                    calib = _calibrate_streamscore_logreg(dfm_all, frac_train=frac_train, C=float(C), class_weight=cw)
                    st.session_state["calib_streamscore"] = calib

            calib = st.session_state.get("calib_streamscore")
            if calib is not None:
                st.success(
                    f"Saved StreamScore model in session state. "
                    f"Test AUC: **{calib.get('test_auc', np.nan):.4f}** | "
                    f"Test AP: **{calib.get('test_ap', np.nan):.4f}**"
                )
                st.caption("Top positive weights (higher → more likely to hit next):")
                st.dataframe(calib["weights_df"].head(30), use_container_width=True, height=420)

                st.download_button(
                    "Download weights_streamscore.csv",
                    data=calib["weights_df"].to_csv(index=False).encode("utf-8"),
                    file_name="weights_streamscore.csv",
                    mime="text/csv",
                )

            st.divider()
            st.subheader("B) OOS StreamSkip mining (disqualify-all-members candidates)")

            min_support_global = st.number_input("Min TRAIN support for GLOBAL pocket", min_value=100, max_value=20000, value=3300, step=100)
            min_support_stream = st.number_input("Min TRAIN support for STREAM pocket", min_value=25, max_value=2000, value=200, step=25)

            use_auto_pockets = st.checkbox("Use broad auto-generated pocket set (recommended)", value=True)
            if use_auto_pockets:
                pockets = _default_streamskip_candidate_pockets()
            else:
                pockets = {
                    "seed_has9 == 1": lambda f: bool(f.get("seed_has9", 0) == 1),
                    "seed_has_worstpair_025 == 1": lambda f: bool(f.get("seed_has_worstpair_025", 0) == 1),
                    "seed_mirrorpair_cnt == 0": lambda f: bool(f.get("seed_mirrorpair_cnt", 0) == 0),
                    "seed_consec_links == 0": lambda f: bool(f.get("seed_consec_links", 0) == 0),
                    "seed_unique == 1": lambda f: bool(f.get("seed_unique", 0) == 1),
                    "seed_even_cnt == 2": lambda f: bool(f.get("seed_even_cnt", 0) == 2),
                    "seed_spread <= 2": lambda f: bool(f.get("seed_spread", 0) <= 2),
                }

            if st.button("Mine OOS StreamSkip pockets now", key="run_streamskip_oos"):
                with st.spinner("Mining candidate pockets on TRAIN and validating on TEST…"):
                    oos = _mine_streamskip_pockets_oos(
                        dfm_all,
                        pockets=pockets,
                        frac_train=frac_train,
                        min_support_global=int(min_support_global),
                        min_support_stream=int(min_support_stream),
                    )
                    st.session_state["streamskip_oos"] = oos

            oos = st.session_state.get("streamskip_oos")
            if oos is not None and not oos.empty:
                st.caption("Candidates are pockets with **H_train=0 AND H_test=0** (time-split). `suggested_tier` is **LOCKED** when support thresholds are met; otherwise treat as **GUARDED** until you validate further.")
                show = oos.sort_values(["scope", "candidate", "N_train"], ascending=[True, False, False])
                st.dataframe(show.head(200), use_container_width=True, height=520)

                st.download_button(
                    "Download streamskip_candidates_oos.csv",
                    data=show.to_csv(index=False).encode("utf-8"),
                    file_name="streamskip_candidates_oos.csv",
                    mime="text/csv",
                )
            elif oos is not None and oos.empty:
                st.warning("No pockets met your support thresholds (or dataset too small for those thresholds).")


    # Build transitions within stream
    df = hist_wf.copy()
    df["member"] = df["result"].apply(as_member)
    df["is_025"] = df["member"].notna().astype(int)

    # seed = previous result in same stream
    df["seed"] = df.groupby("stream")["result"].shift(1)
    df["seed_date"] = df.groupby("stream")["Draw Date"].shift(1)
    df = df.dropna(subset=["seed", "member"]).copy()

    # apply gate to seeds
    if gate_no9:
        df = df[~df["seed"].astype(str).str.contains("9")].copy()

    st.write(f"Total gated 025 hit-events: {len(df):,}")

    # Score each seed and compare to actual member
    lab_rows = []
    for _, r in df.iterrows():
        seed4 = r["seed"]
        s = score_seed(seed4, base_rules, tie_rules)
        target_dow_lab = pd.to_datetime(r["Draw Date"]).day_name() if pd.notna(r["Draw Date"]) else ""
        s_mined, meta_mined = apply_mined_member_layers(
            seed4=seed4,
            stream=r["stream"],
            s_base=s,
            feats=compute_features(seed4),
            draws_since_last_025=r.get("draws_since_last_025", None),
            rolling_025_30=r.get("rolling_025_30", None),
            rolling_025_60=r.get("rolling_025_60", None),
            target_dow=target_dow_lab,
            tier=None,
            mined_dfs=mined_dfs,
            enable_eliminators=enable_elims,
            enable_downranks=enable_1miss,
            enable_locked_rescues=enable_rescues,
            allow_guarded_rescues=allow_guarded,
            enable_rulepack_top3_overrides=enable_rulepack_top3,
        )
        actual = normalize_member_id(r["member"])
        hit1 = int(normalize_member_id(s_mined["top1_final"]) == actual)
        hit2 = int(actual in {normalize_member_id(s_mined["top1_final"]), normalize_member_id(s_mined["top2_final"])})

        lab_rows.append({
            "Stream": r["stream"],
            "SeedDate": r["seed_date"].date() if pd.notna(r["seed_date"]) else None,
            "Seed": seed4,
            "WinnerDate": r["Draw Date"].date() if pd.notna(r["Draw Date"]) else None,
            "ActualWinnerMember": actual,
            "Top1": s_mined["top1_final"],
            "Top1Base": s["top1"],
            "Top2": s_mined["top2_final"],
            "Top2Base": s["top2"],
            "Top3": s_mined["top3_final"],
            "Top3Base": s["top3"],
            "HitTop1": hit1,
            "HitTop2": hit2,
            "TieFired": int(s["tie_fired"]),
            "DeadTie": int(s["dead_tie"]),
            "ForcedPick": s["forced_pick"] if s["forced_pick"] else "",
            "ElimFired": ",".join(meta_mined.get("elim_fired", [])),
            "DownrankFired": meta_mined.get("downrank_fired") or "",
            "RescueFired": meta_mined.get("rescue_fired") or "",
            "RulepackTop3Fired": meta_mined.get("rulepack_top3_fired") or "",
            "SeedSum": s["features"].get("seed_sum"),
            "SeedAbsDiff": s["features"].get("seed_absdiff"),
            "SeedSpread": s["features"].get("seed_spread"),
            "WorstPair025": s["features"].get("seed_has_worstpair_025"),
        })

    lab = pd.DataFrame(lab_rows)
    if lab.empty:
        st.warning("No 025 hit-events found under current settings.")
    else:
        top1_acc = lab["HitTop1"].mean()
        top2_acc = lab["HitTop2"].mean()
        st.write(f"Top-1 accuracy: {top1_acc*100:.2f}%  |  Top-2 capture: {top2_acc*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Actual winner distribution (gated hit-events):")
            st.dataframe(lab["ActualWinnerMember"].value_counts().rename_axis("member").reset_index(name="count"), use_container_width=True)
        with col2:
            st.write("Top-1 pick distribution:")
            st.dataframe(lab["Top1"].value_counts().rename_axis("pick").reset_index(name="count"), use_container_width=True)

        st.dataframe(lab.sort_values(["WinnerDate","Stream"]), use_container_width=True, height=650)

        lab_csv = lab.to_csv(index=False).encode("utf-8")
        st.download_button("Download LAB hit-events (CSV)", data=lab_csv, file_name="core025_lab_hitevents.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Walk-forward Tier Coverage Study (StreamScore percentiles)")
        st.caption(
            "Automates 200+ play-dates: for each date, builds the ranked playlist using ONLY history before that date, "
            "then records where the actual 025-family winning stream landed (Tier/Rank/Percentile)."
        )

        # Choose date range based on available history
        _all_dates = pd.to_datetime(hist_wf["Draw Date"]).dt.date.unique().tolist()
        _all_dates = sorted([d for d in _all_dates if pd.notna(d)])
        if _all_dates:
            _min_d, _max_d = _all_dates[0], _all_dates[-1]
        else:
            _min_d, _max_d = datetime.date.today(), datetime.date.today()

        cA, cB, cC, cD = st.columns([1,1,1,1])
        with cA:
            tierA_pct = st.number_input("Tier A = top % by StreamScore", min_value=1, max_value=90, value=20, step=1, key="tierA_pct")
        with cB:
            tierB_pct = st.number_input("Tier B = next %", min_value=1, max_value=99, value=30, step=1, key="tierB_pct")
        with cC:
            wf_start = st.date_input("Start date", value=_min_d, min_value=_min_d, max_value=_max_d, key="wf_tier_start")
        with cD:
            wf_end = st.date_input("End date", value=_max_d, min_value=_min_d, max_value=_max_d, key="wf_tier_end")

        # Validate percents
        if tierA_pct + tierB_pct >= 100:
            st.error("Tier A% + Tier B% must be < 100 (Tier C is the remainder).")
        else:
            
            selection_mode = st.selectbox(
                "Play-50 selection strategy",
                options=[
                    "Percentile bins (winner-heavy zones)",
                    "Best historical winner-producing rank rows",
                ],
                index=0,
                key="wf_selection_mode",
            )
            cap_streams = st.number_input("Daily stream cap (play method)", min_value=1, max_value=500, value=50, step=1, key="wf_cap_streams")
            bin_size_pct = st.selectbox(
                "Percentile bin size (row-zones) — used only in Percentile-bins mode",
                options=[1, 2, 5, 10],
                index=2,
                key="wf_bin_size_pct",
            )
            fill_to_cap = st.checkbox("Fill to exact cap using best remaining streams", value=True, key="wf_fill_to_cap")
            cRun1, cRun2 = st.columns([1, 1])
            with cRun1:
                run_btn = st.button("Run Walk-forward Tier Study", type="primary", key="run_wf_tier_study")
            with cRun2:
                compare_btn = st.button("Run Gate ON vs OFF Comparison (NO9)", type="secondary", key="run_wf_tier_compare")
            # Cached prior WF outputs (keeps downloads stable across reruns and prevents NameError if UI renders cached views)
            _wf_cache = st.session_state.get("_wf_tier_cache", {}) if isinstance(st.session_state.get("_wf_tier_cache", {}), dict) else {}
            per_event = _wf_cache.get("per_event", pd.DataFrame())
            per_date  = _wf_cache.get("per_date", pd.DataFrame())
            per_stream = _wf_cache.get("per_stream", pd.DataFrame())
            wf_runs = st.session_state.get("_wf_tier_cache_runs", {}) if isinstance(st.session_state.get("_wf_tier_cache_runs", {}), dict) else {}


            wf_sig_base = (
                str(selection_mode),
                str(wf_start), str(wf_end),
                float(tierA_pct), float(tierB_pct),
                int(min_support), int(max_rules),
                json.dumps([{"feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": float(r.weight)} for r in base_rules], sort_keys=True, default=str),
                json.dumps([{"feature": r.feature, "op": r.op, "value": r.value, "pick": r.pick, "weight": float(r.weight)} for r in tie_rules], sort_keys=True, default=str),
                int(cap_streams), int(bin_size_pct), int(fill_to_cap),
            )

            def _wf_tier_compute_base(gate_flag: bool):
                gate_no9 = bool(gate_flag)
                per_stream_rows = []

                # Hit-events = rows where the RESULT is a 025-family member
                events = hist_wf.copy()
                events["Member"] = events["result"].apply(as_member)
                events = events.dropna(subset=["Member"]).copy()
                events["PlayDate"] = pd.to_datetime(events["Draw Date"]).dt.date

                # Restrict to date window
                events = events[(events["PlayDate"] >= wf_start) & (events["PlayDate"] <= wf_end)].copy()

                # No-cheat requirement: playlist on date D uses ONLY history strictly before D
                # Also: if you keep gate_no9 on, apply it to the SEED used for scoring (latest seed before D)

                if events.empty:
                    st.warning("No 025-family hit-events found in that date range.")
                else:
                    # Pre-group events by date for caching playlist build per date
                    events_by_date = {d: g.copy() for d, g in events.groupby("PlayDate")}
                    play_dates = sorted(events_by_date.keys())

                    prog = st.progress(0)
                    status = st.empty()

                    per_event_rows = []
                    per_date_rows = []

                    for i, d in enumerate(play_dates):
                        status.write(f"Building playlist as-of {d} ({i+1}/{len(play_dates)}) …")

                        hist_before = hist_wf[pd.to_datetime(hist_wf["Draw Date"]).dt.date < d].copy()
                        if hist_before.empty:
                            # no prior history — cannot score
                            # Snapshot per-stream-per-day ranked universe for 50-stream play-method analysis
                            try:
                                day_df = idx_map.reset_index().copy()
                                day_df["PlayDate"] = d
                                day_df["UniverseSize"] = int(len(idx_map))
                                day_df["PreGateUniverseSize"] = int(len(pre_gate_keys)) if 'pre_gate_keys' in locals() else int(len(idx_map))
                                day_df["GateExcluded"] = int(len(gated_out_keys)) if 'gated_out_keys' in locals() else 0
                                per_stream_rows.append(day_df)
                            except Exception:
                                pass

                            for _, ev in events_by_date[d].iterrows():
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": ev.get("stream"),
                "WinningStreamKey": canon_stream(ev.get("stream")),
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": None,
                                    "Percentile": None,
                                    "Tier": None,
                                    "FlipRec": None,
                                    "PlayPlan": None,
                                    "PlayMembers": None,
                                    "PlayCount": None,
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
                                    "Reason": "NO_HISTORY_BEFORE_PLAYDATE",
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                })
                            per_date_rows.append({
                                "PlayDate": d,
                                "UniverseSize": 0,
                                "TierA_Size": 0,
                                "TierB_Size": 0,
                                "TierC_Size": 0,
                                "HitEvents": int(len(events_by_date[d])),
                                "Covered_TierA": 0,
                                "Covered_TierAB": 0,
                            })
                            prog.progress(int((i+1)/len(play_dates)*100))
                            continue

                        # Build cadence metrics and latest-per-stream seeds as-of d
                        hist_before = compute_cadence_metrics(hist_before)
                        latest = hist_before.sort_values(["stream", "Draw Date"]).groupby("stream").tail(1).copy()
                        latest['StreamKey'] = latest['stream'].apply(canon_stream)
                        latest_all = latest.copy()  # pre-gate snapshot for diagnostics
                        # Map StreamKey -> seed (pre-gate) for proof columns in diagnostics
                        seed_by_key = dict(zip(
                            latest_all['StreamKey'].dropna().astype(str).tolist(),
                            latest_all['result'].astype(str).tolist()
                        ))
                        seed_has9_by_key = {k: ('9' in str(v)) for k, v in seed_by_key.items()}

                        latest = latest.rename(columns={"result": "seed_result", "Draw Date": "seed_date"})

                        pre_gate_keys = set(latest_all["StreamKey"].dropna().astype(str).tolist())
                        gated_out_keys = set()
                        if gate_no9:
                            latest = latest[~latest["seed_result"].astype(str).str.contains("9")].copy()
                            post_gate_keys = set(latest["StreamKey"].dropna().astype(str).tolist())
                            gated_out_keys = pre_gate_keys - post_gate_keys
                        rows = []
                        target_dow_wf = pd.Timestamp(d).day_name() if pd.notna(d) else ""
                        for _, r in latest.iterrows():
                            seed4 = r["seed_result"]
                            s = score_seed(seed4, base_rules, tie_rules)
                            feats = s["features"]
                            s_mined, meta_mined = apply_mined_member_layers(
                                seed4=seed4,
                                stream=r["stream"],
                                s_base=s,
                                feats=feats,
                                draws_since_last_025=r.get("draws_since_last_025", None),
                                rolling_025_30=r.get("rolling_025_30", None),
                                rolling_025_60=r.get("rolling_025_60", None),
                                target_dow=target_dow_wf,
                                tier=None,
                                mined_dfs=mined_dfs,
                                enable_eliminators=enable_elims,
                                enable_downranks=enable_1miss,
                                enable_locked_rescues=enable_rescues,
                                allow_guarded_rescues=allow_guarded,
                                enable_rulepack_top3_overrides=enable_rulepack_top3,
                            )
                            cadence_row = {
                                "draws_since_last_025": int(r.get("draws_since_last_025", 9999)),
                                "rolling_025_30": int(r.get("rolling_025_30", 0)),
                            }
                            ss = stream_score_row(
                                s, cadence_row,
                                w_match=w_match, w_gap=w_gap, w_roll=w_roll, w_since=w_since,
                                since_mode=since_mode,
                                cooldown_k=cooldown_k, cooldown_penalty=cooldown_penalty
                            )
                            rows.append({
                                "Stream": r["stream"],
                                "SeedDate": r["seed_date"].date() if pd.notna(r["seed_date"]) else None,
                                "Seed": seed4,
                                "StreamScore": float(ss),
                                "PredictedMember": s_mined["top1_final"],
                                "Top1Base": s["top1"],
                                "Top1": s_mined["top1_final"],
                                "Top2": s_mined["top2_final"],
                                "Top2Base": s["top2"],
                                "Top3": s_mined["top3_final"],
                                "Top3Base": s["top3"],
                                "Top1Score": max(s["scores"].values()),
                                "BaseGap(#1-#2)": float(s["base_gap"]),
                                "TieFired": int(s["tie_fired"]),
                                "DeadTie": int(s["dead_tie"]),
                                "ForcedPick": s["forced_pick"] if s.get("forced_pick") else "",
                                "ElimFired": ",".join(meta_mined.get("elim_fired", [])),
                                "DownrankFired": meta_mined.get("downrank_fired") or "",
                                "RescueFired": meta_mined.get("rescue_fired") or "",
                                "RulepackTop3Fired": meta_mined.get("rulepack_top3_fired") or "",
                                "CadenceRolling025_30": int(cadence_row.get("rolling_025_30", 0)),
                                "CadenceDrawsSinceLast025": int(cadence_row.get("draws_since_last_025", 9999)),
                                "SinceLast025(draws)": int(cadence_row.get("draws_since_last_025", 9999)),
                                "Rolling025_30": int(cadence_row.get("rolling_025_30", 0)),
                                "SeedSum": feats.get("seed_sum"),
                                "SeedSpread": feats.get("seed_spread"),
                                "SeedAbsDiff": feats.get("seed_absdiff_sum", feats.get("seed_pairwise_absdiff_sum", 0)),
                                "WorstPair025": feats.get("seed_has_worstpair_025"),
                            })

                        ranked = pd.DataFrame(rows)
                        if ranked.empty:
                            uni_n = 0
                            # record as missing universe
                            for _, ev in events_by_date[d].iterrows():
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": ev.get("stream"),
                "WinningStreamKey": canon_stream(ev.get("stream")),
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": None,
                                    "Percentile": None,
                                    "Tier": None,
                                    "FlipRec": None,
                                    "PlayPlan": None,
                                    "PlayMembers": None,
                                    "PlayCount": None,
                                    "PlayMember": None,
                                    "UniverseSize": 0,
                                    "InUniverse": 0,
                                    "Reason": "UNIVERSE_EMPTY",
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                })
                            per_date_rows.append({
                                "PlayDate": d,
                                "UniverseSize": 0,
                                "TierA_Size": 0,
                                "TierB_Size": 0,
                                "TierC_Size": 0,
                                "HitEvents": int(len(events_by_date[d])),
                                "Covered_TierA": 0,
                                "Covered_TierAB": 0,
                            })
                            prog.progress(int((i+1)/len(play_dates)*100))
                            continue

                        ranked = ranked.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=False).reset_index(drop=True)
                        ranked.insert(0, "Rank", np.arange(1, len(ranked)+1))

                        # FlipRec + PlayMember (same rule as LIVE)
                        ranked["FlipRec"] = ((ranked["BaseGap(#1-#2)"] == 0) & (ranked["TieFired"] == 1) & (ranked["DeadTie"] == 0)).astype(int)
                        # Legacy single-member suggestion (kept): FlipRec means "prefer TOP2"
                        ranked["PlayMember"] = np.where(ranked["FlipRec"] == 1, ranked["Top2"], ranked["PredictedMember"])
                        # NEW default play policy (requested): when TieFired (FlipRec=1), play TOP1 + TOP2
                        ranked["PlayPlan"] = np.where(ranked["FlipRec"] == 1, "TOP1+TOP2", "TOP1")
                        ranked["PlayMembers"] = np.where(
                            ranked["FlipRec"] == 1,
                            ranked["PredictedMember"].astype(str) + " + " + ranked["Top2"].astype(str),
                            ranked["PredictedMember"].astype(str),
                        )
                        ranked["PlayCount"] = np.where(ranked["FlipRec"] == 1, 2, 1).astype(int)

                        # Percentile rank (1 = best). Example: Rank 1 of 80 => 1/80 = 0.0125
                        uni_n = int(len(ranked))
                        ranked["Percentile"] = ranked["Rank"] / uni_n

                        # Tier cutoffs by StreamScore percentiles (requested behavior)
                        # Tier A: top tierA_pct% by StreamScore, Tier B: next tierB_pct%, Tier C: rest
                        qA = ranked["StreamScore"].quantile(1.0 - tierA_pct/100.0)
                        qB = ranked["StreamScore"].quantile(1.0 - (tierA_pct + tierB_pct)/100.0)

                        def _tier_by_score(x):
                            if x >= qA:
                                return "A"
                            if x >= qB:
                                return "B"
                            return "C"

                        ranked["Tier"] = ranked["StreamScore"].apply(_tier_by_score)

                        # Tier sizes (for cost / plays per day)
                        tierA_size = int((ranked["Tier"] == "A").sum())
                        tierB_size = int((ranked["Tier"] == "B").sum())
                        tierC_size = int((ranked["Tier"] == "C").sum())

                        # Evaluate all hit-events on date d
                        covered_A = 0
                        covered_AB = 0

                        ranked = ranked.copy()
                        if "StreamKey" not in ranked.columns:
                            ranked["StreamKey"] = ranked["Stream"].apply(canon_stream)
                        # Precompute key-sets/mappings for out-of-universe diagnostics (avoid NameError)
                        rank_keys = set(ranked["StreamKey"].dropna().astype(str).tolist())
                        rank_display_by_key = {str(k): v for k, v in zip(ranked["StreamKey"], ranked["Stream"]) if pd.notna(k)}
                        latest_display_by_key = {str(k): v for k, v in zip(latest_all["StreamKey"], latest_all["stream"]) if pd.notna(k)}
                        all_keys = set(latest_all["StreamKey"].dropna().astype(str).tolist())
                                                # Reverse-order rank map (sanity-check: is score direction flipped?)
                        rev = ranked.sort_values(["StreamScore", "Top1Score", "BaseGap(#1-#2)"], ascending=True).reset_index(drop=True).copy()
                        rev["Rank_rev"] = np.arange(1, len(rev)+1)
                        rev["Percentile_rev"] = rev["Rank_rev"] / float(len(rev)) if len(rev) else None
                        rev_map = rev.set_index("StreamKey")[["Rank_rev", "Percentile_rev"]]
                        desired_cols = [
                            "Rank", "Percentile", "Tier", "FlipRec", "PlayPlan", "PlayMembers", "PlayCount", "PlayMember",
                            "StreamScore", "PredictedMember", "Top2", "Top3",
                            "Top1Score", "BaseGap(#1-#2)", "TieFired", "DeadTie",
                            "CadenceRolling025_30", "CadenceDrawsSinceLast025"
                        ]
                        idx_map = ranked.set_index("StreamKey").reindex(columns=desired_cols).join(rev_map, how="left")

                        # Snapshot per-stream-per-day ranked universe (for Play-50 reproduction)
                        try:
                            day_df = idx_map.reset_index().copy()
                            # Attach display label + required member columns for downstream selection/capture
                            day_df["Stream"] = day_df["StreamKey"].astype(str).map(rank_display_by_key)
                            if "PredictedMember" in day_df.columns and "Top1" not in day_df.columns:
                                day_df["Top1"] = day_df["PredictedMember"].astype(str)
                            if "PlayDate" not in day_df.columns:
                                day_df["PlayDate"] = d
                            day_df["UniverseSize"] = int(uni_n)
                            day_df["PreGateUniverseSize"] = int(len(pre_gate_keys)) if 'pre_gate_keys' in locals() else int(uni_n)
                            day_df["GateExcluded"] = int(len(gated_out_keys)) if 'gated_out_keys' in locals() else 0
                            per_stream_rows.append(day_df)
                        except Exception:
                            pass


                        # --- Component-only rank maps (LAB diagnostics) ---
                        def _make_rank_map(_df, _col, ascending=False):
                            """Return StreamKey -> 1-based rank when sorting by _col."""
                            if _col not in _df.columns:
                                return {}
                            tmp = _df[["StreamKey", _col]].copy()
                            tmp["StreamKey"] = tmp["StreamKey"].astype(str)
                            tmp[_col] = pd.to_numeric(tmp[_col], errors="coerce")
                            tmp = tmp.sort_values([_col, "StreamKey"], ascending=[ascending, True], na_position="last").reset_index(drop=True)
                            tmp["__rank"] = np.arange(1, len(tmp) + 1)
                            return dict(zip(tmp["StreamKey"].tolist(), tmp["__rank"].astype(int).tolist()))

                        rankmap_top1 = _make_rank_map(ranked, "Top1Score", ascending=False)
                        rankmap_gap = _make_rank_map(ranked, "BaseGap(#1-#2)", ascending=False)
                        rankmap_roll = _make_rank_map(ranked, "CadenceRolling025_30", ascending=False)
                        rankmap_since_due = _make_rank_map(ranked, "CadenceDrawsSinceLast025", ascending=False)
                        rankmap_since_recent = _make_rank_map(ranked, "CadenceDrawsSinceLast025", ascending=True)
                        # SinceBlend (LAB-only): favor extremes in DrawsSinceLast025 (very recent OR very due/never)
                        # Use the better (lower) percentile from asc vs desc orderings to score each stream.
                        stream_keys = ranked["StreamKey"].astype(str).tolist()
                        if uni_n <= 1:
                            rankmap_since_blend = {k: 1 for k in stream_keys}
                        else:
                            _pct_recent = {k: (rankmap_since_recent.get(k, uni_n) / uni_n) for k in stream_keys}
                            _pct_due = {k: (rankmap_since_due.get(k, uni_n) / uni_n) for k in stream_keys}
                            _blend_score = {k: 1.0 - min(_pct_recent[k], _pct_due[k]) for k in stream_keys}
                            _blend_order = sorted(stream_keys, key=lambda k: (-_blend_score[k], k))
                            rankmap_since_blend = {k: i + 1 for i, k in enumerate(_blend_order)}

                        # Capture the day's #1 stream (for driver-audit panels)
                        _top = ranked.iloc[0]
                        top_stream = _top["Stream"]
                        top_stream_key = str(_top.get("StreamKey")) if pd.notna(_top.get("StreamKey")) else canon_stream(top_stream)
                        top_streamscore = float(_top.get("StreamScore"))
                        top_top1score = float(_top.get("Top1Score"))
                        top_basegap = float(_top.get("BaseGap(#1-#2)"))
                        top_roll30 = int(_top.get("CadenceRolling025_30"))
                        top_since = int(_top.get("CadenceDrawsSinceLast025"))
                        top_tiefired = int(_top.get("TieFired"))
                        top_deadtie = int(_top.get("DeadTie"))

                        # Per-day tie/granularity metrics (LAB diagnostics)
                        day_unique_streamscore = int(ranked["StreamScore"].nunique(dropna=True)) if "StreamScore" in ranked.columns else None
                        day_unique_top1score = int(ranked["Top1Score"].nunique(dropna=True)) if "Top1Score" in ranked.columns else None
                        _top_ss_raw = _top.get("StreamScore")
                        _top_t1_raw = _top.get("Top1Score")
                        _top_gap_raw = _top.get("BaseGap(#1-#2)")
                        day_top_score_tie = int((ranked["StreamScore"] == _top_ss_raw).sum()) if "StreamScore" in ranked.columns else None
                        if all(c in ranked.columns for c in ["StreamScore", "Top1Score", "BaseGap(#1-#2)"]):
                            day_top_tuple_tie = int(((ranked["StreamScore"] == _top_ss_raw) & (ranked["Top1Score"] == _top_t1_raw) & (ranked["BaseGap(#1-#2)"] == _top_gap_raw)).sum())
                        else:
                            day_top_tuple_tie = None

                        for _, ev in events_by_date[d].iterrows():
                            stream_raw = ev.get("Stream") if ev.get("Stream") is not None else ev.get("stream")
                            stream_key = canon_stream(stream_raw)
                            if stream_key in idx_map.index:
                                rr = idx_map.loc[stream_key]
                                t = rr["Tier"]
                                prev_seed = seed_by_key.get(stream_key) if stream_key else None
                                prev_seed_has9 = seed_has9_by_key.get(stream_key) if stream_key else None
                                in_uni = 1
                                if t == "A":
                                    covered_A += 1
                                    covered_AB += 1
                                elif t == "B":
                                    covered_AB += 1
                                per_event_rows.append({
                                    "PlayDate": d,
                                    "WinningStream": stream_raw,
                                    "WinningStreamKey": stream_key,
                                    "WinningResult": ev.get("result"),
                                    "WinningMember": ev.get("Member"),
                                    "Rank": int(rr["Rank"]),
                                    "Percentile": float(rr["Percentile"]),
                                    "Tier": t,
                                    "FlipRec": int(rr["FlipRec"]),
                                    "PlayPlan": rr.get("PlayPlan"),
                                    "PlayMembers": rr.get("PlayMembers"),
                                    "PlayCount": int(rr.get("PlayCount")) if pd.notna(rr.get("PlayCount")) else None,
                                    "PlayMember": rr["PlayMember"],
                                    "Reason": "",
                                    "PrevSeed": prev_seed,
                                    "PrevSeedHas9": prev_seed_has9,
                                    "WinnerStreamScore": float(rr["StreamScore"]),
                                    "WinnerTop1Score": float(rr["Top1Score"]),
                                    "WinnerBaseGap": float(rr["BaseGap(#1-#2)"]),
                                    "WinnerRoll025_30": int(rr["CadenceRolling025_30"]),
                                    "WinnerDrawsSinceLast025": int(rr["CadenceDrawsSinceLast025"]),
                                    "WinnerTieFired": int(rr["TieFired"]),
                                    "WinnerDeadTie": int(rr["DeadTie"]),
                                    "Rank_Top1Score": int(rankmap_top1.get(stream_key)) if rankmap_top1 and (stream_key in rankmap_top1) else None,
                                    "Rank_BaseGap": int(rankmap_gap.get(stream_key)) if rankmap_gap and (stream_key in rankmap_gap) else None,
                                    "Rank_Roll025_30": int(rankmap_roll.get(stream_key)) if rankmap_roll and (stream_key in rankmap_roll) else None,
                                    "Rank_Since_due": int(rankmap_since_due.get(stream_key)) if rankmap_since_due and (stream_key in rankmap_since_due) else None,
                                    "Rank_Since_recent": int(rankmap_since_recent.get(stream_key)) if rankmap_since_recent and (stream_key in rankmap_since_recent) else None,
                                    "Rank_Since_blend": int(rankmap_since_blend.get(stream_key)) if rankmap_since_blend and (stream_key in rankmap_since_blend) else None,
                                    "Pct_Top1Score": (float(rankmap_top1.get(stream_key)) / float(uni_n)) if rankmap_top1 and (stream_key in rankmap_top1) and uni_n else None,
                                    "Pct_BaseGap": (float(rankmap_gap.get(stream_key)) / float(uni_n)) if rankmap_gap and (stream_key in rankmap_gap) and uni_n else None,
                                    "Pct_Roll025_30": (float(rankmap_roll.get(stream_key)) / float(uni_n)) if rankmap_roll and (stream_key in rankmap_roll) and uni_n else None,
                                    "Pct_Since_due": (float(rankmap_since_due.get(stream_key)) / float(uni_n)) if rankmap_since_due and (stream_key in rankmap_since_due) and uni_n else None,
                                    "Pct_Since_recent": (float(rankmap_since_recent.get(stream_key)) / float(uni_n)) if rankmap_since_recent and (stream_key in rankmap_since_recent) and uni_n else None,
                                    "Pct_Since_blend": (float(rankmap_since_blend.get(stream_key)) / float(uni_n)) if rankmap_since_blend and (stream_key in rankmap_since_blend) and uni_n else None,
                                    "Rank_rev": (int(rr["Rank_rev"]) if pd.notna(rr.get("Rank_rev")) else None),
                                    "Percentile_rev": (float(rr["Percentile_rev"]) if pd.notna(rr.get("Percentile_rev")) else None),
                                    "BestMatchStream": None,
                                    "MatchScore": None,
                                    "UniverseSize": uni_n,
                                    "InUniverse": in_uni,
                                })
                            else:
                                # Out-of-universe diagnostics
                                if not stream_key:
                                    reason = 'EMPTY_STREAM'
                                elif (gate_no9 and (stream_key in gated_out_keys)):
                                    reason = 'GATED_OUT_SEED_HAS_9'
                                elif (all_keys and (stream_key not in all_keys)):
                                    reason = 'NO_HISTORY_BEFORE_PLAYDATE'
                                elif not rank_keys:
                                    reason = 'UNIVERSE_EMPTY'
                                else:
                                    reason = 'NOT_RANKED'
                                best_match_key = None
                                best_match_stream = None
                                match_score = None
                                if stream_key and rank_keys:
                                    cand = difflib.get_close_matches(stream_key, list(rank_keys), n=1, cutoff=0.60)
                                    if cand:
                                        best_match_key = cand[0]
                                        best_match_stream = rank_display_by_key.get(best_match_key) or latest_display_by_key.get(best_match_key)
                                        match_score = round(difflib.SequenceMatcher(a=stream_key, b=best_match_key).ratio(), 3)
                                prev_seed = seed_by_key.get(stream_key) if stream_key else None
                                prev_seed_has9 = seed_has9_by_key.get(stream_key) if stream_key else None
                                per_event_rows.append({
                                    'PlayDate': d,
                                    'WinningStream': stream_raw,
                                    'WinningStreamKey': stream_key,
                                    'WinningResult': ev.get('result'),
                                    'WinningMember': ev.get('Member'),
                                    'PrevSeed': prev_seed,
                                    'PrevSeedHas9': prev_seed_has9,
                                    'WinnerStreamScore': None,
                                    'WinnerTop1Score': None,
                                    'WinnerBaseGap': None,
                                    'WinnerRoll025_30': None,
                                    'WinnerDrawsSinceLast025': None,
                                    'WinnerTieFired': None,
                                    'WinnerDeadTie': None,
                                    'Rank_Top1Score': None,
                                    'Rank_BaseGap': None,
                                    'Rank_Roll025_30': None,
                                    'Rank_Since_due': None,
                                    'Rank_Since_recent': None,
                                    'Pct_Top1Score': None,
                                    'Pct_BaseGap': None,
                                    'Pct_Roll025_30': None,
                                    'Pct_Since_due': None,
                                    'Pct_Since_recent': None,
                                    'Rank_rev': None,
                                    'Percentile_rev': None,
                                    'Rank': None,
                                    'Percentile': None,
                                    'Tier': None,
                                    'FlipRec': None,
                                    'PlayMember': None,
                                    'UniverseSize': uni_n,
                                    'InUniverse': 0,
                                    'Reason': reason,
                                    'BestMatchStream': best_match_stream,
                                    'MatchScore': match_score,
                                })

                        per_date_rows.append({
                            "PlayDate": d,
                            "UniverseSize": uni_n,
                            "TopStream": top_stream,
                            "TopStreamKey": top_stream_key,
                            "TopStreamScore": top_streamscore,
                            "TopTop1Score": top_top1score,
                            "TopBaseGap": top_basegap,
                            "TopRoll025_30": top_roll30,
                            "TopDrawsSinceLast025": top_since,
                            "TopTieFired": top_tiefired,
                            "TopDeadTie": top_deadtie,
                            "UniqueStreamScore": day_unique_streamscore,
                            "UniqueTop1Score": day_unique_top1score,
                            "TopStreamScoreTieCount": day_top_score_tie,
                            "TopTopTupleTieCount": day_top_tuple_tie,
                            "PreGateUniverseSize": int(len(pre_gate_keys)) if "pre_gate_keys" in locals() else None,
                            "GatedOutStreams": int(len(gated_out_keys)) if "gated_out_keys" in locals() else 0,
                            "TierA_Size": tierA_size,
                            "TierB_Size": tierB_size,
                            "TierC_Size": tierC_size,
                            "HitEvents": int(len(events_by_date[d])),
                            "Covered_TierA": int(covered_A),
                            "Covered_TierAB": int(covered_AB),
                        })

                        prog.progress(int((i+1)/len(play_dates)*100))

                    status.empty()

                    per_event = pd.DataFrame(per_event_rows)
                    per_date = pd.DataFrame(per_date_rows)
                    # ---- Keep Pick-4 digits as 4-char text everywhere (prevents "25" vs "0025" confusion) ----
                    def _pad4(v):
                        if pd.isna(v):
                            return ""
                        s = str(v).strip()
                        # keep only digits (defensive)
                        s = re.sub(r"[^0-9]", "", s)
                        if s == "":
                            return ""
                        return s.zfill(4)

                    def _pad_member_list(v):
                        if pd.isna(v):
                            return ""
                        parts = [normalize_member_id(p.strip()) for p in str(v).split("+") if str(p).strip()]
                        parts = [p for p in parts if p]
                        return " + ".join(parts)

                    for _df in (per_event, per_date):
                        for _c in ["PrevSeed", "WinningResult", "PlayMember", "WinningMember"]:
                            if _c in _df.columns:
                                _df[_c] = _df[_c].apply(_pad4)
                                # Excel-safe text version (leading apostrophe forces text)
                                _df[_c + "_text"] = _df[_c].apply(lambda x: ("'" + x) if x else "")
                        if "PlayMembers" in _df.columns:
                            _df["PlayMembers"] = _df["PlayMembers"].apply(_pad_member_list)
                            _df["PlayMembers_text"] = _df["PlayMembers"].apply(lambda x: ("'" + x) if x else "")
                    # Persist results so downloads don't require re-running after a Streamlit rerun (e.g., clicking download).

                # Build per-stream-per-day table
                if per_stream_rows:
                    per_stream = pd.concat(per_stream_rows, ignore_index=True)
                else:
                    per_stream = pd.DataFrame()
                if not per_stream.empty:
                    per_stream["GateNO9"] = int(gate_no9)
                    per_stream["GateMode"] = "GateON" if gate_no9 else "GateOFF"
                # Ensure per_event has a stable StreamKey column for merges
                if not per_event.empty and "WinningStreamKey" in per_event.columns and "StreamKey" not in per_event.columns:
                    per_event["StreamKey"] = per_event["WinningStreamKey"]
                # Tag outputs with gate mode
                for _df in [per_event, per_date]:
                    if _df is not None and not _df.empty:
                        _df["GateNO9"] = int(gate_no9)
                        _df["GateMode"] = "GateON" if gate_no9 else "GateOFF"
                return per_event, per_date, per_stream

            if run_btn or compare_btn:
                wf_runs = {}
                selected_bins = []

                def _build_run_payload(label: str,
                                       pe_aug: pd.DataFrame,
                                       pd_aug: pd.DataFrame,
                                       ps_sel: pd.DataFrame,
                                       selection_mode: str,
                                       selected_bins: list[str],
                                       selected_ranks: list[int]):
                    sig = wf_sig_base + (label,)
                    return {
                        "sig": sig,
                        "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "gate_mode": label,
                        "selection_mode": str(selection_mode),
                        "cap_streams": int(cap_streams),
                        "bin_size_pct": int(bin_size_pct),
                        "selected_bins": list(selected_bins) if selected_bins else [],
                        "selected_ranks": [int(x) for x in (selected_ranks or [])],
                        "per_event": pe_aug,
                        "per_date": pd_aug,
                        "per_stream": ps_sel,
                    }


                
                if compare_btn:
                    # Run Gate ON first to derive selection rule, then apply same rule to Gate OFF
                    pe_on, pd_on, ps_on = _wf_tier_compute_base(True)

                    selected_bins: list[str] = []
                    selected_ranks: list[int] = []

                    if str(selection_mode).startswith("Percentile"):
                        selected_bins = _wf_suggest_bins_greedy(
                            pe_on, ps_on,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                        )
                        ps_on_sel = _wf_select_cap50_by_bins(
                            ps_on, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        selected_ranks = _wf_suggest_rank_rows_topk(pe_on, cap_streams=int(cap_streams))
                        ps_on_sel = _wf_select_cap50_by_ranks(
                            ps_on, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_on_sel is not None and not ps_on_sel.empty:
                        ps_on_sel["SelectionMode"] = str(selection_mode)

                    pe_on_aug, pd_on_aug = _wf_augment_event_date_with_play50(pe_on, pd_on, ps_on_sel)
                    if not pe_on_aug.empty:
                        pe_on_aug["SelectionMode"] = str(selection_mode)
                    if pd_on_aug is not None and not pd_on_aug.empty:
                        pd_on_aug["SelectionMode"] = str(selection_mode)

                    wf_runs["GateON"] = _build_run_payload(
                        "GateON",
                        pe_on_aug, pd_on_aug, ps_on_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )

                    pe_off, pd_off, ps_off = _wf_tier_compute_base(False)

                    if str(selection_mode).startswith("Percentile"):
                        ps_off_sel = _wf_select_cap50_by_bins(
                            ps_off, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        ps_off_sel = _wf_select_cap50_by_ranks(
                            ps_off, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_off_sel is not None and not ps_off_sel.empty:
                        ps_off_sel["SelectionMode"] = str(selection_mode)

                    pe_off_aug, pd_off_aug = _wf_augment_event_date_with_play50(pe_off, pd_off, ps_off_sel)
                    if not pe_off_aug.empty:
                        pe_off_aug["SelectionMode"] = str(selection_mode)
                    if pd_off_aug is not None and not pd_off_aug.empty:
                        pd_off_aug["SelectionMode"] = str(selection_mode)

                    wf_runs["GateOFF"] = _build_run_payload(
                        "GateOFF",
                        pe_off_aug, pd_off_aug, ps_off_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )


                    # Active view defaults to GateON for diagnostics below
                    per_event = pe_on_aug
                    per_date = pd_on_aug
                    per_stream = ps_on_sel

                    comp = []
                    for lbl in ["GateON", "GateOFF"]:
                        pe_x = wf_runs[lbl]["per_event"]
                        pd_x = wf_runs[lbl]["per_date"]
                        total_ev = int(len(pe_x))
                        in_uni = int(pe_x.get("InUniverse", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        covered_ab = int(pe_x.get("Covered_AB", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        selected_hits = int(pe_x.get("Selected50", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        captured = int(pe_x.get("CapturedMember", pd.Series(dtype=int)).fillna(0).astype(int).sum()) if total_ev else 0
                        comp.append({
                            "GateMode": lbl,
                            "HitEvents": total_ev,
                            "InUniverse": in_uni,
                            "Covered_AB": covered_ab,
                            "Selected50_HitStreams": selected_hits,
                            "CapturedMembers": captured,
                            "CaptureRate(Captured/InUniverse)": (captured / in_uni) if in_uni else 0.0,
                            "SelectedBins": ", ".join(selected_bins) if selected_bins else "",
                        })

                    st.subheader("Gate ON vs OFF (NO9) comparison")
                    st.dataframe(pd.DataFrame(comp))

                else:
                    
                    gate_flag = bool(gate_no9)
                    label = "GateON" if gate_flag else "GateOFF"
                    pe_1, pd_1, ps_1 = _wf_tier_compute_base(gate_flag)

                    selected_bins: list[str] = []
                    selected_ranks: list[int] = []

                    if str(selection_mode).startswith("Percentile"):
                        selected_bins = _wf_suggest_bins_greedy(
                            pe_1, ps_1,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                        )
                        ps_sel = _wf_select_cap50_by_bins(
                            ps_1, selected_bins,
                            cap_streams=int(cap_streams),
                            bin_size_pct=int(bin_size_pct),
                            fill_to_cap=bool(fill_to_cap),
                        )
                    else:
                        selected_ranks = _wf_suggest_rank_rows_topk(pe_1, cap_streams=int(cap_streams))
                        ps_sel = _wf_select_cap50_by_ranks(
                            ps_1, selected_ranks,
                            cap_streams=int(cap_streams),
                            fill_to_cap=bool(fill_to_cap),
                        )

                    if ps_sel is not None and not ps_sel.empty:
                        ps_sel["SelectionMode"] = str(selection_mode)

                    pe_aug, pd_aug = _wf_augment_event_date_with_play50(pe_1, pd_1, ps_sel)
                    if not pe_aug.empty:
                        pe_aug["SelectionMode"] = str(selection_mode)
                    if pd_aug is not None and not pd_aug.empty:
                        pd_aug["SelectionMode"] = str(selection_mode)

                    wf_runs[label] = _build_run_payload(
                        label,
                        pe_aug, pd_aug, ps_sel,
                        selection_mode=str(selection_mode),
                        selected_bins=selected_bins,
                        selected_ranks=selected_ranks,
                    )

                    per_event = pe_aug
                    per_date = pd_aug
                    per_stream = ps_sel

                st.session_state["_wf_tier_cache_runs"] = wf_runs
                st.session_state["_wf_tier_cache"] = {
                    "sig": wf_sig_base + (("GateON" if bool(gate_no9) else "GateOFF") if not compare_btn else "GateON",),
                    "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "per_event": per_event,
                    "per_date": per_date,
                    "per_stream": per_stream,
                    "cap_streams": int(cap_streams),
                    "bin_size_pct": int(bin_size_pct),
                    "selection_mode": str(selection_mode),
                    "selected_bins": list(selected_bins) if "selected_bins" in locals() else [],
                    "selected_ranks": [int(x) for x in (selected_ranks or [])] if "selected_ranks" in locals() else [],
                    "export_schema": "wf_per_event_v2",
                }


                total_events = int(len(per_event))
                in_universe = int(per_event["InUniverse"].sum())
                covered_A = int((per_event["Tier"] == "A").sum())
                covered_AB = int(per_event["Tier"].isin(["A","B"]).sum())

                st.write("### Summary")
                st.write(f"Hit-events analyzed: **{total_events:,}**  |  Winner streams present in universe: **{in_universe:,}**")
                st.write(f"Tier A coverage: **{covered_A}/{total_events} = {covered_A/total_events*100:.2f}%**")
                st.write(f"Tier A+B coverage: **{covered_AB}/{total_events} = {covered_AB/total_events*100:.2f}%**")

                # --- Gate impact (NO9) ---
                gate_excluded = int((per_event['Reason'] == 'GATED_OUT_SEED_HAS_9').sum()) if 'Reason' in per_event.columns else 0
                no_history = int((per_event['Reason'] == 'NO_HISTORY_BEFORE_PLAYDATE').sum()) if 'Reason' in per_event.columns else 0
                st.write('### Gate impact (NO9)')
                if total_events > 0:
                    st.write(f"Winner events excluded by gate (seed has 9): **{gate_excluded:,}/{total_events:,} = {gate_excluded/total_events*100:.2f}%**")
                    st.write(f"Winner events with no prior history (first-day/new stream): **{no_history:,}/{total_events:,} = {no_history/total_events*100:.2f}%**")
                if 'GatedOutStreams' in per_date.columns and len(per_date) > 0:
                    avg_removed = float(per_date['GatedOutStreams'].mean())
                    med_removed = float(per_date['GatedOutStreams'].median())
                    avg_pre = float(per_date['PreGateUniverseSize'].mean()) if 'PreGateUniverseSize' in per_date.columns else float('nan')
                    avg_post = float(per_date['UniverseSize'].mean()) if 'UniverseSize' in per_date.columns else float('nan')
                    st.write(f"Avg streams removed per day by gate: **{avg_removed:.2f}** (median **{med_removed:.0f}**) | Avg universe size: pre-gate **{avg_pre:.1f}**, post-gate **{avg_post:.1f}**")

                # --- Ranking performance within playable universe (non-gated) ---
                st.write('### Ranking performance within playable universe')
                playable = per_event[(per_event.get('Reason','') == '')].copy() if total_events > 0 else per_event.iloc[0:0].copy()
                n_playable = int(len(playable))
                if n_playable == 0:
                    st.write('No playable (non-gated) winner events in the selected window.')
                else:
                    top1 = int((playable['Rank'] == 1).sum())
                    top2 = int((playable['Rank'] <= 2).sum())
                    top3 = int((playable['Rank'] <= 3).sum())
                    avg_rank = float(playable['Rank'].mean())
                    med_rank = float(playable['Rank'].median())
                    p90_rank = float(playable['Rank'].quantile(0.90))
                    st.write(f"Playable winner events: **{n_playable:,}** | Top1: **{top1}/{n_playable} = {top1/n_playable*100:.2f}%** | Top2: **{top2}/{n_playable} = {top2/n_playable*100:.2f}%** | Top3: **{top3}/{n_playable} = {top3/n_playable*100:.2f}%**")
                    st.write(f"Rank stats (playable winners): avg **{avg_rank:.2f}**, median **{med_rank:.0f}**, 90th pct **{p90_rank:.0f}**")


                st.markdown("---")
                st.subheader("Model sanity + lift + driver audit (LAB-only)")

                # 1) Direction sanity-check: are we ranking the correct end of StreamScore?
                st.write("### 1) Direction sanity-check (normal vs reversed StreamScore ordering)")
                if n_playable == 0:
                    st.info("No playable winner events available for sanity-check in this window.")
                else:
                    # Normal
                    _n = playable.copy()
                    _n = _n.dropna(subset=["Rank"])
                    top1_n = int((_n["Rank"] == 1).sum())
                    top2_n = int((_n["Rank"] <= 2).sum())
                    top3_n = int((_n["Rank"] <= 3).sum())
                    avg_n = float(_n["Rank"].mean())
                    med_n = float(_n["Rank"].median())
                    p90_n = float(_n["Rank"].quantile(0.90))

                    # Reversed
                    _r = playable.copy()
                    _r = _r.dropna(subset=["Rank_rev"])
                    top1_r = int((_r["Rank_rev"] == 1).sum())
                    top2_r = int((_r["Rank_rev"] <= 2).sum())
                    top3_r = int((_r["Rank_rev"] <= 3).sum())
                    avg_r = float(_r["Rank_rev"].mean())
                    med_r = float(_r["Rank_rev"].median())
                    p90_r = float(_r["Rank_rev"].quantile(0.90))

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Normal (higher StreamScore = better)**")
                        st.write(f"Top1: **{top1_n}/{n_playable} = {top1_n/n_playable*100:.2f}%**")
                        st.write(f"Top2: **{top2_n}/{n_playable} = {top2_n/n_playable*100:.2f}%**")
                        st.write(f"Top3: **{top3_n}/{n_playable} = {top3_n/n_playable*100:.2f}%**")
                        st.write(f"Rank stats: avg **{avg_n:.2f}**, median **{med_n:.0f}**, p90 **{p90_n:.0f}**")
                    with c2:
                        st.write("**Reversed (lower StreamScore = better)**")
                        st.write(f"Top1: **{top1_r}/{n_playable} = {top1_r/n_playable*100:.2f}%**")
                        st.write(f"Top2: **{top2_r}/{n_playable} = {top2_r/n_playable*100:.2f}%**")
                        st.write(f"Top3: **{top3_r}/{n_playable} = {top3_r/n_playable*100:.2f}%**")
                        st.write(f"Rank stats: avg **{avg_r:.2f}**, median **{med_r:.0f}**, p90 **{p90_r:.0f}**")

                # 2) Lift check: where do playable winners land by StreamScore percentile?
                st.write("### 2) Score lift check (winner placement by StreamScore percentile)")
                if n_playable == 0:
                    st.info("No playable winner events available for lift check in this window.")
                else:
                    def _bin_from_pct(p):
                        if pd.isna(p):
                            return None
                        if p <= 0.20:
                            return "Top 20%"
                        if p <= 0.50:
                            return "20–50%"
                        if p <= 0.80:
                            return "50–80%"
                        return "Bottom 20%"

                    lift = playable.copy()
                    lift["PctBin_Normal"] = lift["Percentile"].apply(_bin_from_pct)
                    lift["PctBin_Reversed"] = lift["Percentile_rev"].apply(_bin_from_pct) if "Percentile_rev" in lift.columns else None

                    def _summ_bin(colname):
                        vc = lift[colname].value_counts(dropna=False)
                        order = ["Top 20%", "20–50%", "50–80%", "Bottom 20%"]
                        out = []
                        for k in order:
                            c = int(vc.get(k, 0))
                            out.append({"Bin": k, "Count": c, "Pct": (c / n_playable * 100.0)})
                        return pd.DataFrame(out)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Normal ordering**")
                        st.dataframe(_summ_bin("PctBin_Normal"), use_container_width=True, height=190)
                    with c2:
                        st.write("**Reversed ordering**")
                        if "Percentile_rev" in lift.columns:
                            st.dataframe(_summ_bin("PctBin_Reversed"), use_container_width=True, height=190)
                        else:
                            st.info("Reversed percentiles not available (this indicates Rank_rev was not recorded).")

                    # Expected random baseline (for context)
                    st.caption("Random baseline (no signal): roughly 20% / 30% / 30% / 20% across the bins.")

                # 3) Driver audit: compare #1 stream vs winner stream on miss events
                st.write("### 3) Driver audit (why the day’s #1 stream beat the winner on miss events)")
                if n_playable == 0:
                    st.info("No playable winner events available for driver audit in this window.")
                else:
                    bad_rank_threshold = st.number_input(
                        "Define a miss as Winner Rank >",
                        min_value=1,
                        max_value=500,
                        value=25,
                        step=1,
                        key="bad_rank_threshold_driver_audit",
                    )

                    miss = playable[(playable["Rank"].notna()) & (playable["Rank"] > bad_rank_threshold)].copy()
                    if miss.empty:
                        st.success("No miss events under this threshold in the selected window.")
                    else:
                        # Join in the day’s #1 stream details from per-date summary
                        join_cols = [
                            "PlayDate", "UniverseSize",
                            "TopStream", "TopStreamKey", "TopStreamScore",
                            "TopTop1Score", "TopBaseGap", "TopRoll025_30", "TopDrawsSinceLast025",
                            "TopTieFired", "TopDeadTie"
                        ]
                        join_cols = [c for c in join_cols if c in per_date.columns]
                        miss = miss.merge(per_date[join_cols], on="PlayDate", how="left")

                        # Helper to compute StreamScore component contributions (matches the stream_score_row formula)
                        def _since_component(x):
                            try:
                                xv = float(x)
                            except Exception:
                                return 0.0
                            since_norm_local = min(max(xv, 0.0), 50.0) / 50.0
                            if since_mode == "recent":
                                return 1.0 - since_norm_local
                            if since_mode == "blend":
                                return abs(since_norm_local - 0.5) * 2.0
                            return since_norm_local


                        def _score_parts(match_strength, gap, roll, since, tie_fired, dead_tie):
                            base = (w_match * float(match_strength)) + (w_gap * float(gap)) + (w_roll * float(roll)) + (w_since * float(_since_component(since)))
                            cooldown = float(cooldown_penalty) if float(since) <= float(cooldown_k) else 0.0
                            tie_pen = 0.0
                            if int(dead_tie) == 1:
                                tie_pen = 0.5
                            elif int(tie_fired) == 1 and float(gap) == 0.0:
                                tie_pen = 0.2
                            return {
                                "match": w_match * float(match_strength),
                                "gap": w_gap * float(gap),
                                "roll": w_roll * float(roll),
                                "since": w_since * float(_since_component(since)),
                                "cooldown_pen": cooldown,
                                "tie_pen": tie_pen,
                                "base_sum": base,
                                "final_calc": base - cooldown - tie_pen,
                            }

                        # Winner parts
                        miss["W_match"] = miss["WinnerTop1Score"]
                        miss["W_gap"] = miss["WinnerBaseGap"]
                        miss["W_roll"] = miss["WinnerRoll025_30"]
                        miss["W_since"] = miss["WinnerDrawsSinceLast025"]
                        miss["W_tiefired"] = miss["WinnerTieFired"]
                        miss["W_deadtie"] = miss["WinnerDeadTie"]

                        # Top parts
                        miss["T_match"] = miss["TopTop1Score"]
                        miss["T_gap"] = miss["TopBaseGap"]
                        miss["T_roll"] = miss["TopRoll025_30"]
                        miss["T_since"] = miss["TopDrawsSinceLast025"]
                        miss["T_tiefired"] = miss["TopTieFired"]
                        miss["T_deadtie"] = miss["TopDeadTie"]

                        # Compute contribution deltas
                        deltas = []
                        for _, row in miss.iterrows():
                            wp = _score_parts(row["W_match"], row["W_gap"], row["W_roll"], row["W_since"], row["W_tiefired"], row["W_deadtie"])
                            tp = _score_parts(row["T_match"], row["T_gap"], row["T_roll"], row["T_since"], row["T_tiefired"], row["T_deadtie"])
                            deltas.append({
                                "delta_match": tp["match"] - wp["match"],
                                "delta_gap": tp["gap"] - wp["gap"],
                                "delta_roll": tp["roll"] - wp["roll"],
                                "delta_since": tp["since"] - wp["since"],
                                "delta_cooldown_pen": tp["cooldown_pen"] - wp["cooldown_pen"],
                                "delta_tie_pen": tp["tie_pen"] - wp["tie_pen"],
                                "delta_streamscore": float(row["TopStreamScore"]) - float(row["WinnerStreamScore"]),
                            })
                        deltas_df = pd.DataFrame(deltas)
                        miss = pd.concat([miss.reset_index(drop=True), deltas_df.reset_index(drop=True)], axis=1)

                        # Which component most often drove the gap?
                        comp_cols = ["delta_match", "delta_gap", "delta_roll", "delta_since", "delta_cooldown_pen", "delta_tie_pen"]
                        miss["PrimaryDriver"] = miss[comp_cols].abs().idxmax(axis=1).str.replace("delta_", "", regex=False)

                        driver_counts = miss["PrimaryDriver"].value_counts().rename_axis("Driver").reset_index(name="Count")
                        driver_counts["Pct"] = (driver_counts["Count"] / len(miss) * 100.0)

                        st.write(f"Miss events in window: **{len(miss):,}** (Winner Rank > {bad_rank_threshold})")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.write("Primary driver frequency:")
                            st.dataframe(driver_counts, use_container_width=True, height=260)
                        with c2:
                            st.write("Average deltas (Top stream minus Winner stream):")
                            avg_d = miss[["delta_streamscore"] + comp_cols].mean().to_frame("AvgDelta").reset_index().rename(columns={"index": "Metric"})
                            st.dataframe(avg_d, use_container_width=True, height=260)

                        st.write("Worst miss events (highest winner ranks):")
                        show_cols = [
                            "PlayDate", "WinningStream", "Rank", "WinnerStreamScore",
                            "TopStream", "TopStreamScore",
                            "delta_streamscore", "PrimaryDriver",
                            "delta_match", "delta_gap", "delta_roll", "delta_since",
                            "delta_cooldown_pen", "delta_tie_pen",
                        ]
                        show_cols = [c for c in show_cols if c in miss.columns]
                        st.dataframe(
                            miss.sort_values(["Rank"], ascending=False)[show_cols].head(60),
                            use_container_width=True,
                            height=520
                        )

                # --- 4) Component lift leaderboard (LAB-only) ---
                st.write("### 4) Component lift leaderboard (winner placement by single component ranks)")
                if playable.empty:
                    st.info("No playable winner events available for component lift (all winners were out-of-universe or gated).")
                else:
                    def _kpi_from_rankcol(_df, _label, _col):
                        if _col not in _df.columns:
                            return None
                        x = pd.to_numeric(_df[_col], errors="coerce").dropna()
                        n = int(len(x))
                        if n == 0:
                            return None
                        top1 = int((x == 1).sum())
                        top2 = int((x <= 2).sum())
                        top3 = int((x <= 3).sum())
                        return {
                            "Component": _label,
                            "N": n,
                            "Top1": top1,
                            "Top1%": (top1 / n * 100.0),
                            "Top2": top2,
                            "Top2%": (top2 / n * 100.0),
                            "Top3": top3,
                            "Top3%": (top3 / n * 100.0),
                            "AvgRank": float(x.mean()),
                            "MedianRank": float(x.median()),
                            "P90Rank": float(x.quantile(0.90)),
                        }

                    comp_rows = []
                    comp_rows.append(_kpi_from_rankcol(playable, "StreamScore (normal)", "Rank"))
                    comp_rows.append(_kpi_from_rankcol(playable, "StreamScore (reversed)", "Rank_rev"))
                    comp_rows.append(_kpi_from_rankcol(playable, "Top1Score only (desc)", "Rank_Top1Score"))
                    comp_rows.append(_kpi_from_rankcol(playable, "BaseGap only (desc)", "Rank_BaseGap"))
                    comp_rows.append(_kpi_from_rankcol(playable, "CadenceRolling025_30 only (desc)", "Rank_Roll025_30"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 only (due: desc)", "Rank_Since_due"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 only (recent: asc)", "Rank_Since_recent"))
                    comp_rows.append(_kpi_from_rankcol(playable, "DrawsSinceLast025 SinceBlend (extremes)", "Rank_Since_blend"))
                    comp_rows = [r for r in comp_rows if r is not None]
                    if not comp_rows:
                        st.info("Component lift leaderboard unavailable (missing component rank columns).")
                    else:
                        comp_df = pd.DataFrame(comp_rows)
                        comp_df = comp_df.sort_values(["Top3%", "Top2%", "Top1%"], ascending=False)
                        st.dataframe(comp_df, use_container_width=True, height=320)
                        st.caption("Interpretation: higher Top1/Top2/Top3% indicates the component alone places winners nearer the top. This is LAB-only diagnostics; LIVE ranking remains unchanged.")

                # --- 5) Tie + granularity diagnostics (LAB-only) ---
                st.write("### 5) Tie + granularity diagnostics (how often scores are tied)")
                if per_date.empty or ("UniqueStreamScore" not in per_date.columns):
                    st.info("Tie diagnostics unavailable (missing per-day tie/granularity metrics).")
                else:
                    tie = per_date.copy()
                    tie["TopStreamScoreTieCount"] = pd.to_numeric(tie.get("TopStreamScoreTieCount"), errors="coerce")
                    tie["TopTopTupleTieCount"] = pd.to_numeric(tie.get("TopTopTupleTieCount"), errors="coerce")
                    tie["UniqueStreamScore"] = pd.to_numeric(tie.get("UniqueStreamScore"), errors="coerce")
                    tie["UniqueTop1Score"] = pd.to_numeric(tie.get("UniqueTop1Score"), errors="coerce")
                    tie["TopScoreTied"] = (tie["TopStreamScoreTieCount"].fillna(0) > 1).astype(int)
                    tie["TopTupleTied"] = (tie["TopTopTupleTieCount"].fillna(0) > 1).astype(int)

                    c1, c2 = st.columns(2)
                    with c1:
                        n_days = int(len(tie))
                        st.write(f"Days analyzed: **{n_days:,}**")
                        st.write(f"Days with top StreamScore tie: **{int(tie['TopScoreTied'].sum()):,}/{n_days:,} = {tie['TopScoreTied'].mean()*100.0:.2f}%**")
                        st.write(f"Days with top (StreamScore,Top1Score,Gap) tie: **{int(tie['TopTupleTied'].sum()):,}/{n_days:,} = {tie['TopTupleTied'].mean()*100.0:.2f}%**")
                        st.write(f"Unique StreamScores/day: avg **{tie['UniqueStreamScore'].mean():.1f}**, median **{tie['UniqueStreamScore'].median():.0f}**")
                        st.write(f"Unique Top1Scores/day: avg **{tie['UniqueTop1Score'].mean():.1f}**, median **{tie['UniqueTop1Score'].median():.0f}**")

                    with c2:
                        dist = tie["TopStreamScoreTieCount"].value_counts(dropna=True).sort_index().reset_index()
                        dist.columns = ["TopStreamScoreTieCount", "Days"]
                        st.write("Distribution: how many streams share the day's #1 StreamScore")
                        st.dataframe(dist, use_container_width=True, height=220)

                    st.write("Lowest-granularity days (fewest unique StreamScores):")
                    show_cols = ["PlayDate", "UniverseSize", "UniqueStreamScore", "UniqueTop1Score", "TopStream", "TopStreamScore", "TopStreamScoreTieCount", "TopTopTupleTieCount"]
                    show_cols = [c for c in show_cols if c in tie.columns]
                    st.dataframe(tie.sort_values(["UniqueStreamScore", "UniqueTop1Score"], ascending=True)[show_cols].head(50), use_container_width=True, height=340)
                # Plays per day (average tier sizes)
                if not per_date.empty and per_date["UniverseSize"].sum() > 0:
                    avg_uni = per_date["UniverseSize"].mean()
                    avg_A = per_date["TierA_Size"].mean()
                    avg_AB = (per_date["TierA_Size"] + per_date["TierB_Size"]).mean()
                    st.write(
                        f"Average streams/day in universe: **{avg_uni:.1f}**  |  "
                        f"Tier A plays/day: **{avg_A:.1f}**  |  Tier A+B plays/day: **{avg_AB:.1f}**"
                    )

                st.write("### Winner rank distribution (first 200 rows)")
                st.dataframe(per_event.sort_values(["PlayDate","Rank"], na_position="last").head(200), use_container_width=True, height=420)

            # ============================================================
            # Out-of-universe diagnostics table
            # ============================================================
            st.subheader('Out-of-universe diagnostics (why some winners were not ranked)')
            if isinstance(per_event, pd.DataFrame) and ('InUniverse' in per_event.columns):
                out_diag = per_event[per_event['InUniverse'] == 0].copy()
            else:
                out_diag = per_event.iloc[0:0].copy()

            if out_diag.empty:
                st.success('No out-of-universe winners in this view.')
            else:
                # Ensure key columns exist even for older rows
                if 'WinningStreamKey' not in out_diag.columns:
                    out_diag['WinningStreamKey'] = out_diag['WinningStream'].apply(canon_stream)
                if 'Reason' not in out_diag.columns:
                    out_diag['Reason'] = 'NOT_RANKED'
                if 'BestMatchStream' not in out_diag.columns:
                    out_diag['BestMatchStream'] = None
                if 'MatchScore' not in out_diag.columns:
                    out_diag['MatchScore'] = None

                diag_cols = [
                    'PlayDate', 'WinningStream', 'WinningStreamKey', 'WinningResult', 'WinningMember',
                    'PrevSeed', 'PrevSeedHas9',
                    'UniverseSize', 'Reason', 'BestMatchStream', 'MatchScore'
                ]
                diag_cols = [c for c in diag_cols if c in out_diag.columns]
                st.dataframe(out_diag[diag_cols].sort_values(['Reason','PlayDate'], ascending=[True, False]), use_container_width=True)

                st.caption(
                    'Interpretation: '
                    'GATED_OUT_SEED_HAS_9 = excluded by the core gate (seed contains 9). '
                    'NO_HISTORY_BEFORE_PLAYDATE = stream label/key was not present in history prior to that play date (new stream or label drift). '
                    'UNIVERSE_EMPTY = no ranked streams could be built for that play date.'
                )

                csv_bytes = out_diag[diag_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download out-of-universe diagnostics CSV',
                    data=csv_bytes,
                    file_name=f'core025_out_of_universe_diagnostics_{datetime.date.today().isoformat()}.csv',
                    mime='text/csv',
                    use_container_width=True,
                )


                # Coverage curve by top X% (optional quick view)
                # Ensure total_events exists even after reruns (download clicks cause reruns)
                total_events = int(len(per_event)) if isinstance(per_event, pd.DataFrame) else 0

                st.write("### Coverage curve by top-% cutoff (by Rank)")
                cut_pcts = [5,10,15,20,25,30,40,50]
                curve = []
                for p in cut_pcts:
                    # covered if Rank <= ceil(p% of universe size)
                    cov = 0
                    for _, row in per_event.dropna(subset=["Rank","UniverseSize"]).iterrows():
                        n = int(row["UniverseSize"])
                        k = max(1, int(np.ceil(n * (p/100.0))))
                        if int(row["Rank"]) <= k:
                            cov += 1
                    curve.append({
                        "Top%": p,
                        "Covered": cov,
                        "Total": total_events,
                        "Coverage%": (cov/total_events*100.0) if total_events else 0.0,
                    })
                curve_df = pd.DataFrame(curve)
                st.dataframe(curve_df, use_container_width=True)

                st.markdown("---")
                st.subheader("Walk-forward Row Percentile Map (row-by-row)")
                st.caption(
                    "This is the row-based view: for each winning 025-family hit-event, we record the exact row (rank) "
                    "where the winning stream landed in the full ranked playlist for that day (walk-forward, no cheating). "
                    "Use this to decide: 'Play rows 1..K' instead of percent chunks."
                )

                per_event_in = per_event[(per_event["InUniverse"] == 1) & per_event["Rank"].notna() & per_event["UniverseSize"].notna()].copy()
                if per_event_in.empty:
                    st.warning("No in-universe events to build a row map (all winners were out-of-universe in this window).")
                else:
                    total_in = int(len(per_event_in))
                    max_rank = int(per_event_in["Rank"].max())

                    # Exact row win map (how often the winner is at each row)
                    row_map = (
                        per_event_in.groupby("Rank")
                        .size()
                        .rename("Wins")
                        .reset_index()
                        .sort_values("Rank")
                    )
                    row_map["Win%_of_events"] = row_map["Wins"] / total_events * 100.0
                    row_map["Win%_of_in_universe"] = row_map["Wins"] / total_in * 100.0

                    st.write("### Exact Row Win Map (top 50 rows shown)")
                    st.dataframe(row_map.head(50), use_container_width=True, height=520)

                    # Coverage curve for playing rows 1..K (fixed plays/day = K)
                    st.write("### Coverage curve: play rows 1..K")
                    K_max = int(min(50, max_rank))
                    cov_rows = []
                    wins_by_rank = dict(zip(row_map["Rank"].astype(int), row_map["Wins"].astype(int)))
                    running = 0
                    for k in range(1, K_max + 1):
                        running += wins_by_rank.get(k, 0)
                        cov_rows.append({
                            "TopRows(K)": k,
                            "Covered": int(running),
                            "TotalEvents": int(total_events),
                            "Coverage%": (running / total_events * 100.0) if total_events else 0.0,
                            "Coverage_InUniverse%": (running / total_in * 100.0) if total_in else 0.0,
                        })
                    cov_rows_df = pd.DataFrame(cov_rows)

                    if str(selection_mode).startswith("Best historical"):
                        _suggested_rows = _wf_suggest_rank_rows_topk(per_event, cap_streams=int(cap_streams))
                        st.markdown(
                            "**Suggested top rows (non-contiguous):** " + (", ".join([str(x) for x in _suggested_rows]) if _suggested_rows else "(none)")
                        )
                    st.dataframe(cov_rows_df, use_container_width=True, height=520)

                    # Percentile bin map (smooth view)
                    st.write("### Percentile Bin Map (5% bins)")
                    per_event_in["PctBin"] = (np.floor(per_event_in["Percentile"] * 20) / 20).clip(0, 0.95)
                    per_event_in["PctBinLabel"] = per_event_in["PctBin"].apply(lambda x: f"{int(x*100)}–{int(x*100+5)}%")
                    bin_map = (
                        per_event_in.groupby("PctBinLabel")
                        .size()
                        .rename("Wins")
                        .reset_index()
                    )
                    # Ensure bins are ordered
                    bin_order = [f"{i}–{i+5}%" for i in range(0, 100, 5)]
                    bin_map["PctBinLabel"] = pd.Categorical(bin_map["PctBinLabel"], categories=bin_order, ordered=True)
                    bin_map = bin_map.sort_values("PctBinLabel")
                    bin_map["Win%_of_events"] = bin_map["Wins"] / total_events * 100.0
                    bin_map["Win%_of_in_universe"] = bin_map["Wins"] / total_in * 100.0
                    st.dataframe(bin_map, use_container_width=True, height=520)

                    st.caption(
                        "Tip: If you want a single stable rule, pick a K from the 'rows 1..K' curve. "
                        "Row-based rules keep daily play counts fixed even when the number of streams varies."
                    )

                # Downloads
                st.download_button(
                    "Download per-event tier results (CSV)",
                    data=per_event.to_csv(index=False).encode("utf-8"),
                    file_name=_wf_download_name("tier_per_event", _wf_cache, rules_meta),
                    mime="text/csv",
                )
                st.download_button(
                    "Download per-date tier summary (CSV)",
                    data=per_date.to_csv(index=False).encode("utf-8"),
                    file_name=_wf_download_name("tier_per_date", _wf_cache, rules_meta),
                    mime="text/csv",
                )
            if per_stream is not None and not per_stream.empty:
                st.download_button(
                    "Download per-stream (per day)",
                    data=per_stream.to_csv(index=False),
                    file_name=_wf_download_name("tier_per_stream", _wf_cache, rules_meta),
                    mime="text/csv",
                )


    # ----------------------------------------------------------------------------------
    # Cached downloads: clicking a Streamlit download button triggers a rerun.
    # Without caching, users would need to re-run the walk-forward each time.
    # This section keeps the last walk-forward tables available for repeated downloads.
    # ----------------------------------------------------------------------------------
    wf_cache = st.session_state.get("_wf_tier_cache")
    if (
        isinstance(wf_cache, dict)
        and isinstance(wf_cache.get("per_event"), pd.DataFrame)
        and isinstance(wf_cache.get("per_date"), pd.DataFrame)
    ):
        with st.expander("Downloads (cached — no need to re-run Walk-forward)", expanded=False):
            st.caption(
                f"Cached from last run at {wf_cache.get('ran_at','(unknown time)')} | schema: {wf_cache.get('export_schema', 'wf_per_event_v1')}. "
                "After any rerun (including downloads), these files remain available."
            )

            per_event_c = wf_cache["per_event"]
            per_date_c = wf_cache["per_date"]
            per_stream_c = wf_cache.get("per_stream", pd.DataFrame())

            # Out-of-universe diagnostics (derived)
            if "InUniverse" in per_event_c.columns:
                try:
                    out_diag_c = per_event_c[per_event_c["InUniverse"].astype(int) == 0].copy()
                except Exception:
                    out_diag_c = per_event_c[per_event_c["InUniverse"] == 0].copy()
            else:
                out_diag_c = per_event_c.iloc[0:0].copy()

            # Rebuild Component Lift table (derived) for export convenience
            component_lift_df = None
            try:
                playable_c = per_event_c.copy()
                if "InUniverse" in playable_c.columns:
                    playable_c = playable_c[playable_c["InUniverse"].astype(int) == 1]
                if "Reason" in playable_c.columns:
                    playable_c = playable_c[playable_c["Reason"].fillna("").astype(str).str.strip() == ""]
                components = [
                    ("StreamScore (normal)", "Rank"),
                    ("StreamScore (reversed)", "Rank_rev"),
                    ("Top1Score only (desc)", "Rank_Top1Score"),
                    ("BaseGap only (desc)", "Rank_BaseGap"),
                    ("CadenceRolling025_30 only (desc)", "Rank_Roll025_30"),
                    ("DrawsSinceLast025 only (due: desc)", "Rank_Since_due"),
                    ("DrawsSinceLast025 only (recent: asc)", "Rank_Since_recent"),
                    ("DrawsSinceLast025 SinceBlend (extremes)", "Rank_Since_blend"),
                ]
                rows = []
                N = int(len(playable_c))
                for comp_name, rank_col in components:
                    if rank_col not in playable_c.columns or N == 0:
                        continue
                    r = pd.to_numeric(playable_c[rank_col], errors="coerce")
                    r = r.dropna().astype(int)
                    if len(r) == 0:
                        continue
                    rows.append(
                        {
                            "Component": comp_name,
                            "N": N,
                            "Top1": int((r <= 1).sum()),
                            "Top1%": float((r <= 1).sum() / N * 100.0),
                            "Top2": int((r <= 2).sum()),
                            "Top2%": float((r <= 2).sum() / N * 100.0),
                            "Top3": int((r <= 3).sum()),
                            "Top3%": float((r <= 3).sum() / N * 100.0),
                            "AvgRank": float(r.mean()),
                            "MedianRank": float(r.median()),
                            "P90Rank": float(r.quantile(0.90)),
                        }
                    )
                if rows:
                    component_lift_df = pd.DataFrame(rows).sort_values(
                        by=["Top3%", "Top2%", "Top1%"], ascending=False
                    )
            except Exception:
                component_lift_df = None

            # Rebuild Tie diagnostics summary (derived) for export convenience
            tie_summary_df = None
            try:
                if "TopStreamScoreTieCount" in per_date_c.columns:
                    days_total = int(len(per_date_c))
                    ties = int((pd.to_numeric(per_date_c["TopStreamScoreTieCount"], errors="coerce") > 1).sum())
                    tie_summary_df = pd.DataFrame(
                        [
                            {
                                "DaysAnalyzed": days_total,
                                "DaysWithTopStreamScoreTie": ties,
                                "TieRate%": (ties / days_total * 100.0) if days_total else 0.0,
                                "UniqueStreamScoresPerDay_Avg": float(pd.to_numeric(per_date_c.get("UniqueStreamScore"), errors="coerce").mean())
                                if "UniqueStreamScore" in per_date_c.columns
                                else None,
                                "UniqueStreamScoresPerDay_Median": float(pd.to_numeric(per_date_c.get("UniqueStreamScore"), errors="coerce").median())
                                if "UniqueStreamScore" in per_date_c.columns
                                else None,
                                "UniqueTop1ScoresPerDay_Avg": float(pd.to_numeric(per_date_c.get("UniqueTop1Score"), errors="coerce").mean())
                                if "UniqueTop1Score" in per_date_c.columns
                                else None,
                                "UniqueTop1ScoresPerDay_Median": float(pd.to_numeric(per_date_c.get("UniqueTop1Score"), errors="coerce").median())
                                if "UniqueTop1Score" in per_date_c.columns
                                else None,
                            }
                        ]
                    )
            except Exception:
                tie_summary_df = None

            # Individual table downloads
            st.download_button(
                "Download walk-forward per-event (CSV)",
                data=per_event_c.to_csv(index=False).encode("utf-8"),
                file_name=_wf_download_name("tier_per_event", _wf_cache, rules_meta),
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download walk-forward per-date (CSV)",
                data=per_date_c.to_csv(index=False).encode("utf-8"),
                file_name=_wf_download_name("tier_per_date", _wf_cache, rules_meta),
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download out-of-universe diagnostics (CSV)",
                data=out_diag_c.to_csv(index=False).encode("utf-8"),
                file_name="core025_out_of_universe_diagnostics.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if isinstance(component_lift_df, pd.DataFrame) and len(component_lift_df) > 0:
                st.download_button(
                    "Download component lift leaderboard (CSV)",
                    data=component_lift_df.to_csv(index=False).encode("utf-8"),
                    file_name="core025_component_lift_leaderboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if isinstance(tie_summary_df, pd.DataFrame) and len(tie_summary_df) > 0:
                st.download_button(
                    "Download tie diagnostics summary (CSV)",
                    data=tie_summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="core025_tie_diagnostics_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # One-click bundle
            try:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr("core025_walkforward_tier_per_event.csv", per_event_c.to_csv(index=False))
                    z.writestr("core025_walkforward_tier_per_date.csv", per_date_c.to_csv(index=False))
                    if per_stream_c is not None and not per_stream_c.empty:
                        z.writestr("core025_walkforward_tier_per_stream.csv", per_stream_c.to_csv(index=False))
                    z.writestr("core025_out_of_universe_diagnostics.csv", out_diag_c.to_csv(index=False))
                    if isinstance(component_lift_df, pd.DataFrame) and len(component_lift_df) > 0:
                        z.writestr("core025_component_lift_leaderboard.csv", component_lift_df.to_csv(index=False))
                    if isinstance(tie_summary_df, pd.DataFrame) and len(tie_summary_df) > 0:
                        z.writestr("core025_tie_diagnostics_summary.csv", tie_summary_df.to_csv(index=False))
                st.download_button(
                    "Download ALL cached LAB tables (.zip)",
                    data=buf.getvalue(),
                    file_name=_wf_download_name("lab_tables", _wf_cache, rules_meta, ext="zip"),
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as _zip_e:
                st.info(f"Zip bundle unavailable: {_zip_e}")

            wf_runs_cache = st.session_state.get("_wf_tier_cache_runs", {})
            if isinstance(wf_runs_cache, dict) and wf_runs_cache:
                with st.expander("Downloads (Gate ON vs OFF caches)", expanded=False):
                    st.caption("Available cached runs:")
                    st.write(", ".join(sorted(list(wf_runs_cache.keys()))))
                    # Individual downloads
                    
                    for _lbl, _cache in wf_runs_cache.items():
                        _sel_mode = str(_cache.get("selection_mode", ""))
                        _bins_list = _cache.get("selected_bins", []) or []
                        _ranks_list = _cache.get("selected_ranks", []) or []
                        _rule = ", ".join(_bins_list) if _bins_list else (", ".join([str(x) for x in _ranks_list]) if _ranks_list else "(none)")
                        st.markdown(
                            f"**{_lbl}** — ran_at: {_cache.get('ran_at','')}  |  cap: {_cache.get('cap_streams','')}  |  mode: {_sel_mode}  |  bin: {_cache.get('bin_size_pct','')}  |  rule: {_rule}"
                        )
                        _pe = _cache.get("per_event", pd.DataFrame())
                        _pd = _cache.get("per_date", pd.DataFrame())
                        _ps = _cache.get("per_stream", pd.DataFrame())
                        c1, c2, c3 = st.columns([1, 1, 1])
                        with c1:
                            st.download_button(
                                f"Download per-event ({_lbl})",
                                data=_pe.to_csv(index=False),
                                file_name=_wf_download_name("tier_per_event", _cache, rules_meta),
                                mime="text/csv",
                            )
                        with c2:
                            st.download_button(
                                f"Download per-date ({_lbl})",
                                data=_pd.to_csv(index=False),
                                file_name=_wf_download_name("tier_per_date", _cache, rules_meta),
                                mime="text/csv",
                            )
                        with c3:
                            if _ps is not None and not _ps.empty:
                                st.download_button(
                                    f"Download per-stream ({_lbl})",
                                    data=_ps.to_csv(index=False),
                                    file_name=_wf_download_name("tier_per_stream", _cache, rules_meta),
                                    mime="text/csv",
                                )

                    # Combined zip of all cached runs
                    try:
                        zbuf2 = io.BytesIO()
                        with zipfile.ZipFile(zbuf2, mode="w", compression=zipfile.ZIP_DEFLATED) as z2:
                            for _lbl, _cache in wf_runs_cache.items():
                                _pe = _cache.get("per_event", pd.DataFrame())
                                _pd = _cache.get("per_date", pd.DataFrame())
                                _ps = _cache.get("per_stream", pd.DataFrame())
                                z2.writestr(f"core025_walkforward_tier_per_event_{_lbl}.csv", _pe.to_csv(index=False))
                                z2.writestr(f"core025_walkforward_tier_per_date_{_lbl}.csv", _pd.to_csv(index=False))
                                if _ps is not None and not _ps.empty:
                                    z2.writestr(f"core025_walkforward_tier_per_stream_{_lbl}.csv", _ps.to_csv(index=False))
                        zbuf2.seek(0)
                        st.download_button(
                            "Download ALL cached runs (zip)",
                            data=zbuf2.getvalue(),
                            file_name=_wf_download_name("gate_runs_bundle", _wf_cache, rules_meta, ext="zip"),
                            mime="application/zip",
                        )
                    except Exception:
                        st.info("Gate-runs zip bundle unavailable.")
