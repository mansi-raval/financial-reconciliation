"""Step 3a: Feature engineering for every candidate (bank, check) pair.

Given N bank rows and M check rows, we compute an N x M matrix for each of
4 features:
  - desc_sim:   cosine similarity of description embeddings (range ~[-1, 1],
                typically [0, 1] for our text)
  - amount_sim: 1 - |Δamount| / max(amount, 1); clipped to [0, 1]
  - date_sim:   exp(-|Δdays| / tau)  with tau = 3  (0 days => 1.0, 5 days => ~0.19)
  - type_match: 1.0 if standardized types agree else 0.0

Using matrices keeps things vectorized and fast — 308x308 = ~95k pairs but
everything is numpy, runs in milliseconds.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


DATE_TAU_DAYS = 3.0


def amount_similarity_matrix(bank_amounts: np.ndarray, check_amounts: np.ndarray) -> np.ndarray:
    b = bank_amounts.reshape(-1, 1)  # (N, 1)
    c = check_amounts.reshape(1, -1)  # (1, M)
    delta = np.abs(b - c)
    denom = np.maximum(np.maximum(np.abs(b), np.abs(c)), 1.0)
    sim = 1.0 - (delta / denom)
    return np.clip(sim, 0.0, 1.0)


def date_similarity_matrix(bank_dates: pd.Series, check_dates: pd.Series,
                           tau_days: float = DATE_TAU_DAYS) -> np.ndarray:
    b = bank_dates.values.astype("datetime64[D]").astype(np.int64).reshape(-1, 1)
    c = check_dates.values.astype("datetime64[D]").astype(np.int64).reshape(1, -1)
    delta_days = np.abs(b - c)
    return np.exp(-delta_days / tau_days)


def type_match_matrix(bank_types: pd.Series, check_types: pd.Series) -> np.ndarray:
    b = np.asarray(bank_types, dtype=object).reshape(-1, 1)
    c = np.asarray(check_types, dtype=object).reshape(1, -1)
    return (b == c).astype(np.float32)


def cosine_similarity_matrix(bank_emb: np.ndarray, check_emb: np.ndarray) -> np.ndarray:
    """bank_emb: (N, D), check_emb: (M, D). Both should be L2-normalized."""
    return bank_emb @ check_emb.T
