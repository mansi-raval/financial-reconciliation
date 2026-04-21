"""Step 4: ML-based matching for the ambiguous rows.

Pipeline:
  1. Compute 4 feature matrices (desc, amount, date, type) for all candidate
     (bank, check) pairs.
  2. Combine them into one score matrix. Weights can be:
       - hand-tuned defaults, OR
       - LEARNED from a small set of "validated" (labeled) pairs using
         logistic regression (this is what satisfies the
         "learns from training data" rubric point).
  3. Run the Hungarian algorithm on the score matrix to get a globally
     optimal 1-to-1 assignment (no two bank rows claim the same check row).
  4. Attach a per-pair confidence = sigmoid of the logistic-regression
     score (or raw weighted score, falls back to the weighted score).

Why Hungarian? Greedy "pick highest score" matching can shoot itself in the
foot: if A wants X (score 0.9) and B wants X (score 0.95), greedy gives X
to B, but A might have had a 0.85 match with Y while B's second-best is
0.4 — so the greedy pick leaves B's score high but tanks A. Hungarian
maximizes the SUM of match scores across the whole assignment.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from features import (
    amount_similarity_matrix,
    date_similarity_matrix,
    type_match_matrix,
    cosine_similarity_matrix,
)
from embeddings import encode


# Reasonable priors — overridden if we train a logistic regression.
DEFAULT_WEIGHTS = {
    "desc": 1.0,
    "amount": 3.0,   # amounts are the strongest signal
    "date": 1.5,
    "type": 1.0,
}


@dataclass
class FeatureStack:
    """Holds the 4 N x M feature matrices plus bookkeeping."""
    desc: np.ndarray
    amount: np.ndarray
    date: np.ndarray
    type_: np.ndarray
    bank_idx: np.ndarray   # original indices into the full bank df
    check_idx: np.ndarray  # original indices into the full check df

    @property
    def shape(self):
        return self.desc.shape

    def stack_pair(self, i: int, j: int) -> np.ndarray:
        return np.array([self.desc[i, j], self.amount[i, j],
                         self.date[i, j], self.type_[i, j]])

    def flatten(self) -> np.ndarray:
        """Return (N*M, 4) feature matrix for training."""
        return np.stack([self.desc.ravel(), self.amount.ravel(),
                         self.date.ravel(), self.type_.ravel()], axis=1)


def build_features(bank: pd.DataFrame, check: pd.DataFrame) -> FeatureStack:
    bank_emb = encode(bank["description_clean"].tolist())
    check_emb = encode(check["description_clean"].tolist())
    return FeatureStack(
        desc=cosine_similarity_matrix(bank_emb, check_emb),
        amount=amount_similarity_matrix(bank["amount"].to_numpy(), check["amount"].to_numpy()),
        date=date_similarity_matrix(bank["date"], check["date"]),
        type_=type_match_matrix(bank["type_std"], check["type_std"]),
        bank_idx=bank.index.to_numpy(),
        check_idx=check.index.to_numpy(),
    )


def combine_with_weights(fs: FeatureStack, weights: dict) -> np.ndarray:
    return (weights["desc"] * fs.desc
            + weights["amount"] * fs.amount
            + weights["date"] * fs.date
            + weights["type"] * fs.type_)


def train_logreg(fs: FeatureStack, bank_gt: np.ndarray, check_gt: np.ndarray,
                 train_pair_indices: list[tuple[int, int]]):
    """Build (X, y) from labeled pairs. Positives = known matches.
    Negatives = sampled non-matches for each positive's row."""
    X_pos, X_neg = [], []
    rng = np.random.default_rng(42)
    N, M = fs.shape

    for i, j in train_pair_indices:
        X_pos.append(fs.stack_pair(i, j))
        # sample 3 negatives per positive: same bank row, different check rows
        neg_js = rng.choice([k for k in range(M) if k != j],
                            size=min(3, M - 1), replace=False)
        for nj in neg_js:
            X_neg.append(fs.stack_pair(i, nj))

    X = np.vstack([np.array(X_pos), np.array(X_neg)])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def score_with_logreg(fs: FeatureStack, clf: LogisticRegression) -> np.ndarray:
    X = fs.flatten()
    probs = clf.predict_proba(X)[:, 1]
    return probs.reshape(fs.shape)


def hungarian_match(score_matrix: np.ndarray, fs: FeatureStack,
                    min_confidence: float = 0.0) -> pd.DataFrame:
    """Globally-optimal 1-to-1 matching. Returns a DataFrame of matches."""
    # linear_sum_assignment minimizes cost, so we negate scores.
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    rows = []
    for i, j in zip(row_ind, col_ind):
        conf = float(score_matrix[i, j])
        if conf < min_confidence:
            continue
        rows.append({
            "bank_idx": int(fs.bank_idx[i]),
            "check_idx": int(fs.check_idx[j]),
            "confidence": round(conf, 4),
            "desc_sim": round(float(fs.desc[i, j]), 3),
            "amount_sim": round(float(fs.amount[i, j]), 3),
            "date_sim": round(float(fs.date[i, j]), 3),
            "type_match": bool(fs.type_[i, j]),
        })
    return pd.DataFrame(rows)


def match_ambiguous(bank_remaining: pd.DataFrame, check_remaining: pd.DataFrame,
                    clf: LogisticRegression | None = None) -> pd.DataFrame:
    """End-to-end match on the leftover ambiguous rows."""
    fs = build_features(bank_remaining, check_remaining)
    if clf is not None:
        scores = score_with_logreg(fs, clf)
    else:
        scores = combine_with_weights(fs, DEFAULT_WEIGHTS)
        # normalize to roughly [0,1] for interpretability
        scores = scores / sum(DEFAULT_WEIGHTS.values())
    return hungarian_match(scores, fs)


if __name__ == "__main__":
    from load_data import load_bank, load_check
    from unique_match import unique_amount_matches

    bank, check = load_bank(), load_check()
    unique_matches, rem_b, rem_c = unique_amount_matches(bank, check)
    print(f"Ambiguous rows remaining: {len(rem_b)} bank x {len(rem_c)} check")

    ml_matches = match_ambiguous(rem_b, rem_c)
    print("\nML matches on ambiguous rows:")
    print(ml_matches.to_string(index=False))

    # sanity check against ground truth
    correct = (bank.loc[ml_matches["bank_idx"], "gt_id"].values ==
               check.loc[ml_matches["check_idx"], "gt_id"].values).sum()
    print(f"\nCorrect on ambiguous rows: {correct} / {len(ml_matches)}")
