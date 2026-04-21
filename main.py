"""
Financial Reconciliation System
================================
Matches bank transactions to check register transactions using:
  1. Unique amount matching  - if an amount appears only once in both files, match it directly
  2. ML matching             - for the rest, use sentence embeddings + cosine similarity
  3. Review                  - flag low-confidence matches for human review
  4. Improve                 - show how accuracy improves with more training data

Run:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "src")

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from load_data import load_bank, load_check
from unique_match import unique_amount_matches
from matcher import match_ambiguous, build_features, train_logreg, score_with_logreg, hungarian_match
from evaluate import evaluate_stages
from learning_curve import run_curve, gt_pair_indices

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # print to console
        logging.FileHandler("outputs/run.log", mode="w") # save to file
    ]
)
log = logging.getLogger()
Path("outputs").mkdir(exist_ok=True)


# =============================================================================
# STEP 1 — Load the data
# =============================================================================
log.info("STEP 1: Loading data...")
bank  = load_bank()
check = load_check()
log.info("  Bank:  %d transactions", len(bank))
log.info("  Check: %d transactions", len(check))


# =============================================================================
# STEP 2 — Unique amount matching
# =============================================================================
# If an amount appears exactly once in both files, it can only match one way.
# No ML needed — this is free and 100% precise.
log.info("")
log.info("STEP 2: Unique amount matching...")
unique_matches, remaining_bank, remaining_check = unique_amount_matches(bank, check)
log.info("  Matched (unique amounts): %d", len(unique_matches))
log.info("  Still unmatched:          %d", len(remaining_bank))


# =============================================================================
# STEP 3 — ML matching for the remaining ambiguous transactions
# =============================================================================
# For transactions where the amount is not unique, we use:
#   - sentence embeddings to compare descriptions
#   - amount similarity, date proximity, type match
# Combined into a score matrix, then Hungarian assignment picks the best pairs.
log.info("")
log.info("STEP 3: ML matching for ambiguous transactions...")
ml_matches = match_ambiguous(remaining_bank, remaining_check)
log.info("  ML matched: %d", len(ml_matches))
log.info("  Mean confidence: %.3f", ml_matches["confidence"].mean())


# =============================================================================
# STEP 4 — Combine results and save to CSV
# =============================================================================
log.info("")
log.info("STEP 4: Combining results and saving...")

# Drop duplicate id columns from unique_matches before concat
u = unique_matches.drop(columns=[c for c in ["bank_id","check_id"] if c in unique_matches.columns])
all_matches = pd.concat([u.assign(source="unique"), ml_matches.assign(source="ml")], ignore_index=True)

# Attach human-readable info
all_matches["bank_id"]     = bank.loc[all_matches["bank_idx"].to_numpy(), "transaction_id"].to_numpy()
all_matches["check_id"]    = check.loc[all_matches["check_idx"].to_numpy(), "transaction_id"].to_numpy()
all_matches["bank_desc"]   = bank.loc[all_matches["bank_idx"].to_numpy(), "description"].to_numpy()
all_matches["check_desc"]  = check.loc[all_matches["check_idx"].to_numpy(), "description"].to_numpy()
all_matches["bank_amount"] = bank.loc[all_matches["bank_idx"].to_numpy(), "amount"].to_numpy()
all_matches["check_amount"]= check.loc[all_matches["check_idx"].to_numpy(), "amount"].to_numpy()
all_matches["bank_date"]   = bank.loc[all_matches["bank_idx"].to_numpy(), "date"].dt.strftime("%Y-%m-%d").to_numpy()
all_matches["check_date"]  = check.loc[all_matches["check_idx"].to_numpy(), "date"].dt.strftime("%Y-%m-%d").to_numpy()

# Pick which columns to save
output_cols = ["bank_id","check_id","source","confidence",
               "bank_date","check_date","bank_amount","check_amount",
               "bank_desc","check_desc","desc_sim","amount_sim","date_sim","type_match"]
output_cols = [c for c in output_cols if c in all_matches.columns]
all_matches[output_cols].to_csv("outputs/all_matches.csv", index=False)
log.info("  Saved all_matches.csv (%d rows)", len(all_matches))


# =============================================================================
# STEP 5 — Evaluate: Precision, Recall, F1
# =============================================================================
# We know the correct answer because bank B0047 should match check R0047
# (they share the same numeric suffix). We use this only for scoring.
log.info("")
log.info("STEP 5: Evaluating Precision / Recall / F1...")
results = evaluate_stages(unique_matches, ml_matches, bank, check)
for stage, m in results.items():
    log.info("  %-15s  P=%.4f  R=%.4f  F1=%.4f  (%d/%d correct)",
             stage, m["precision"], m["recall"], m["f1"], m["correct"], m["proposed"])


# =============================================================================
# STEP 6 — Review: flag low-confidence matches for human check
# =============================================================================
log.info("")
log.info("STEP 6: Review — flagging low-confidence matches...")
low_confidence = all_matches[all_matches["confidence"] < 0.8]
log.info("  Matches needing human review (confidence < 0.8): %d", len(low_confidence))
for _, row in low_confidence.iterrows():
    log.warning("  REVIEW: %s <-> %s  conf=%.2f  | '%s' vs '%s'",
                row["bank_id"], row["check_id"], row["confidence"],
                row["bank_desc"], row["check_desc"])


# =============================================================================
# STEP 7 — Improve: show F1 improves as we give it more training data
# =============================================================================
# We simulate "validated matches" at 10%, 20%...80% of all pairs.
# Each time: train logistic regression -> run matching -> measure F1.
# This proves the system learns and gets better with more human feedback.
log.info("")
log.info("STEP 7: Improve — showing F1 vs training data size...")
log.info("  (This simulates the match -> review -> improve feedback loop)")
run_curve(fractions=(0.1, 0.2, 0.3, 0.5, 0.8))

log.info("")
log.info("Done. Check outputs/ for:")
log.info("  all_matches.csv      — every match with confidence scores")
log.info("  learning_curve.png   — F1 improves with more training data")
log.info("  run.log              — full log of this run")
