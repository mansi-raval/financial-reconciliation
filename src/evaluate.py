"""Step 5: Precision / Recall / F1 — faithful to the paper's definitions.

From the paper (equations 3, 4, 5):

  Precision = |{accepted} ∩ {reconciled}| / |{reconciled}|
            = correctly matched / all matches the system proposed

  Recall    = |{accepted} ∩ {reconciled}| / |{accepted}|
            = correctly matched / all transactions that SHOULD be matched

  F1        = 2 * (Precision * Recall) / (Precision + Recall)

Key point on the denominator for Recall:
  - For the UNIQUE stage  → denominator is 308 (responsible for all of them)
  - For the ML stage      → denominator is 22  (only responsible for the
                            ambiguous rows passed to it, NOT all 308)
  - For COMBINED          → denominator is 308

Using 308 as the ML denominator would give recall = 22/308 = 0.07,
which looks wrong — the ML stage got every row it was given correct.
Each stage is evaluated against what it was actually responsible for.

Ground truth: bank transaction B0047 should match check R0047.
The shared numeric suffix is our answer key (never used for matching).
"""

from __future__ import annotations
import pandas as pd


def evaluate(matches: pd.DataFrame, bank: pd.DataFrame, check: pd.DataFrame,
             total_responsible: int) -> dict:
    """
    matches          — DataFrame with bank_idx and check_idx columns
    bank / check     — full DataFrames (for gt_id lookup)
    total_responsible — how many transactions THIS stage was responsible for
                        (= denominator of recall)
    """
    if len(matches) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "correct": 0, "proposed": 0, "responsible": total_responsible}

    b_gt = bank.loc[matches["bank_idx"], "gt_id"].to_numpy()
    c_gt = check.loc[matches["check_idx"], "gt_id"].to_numpy()

    correct  = int((b_gt == c_gt).sum())   # |{accepted} ∩ {reconciled}|
    proposed = len(matches)                 # |{reconciled}|

    precision = correct / proposed                   # eq. 3
    recall    = correct / total_responsible          # eq. 4
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0  # eq. 5

    return {
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "correct":     correct,
        "proposed":    proposed,
        "responsible": total_responsible,
    }


def evaluate_stages(unique_matches: pd.DataFrame, ml_matches: pd.DataFrame,
                    bank: pd.DataFrame, check: pd.DataFrame) -> dict:
    """
    Evaluate each stage with the correct recall denominator:
      unique stage → responsible for all 308
      ml stage     → responsible only for the ambiguous rows it received
      combined     → responsible for all 308
    """
    total           = len(bank)                  # 308
    ml_responsible  = len(ml_matches)            # 22 (rows the ML stage saw)

    all_matches = pd.concat(
        [unique_matches[["bank_idx", "check_idx"]],
         ml_matches[["bank_idx", "check_idx"]]],
        ignore_index=True
    )

    return {
        "unique_stage": evaluate(unique_matches, bank, check, total),
        "ml_stage":     evaluate(ml_matches,     bank, check, ml_responsible),
        "combined":     evaluate(all_matches,    bank, check, total),
    }


if __name__ == "__main__":
    from load_data import load_bank, load_check
    from unique_match import unique_amount_matches
    from matcher import match_ambiguous

    bank, check = load_bank(), load_check()
    unique_m, rem_b, rem_c = unique_amount_matches(bank, check)
    ml_m = match_ambiguous(rem_b, rem_c)

    results = evaluate_stages(unique_m, ml_m, bank, check)
    for stage, metrics in results.items():
        print(f"{stage:15s}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"F1={metrics['f1']:.4f}  ({metrics['correct']}/{metrics['proposed']} "
              f"correct out of {metrics['ground_truth']} GT)")
