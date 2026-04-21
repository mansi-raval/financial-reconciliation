"""Step 6: Learning curve — how does F1 scale with the amount of training data?

We simulate "validated matches" at fractions of the ground truth:
  10%, 20%, 30%, ..., 80% of the 308 pairs are revealed as labeled training
  data. We fit logistic regression on those, run Hungarian matching on the
  FULL 308x308 score matrix, and report F1 on all 308.

Outputs:
  - console table of (train_frac, precision, recall, f1)
  - a matplotlib PNG saved to outputs/learning_curve.png

Also reports a baseline using hand-tuned weights (no training).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_bank, load_check
from matcher import (
    build_features, combine_with_weights, hungarian_match,
    train_logreg, score_with_logreg, DEFAULT_WEIGHTS,
)
from evaluate import evaluate


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def gt_pair_indices(bank: pd.DataFrame, check: pd.DataFrame) -> list[tuple[int, int]]:
    """Return (i, j) index pairs where bank.iloc[i].gt_id == check.iloc[j].gt_id."""
    check_pos = {gt: pos for pos, gt in enumerate(check["gt_id"].to_numpy())}
    pairs = []
    for i, gt in enumerate(bank["gt_id"].to_numpy()):
        if gt in check_pos:
            pairs.append((i, check_pos[gt]))
    return pairs


def run_curve(fractions=(0.1, 0.2, 0.3, 0.5, 0.8), seed: int = 0):
    bank, check = load_bank(), load_check()
    fs = build_features(bank.reset_index(drop=True), check.reset_index(drop=True))
    pairs = gt_pair_indices(bank.reset_index(drop=True), check.reset_index(drop=True))

    # Baseline: hand-tuned weights, no training
    baseline_scores = combine_with_weights(fs, DEFAULT_WEIGHTS)
    baseline_matches = hungarian_match(baseline_scores, fs)
    baseline_metrics = evaluate(baseline_matches, bank, check)
    print(f"{'baseline (no training)':30s}  "
          f"P={baseline_metrics['precision']:.4f}  R={baseline_metrics['recall']:.4f}  "
          f"F1={baseline_metrics['f1']:.4f}")

    rng = np.random.default_rng(seed)
    results = []
    for frac in fractions:
        k = max(5, int(frac * len(pairs)))
        train_idx = rng.choice(len(pairs), size=k, replace=False)
        train_pairs = [pairs[i] for i in train_idx]

        clf = train_logreg(fs, bank["gt_id"].to_numpy(), check["gt_id"].to_numpy(), train_pairs)
        scores = score_with_logreg(fs, clf)
        matches = hungarian_match(scores, fs)
        m = evaluate(matches, bank, check)
        results.append({"train_frac": frac, "train_n": k, **m})
        print(f"train_frac={frac:>4.0%} (n={k:>4d})          "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)

    # plot
    plt.figure(figsize=(7, 4))
    plt.plot(df["train_frac"] * 100, df["f1"], marker="o", label="Trained logreg F1")
    plt.axhline(baseline_metrics["f1"], ls="--", color="gray",
                label=f"Baseline (hand-tuned) F1={baseline_metrics['f1']:.3f}")
    plt.xlabel("Training pairs (% of ground truth)")
    plt.ylabel("F1")
    plt.title("Reconciliation F1 vs training data size")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png = OUTPUT_DIR / "learning_curve.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved plot -> {out_png}")
    return df, baseline_metrics


if __name__ == "__main__":
    run_curve()
