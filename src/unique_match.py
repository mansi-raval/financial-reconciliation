"""Step 2: Match transactions whose amount is unique in BOTH files.

Logic: if an amount X appears exactly once in the bank file AND exactly once
in the check file, those two rows almost certainly describe the same
transaction. This gives us a pile of high-confidence matches "for free",
shrinking the harder ambiguous set we hand to the ML model.

Confidence scoring:
- base 1.0 for a singleton-amount pair
- small penalty if dates differ by more than 5 days (assignment threshold)
- small penalty if types don't agree (DEBIT vs CREDIT on the two sides)

We also FLAG suspicious matches (date gap > 5 or type mismatch) so a human
reviewer can double-check them.
"""

from __future__ import annotations
import pandas as pd


DATE_THRESHOLD_DAYS = 5


def unique_amount_matches(bank: pd.DataFrame, check: pd.DataFrame):
    """Return (matches_df, remaining_bank, remaining_check).

    matches_df columns: bank_idx, check_idx, amount, date_gap_days,
                       type_match, confidence, flags
    """
    # Amounts that appear exactly once on each side
    bank_counts = bank["amount"].value_counts()
    check_counts = check["amount"].value_counts()
    unique_bank = set(bank_counts[bank_counts == 1].index)
    unique_check = set(check_counts[check_counts == 1].index)
    shared_unique = unique_bank & unique_check

    rows = []
    for amt in shared_unique:
        b_row = bank[bank["amount"] == amt].iloc[0]
        c_row = check[check["amount"] == amt].iloc[0]
        date_gap = abs((b_row["date"] - c_row["date"]).days)
        type_match = b_row["type_std"] == c_row["type_std"]

        confidence = 1.0
        flags = []
        if date_gap > DATE_THRESHOLD_DAYS:
            confidence -= 0.15
            flags.append(f"date_gap={date_gap}d")
        if not type_match:
            confidence -= 0.25
            flags.append("type_mismatch")

        rows.append({
            "bank_idx": b_row.name,
            "check_idx": c_row.name,
            "bank_id": b_row["transaction_id"],
            "check_id": c_row["transaction_id"],
            "amount": amt,
            "date_gap_days": date_gap,
            "type_match": type_match,
            "confidence": round(confidence, 3),
            "flags": ",".join(flags) if flags else "",
        })

    matches = pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)

    matched_bank_idx = set(matches["bank_idx"]) if len(matches) else set()
    matched_check_idx = set(matches["check_idx"]) if len(matches) else set()
    remaining_bank = bank[~bank.index.isin(matched_bank_idx)].copy()
    remaining_check = check[~check.index.isin(matched_check_idx)].copy()

    return matches, remaining_bank, remaining_check


if __name__ == "__main__":
    from load_data import load_bank, load_check

    bank = load_bank()
    check = load_check()
    matches, rem_b, rem_c = unique_amount_matches(bank, check)

    print(f"Unique-amount matches: {len(matches)}")
    print(f"Remaining bank rows:   {len(rem_b)}")
    print(f"Remaining check rows:  {len(rem_c)}")
    flagged = matches[matches["flags"] != ""]
    print(f"Flagged for review:    {len(flagged)}")
    print("\nSample matches:")
    print(matches.head(10).to_string(index=False))
    # quick accuracy check against ground truth
    correct = (bank.loc[matches["bank_idx"], "gt_id"].values ==
               check.loc[matches["check_idx"], "gt_id"].values).sum()
    print(f"\nGround-truth correct: {correct} / {len(matches)}")
