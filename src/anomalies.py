"""Flag anomalous / suspicious matches.

A match is flagged as anomalous when any of the following hold:
  - confidence < 0.5                     -> system doesn't trust the pair
  - date gap > 5 days                    -> beyond stated drift tolerance
  - amount difference > 1% of amount     -> beyond rounding-noise tolerance
  - type mismatch                        -> DEBIT vs CREDIT
  - description similarity < 0.15        -> semantics disagree even though
                                            amount/date matched (e.g. bank
                                            says GYM MEMBERSHIP, register
                                            says Music subscription)

The description check is the key to catching the intentional anomalies in
the dataset: amounts and dates can line up coincidentally when both files
describe the same time period, but the semantic content of the descriptions
shouldn't be wildly different.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from embeddings import encode


DESC_SIM_FLOOR = 0.15


def flag(matches: pd.DataFrame, bank: pd.DataFrame, check: pd.DataFrame) -> pd.DataFrame:
    m = matches.copy()
    bank_rows = bank.loc[m["bank_idx"].to_numpy()].reset_index(drop=True)
    check_rows = check.loc[m["check_idx"].to_numpy()].reset_index(drop=True)

    m["date_gap_days"] = (bank_rows["date"].to_numpy() - check_rows["date"].to_numpy())
    m["date_gap_days"] = pd.Series(m["date_gap_days"]).dt.days.abs().to_numpy()
    m["amount_diff_pct"] = (
        (bank_rows["amount"].to_numpy() - check_rows["amount"].to_numpy()).__abs__()
        / bank_rows["amount"].abs().clip(lower=1).to_numpy() * 100
    )
    m["type_agrees"] = bank_rows["type_std"].to_numpy() == check_rows["type_std"].to_numpy()

    # Semantic description similarity (pairwise, only for matched rows)
    bank_emb = encode(bank_rows["description_clean"].tolist())
    check_emb = encode(check_rows["description_clean"].tolist())
    m["desc_sim"] = (bank_emb * check_emb).sum(axis=1)

    issues = []
    for _, r in m.iterrows():
        flags = []
        if r.get("confidence", 1.0) < 0.5:
            flags.append("low_confidence")
        if r["date_gap_days"] > 5:
            flags.append(f"date_gap={int(r['date_gap_days'])}d")
        if r["amount_diff_pct"] > 1.0:
            flags.append(f"amount_diff={r['amount_diff_pct']:.2f}%")
        if not r["type_agrees"]:
            flags.append("type_mismatch")
        if r["desc_sim"] < DESC_SIM_FLOOR:
            flags.append(f"desc_mismatch(sim={r['desc_sim']:.2f})")
        issues.append(",".join(flags))
    m["anomaly_flags"] = issues
    return m[m["anomaly_flags"] != ""]


if __name__ == "__main__":
    from load_data import load_bank, load_check
    bank, check = load_bank(), load_check()
    matches = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parents[1] /
                          "outputs" / "all_matches.csv")
    anomalies = flag(matches, bank, check)
    print(f"Anomalous matches: {len(anomalies)}")
    if len(anomalies):
        bank_desc = bank.loc[anomalies["bank_idx"].to_numpy(), "description"].to_numpy()
        check_desc = check.loc[anomalies["check_idx"].to_numpy(), "description"].to_numpy()
        anomalies = anomalies.assign(bank_desc=bank_desc, check_desc=check_desc)
        cols = ["bank_id", "check_id", "bank_desc", "check_desc",
                "desc_sim", "amount_diff_pct", "date_gap_days", "anomaly_flags"]
        cols = [c for c in cols if c in anomalies.columns]
        print(anomalies[cols].to_string(index=False))
