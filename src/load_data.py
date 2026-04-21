"""Step 1: Load the two CSVs and prepare them for matching.

Key normalizations done here (so later stages see clean, comparable data):
- dates -> pandas datetime
- descriptions -> lowercased, collapsed whitespace
- transaction types -> standardized ('DR' / 'DEBIT' -> 'DEBIT', 'CR'/'CREDIT' -> 'CREDIT')
- a numeric suffix extracted from transaction_id to serve as GROUND TRUTH
  (B0047 / R0047 share suffix 47). We use this ONLY for evaluation — never
  as a feature for matching.
"""

from pathlib import Path
import re
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _normalize_type(t: str) -> str:
    t = str(t).strip().upper()
    if t in {"DR", "DEBIT"}:
        return "DEBIT"
    if t in {"CR", "CREDIT"}:
        return "CREDIT"
    return t


def _clean_description(s: str) -> str:
    s = str(s).lower().strip()
    return re.sub(r"\s+", " ", s)


def _suffix(tid: str) -> int:
    m = re.search(r"(\d+)$", str(tid))
    return int(m.group(1)) if m else -1


def load_bank(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "bank_statements.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["description_clean"] = df["description"].map(_clean_description)
    df["type_std"] = df["type"].map(_normalize_type)
    df["gt_id"] = df["transaction_id"].map(_suffix)
    return df


def load_check(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "check_register.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["description_clean"] = df["description"].map(_clean_description)
    df["type_std"] = df["type"].map(_normalize_type)
    df["gt_id"] = df["transaction_id"].map(_suffix)
    return df


if __name__ == "__main__":
    bank = load_bank()
    check = load_check()
    print(f"bank:  {len(bank)} rows, {bank.columns.tolist()}")
    print(f"check: {len(check)} rows, {check.columns.tolist()}")
    print("\nBank sample:")
    print(bank[["transaction_id", "date", "description_clean", "amount", "type_std"]].head())
    print("\nCheck sample:")
    print(check[["transaction_id", "date", "description_clean", "amount", "type_std"]].head())
    # quick sanity: do suffixes have 1-1 correspondence?
    overlap = set(bank["gt_id"]) & set(check["gt_id"])
    print(f"\nGround-truth suffix overlap: {len(overlap)} / {len(bank)}")
