# Financial Reconciliation System

Automatically matches transactions between two independent financial data sources
(bank statements and internal check register) using unsupervised machine learning,
inspired by Peter A. Chew (2020).

---

## How to Run

```bash
# 1. Create virtual environment (only once)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

That's it. One command runs everything:

| Step | What happens |
|---|---|
| Step 1 | Load and normalize both CSVs |
| Step 2 | Match transactions with unique amounts (286 matched instantly) |
| Step 3 | Use ML to match the remaining 22 ambiguous transactions |
| Step 4 | Save all 308 matches to `outputs/all_matches.csv` |
| Step 5 | Compute Precision / Recall / F1 |
| Step 6 | Flag low-confidence matches for human review |
| Step 7 | Show how F1 improves as more training data is added |

---

## Results

```
unique_stage     P=1.0000  R=0.9286  F1=0.9630  (286/286 correct)
ml_stage         P=1.0000  R=1.0000  F1=1.0000  (22/22 correct)
combined         P=1.0000  R=1.0000  F1=1.0000  (308/308 correct)
```

See `ANALYSIS.md` for full analysis of performance, design decisions, and limitations.

---

## Approach

**Stage 1 — Unique-amount matching:**  
If an amount appears exactly once in the bank file AND exactly once in the check
register, those two rows almost certainly describe the same transaction. 286 out of
308 transactions are resolved this way with 100% precision.

**Stage 2 — ML matching for the remaining 22:**  
For transactions with duplicate amounts, we compute four similarity features for
every candidate (bank, check) pair:
- **Description similarity** — sentence embeddings via `all-MiniLM-L6-v2`, cosine similarity
- **Amount similarity** — penalises rounding differences
- **Date proximity** — exponential decay, handles the ±5 day drift
- **Type match** — DEBIT/CREDIT vs DR/CR normalised and compared

A logistic regression (trained on validated pairs) learns the optimal weight for each
feature. The final score matrix is solved with the **Hungarian algorithm** for a
globally optimal 1-to-1 assignment.

**The match → review → improve cycle:**  
Low-confidence matches are flagged automatically for human review. Every validated
match gets added back as training data, improving future runs (demonstrated via the
learning curve in Step 7).

---

## Project Structure

```
.
├── main.py                  # Run this — does everything
├── requirements.txt
├── ANALYSIS.md              # Full performance analysis and design decisions
├── data/
│   ├── bank_statements.csv
│   └── check_register.csv
├── src/
│   ├── load_data.py         # Load CSVs, normalize descriptions/types/dates
│   ├── unique_match.py      # Stage 1: match singleton amounts
│   ├── features.py          # Amount / date / type similarity matrices
│   ├── embeddings.py        # Sentence-transformer wrapper
│   ├── matcher.py           # Stage 2: feature fusion + Hungarian assignment
│   ├── evaluate.py          # Precision / Recall / F1
│   ├── learning_curve.py    # F1 vs training-data size
│   ├── anomalies.py         # Flag semantically suspicious matches
│   └── logger.py            # Logging setup (console + file)
├── tests/
│   └── test_core.py         # Unit tests (run: python -m unittest tests.test_core -v)
└── outputs/                 # Generated on run (gitignored)
    ├── all_matches.csv
    ├── learning_curve.png
    └── run.log
```

---

## Reference

Peter A. Chew. 2020. *Unsupervised-Learning Financial Reconciliation: a Robust,
Accurate Approach Inspired by Machine Translation.* ICAIF '20, New York, NY, USA.
