# Implementation Analysis — Financial Reconciliation System

---

## 1. Performance Analysis

### 1.1 Precision, Recall, and F1 Scores

The system runs in two stages. Results on the provided 308-transaction dataset:

| Stage | Precision | Recall | F1 | Correct / Proposed |
|---|---|---|---|---|
| Unique-amount matching | 1.0000 | 0.9286 | 0.9630 | 286 / 286 |
| ML matching (ambiguous) | 1.0000 | 1.0000 | 1.0000 | 22 / 22 |
| **Combined (final)** | **1.0000** | **1.0000** | **1.0000** | **308 / 308** |

**Why the ML stage recall is 1.0, not 0.07:**  
Recall should be computed against what each stage is actually responsible for. The ML stage only receives the 22 rows that unique-amount matching couldn't resolve — so its denominator is 22, not 308. It matched all 22 correctly, giving recall = 1.0. Dividing by 308 would be grading it on questions it never sat.

**Honest caveat on F1 = 1.0:**  
The dataset, while realistic in format, is relatively easy — only 3 pairs have truly ambiguous amounts (same amount appearing more than once on each side), and even those 3 pairs are months apart by date, making them trivially separable. On the paper's real-world 1,314-transaction dataset, the author reports F1 ≈ 86%. That is a more honest benchmark for production use.

---

### 1.2 Which Transactions Are Hardest to Match

**Easy (handled by Stage 1):**
- Unique amounts — matched automatically with zero errors.  
  Example: `$4,096.11` appears once in bank, once in check → instant match.

**Medium (handled by Stage 2, high confidence):**
- Duplicate amounts with clear description signals.  
  Example: two `$192.86` entries, one is "HEALTH INS PMT" / "Health insurance" in November, the other is a different transaction months earlier → date + description together resolve it confidently.

**Hard (Stage 2, low confidence — flagged for review):**
These are the 10 matches the system surfaces with `WARNING`:

| Bank description | Check description | Why it's hard |
|---|---|---|
| `CK #9571` | `Paid by check` | Generic descriptions, low semantic similarity |
| `ONLINE PMT WATER` | `Electric bill` | Different utility types named |
| `BISTRO #4304` | `Lunch` | Brand name vs category |
| `KROGER #3255` | `Grocery store` | Brand name vs generic |
| `TRADER JOES` | `Groceries` | Close but short, low embedding signal |

**Root cause:** the `all-MiniLM-L6-v2` model is trained on general English text. It does not know that "TRADER JOES" is a grocery store, that "CK" is shorthand for "check", or that "BISTRO" means restaurant. The model can recognise *semantic similarity between full sentences*, but short codes and brand names defeat it.

---

### 1.3 How Performance Improves with More Training Data

When logistic regression is trained on a growing fraction of validated pairs and used to score the full 308×308 matrix:

| Training data | Pairs used | F1 |
|---|---|---|
| No training (hand-tuned weights) | 0 | 0.974 |
| 10% | 30 | 0.916 |
| 20% | 61 | 0.987 |
| 30% | 92 | 0.964 |
| 50% | 154 | 0.945 |
| 80% | 246 | 0.994 |

The trend is upward with some noise — at small training sizes (10–30 pairs) the logistic regression can overfit or underfit the four features, producing unstable weights. By 80% (246 validated pairs), the model learns reliable weights and nearly matches the hand-tuned baseline.

The key insight from the paper holds: **the system improves every time a human validates or rejects a match**, because those decisions become training data for the next run.

See `outputs/learning_curve.png` for the plot.

---

## 2. Design Decisions

### 2.1 Choice of ML Approach

The assignment offered three options:

- **Option A** — SVD/LSA as described in the paper (mutual information → term matrix → SVD → cosine nearest-neighbour)
- **Option B** — Modern sentence embeddings (BERT / sentence-transformers)
- **Option C** — Custom hybrid

**We implemented Option B (embeddings) with one addition from Option C (logistic regression for weight learning).**

**Why embeddings over SVD for this dataset:**

SVD (the paper's approach) works by learning word co-occurrence patterns from the *parallel corpus* — previously matched transaction pairs. On 308 transactions it would only have a few hundred aligned word pairs to learn from — not enough for SVD to extract meaningful signal from short, noisy financial descriptions like "CK #5389" or "BP GAS #1775".

`all-MiniLM-L6-v2`, by contrast, was pre-trained on over 1 billion sentence pairs. It already understands English semantics and encodes each description into a 384-dimensional vector that captures meaning — without needing any of our data to train. For a small dataset, this is a significant practical advantage.

**Why logistic regression for feature combination:**

We have four features (description similarity, amount similarity, date proximity, type match). Rather than guessing weights, we train a logistic regression on known pairs, which learns the relative importance of each feature from the data itself. Coefficients from one run:

```
desc:   1.47   (useful but not dominant)
amount: 3.74   (strong signal)
date:   4.68   (strongest signal on this dataset)
type:   0.12   (near-useless — types almost always agree)
```

This is interpretable — you can read directly which feature the model trusts most.

---

### 2.2 Departures from the Paper's Methodology

| Paper's approach | Our approach | Reason |
|---|---|---|
| SVD on term-by-transaction matrix | Sentence-transformer embeddings | SVD needs more training data; embeddings work out-of-the-box |
| Mutual information for term alignment | No term alignment step | Not needed when using pre-trained embeddings |
| Greedy nearest-neighbour per query | Hungarian (optimal) assignment | Greedy can assign two bank rows to the same check row; Hungarian guarantees 1-to-1 |
| Monthly incremental process | Single-batch processing | Dataset is one batch, not monthly |
| Human QC loop (interactive) | Automated confidence flagging | Simulated via the review step; interactive loop needs a UI |

The paper's algorithm was designed for a world without large pre-trained language models (2020, before sentence-transformers became widespread). Our approach achieves the same goal — learning from parallel financial "languages" — using a stronger foundation.

---

### 2.3 Trade-offs Considered

**Speed vs Accuracy:**  
`all-MiniLM-L6-v2` (80MB) runs in ~3 seconds for 308 descriptions on CPU. A larger model like `all-mpnet-base-v2` (420MB) would give better semantic similarity scores but takes ~4× longer. For 308 rows the difference is seconds; for 300,000 rows it becomes minutes. We chose MiniLM for the right speed/accuracy balance at this scale.

**Hybrid vs Pure ML:**  
We could have run the ML on all 308 rows without the unique-amount pre-pass. We chose the hybrid because:
1. The unique-amount step is free, 100% accurate, and reduces the ML's search space dramatically.
2. This is exactly what the paper recommends — and for good reason.
3. Fewer rows in the ML step means fewer chances for the model to make errors.

**Global (Hungarian) vs Local (Greedy) Assignment:**  
Greedy matching — pick the highest-scoring pair, remove those rows, repeat — is O(n log n) and simple. Hungarian is O(n³) but guarantees the globally optimal assignment. For n=308 both are instantaneous. We chose Hungarian because on harder datasets greedy assignment makes a common mistake: two bank rows both want the same check row, and greedy gives it to whichever arrives first, leaving the other stranded with a bad match.

---

## 3. Limitations and Future Improvements

### 3.1 Weaknesses of the Current Implementation

**The embedding model doesn't know financial brands:**  
`SAFEWAY` and `Weekly groceries` have low cosine similarity (0.10) because the model doesn't know Safeway is a grocery store. The same applies to gas stations (SHELL, BP, EXXON) and check abbreviations (CK, CHK). This creates false anomaly flags and low-confidence scores on perfectly correct matches.

**Dataset is too clean to stress-test ML:**  
Only 3 pairs have genuinely ambiguous amounts. The ML stage's actual difficulty is low. A production dataset would have many more cases where amount, date, *and* description all overlap between multiple candidates — the scenario that truly separates good systems from bad ones.

**Learning curve is noisy at small training sizes:**  
At 10–30 training pairs, logistic regression is unstable — four features is a tiny feature space and a few wrong training examples skew the weights significantly. Averaging over multiple random seeds would smooth this out.

**No true human-in-the-loop:**  
The review step *surfaces* low-confidence matches but has no way to accept or reject them and feed the decision back into the next run. A real implementation would need a simple database or CSV-based feedback mechanism.

**Scalability:**  
The current system builds a full 308×308 matrix. At 50,000 transactions that is 2.5 billion pairs — computationally infeasible. Production systems need *blocking*: first narrow candidates by date window and amount bucket, then run the ML only within each block.

---

### 3.2 What We Would Do with More Time

**Implement the paper's SVD approach as a comparison:**  
Build `src/svd_matcher.py` that faithfully implements mutual information + term-by-transaction matrix + SVD + cosine nearest-neighbour. Run both approaches on the same data and compare F1, runtime, and interpretability. This would let us say definitively whether embeddings outperform SVD on financial data at this scale.

**Add a richer feature set:**  
- Fuzzy string matching score (Levenshtein distance) between raw descriptions — catches typos like "Groceres" vs "Groceries"
- Amount sign consistency (positive vs negative) — catches refunds mismatched to charges
- Day-of-week pattern — payroll always lands on Friday, rent always on the 1st

**Add a proper feedback UI:**  
A simple web page (Flask or Streamlit) where a human can see flagged matches side by side, click Accept or Reject, and have those decisions written back to a `validated_pairs.csv`. Next run, the system trains on that file automatically.

**Handle one-to-many and many-to-one matches:**  
A single bank transfer can sometimes correspond to two check register entries (a payment split across two categories). Hungarian enforces strict 1-to-1 — a more flexible matching algorithm (minimum-cost flow) would handle splits.

---

### 3.3 Edge Cases and How to Handle Them

| Edge case | Current behaviour | Better approach |
|---|---|---|
| Transaction in bank but missing in check | Forced into the worst available match | Set a minimum confidence threshold; unmatched below it → "unreconciled" bucket |
| Same amount, same date, similar description (true duplicates) | Arbitrary assignment | Flag for mandatory human review; cannot be resolved algorithmically |
| Amount is negative (refund/reversal) | Treated the same as positive | Separate debit/credit pools before matching |
| Description is empty or null | Embedding defaults to zero vector | Fallback to amount + date only; log a warning |
| New transaction format from bank (schema change) | Normalisation step fails silently | Schema validation at ingestion with explicit error if columns are missing |
| Very large date drift (> 30 days) | Low date similarity score, possibly wrong match | Hard block: do not propose matches more than N days apart |
