"""Step 3b: Encode transaction descriptions with sentence-transformers.

Model: all-MiniLM-L6-v2
  - 80 MB on disk, 384-dim output
  - Trained on 1B+ sentence pairs (NLI, Reddit, StackExchange etc.)
  - Runs on CPU in ~milliseconds for a few hundred short strings

Why this model over others:
  - Tiny and fast (good for Intel Mac / no GPU)
  - Strong semantic similarity on short phrases — which is exactly our case
  - Normalizes output so cosine similarity = dot product (faster math)

We also augment each description with its transaction type so semantically
identical descriptions with opposite directionality (e.g., a "refund") don't
incorrectly match a same-worded debit.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings as a (len(texts), 384) float32 array."""
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


if __name__ == "__main__":
    samples = ["bp gas #1775", "fill up", "kroger #6864", "food shopping", "trader joes"]
    emb = encode(samples)
    print(f"embeddings shape: {emb.shape}")
    sims = emb @ emb.T
    print("similarity matrix:")
    for i, s in enumerate(samples):
        row = " ".join(f"{v:+.2f}" for v in sims[i])
        print(f"  {s:20s} {row}")
