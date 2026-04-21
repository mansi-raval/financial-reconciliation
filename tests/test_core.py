"""Unit tests for the critical reconciliation primitives.

Run with:  python -m pytest tests/ -v
(or just: python -m unittest tests.test_core)
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_data import _normalize_type, _clean_description, _suffix
from features import (
    amount_similarity_matrix,
    date_similarity_matrix,
    type_match_matrix,
    cosine_similarity_matrix,
)
from unique_match import unique_amount_matches


class TestNormalization(unittest.TestCase):
    def test_type_mapping(self):
        self.assertEqual(_normalize_type("DR"), "DEBIT")
        self.assertEqual(_normalize_type("dr"), "DEBIT")
        self.assertEqual(_normalize_type("CREDIT"), "CREDIT")
        self.assertEqual(_normalize_type("CR"), "CREDIT")

    def test_description_cleaning(self):
        self.assertEqual(_clean_description("  BP GAS  #1775 "), "bp gas #1775")
        self.assertEqual(_clean_description("Trader\tJoes"), "trader joes")

    def test_suffix_extraction(self):
        self.assertEqual(_suffix("B0047"), 47)
        self.assertEqual(_suffix("R0308"), 308)
        self.assertEqual(_suffix("noid"), -1)


class TestFeatureMatrices(unittest.TestCase):
    def test_amount_similarity_perfect(self):
        m = amount_similarity_matrix(np.array([100.0, 50.0]), np.array([100.0, 50.0]))
        self.assertAlmostEqual(m[0, 0], 1.0)
        self.assertAlmostEqual(m[1, 1], 1.0)
        self.assertLess(m[0, 1], 1.0)

    def test_amount_similarity_rounding(self):
        m = amount_similarity_matrix(np.array([100.00]), np.array([100.01]))
        self.assertGreater(m[0, 0], 0.999)

    def test_date_similarity_decay(self):
        b = pd.to_datetime(pd.Series(["2023-01-10"]))
        c = pd.to_datetime(pd.Series(["2023-01-10", "2023-01-13"]))
        m = date_similarity_matrix(b, c, tau_days=3.0)
        self.assertAlmostEqual(m[0, 0], 1.0)
        self.assertAlmostEqual(m[0, 1], float(np.exp(-1)), places=3)

    def test_type_match(self):
        b = pd.Series(["DEBIT", "CREDIT"])
        c = pd.Series(["DEBIT", "CREDIT"])
        m = type_match_matrix(b, c)
        np.testing.assert_array_equal(m, np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_cosine_symmetric(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        m = cosine_similarity_matrix(emb, emb)
        np.testing.assert_allclose(m, np.eye(2))


class TestUniqueMatching(unittest.TestCase):
    def _mk(self, rows, prefix):
        df = pd.DataFrame(rows)
        df["description_clean"] = df["description"].str.lower()
        df["type_std"] = "DEBIT"
        df["gt_id"] = df["transaction_id"].str.extract(r"(\d+)$")[0].astype(int)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_singleton_amounts_match(self):
        bank = self._mk([
            {"transaction_id": "B0001", "date": "2023-01-01", "description": "A", "amount": 10.0, "type": "DEBIT"},
            {"transaction_id": "B0002", "date": "2023-01-02", "description": "B", "amount": 20.0, "type": "DEBIT"},
        ], "B")
        check = self._mk([
            {"transaction_id": "R0001", "date": "2023-01-01", "description": "a", "amount": 10.0, "type": "DR"},
            {"transaction_id": "R0002", "date": "2023-01-02", "description": "b", "amount": 20.0, "type": "DR"},
        ], "R")
        matches, rem_b, rem_c = unique_amount_matches(bank, check)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len(rem_b), 0)
        self.assertEqual(len(rem_c), 0)

    def test_duplicate_amounts_excluded(self):
        bank = self._mk([
            {"transaction_id": "B0001", "date": "2023-01-01", "description": "A", "amount": 10.0, "type": "DEBIT"},
            {"transaction_id": "B0002", "date": "2023-01-02", "description": "B", "amount": 10.0, "type": "DEBIT"},
            {"transaction_id": "B0003", "date": "2023-01-03", "description": "C", "amount": 99.0, "type": "DEBIT"},
        ], "B")
        check = self._mk([
            {"transaction_id": "R0001", "date": "2023-01-01", "description": "a", "amount": 10.0, "type": "DEBIT"},
            {"transaction_id": "R0002", "date": "2023-01-02", "description": "b", "amount": 10.0, "type": "DEBIT"},
            {"transaction_id": "R0003", "date": "2023-01-03", "description": "c", "amount": 99.0, "type": "DEBIT"},
        ], "R")
        matches, rem_b, rem_c = unique_amount_matches(bank, check)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]["amount"], 99.0)
        self.assertEqual(len(rem_b), 2)
        self.assertEqual(len(rem_c), 2)


if __name__ == "__main__":
    unittest.main()
