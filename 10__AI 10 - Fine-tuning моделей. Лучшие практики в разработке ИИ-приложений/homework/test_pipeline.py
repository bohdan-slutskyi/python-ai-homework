"""Simple automated checks for preparation, leakage and classification metrics."""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from prepare_data import normalize
from train import DATA, ROOT, classification_metrics, load_records, split_stratified


class DataTests(unittest.TestCase):
    def test_cleaning(self):
        self.assertEqual(normalize("  Great   product! &amp; good  "), "Great product! & good")
        self.assertEqual(normalize("A <b>nice</b> case"), "A nice case")

    def test_dataset_counts_and_hash(self):
        rows = load_records()
        provenance = json.loads((ROOT / "data" / "provenance.json").read_text())
        self.assertEqual(len(rows), provenance["clean_rows"])
        self.assertEqual(hashlib.sha256(DATA.read_bytes()).hexdigest(), provenance["clean_tsv_sha256"])

    def test_stratification_determinism_and_no_overlap(self):
        records = load_records()
        one = split_stratified(records)
        two = split_stratified(records)
        self.assertEqual(one, two)
        self.assertEqual(sum(map(len, one)), len(records))
        train, val, test = one
        for a, b in ((train, val), (train, test), (val, test)):
            self.assertFalse({x.casefold() for x, _ in a} & {x.casefold() for x, _ in b})
        for subset in one:
            self.assertLess(abs(sum(y for _, y in subset) / len(subset) - 0.5), 0.04)

    def test_confusion_matrix(self):
        logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
        labels = [0, 1, 1, 0]
        metric = classification_metrics(logits, labels)
        self.assertEqual(metric["confusion_matrix"], {"TN": 1, "FP": 1, "FN": 1, "TP": 1})
        self.assertEqual(metric["accuracy"], 0.5)
        self.assertEqual(metric["f1_positive"], 0.5)


if __name__ == "__main__":
    unittest.main()
