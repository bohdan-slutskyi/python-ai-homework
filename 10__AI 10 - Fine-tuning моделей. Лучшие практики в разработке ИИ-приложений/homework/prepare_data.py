"""Prepare a small real review dataset for fine-tuning (stdlib only).

Source: UCI Sentiment Labelled Sentences (Kotzias et al., KDD 2015),
the amazon_cells_labelled.txt portion (500 positive, 500 negative).
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
MEMBER = "sentiment labelled sentences/amazon_cells_labelled.txt"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "reviews.tsv"
PROVENANCE = ROOT / "data" / "provenance.json"


def normalize(text: str) -> str:
    """Normalize simple review noise without changing sentiment words."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw_lines = archive.read(MEMBER).decode("utf-8").splitlines()

    unique: dict[str, tuple[str, int]] = {}
    duplicates = 0
    conflicts = 0
    for line in raw_lines:
        sentence, raw_label = line.rsplit("\t", maxsplit=1)
        text = normalize(sentence)
        if not text or raw_label not in {"0", "1"}:
            raise ValueError("Unexpected/empty data row")
        label = int(raw_label)
        key = text.casefold()
        if key in unique:
            duplicates += 1
            if unique[key][1] != label:
                # Never keep contradictory labels.
                conflicts += 1
                raise ValueError(f"Conflicting labels for duplicate: {text!r}")
            continue
        unique[key] = (text, label)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    # Preserve original order. No split is performed before deduplication.
    with DATA.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "label"])
        writer.writerows(unique.values())

    counts = Counter(label for _, label in unique.values())
    provenance = {
        "source": "UCI Machine Learning Repository — Sentiment Labelled Sentences",
        "url": URL,
        "subset": MEMBER,
        "publication": "Kotzias et al. (2015), From Group to Individual Labels using Deep Features, KDD",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "clean_tsv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "raw_rows": len(raw_lines),
        "clean_rows": len(unique),
        "removed_duplicates": duplicates,
        "conflicting_duplicates": conflicts,
        "label_counts": {"negative": counts[0], "positive": counts[1]},
        "cleaning": ["HTML character unescape", "Unicode NFKC", "strip HTML tags", "normalize whitespace", "case-insensitive deduplication"],
    }
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
