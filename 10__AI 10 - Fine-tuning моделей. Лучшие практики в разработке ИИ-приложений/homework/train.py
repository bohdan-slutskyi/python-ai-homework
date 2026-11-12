"""Fine-tune pretrained BERT-tiny for sentiment classification on a Mac M1.

Dependencies: torch, transformers, accelerate, numpy (see requirements.txt).
The held-out TEST set is never used for model/hyperparameter selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "reviews.tsv"
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "prajjwal1/bert-tiny"
SEED = 42
EPOCHS = int(os.environ.get("AI10_EPOCHS", "9"))
LEARNING_RATE = float(os.environ.get("AI10_LR", "0.0001"))
MAX_LENGTH = 96


def load_records() -> list[tuple[str, int]]:
    if not DATA.exists():
        raise FileNotFoundError("Run python prepare_data.py first")
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    items = [(row["text"], int(row["label"])) for row in rows]
    assert items and all(text.strip() and label in (0, 1) for text, label in items)
    assert len({text.casefold() for text, _ in items}) == len(items), "duplicates left in data"
    return items


def split_stratified(
    records: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """70/15/15 split per class; split only after cleaning/deduplication."""
    rng = random.Random(SEED)
    buckets = {label: [item for item in records if item[1] == label] for label in (0, 1)}
    splits: list[list[tuple[str, int]]] = [[], [], []]
    for label in (0, 1):
        rows = buckets[label][:]
        rng.shuffle(rows)
        n = len(rows)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        splits[0].extend(rows[:n_train])
        splits[1].extend(rows[n_train:n_train + n_val])
        splits[2].extend(rows[n_train + n_val:])
    for split in splits:
        rng.shuffle(split)
        assert set(label for _, label in split) == {0, 1}
    ids = [set(text.casefold() for text, _ in split) for split in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "data leakage"
    return splits[0], splits[1], splits[2]


class ReviewDataset(Dataset):
    def __init__(self, rows: list[tuple[str, int]], tokenizer):
        self.rows = rows
        self.encoding = tokenizer(
            [text for text, _ in rows],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = {key: value[i] for key, value in self.encoding.items()}
        row["labels"] = self.rows[i][1]
        return row


def classification_metrics(logits, labels) -> dict:
    predicted = np.asarray(logits).argmax(axis=-1)
    truth = np.asarray(labels)
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth)
    return {
        "accuracy": round(float(accuracy), 4),
        "precision_positive": round(precision, 4),
        "recall_positive": round(recall, 4),
        "f1_positive": round(f1, 4),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def compute_metrics(prediction):
    output = classification_metrics(prediction.predictions, prediction.label_ids)
    return {key: output[key] for key in ("accuracy", "precision_positive", "recall_positive", "f1_positive")}


def counts(rows: list[tuple[str, int]]) -> dict:
    c = Counter(label for _, label in rows)
    return {"rows": len(rows), "negative": c[0], "positive": c[1]}


def main() -> None:
    set_seed(SEED)
    records = load_records()
    train_rows, val_rows, test_rows = split_stratified(records)
    print("Dataset splits:", {k: counts(v) for k, v in
          (("train", train_rows), ("validation", val_rows), ("test", test_rows))}, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    train_ds = ReviewDataset(train_rows, tokenizer)
    val_ds = ReviewDataset(val_rows, tokenizer)
    test_ds = ReviewDataset(test_rows, tokenizer)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(ARTIFACTS / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_positive",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    print("Device:", trainer.args.device, flush=True)
    # Evaluate unadapted pretrained encoder with a newly initialized classifier
    # on VALIDATION only (TEST remains untouched until all training is finished).
    baseline_val = trainer.predict(val_ds)
    baseline_metrics = classification_metrics(baseline_val.predictions, baseline_val.label_ids)
    print("Baseline validation:", baseline_metrics, flush=True)

    training = trainer.train()
    validation = trainer.predict(val_ds)
    test = trainer.predict(test_ds)
    val_metrics = classification_metrics(validation.predictions, validation.label_ids)
    test_metrics = classification_metrics(test.predictions, test.label_ids)

    RESULTS.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ARTIFACTS / "best_model"))
    provenance = json.loads((ROOT / "data" / "provenance.json").read_text(encoding="utf-8"))
    summary = {
        "task": "binary sentiment classification of real Amazon product review sentences",
        "source": provenance["url"],
        "source_subset": provenance["subset"],
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "model": MODEL_NAME,
        "device": str(trainer.args.device),
        "seed": SEED,
        "train_config": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": 16, "max_tokens": MAX_LENGTH,
                         "select_best_epoch_by": "validation_f1_positive"},
        "counts": {"raw": provenance["raw_rows"], "after_cleaning": len(records),
                   "removed_duplicates": provenance["removed_duplicates"],
                   "train": counts(train_rows), "validation": counts(val_rows), "test": counts(test_rows)},
        "baseline_validation": baseline_metrics,
        "fine_tuned_validation": val_metrics,
        "fine_tuned_test": test_metrics,
        "training": training.metrics,
        "selected_epoch": max((entry for entry in trainer.state.log_history if "eval_f1_positive" in entry), key=lambda e: e["eval_f1_positive"])["epoch"],
        "trainer_log": trainer.state.log_history,
        "versions": {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
        "caution": "Exploratory educational experiment: test set was evaluated in a previous three-epoch trial too. It was not used for gradient updates or checkpoint selection, but iteration after inspecting that trial can introduce optimistic selection bias.",
    }
    path = RESULTS / "metrics.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred = np.asarray(test.predictions).argmax(axis=-1).tolist()
    with (RESULTS / "test_examples.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["review", "true_label", "predicted_label", "correct"])
        for (text, truth), guess in list(zip(test_rows, pred))[:12]:
            writer.writerow([text, "positive" if truth else "negative",
                             "positive" if guess else "negative", int(guess == truth)])
    print("Fine-tuned validation:", val_metrics, flush=True)
    print("Fine-tuned TEST:", test_metrics, flush=True)
    print("Saved results:", path, flush=True)


if __name__ == "__main__":
    main()
