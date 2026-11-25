#!/usr/bin/env python3
"""
Better Me AI - Safety Classifier Evaluation Tool
Evaluates the BERT safety classifier against labeled test cases in safety_cases.csv.
Computes Confusion Matrix, Precision, Recall, F1-Score, and highlights High-Risk Recall.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.safety_service import safety_service

CLASSES = ["Safe", "Moderate", "High Risk"]


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Safety evaluation dataset not found at: {csv_path}")

    cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            text = (row.get("text") or "").strip()
            expected = (row.get("expected") or "").strip()
            if text and expected:
                cases.append({"index": idx, "text": text, "expected": expected})
    return cases


def normalize_class(name: str) -> str:
    cleaned = name.strip().lower()
    for c in CLASSES:
        if c.lower() == cleaned:
            return c
    return name


def run_evaluation(csv_path: Path = None):
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "safety_cases.csv"

    print("=" * 70)
    print("🛡️  BETTER ME — SAFETY CLASSIFIER EVALUATION REPORT")
    print("=" * 70)
    print(f"Loading dataset from: {csv_path}")

    cases = load_dataset(csv_path)
    if not cases:
        print("[ERROR] No valid test rows found in dataset!")
        return 1

    print(f"Loaded {len(cases)} test cases.\n")

    # Load model once
    safety_service.load_model()

    # Track predictions and confusion matrix
    # matrix[expected][predicted] = count
    matrix = {exp: {pred: 0 for pred in CLASSES} for exp in CLASSES}
    details = []

    correct_count = 0
    total_count = len(cases)

    for item in cases:
        text = item["text"]
        expected = normalize_class(item["expected"])

        pred_result = safety_service.predict(text)
        predicted = normalize_class(pred_result.risk_level)

        if expected not in matrix:
            matrix[expected] = {p: 0 for p in CLASSES}
        if predicted not in matrix[expected]:
            matrix[expected][predicted] = 0

        matrix[expected][predicted] += 1
        is_correct = (expected == predicted)
        if is_correct:
            correct_count += 1

        details.append({
            "index": item["index"],
            "text": text,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": pred_result.probabilities.get(pred_result.risk_level, 0.0)
        })

    # --- 1. Confusion Matrix Display ---
    print("\n--- 1. CONFUSION MATRIX (Rows: Expected, Columns: Predicted) ---")
    title_col = "Expected \\ Predicted"
    header = f"{title_col:<24} | " + " | ".join(f"{c:<10}" for c in CLASSES) + " | Total"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for exp in CLASSES:
        row_counts = [matrix[exp].get(p, 0) for p in CLASSES]
        total_exp = sum(row_counts)
        row_str = f"{exp:<24} | " + " | ".join(f"{cnt:<10}" for cnt in row_counts) + f" | {total_exp}"
        print(row_str)

    print("-" * len(header))
    col_totals = [sum(matrix[exp].get(p, 0) for exp in CLASSES) for p in CLASSES]
    print(f"{'Total Predicted':<24} | " + " | ".join(f"{cnt:<10}" for cnt in col_totals) + f" | {total_count}")

    # --- 2. Per-Class Metrics ---
    print("\n--- 2. CLASSIFICATION METRICS ---")
    print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)

    per_class = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0

    for c in CLASSES:
        tp = matrix[c].get(c, 0)
        fp = sum(matrix[other].get(c, 0) for other in CLASSES if other != c)
        fn = sum(matrix[c].get(other, 0) for other in CLASSES if other != c)
        support = sum(matrix[c].get(other, 0) for other in CLASSES)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        print(f"{c:<16} | {precision:>9.2%} | {recall:>9.2%} | {f1:>9.2%} | {support:>8}")

    n_classes = len(CLASSES)
    macro_p /= n_classes
    macro_r /= n_classes
    macro_f1 /= n_classes
    overall_acc = (correct_count / total_count) if total_count > 0 else 0.0

    print("-" * 65)
    print(f"{'Overall Accuracy':<16} | {'-':<10} | {'-':<10} | {overall_acc:>9.2%} | {total_count:>8}")
    print(f"{'Macro Average':<16} | {macro_p:>9.2%} | {macro_r:>9.2%} | {macro_f1:>9.2%} | {total_count:>8}")

    # --- 3. High-Risk Safety Gate Highlight ---
    high_risk_metrics = per_class.get("High Risk", {"recall": 0.0, "precision": 0.0, "support": 0})
    hr_recall = high_risk_metrics["recall"]
    hr_support = high_risk_metrics["support"]

    print("\n" + "=" * 70)
    print("🚨 CLINICAL SAFETY AUDIT: HIGH-RISK CRISIS RECALL")
    print("=" * 70)
    print(f"High-Risk Recall:    {hr_recall:.2%} ({int(round(hr_recall * hr_support))}/{hr_support} cases identified)")
    print(f"High-Risk Precision: {high_risk_metrics['precision']:.2%}")

    if hr_recall == 1.0:
        print("\n✅ PASS: 100% of suicidal / acute self-harm triggers were caught.")
        print("   Deterministic crisis gate & BERT safety classifier protected user safely.")
    else:
        missed = [d for d in details if d["expected"] == "High Risk" and not d["correct"]]
        print(f"\n❌ WARNING: {len(missed)} High-Risk case(s) failed to trigger High Risk risk level!")
        for m in missed:
            print(f"   - Row {m['index']}: '{m['text']}' (Predicted: {m['predicted']})")

    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(run_evaluation(path))
