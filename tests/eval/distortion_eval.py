#!/usr/bin/env python3
"""
Better Me AI - Cognitive Distortion Classifier Evaluation Tool
Evaluates the BERT cognitive distortion classifier against labeled test cases in distortion_cases.csv.
Computes Confusion Matrix, per-class Precision, Recall, F1-Score, and Macro/Weighted averages.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.distortion_service import distortion_service, CANONICAL_DISTORTIONS

CANONICAL_CLASSES = list(CANONICAL_DISTORTIONS.values())


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Distortion evaluation dataset not found at: {csv_path}")

    cases = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            text = (row.get("text") or "").strip()
            expected = (row.get("expected") or "").strip()
            if text and expected:
                cases.append({"index": idx, "text": text, "expected": expected})
    return cases


def normalize_distortion(name: str) -> str:
    cleaned = name.strip().lower()
    for c in CANONICAL_CLASSES:
        if c.lower() == cleaned:
            return c
    return name


def run_evaluation(csv_path: Path = None):
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "distortion_cases.csv"

    print("=" * 80)
    print("🧠 BETTER ME — COGNITIVE DISTORTION CLASSIFIER EVALUATION REPORT")
    print("=" * 80)
    print(f"Loading dataset from: {csv_path}")

    cases = load_dataset(csv_path)
    if not cases:
        print("[ERROR] No valid test rows found in dataset!")
        return 1

    print(f"Loaded {len(cases)} test cases.\n")

    # Load model once
    distortion_service.load_model()

    # Determine all unique classes present in expected dataset + canonical
    all_classes = CANONICAL_CLASSES.copy()
    for item in cases:
        exp = normalize_distortion(item["expected"])
        if exp not in all_classes:
            all_classes.append(exp)

    # matrix[expected][predicted] = count
    matrix = {exp: {pred: 0 for pred in all_classes} for exp in all_classes}
    details = []

    correct_count = 0
    total_count = len(cases)

    for item in cases:
        text = item["text"]
        expected = normalize_distortion(item["expected"])

        pred_result = distortion_service.predict(text)
        predicted = normalize_distortion(pred_result.predicted_class)

        if predicted not in all_classes:
            all_classes.append(predicted)
            for k in matrix:
                matrix[k][predicted] = 0
            matrix[predicted] = {p: 0 for p in all_classes}

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
            "confidence": pred_result.confidence
        })

    # Filter classes that have at least 1 expected or 1 predicted occurrence to keep matrix clean
    active_classes = [c for c in all_classes if sum(matrix[c].values()) > 0 or any(matrix[exp][c] > 0 for exp in all_classes)]
    if not active_classes:
        active_classes = CANONICAL_CLASSES

    # --- 1. Confusion Matrix Display ---
    col_width = 12
    print("\n--- 1. CONFUSION MATRIX (Rows: Expected, Columns: Predicted) ---")
    header_cols = [f"{c[:col_width]:<{col_width}}" for c in active_classes]
    title_col = "Expected \\ Pred"
    header = f"{title_col:<26} | " + " | ".join(header_cols) + " | Total"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for exp in active_classes:
        row_counts = [matrix[exp].get(p, 0) for p in active_classes]
        total_exp = sum(row_counts)
        row_cells = [f"{cnt:<{col_width}}" for cnt in row_counts]
        row_str = f"{exp[:25]:<26} | " + " | ".join(row_cells) + f" | {total_exp}"
        print(row_str)

    print("-" * len(header))
    col_totals = [sum(matrix[exp].get(p, 0) for exp in active_classes) for p in active_classes]
    col_cells = [f"{cnt:<{col_width}}" for cnt in col_totals]
    print(f"{'Total Predicted':<26} | " + " | ".join(col_cells) + f" | {total_count}")

    # --- 2. Per-Class Metrics ---
    print("\n--- 2. CLASSIFICATION METRICS ---")
    print(f"{'Distortion Class':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)

    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f1 = 0.0
    evaluated_classes_count = 0

    for c in active_classes:
        tp = matrix[c].get(c, 0)
        fp = sum(matrix[other].get(c, 0) for other in active_classes if other != c)
        fn = sum(matrix[c].get(other, 0) for other in active_classes if other != c)
        support = sum(matrix[c].get(other, 0) for other in active_classes)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if support > 0 or (tp + fp) > 0:
            macro_p += precision
            macro_r += recall
            macro_f1 += f1
            weighted_p += precision * support
            weighted_r += recall * support
            weighted_f1 += f1 * support
            evaluated_classes_count += 1

        print(f"{c:<28} | {precision:>9.2%} | {recall:>9.2%} | {f1:>9.2%} | {support:>8}")

    overall_acc = (correct_count / total_count) if total_count > 0 else 0.0
    if evaluated_classes_count > 0:
        macro_p /= evaluated_classes_count
        macro_r /= evaluated_classes_count
        macro_f1 /= evaluated_classes_count
    if total_count > 0:
        weighted_p /= total_count
        weighted_r /= total_count
        weighted_f1 /= total_count

    print("-" * 75)
    print(f"{'Overall Accuracy':<28} | {'-':<10} | {'-':<10} | {overall_acc:>9.2%} | {total_count:>8}")
    print(f"{'Macro Average':<28} | {macro_p:>9.2%} | {macro_r:>9.2%} | {macro_f1:>9.2%} | {total_count:>8}")
    print(f"{'Weighted Average':<28} | {weighted_p:>9.2%} | {weighted_r:>9.2%} | {weighted_f1:>9.2%} | {total_count:>8}")

    # --- 3. Summary & Misclassifications ---
    misclassified = [d for d in details if not d["correct"]]
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: {correct_count}/{total_count} ({overall_acc:.2%}) correct predictions.")
    if misclassified:
        print(f"\nMisclassified Cases ({len(misclassified)}):")
        for m in misclassified:
            print(f"   - Row {m['index']}: '{m['text'][:50]}...'")
            print(f"     Expected:  {m['expected']}")
            print(f"     Predicted: {m['predicted']} (Confidence: {m['confidence']:.2%})")
    else:
        print("✅ All test cases predicted correctly!")

    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    sys.exit(run_evaluation(path))
