import csv
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from milestone_3.entities import Entity

RESULTS_DIR = Path("milestone_3/new_ground_truth/test_results")
ENTITIES_DIR = RESULTS_DIR / "entities"
SCORES_CSV = RESULTS_DIR / "scores.csv"
SCORES_SUMMARY_CSV = RESULTS_DIR / "scores_summary.csv"

METRIC_NAMES = ["TP", "FP", "FN", "precision", "recall", "f1"]
TARGETS = ["PER", "ORG"]
MODELS = ["rule_based", "flair", "spacy", "improved_spacy", "bert", "distilbert"]

def make_score_row(filename: str, model_name: str, eval_result: dict) -> dict:
    """Flatten the evaluation result into a CSV-friendly row."""
    row = {
        "filename": filename,
        "model": model_name,
        "macro_f1": eval_result["macro_f1"],
    }

    per_label = eval_result["per_label"]

    for label, stats in per_label.items():
        for metric_name in METRIC_NAMES:
            key = f"{metric_name}_{label}"
            row[key] = stats.get(metric_name, 0.0)

    return row


def load_doc_entities(json_path: Path):
    """Load the ground truth and predictions from the saved entity JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("filename", json_path.name)
    ground_truth = [Entity(**e) for e in data.get("ground_truth", [])]

    predictions_by_model = {}
    for model_name in MODELS:
        ents_data = data.get(model_name)
        if ents_data is None:
            continue
        predictions_by_model[model_name] = [Entity(**e) for e in ents_data]

    return filename, ground_truth, predictions_by_model


def evaluate(ground_truth: list, predictions: list) -> dict:
    """Calculate Precision, Recall, and F1 based on exact match (label, start, end)."""
    gt = {(e.label, e.start, e.end) for e in ground_truth}
    pred = {(e.label, e.start, e.end) for e in predictions}

    metrics = {}
    f1_sum = 0.0

    for lab in TARGETS:
        gt_lab = {(l, s, e) for (l, s, e) in gt if l == lab}
        pred_lab = {(l, s, e) for (l, s, e) in pred if l == lab}

        tp = len(pred_lab & gt_lab)
        fp = len(pred_lab - gt_lab)
        fn = len(gt_lab - pred_lab)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        
        f1_sum += f1

        metrics[lab] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = f1_sum / len(TARGETS) if TARGETS else 0.0

    return {
        "macro_f1": macro_f1,
        "per_label": metrics,
    }


def aggregate_results(score_rows: list) -> dict:
    """Aggregate results across all documents to get global metrics per model."""
    summary = {}

    for row in score_rows:
        model = row["model"]
        
        if model not in summary:
            summary[model] = {
                "num_docs": 0,
                "macro_f1_sum": 0.0,
                "per_label_counts": {lab: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for lab in TARGETS},
            }
        
        info = summary[model]
        info["num_docs"] += 1
        info["macro_f1_sum"] += float(row.get("macro_f1", 0.0))

        for lab in TARGETS:
            for m in ["TP", "FP", "FN"]:
                key = f"{m}_{lab}"
                info["per_label_counts"][lab][m] += float(row.get(key, 0.0))

    for model, info in summary.items():
        num_docs = info["num_docs"]
        info["macro_f1_mean"] = info["macro_f1_sum"] / num_docs if num_docs > 0 else 0.0

        per_label_metrics = {}
        for lab in TARGETS:
            counts = info["per_label_counts"][lab]
            TP = counts["TP"]
            FP = counts["FP"]
            FN = counts["FN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom > 0 else 0.0

            per_label_metrics[lab] = {
                "TP": TP, "FP": FP, "FN": FN,
                "precision": precision, "recall": recall, "f1": f1,
            }

        macro_f1_labels = (
            sum(v["f1"] for v in per_label_metrics.values()) / len(TARGETS)
            if TARGETS else 0.0
        )

        info["per_label"] = per_label_metrics
        info["macro_f1_macro_over_labels"] = macro_f1_labels

        del info["per_label_counts"]
        del info["macro_f1_sum"]

    return summary


def write_summary_csv(summary: dict):
    base_fields = ["model", "num_docs", "macro_f1_mean", "macro_f1_macro_over_labels"]
    metric_fields = [f"{m}_{lab}" for lab in TARGETS for m in METRIC_NAMES]
    fieldnames = base_fields + metric_fields

    rows = []
    for model, info in summary.items():
        row = {
            "model": model,
            "num_docs": info["num_docs"],
            "macro_f1_mean": info["macro_f1_mean"],
            "macro_f1_macro_over_labels": info["macro_f1_macro_over_labels"],
        }

        for lab, stats in info["per_label"].items():
            for metric_name in METRIC_NAMES:
                key = f"{metric_name}_{lab}"
                row[key] = stats[metric_name]

        rows.append(row)

    with SCORES_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict):
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY ({len(summary)} models)")
    print("="*60)
    for model, info in summary.items():
        print(f"Model: {model}")
        print(f"  Documents: {info['num_docs']}")
        print(f"  Mean Macro-F1 (avg of doc scores):   {info['macro_f1_mean']:.4f}")
        print(f"  Global Macro-F1 (aggregated counts): {info['macro_f1_macro_over_labels']:.4f}")
        
        for lab, stats in info["per_label"].items():
            print(
                f"    {lab}: F1={stats['f1']:.4f}, P={stats['precision']:.4f}, R={stats['recall']:.4f} "
                f"(TP={stats['TP']:.0f}, FP={stats['FP']:.0f}, FN={stats['FN']:.0f})"
            )
        print("-" * 30)

def main():
    if not ENTITIES_DIR.exists():
        print(f"Error: Entities directory not found at {ENTITIES_DIR}")
        print("Please run gt_pipeline.py first.")
        return

    entity_files = sorted(ENTITIES_DIR.glob("*_entities.json"))
    if not entity_files:
        print(f"No entity JSON files found in {ENTITIES_DIR}")
        return

    print(f"Found {len(entity_files)} files to evaluate...")
    score_rows = []

    for json_file in entity_files:
        filename, ground_truth, preds = load_doc_entities(json_file)

        for model in preds:
            eval_result = evaluate(ground_truth, preds[model])
            row = make_score_row(filename, model, eval_result)
            score_rows.append(row)

    base_fields = ["filename", "model", "macro_f1"]
    metric_fields = [f"{m}_{lab}" for lab in TARGETS for m in METRIC_NAMES]
    
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + metric_fields)
        writer.writeheader()
        writer.writerows(score_rows)
    
    print(f"Detailed scores saved to {SCORES_CSV}")

    summary = aggregate_results(score_rows)
    write_summary_csv(summary)
    print(f"Summary saved to {SCORES_SUMMARY_CSV}")
    
    print_summary(summary)

if __name__ == "__main__":
    main()