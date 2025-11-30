import csv
import json
from pathlib import Path

from milestone_2.entities import Entity

RESULTS_DIR = Path("milestone_2/results")
ENTITIES_DIR = RESULTS_DIR / "entities"
SCORES_CSV = RESULTS_DIR / "scores.csv"
SCORES_SUMMARY_CSV = RESULTS_DIR / "scores_summary.csv"

RESULTS_DIR.mkdir(exist_ok=True)
ENTITIES_DIR.mkdir(exist_ok=True)

METRIC_NAMES = ["TP", "FP", "FN", "precision", "recall", "f1"]
TARGETS = ["LOC", "PER", "ORG"]
MODELS = ["rule_based", "flair", "spacy"]


def make_score_row(filename: str, model_name: str, eval_result: dict) -> dict:
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
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filename = data.get("filename", json_path.name)
    ground_truth = [Entity(**e) for e in data.get("ground_truth", [])]

    predictions_by_model: dict[str, list[Entity]] = {}
    for model_name in MODELS:
        ents_data = data.get(model_name)
        if ents_data is None:
            continue
        predictions_by_model[model_name] = [Entity(**e) for e in ents_data]

    return filename, ground_truth, predictions_by_model


def evaluate(ground_truth: list[Entity], predictions: list[Entity]) -> dict:
    gt = {(e.label, e.start, e.end) for e in ground_truth}
    pred = {(e.label, e.start, e.end) for e in predictions}

    metrics = {}
    precision = {}
    recall = {}
    f1_score = {}
    mean_f1 = 0.0

    for lab in TARGETS:
        tp = len({(x, y, z) for (x, y, z) in (pred & gt) if x == lab})
        fp = len({(x, y, z) for (x, y, z) in (pred - gt) if x == lab})
        fn = len({(x, y, z) for (x, y, z) in (gt - pred) if x == lab})

        precision[lab] = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall[lab] = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        num = 2 * precision[lab] * recall[lab]
        denom = precision[lab] + recall[lab]
        f1_score[lab] = num / denom if denom != 0 else 0.0
        mean_f1 += f1_score[lab]

        metrics[lab] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision[lab],
            "recall": recall[lab],
            "f1": f1_score[lab],
        }

    macro_f1 = mean_f1 / len(TARGETS)

    return {
        "macro_f1": macro_f1,
        "per_label": metrics,
    }


def aggregate_results(score_rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}

    for row in score_rows:
        model = row["model"]
        model_info = summary.setdefault(
            model,
            {
                "num_docs": 0,
                "macro_f1_sum": 0.0,
                "per_label_counts": {
                    lab: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for lab in TARGETS
                },
            },
        )

        model_info["num_docs"] += 1
        model_info["macro_f1_sum"] += float(row.get("macro_f1", 0.0))

        for lab in TARGETS:
            for m in ["TP", "FP", "FN"]:
                key = f"{m}_{lab}"
                model_info["per_label_counts"][lab][m] += float(row.get(key, 0.0))

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
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        macro_f1_labels = (
            sum(v["f1"] for v in per_label_metrics.values()) / len(TARGETS)
            if TARGETS
            else 0.0
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
    for model, info in summary.items():
        print(f"Model: {model}")
        print(f"  Documents: {info['num_docs']}")
        print(f"  Mean macro-F1 over documents: {info['macro_f1_mean']:.4f}")
        print(
            f"  Macro-F1 over labels (from aggregated counts): "
            f"{info['macro_f1_macro_over_labels']:.4f}"
        )
        for lab, stats in info["per_label"].items():
            print(
                f"    {lab}: F1={stats['f1']:.4f}, "
                f"P={stats['precision']:.4f}, R={stats['recall']:.4f} "
                f"(TP={stats['TP']:.0f}, FP={stats['FP']:.0f}, FN={stats['FN']:.0f})"
            )
        print()


def main():
    entity_files = sorted(ENTITIES_DIR.glob("*_entities.json"))

    score_rows: list[dict] = []

    for json_file in entity_files:
        filename, ground_truth, predictions_by_model = load_doc_entities(json_file)

        for model_name, preds in predictions_by_model.items():
            eval_result = evaluate(ground_truth, preds)
            row = make_score_row(filename, model_name, eval_result)
            score_rows.append(row)

    base_fields = ["filename", "model", "macro_f1"]
    metric_fields = [f"{m}_{lab}" for lab in TARGETS for m in METRIC_NAMES]
    fieldnames = base_fields + metric_fields

    with SCORES_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    summary = aggregate_results(score_rows)
    write_summary_csv(summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
