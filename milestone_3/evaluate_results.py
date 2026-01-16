import csv
import json
from pathlib import Path

from milestone_3.entities import Entity

RESULTS_DIR = Path("milestone_3/results")
ENTITIES_DIR = RESULTS_DIR / "entities"
SCORES_CSV = RESULTS_DIR / "scores.csv"
SCORES_SUMMARY_CSV = RESULTS_DIR / "scores_summary.csv"

SCORES_SET_CSV = RESULTS_DIR / "scores_set.csv"
SCORES_SET_SUMMARY_CSV = RESULTS_DIR / "scores_set_summary.csv"

RESULTS_DIR.mkdir(exist_ok=True)
ENTITIES_DIR.mkdir(exist_ok=True)

METRIC_NAMES = ["TP", "FP", "FN", "precision", "recall", "f1"]
TARGETS = ["PER", "ORG"]
MODELS = ["rule_based", "flair", "spacy", "improved_spacy", "bert", "distilbert"]


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


def evaluate_span_strict(ground_truth: list[Entity], predictions: list[Entity]) -> dict:
    gt = {(e.label, e.start, e.end) for e in ground_truth}
    pred = {(e.label, e.start, e.end) for e in predictions}

    metrics = {}
    mean_f1 = 0.0

    for lab in TARGETS:
        tp = len({(x, y, z) for (x, y, z) in (pred & gt) if x == lab})
        fp = len({(x, y, z) for (x, y, z) in (pred - gt) if x == lab})
        fn = len({(x, y, z) for (x, y, z) in (gt - pred) if x == lab})

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        num = 2 * precision * recall
        denom = precision+ recall
        f1_score = num / denom if denom != 0 else 0.0
        mean_f1 += f1_score

        metrics[lab] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }

    macro_f1 = mean_f1 / len(TARGETS)

    return {"macro_f1": macro_f1, "per_label": metrics}


def _normalize_text(s: str) -> str:
    # Light normalization for entity set comparison
    return " ".join((s or "").split()).strip().lower()


def evaluate_entity_set(ground_truth: list[Entity], predictions: list[Entity]) -> dict:
    gt = {(e.label, _normalize_text(getattr(e, "text", ""))) for e in ground_truth}
    pred = {(e.label, _normalize_text(getattr(e, "text", ""))) for e in predictions}

    metrics = {}
    mean_f1 = 0.0

    for lab in TARGETS:
        tp = len({t for t in (pred & gt) if t[0] == lab})
        fp = len({t for t in (pred - gt) if t[0] == lab})
        fn = len({t for t in (gt - pred) if t[0] == lab})

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        num = 2 * precision * recall
        denom = precision+ recall
        f1_score = num / denom if denom != 0 else 0.0
        mean_f1 += f1_score

        metrics[lab] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }

    macro_f1 = mean_f1 / len(TARGETS)
    return {"macro_f1": macro_f1, "per_label": metrics}


def aggregate_results(score_rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}

    for row in score_rows:
        model = row["model"]
        model_info = summary.setdefault(
            model,
            {
                "num_docs": 0,
                "macro_f1_sum": 0.0,
                "per_label_counts": {lab: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for lab in TARGETS},
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

        macro_f1_labels = sum(v["f1"] for v in per_label_metrics.values()) / len(TARGETS) if TARGETS else 0.0

        info["per_label"] = per_label_metrics
        info["macro_f1_macro_over_labels"] = macro_f1_labels

        del info["per_label_counts"]
        del info["macro_f1_sum"]

    return summary


def write_summary_csv(summary: dict, out_path: Path):
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

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict, title: str):
    print(title)
    for model, info in summary.items():
        print(f"Model: {model}")
        print(f"  Documents: {info['num_docs']}")
        print(f"  Mean macro-F1 over documents: {info['macro_f1_mean']:.4f}")
        print(f"  Macro-F1 over labels (from aggregated counts): {info['macro_f1_macro_over_labels']:.4f}")
        for lab, stats in info["per_label"].items():
            print(
                f"    {lab}: F1={stats['f1']:.4f}, "
                f"P={stats['precision']:.4f}, R={stats['recall']:.4f} "
                f"(TP={stats['TP']:.0f}, FP={stats['FP']:.0f}, FN={stats['FN']:.0f})"
            )
        print()


def write_scores_csv(score_rows: list[dict], out_path: Path):
    base_fields = ["filename", "model", "macro_f1"]
    metric_fields = [f"{m}_{lab}" for lab in TARGETS for m in METRIC_NAMES]
    fieldnames = base_fields + metric_fields

    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)


def main():
    entity_files = sorted(ENTITIES_DIR.glob("*_entities.json"))

    score_rows_strict: list[dict] = []
    score_rows_set: list[dict] = []

    for json_file in entity_files:
        filename, ground_truth, predictions_by_model = load_doc_entities(json_file)

        for model_name, preds in predictions_by_model.items():
            eval_strict = evaluate_span_strict(ground_truth, preds)
            score_rows_strict.append(make_score_row(filename, model_name, eval_strict))

            eval_set = evaluate_entity_set(ground_truth, preds)
            score_rows_set.append(make_score_row(filename, model_name, eval_set))

    write_scores_csv(score_rows_strict, SCORES_CSV)
    write_scores_csv(score_rows_set, SCORES_SET_CSV)

    summary_strict = aggregate_results(score_rows_strict)
    summary_set = aggregate_results(score_rows_set)

    write_summary_csv(summary_strict, SCORES_SUMMARY_CSV)
    write_summary_csv(summary_set, SCORES_SET_SUMMARY_CSV)

    print_summary(summary_strict, "=== Strict span evaluation (label + start/end) ===")
    print_summary(summary_set, "=== Entity-set evaluation (label + normalized text; ignore spans) ===")


if __name__ == "__main__":
    main()
