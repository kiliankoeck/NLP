import csv
import json
import os
from pathlib import Path
import logging

from milestone_2.entities import Entity
from milestone_2.ml_flair.flair_ner import FlairNer
from milestone_2.ml_spacy.spacy_ner import SpacyNer
from milestone_2.preprocessing_gerparcor.xmi_parser import XmiParser
from milestone_2.rule_based.rule_based_ner import RuleBasedNER

LOG_DIR = Path("logs")
RAW_XMI_DIR = Path("data/raw_xmi")

RESULTS_DIR = Path("milestone_2/results")
ENTITIES_DIR = RESULTS_DIR / "entities"
SCORES_CSV = RESULTS_DIR / "scores.csv"
RESULTS_DIR.mkdir(exist_ok=True)
ENTITIES_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "ner_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
METRIC_NAMES = ["TP", "FP", "FN", "precision", "recall", "f1"]
TARGETS = ["LOC", "PER", "ORG"]

def initialize_rulebased_ner():
    geonames_dir = Path("milestone_2/rule_based/location_data")
    rulebased_ner = RuleBasedNER(geonames_dir)
    return rulebased_ner


def main():

    logger.info("Initializing classes")
    xmi_parser = XmiParser()

    rule_based_ner = initialize_rulebased_ner()
    flair_ner = FlairNer()
    spacy_ner = SpacyNer()

    models = {
        "rule_based": rule_based_ner.annotate,
        "flair": flair_ner.annotate,
        "spacy": spacy_ner.annotate,
    }

    logger.info("loading files")

    if RAW_XMI_DIR.is_file():
        files = [RAW_XMI_DIR]
    else:
        files = [p for p in RAW_XMI_DIR.glob("*.xmi")]

    logger.info(f"loaded {len(files)} files")

    score_rows = []

    for i, f in enumerate(files):
        logger.info("Processing file {}/{}".format(i+1, len(files)))

        try:

            gerparcor_data = xmi_parser.parse(f)

            plain_text = gerparcor_data["text"]

            ground_truth: list[Entity] = [Entity(**e) if isinstance(e, dict) else e for e in gerparcor_data["entities"]]

            predictions_by_model: dict[str, list[Entity]] = {}
            eval_by_model: dict[str, dict] = {}
            for model_name, annotator in models.items():
                preds = annotator(plain_text)
                predictions_by_model[model_name] = preds
                eval_result = evaluate(ground_truth, preds)
                eval_by_model[model_name] = eval_result
                score_rows.append(make_score_row(f.name, model_name, eval_result))

            save_entities_for_doc(f, ground_truth, predictions_by_model)

        except Exception as e:
            logger.exception(f"Error parsing file {f}")

    logger.info("Saving results")
    scores_csv = RESULTS_DIR / "scores.csv"
    base_fields = ["filename", "model", "macro_f1"]
    metric_fields = [f"{m}_{lab}" for lab in TARGETS for m in METRIC_NAMES]
    fieldnames = base_fields + metric_fields

    with open(scores_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    logger.info("Done")


def evaluate(ground_truth: list[Entity], predictions: list[Entity]) -> dict:
    gt = {(e.label, e.start, e.end) for e in ground_truth}
    pred = {(e.label, e.start, e.end) for e in predictions}

    metrics = {}
    precision = {}
    recall = {}
    f1_score = {}
    mean_f1 = 0.0

    tp, fp, fn = 0.0, 0.0, 0.0

    for lab in TARGETS:
        tp = len({(x, y, z) for (x, y, z) in (pred & gt) if x == lab})
        fp = len({(x, y, z) for (x, y, z) in (pred - gt) if x == lab})
        fn = len({(x, y, z) for (x, y, z) in (gt - pred) if x == lab})

        precision[lab] = tp / (tp + fp) if fp + tp != 0 else 0.0
        recall[lab] = tp / (tp + fn) if tp + fn != 0 else 0.0

        num = 2 * precision[lab] * recall[lab]
        denom = precision[lab] + recall[lab]
        f1_score[lab] = num / denom if denom != 0 else 0.0
        mean_f1 += f1_score[lab]

        metrics[lab] = {"TP": tp, "FP": fp, "FN": fn,
                        "precision": precision[lab],
                        "recall": recall[lab],
                        "f1": f1_score[lab]}

    macro_f1 = mean_f1 / len(TARGETS)

    return {
        "macro_f1": macro_f1,
        "per_label": metrics,
    }

def make_score_row(filename: str, model_name: str, eval_result: dict) -> dict:
    row = {
        "filename": filename,
        "model": model_name,
        "macro_f1": eval_result["macro_f1"],
    }

    per_label = eval_result["per_label"]  # dict[label] -> metrics dict

    for label, stats in per_label.items():
        for metric_name in METRIC_NAMES:
            key = f"{metric_name}_{label}"   # e.g. "TP_LOC", "precision_PER"
            row[key] = stats.get(metric_name, 0.0)

    return row

def save_entities_for_doc(doc_path: Path,
                          ground_truth: list[Entity],
                          predictions_by_model: dict[str, list[Entity]]) -> None:
    doc_id = doc_path.stem
    payload = {
        "filename": doc_path.name,
        "ground_truth": [e.to_dict() for e in ground_truth],
    }
    for model_name, ents in predictions_by_model.items():
        payload[model_name] = [e.to_dict() for e in ents]

    out_file = ENTITIES_DIR / f"{doc_id}_entities.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
