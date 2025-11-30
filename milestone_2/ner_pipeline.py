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
        files = [p for p in RAW_XMI_DIR.iterdir() if p.is_file()]

    logger.info(f"loaded {len(files)} files")


    for i, f in enumerate(files):
        logger.info("Processing file {}/{}".format(i+1, len(files)))

        try:

            gerparcor_data = xmi_parser.parse(f)

            plain_text = gerparcor_data["text"]

            ground_truth: list[Entity] = [Entity(**e) if isinstance(e, dict) else e for e in gerparcor_data["entities"]]

            predictions_by_model: dict[str, list[Entity]] = {}
            for model_name, annotator in models.items():
                preds = annotator(plain_text)
                predictions_by_model[model_name] = preds

            save_entities_for_doc(f, ground_truth, predictions_by_model)

        except Exception as e:
            logger.exception(f"Error parsing file {f}")


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
