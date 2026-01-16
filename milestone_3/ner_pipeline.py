import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import logging
import random
from typing import List, Optional, Sequence

from milestone_3.entities import AnnotatedProtocol, Entity
from milestone_3.ml_flair.flair_ner import FlairNer
from milestone_3.ml_spacy.spacy_ner import SpacyNer
from milestone_3.ml_spacy.improved_spacy_ner import ImprovedSpacyNer
from milestone_3.ml_bert.bert_ner import BertNer
from milestone_3.ml_bert.distilbert_ner import DistilBertNer

from milestone_3.rule_based.rule_based_ner import RuleBasedNER

LOG_DIR = Path("logs")
DATASET_DIR = Path("milestone_3/annotated_data")

RESULTS_DIR = Path("milestone_3/results")
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
TARGETS = [ "PER", "ORG"]

def initialize_rulebased_ner():
    geonames_dir = Path("milestone_3/rule_based/location_data")
    rulebased_ner = RuleBasedNER(geonames_dir)
    return rulebased_ner

def extract_annotated_protocols(
    json_file_paths: Sequence[Path],
) -> List[AnnotatedProtocol]:
    rng = random.Random()
    protocols: List[AnnotatedProtocol] = []

    for path in json_file_paths:
        try:
            json_str = path.read_text(encoding="utf-8")
        except OSError as e:
            raise RuntimeError(f"Could not read file: {path}") from e

        try:
            tasks = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {path}") from e


        for task in tasks:
            annotations = task.get("annotations", [])
            if not annotations:
                continue

            annotation = rng.choice(annotations)

            filename = (
                (task.get("meta") or {}).get("filename")
                or task.get("file_upload")
                or f"task_{task.get('id', 'unknown')}"
            )

            full_text = (task.get("data") or {}).get("text", "")

            entities: List[Entity] = []
            for res in annotation.get("result", []):
                if res.get("type") != "labels":
                    continue

                value = res.get("value", {})
                labels = value.get("labels", [])

                if not labels:
                    continue

                entities.append(
                    Entity(
                        text=value.get("text", ""),
                        start=value.get("start"),
                        end=value.get("end"),
                        label=labels[0],
                    )
                )

            protocols.append(
                AnnotatedProtocol(
                    filename=filename,
                    full_text=full_text,
                    entities=entities,
                )
            )

    return protocols


def save_entities_for_doc(doc_id: str,original_filename: str,
                          ground_truth: list[Entity],
                          predictions_by_model: dict[str, list[Entity]]) -> None:
    payload = {
        "filename": original_filename,
        "ground_truth": [e.to_dict() for e in ground_truth],
    }
    for model_name, ents in predictions_by_model.items():
        payload[model_name] = [e.to_dict() for e in ents]

    out_file = ENTITIES_DIR / f"{doc_id}_entities.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def safe_doc_id(filename: str) -> str:
    stem = Path(filename).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)

def main():

    logger.info("Initializing classes")

    rule_based_ner = initialize_rulebased_ner()
    flair_ner = FlairNer()
    spacy_ner = SpacyNer()
    improved_spacy_ner = ImprovedSpacyNer()
    bert_ner = BertNer()
    distilbert_ner = DistilBertNer()

    models = {
        "rule_based": rule_based_ner.annotate,
        "flair": flair_ner.annotate,
        "spacy": spacy_ner.annotate,
        "improved_spacy": improved_spacy_ner.annotate,
        "bert": bert_ner.annotate,
        "distilbert": distilbert_ner.annotate
    }

    logger.info("loading files")

    if DATASET_DIR.is_file():
        files = [DATASET_DIR]
    else:
        files = [p for p in DATASET_DIR.iterdir() if p.is_file()]

    logger.info(f"loaded {len(files)} files")

    
    protocols = extract_annotated_protocols(files)
    logger.info("Extracted %d annotated protocols", len(protocols))

    for i, protocol in enumerate(protocols, start=1):
        logger.info("Processing protocol %d/%d (%s)", i, len(protocols), protocol.filename)

        try:
            plain_text = protocol.full_text
            ground_truth: List[Entity] = protocol.entities

            predictions_by_model: dict[str, list[Entity]] = {}
            for model_name, annotator in models.items():
                preds = annotator(plain_text)
                predictions_by_model[model_name] = preds
            doc_id = safe_doc_id(protocol.filename)
            save_entities_for_doc(
                doc_id=doc_id,
                original_filename=protocol.filename,
                ground_truth=ground_truth,
                predictions_by_model=predictions_by_model,
            )

        except Exception as e:
            logger.exception("Error processing protocol: %s", protocol.filename)

if __name__ == "__main__":
    main()
