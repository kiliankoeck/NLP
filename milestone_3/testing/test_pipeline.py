import json
import logging
from pathlib import Path

# Imports
from milestone_3.entities import Entity
from milestone_3.preprocessing_gerparcor.xmi_parser import XmiParser

# Models
from milestone_3.ml_flair.flair_ner import FlairNer
from milestone_3.ml_spacy.spacy_ner import SpacyNer
from milestone_3.rule_based.rule_based_ner import RuleBasedNER
# New Models
from milestone_3.ml_spacy.improved_spacy_ner import ImprovedSpacyNer
from milestone_3.ml_bert.bert_ner import BertNer
from milestone_3.ml_bert.distilbert_ner import DistilBertNer

# --- SAFETY CONFIGURATION ---
# 1. Read from the test set
RAW_XMI_DIR = Path("data/test_set") 
# 2. Save to a separate folder so we don't overwrite main results
RESULTS_DIR = Path("milestone_3/testing/test_results") 

ENTITIES_DIR = RESULTS_DIR / "entities"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_rulebased_ner():
    geonames_dir = Path("milestone_3/rule_based/location_data")
    return RuleBasedNER(geonames_dir)

def main():
    logger.info("Initializing Test Pipeline...")
    
    # Initialize all models
    xmi_parser = XmiParser()
    rule_based_ner = initialize_rulebased_ner()
    flair_ner = FlairNer()
    spacy_ner = SpacyNer()
    improved_spacy_ner = ImprovedSpacyNer()
    bert_ner = BertNer()
    distilbert_ner = DistilBertNer()

    # Register all models
    models = {
        "rule_based": rule_based_ner.annotate,
        "flair": flair_ner.annotate,
        "spacy": spacy_ner.annotate,
        "improved_spacy": improved_spacy_ner.annotate, 
        "bert": bert_ner.annotate,
        "distilbert": distilbert_ner.annotate
    }

    logger.info(f"Looking for files in {RAW_XMI_DIR}")
    files = list(RAW_XMI_DIR.glob("*.xmi"))
    logger.info(f"Found {len(files)} files")

    for i, f in enumerate(files):
        logger.info(f"Processing {i+1}/{len(files)}: {f.name}")
        try:
            # Parse Ground Truth
            data = xmi_parser.parse(f)
            text = data["text"]
            ground_truth = [Entity(**e) if isinstance(e, dict) else e for e in data["entities"]]

            # Run Predictions
            preds_by_model = {}
            for name, annotate_func in models.items():
                preds_by_model[name] = annotate_func(text)

            # Save
            save_results(f, ground_truth, preds_by_model)
            
        except Exception as e:
            logger.exception(f"Failed on {f.name}")

def save_results(path, gt, preds):
    doc_id = path.stem
    payload = {
        "filename": path.name, 
        "ground_truth": [e.to_dict() for e in gt]
    }
    for model, ents in preds.items():
        payload[model] = [e.to_dict() for e in ents]

    with open(ENTITIES_DIR / f"{doc_id}_entities.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()