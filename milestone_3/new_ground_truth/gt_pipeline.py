import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from milestone_3.entities import Entity
from milestone_3.preprocessing_gerparcor.xmi_parser import XmiParser

from milestone_3.ml_flair.flair_ner import FlairNer
from milestone_3.ml_spacy.spacy_ner import SpacyNer
from milestone_3.rule_based.rule_based_ner import RuleBasedNER
from milestone_3.ml_spacy.improved_spacy_ner import ImprovedSpacyNer
from milestone_3.ml_bert.bert_ner import BertNer
from milestone_3.ml_bert.distilbert_ner import DistilBertNer

XMI_DIR = Path("data/testfiles_xmi")
ANNOTATED_DIR = Path("milestone_3/annotated_data/annotated_data_subset")

JSON_BUNDESRAT = ANNOTATED_DIR / "filtered_subset_bundesrat.json"
JSON_NATIONALRAT = ANNOTATED_DIR / "filtered_subset_nationalrat.json"


RESULTS_DIR = Path("milestone_3/new_ground_truth/test_results") 
ENTITIES_DIR = RESULTS_DIR / "entities"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_rulebased_ner():
    geonames_dir = Path("milestone_3/rule_based/location_data")
    return RuleBasedNER(geonames_dir)

def parse_label_studio_task(task: dict):
    """
    Extracts text and ground truth entities from a single Label Studio task dict.
    """
    text = task.get("data", {}).get("text")
    if not text:
        text = task.get("text")
    
    ground_truth = []
    annotations = task.get("annotations", [])
    
    if annotations:
        result_list = annotations[0].get("result", [])
        for item in result_list:
            if item.get("type") == "labels":
                value = item.get("value", {})
                
                start = value.get("start")
                end = value.get("end")
                ent_text = value.get("text", "")
                labels_list = value.get("labels", [])
                label_str = labels_list[0] if labels_list else "UNKNOWN"
                
                if start is not None and end is not None:
                    entity = Entity(
                        text=ent_text,
                        label=label_str,
                        start=start,
                        end=end
                    )
                    ground_truth.append(entity)
    
    return text, ground_truth

def save_results(filename, gt, preds):
    doc_id = Path(filename).stem
    payload = {
        "filename": filename, 
        "ground_truth": [e.to_dict() for e in gt]
    }
    for model, ents in preds.items():
        payload[model] = [e.to_dict() for e in ents]

    with open(ENTITIES_DIR / f"{doc_id}_entities.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    logger.info("Initializing Test Pipeline with HYBRID Data Source...")

    xmi_parser = XmiParser() 
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

    if not XMI_DIR.exists():
        logger.error(f"XMI Directory not found: {XMI_DIR}")
        return
    xmi_files = sorted(list(XMI_DIR.glob("*.xmi")))
    logger.info(f"Found {len(xmi_files)} XMI files.")

    tasks = []

    if JSON_BUNDESRAT.exists():
        with open(JSON_BUNDESRAT, "r", encoding="utf-8") as f:
            bundesrat_data = json.load(f)
            if isinstance(bundesrat_data, list):
                tasks.extend(bundesrat_data)
                logger.info(f"Loaded {len(bundesrat_data)} tasks from Bundesrat JSON")
    else:
        logger.error(f"Missing Bundesrat JSON: {JSON_BUNDESRAT}")


    if JSON_NATIONALRAT.exists():
        with open(JSON_NATIONALRAT, "r", encoding="utf-8") as f:
            nationalrat_data = json.load(f)
            if isinstance(nationalrat_data, list):
                tasks.extend(nationalrat_data)
                logger.info(f"Loaded {len(nationalrat_data)} tasks from Nationalrat JSON")
    else:
        logger.error(f"Missing Nationalrat JSON: {JSON_NATIONALRAT}")

    if len(tasks) != len(xmi_files):
        logger.warning(f"MISMATCH: Found {len(xmi_files)} files but {len(tasks)} annotated tasks.")
        logger.warning("Proceeding with the minimum length of both lists.")
    
    for xmi_path, task in zip(xmi_files, tasks):
        logger.info(f"Processing: {xmi_path.name}")
        
        try:
            json_text, ground_truth = parse_label_studio_task(task)
            
            if json_text and len(json_text) > 10:
                text = json_text
            else:
                logger.info(f"  Text missing in JSON for {xmi_path.name}, parsing XMI...")
                data_xmi = xmi_parser.parse(xmi_path)
                text = data_xmi["text"]

            if not text:
                logger.warning(f"  Skipping {xmi_path.name} (No text found).")
                continue
            preds_by_model = {}
            for name, annotate_func in models.items():
                preds_by_model[name] = annotate_func(text)

            save_results(xmi_path.name, ground_truth, preds_by_model)
            
        except Exception as e:
            logger.exception(f"Failed on {xmi_path.name}")

if __name__ == "__main__":
    main()