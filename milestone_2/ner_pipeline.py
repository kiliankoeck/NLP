import os
from pathlib import Path

from milestone_2.entities import Entity
from milestone_2.ml_flair.flair_ner import FlairNer
from milestone_2.ml_spacy.spacy_ner import SpacyNer
from milestone_2.preprocessing_gerparcor.xmi_parser import XmiParser
from milestone_2.rule_based.rule_based_ner import RuleBasedNER


def initialize_rulebased_ner():
    print("Initializing RuleBased NER")

    geonames_dir = Path("./rule_based/location_data")
    rulebased_ner = RuleBasedNER(geonames_dir)

    print("Finished Initializing RuleBased NER")
    return rulebased_ner


def main():
    gerparcor_dir = Path("../data/test_set")

    xmi_parser = XmiParser()
    rule_based_ner = initialize_rulebased_ner()
    #flair_ner = FlairNer()
    spacy_ner = SpacyNer()

    files = [gerparcor_dir] if os.path.isfile(gerparcor_dir) else [os.path.join(gerparcor_dir, f) for f in
                                                                   os.listdir(gerparcor_dir) if f.endswith(".xmi")]
    for f in files:
        try:

            gerparcor_data = xmi_parser.parse(f)

            plain_text = gerparcor_data["text"]

            # A list of {text: string, label: string, begin: int, end: int}
            ground_truth: list[Entity] = gerparcor_data["entities"]

            # Also a list of {text: string, label: string, begin: int, end: int}
            rule_based_res: list[Entity] = rule_based_ner.annotate(plain_text)
            #flair_res: list[Entity] = flair_ner.annotate(plain_text)
            spacy_res: list[Entity] = spacy_ner.annotate(plain_text)

            evaluation_rule_based = evaluate(ground_truth, rule_based_res)
            #evaluation_flair = evaluate(ground_truth, flair_res)
            evaluation_spacy = evaluate(ground_truth, spacy_res)
            print(evaluation_spacy)
            print(evaluation_rule_based)
            #print(evaluation_flair)

            # TODO: correct filename (kilian)
            # save_results(rule_based_res, f.name + "rule_based_ner.json")

        except Exception as e:
            print(f"Error parsing {f}: {e}")


def evaluate(ground_truth: list[Entity], predictions: list[Entity]) -> float:
    # Compare if
    # 1) both lists have the same tokens (not only same string but also same starting and endpoint)
    # 2) both lists have the same entities assigned to these tokens
    TARGETS = ["LOC", "PER", "ORG"]

    ground_truth = [Entity(**e) if isinstance(e, dict) else e for e in ground_truth]
    predictions = [Entity(**e) if isinstance(e, dict) else e for e in predictions]

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

        metrics[lab] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }

        precision[lab] = tp / (tp + fp) if fp + tp != 0 else 0.0
        recall[lab] = tp / (tp + fn) if tp + fn != 0 else 0.0

        num = 2 * precision[lab] * recall[lab]
        denom = precision[lab] + recall[lab]
        f1_score[lab] = num / denom if denom != 0 else 0.0
        mean_f1 += f1_score[lab]

    return mean_f1 / 3.0


# TODO: maybe save results of each ner and groundtruth to files.
def save_results(list1: list[Entity], filename: str):
    return


if __name__ == "__main__":
    main()
