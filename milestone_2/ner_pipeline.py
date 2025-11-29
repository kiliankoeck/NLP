import os
from pathlib import Path

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

    files = [gerparcor_dir] if os.path.isfile(gerparcor_dir) else [os.path.join(gerparcor_dir, f) for f in os.listdir(gerparcor_dir) if f.endswith(".xmi")]
    for f in files:
        try:
            gerparcor_data = xmi_parser.parse(f)
            rule_based_res = rule_based_ner.annotate(gerparcor_data["text"])
            for e in rule_based_res:
                print(e)
        except Exception as e:
            print(f"Error parsing {f}: {e}")



if __name__ == "__main__":
    main()
