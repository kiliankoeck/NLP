"""
Extracts text, tokens, and bio tags from XMI files
 
** input and output folders are for the test set
** need to change that for the full dataset  
"""
from typing import List, Tuple
import xml.etree.ElementTree as ET
import os
import json

#namespace in corpus
NS = {
    "cas": "http:///uima/cas.ecore",
    "token": "http:///org/texttechnologylab/annotation/token.ecore",
    "ner": "http:///de/tudarmstadt/ukp/dkpro/core/api/ner/type.ecore",
}

VALID_NER = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}

def _ns(tag, prefix):
    return f".//{{{NS[prefix]}}}{tag}"

def extract_from_xmi(xmi_path: str):
    tree = ET.parse(xmi_path)
    root = tree.getroot()

    #raw text
    sofa_elem = None
    for s in root.findall(".//"):
        if s.tag.endswith("Sofa") and "sofaString" in s.attrib:
            sofa_elem = s
            break
    if sofa_elem is None:
        sofa_elem = root.find(_ns("Sofa", "cas"))
    if sofa_elem is None or "sofaString" not in sofa_elem.attrib:
        raise ValueError(f"No sofaString found in {xmi_path}")
    raw_text = sofa_elem.attrib["sofaString"]

    #Tokens
    tokens = []
    for tok in root.findall(_ns("Token", "token")):
        try:
            b, e = int(tok.attrib["begin"]), int(tok.attrib["end"])
        except KeyError:
            continue
        tokens.append((b, e, raw_text[b:e]))
    if not tokens:
        for elem in root.findall(".//"):
            if elem.tag.endswith("Token") and "begin" in elem.attrib:
                b, e = int(elem.attrib["begin"]), int(elem.attrib["end"])
                tokens.append((b, e, raw_text[b:e]))

    #NER
    entities = []
    for ne in root.findall(_ns("NamedEntity", "ner")):
        try:
            b, e = int(ne.attrib["begin"]), int(ne.attrib["end"])
        except KeyError:
            continue
        val = ne.attrib.get("value", "").strip().upper()
        if val in VALID_NER:
            entities.append((b, e, VALID_NER[val]))
    if not entities:
        for elem in root.findall(".//"):
            if elem.tag.endswith("NamedEntity") and "begin" in elem.attrib:
                b, e = int(elem.attrib["begin"]), int(elem.attrib["end"])
                val = elem.attrib.get("value", "").strip().upper()
                if val in VALID_NER:
                    entities.append((b, e, VALID_NER[val]))

    #BIO tags
    bio_tags = ["O"] * len(tokens)
    for ent_b, ent_e, ent_label in entities:
        started = False
        for i, (tb, te, _) in enumerate(tokens):
            if tb >= ent_b and te <= ent_e:
                if not started:
                    bio_tags[i] = f"B-{ent_label}"
                    started = True
                else:
                    bio_tags[i] = f"I-{ent_label}"
            elif tb < ent_e and te > ent_b and not (tb >= ent_b and te <= ent_e):
                if not started:
                    bio_tags[i] = f"B-{ent_label}"
                    started = True
                else:
                    bio_tags[i] = f"I-{ent_label}"

    return {"text": raw_text, "tokens": tokens, "bio_tags": bio_tags}

def main(input_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    files = [input_path] if os.path.isfile(input_path) else [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".xmi")]

    for f in files:
        try:
            data = extract_from_xmi(f)
            out_file = os.path.join(output_folder, os.path.basename(f).replace(".xmi", ".json"))
            with open(out_file, "w", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False, indent=2)
            print(f"Parsed {f} -> {out_file}")
        except Exception as e:
            print(f"Error parsing {f}: {e}")

if __name__ == "__main__":
    import sys
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "data/test_set/"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "data/test_set_json/"
    main(input_folder, output_folder)
