"""
Extracts text, tokens, and bio tags from XMI files

** input and output folders are for the test set
** need to change that for the full dataset
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# namespace in corpus
NS = {
    "cas": "http:///uima/cas.ecore",
    "token": "http:///org/texttechnologylab/annotation/token.ecore",
    "ner": "http:///de/tudarmstadt/ukp/dkpro/core/api/ner/type.ecore",
}

VALID_NER = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}

class XmiParser:

    def __init__(self):
        return

    def _ns(self, tag, prefix):
        return f".//{{{NS[prefix]}}}{tag}"


    def parse(self, xmi_path: Path):
        tree = ET.parse(xmi_path)
        root = tree.getroot()

        # raw text
        sofa_elem = None
        for s in root.findall(".//"):
            if s.tag.endswith("Sofa") and "sofaString" in s.attrib:
                sofa_elem = s
                break
        if sofa_elem is None:
            sofa_elem = root.find(self._ns("Sofa", "cas"))
        if sofa_elem is None or "sofaString" not in sofa_elem.attrib:
            raise ValueError(f"No sofaString found in {xmi_path}")
        raw_text = sofa_elem.attrib["sofaString"]

        # NER
        entities = []
        for ne in root.findall(self._ns("NamedEntity", "ner")):
            try:
                b, e = int(ne.attrib["begin"]), int(ne.attrib["end"])
            except KeyError:
                continue
            val = ne.attrib.get("value", "").strip().upper()
            if val in VALID_NER:
                entities.append({"start": b, "end": e, "text": raw_text[b:e], "label": VALID_NER[val]})
        if not entities:
            for elem in root.findall(".//"):
                if elem.tag.endswith("NamedEntity") and "begin" in elem.attrib:
                    b, e = int(elem.attrib["begin"]), int(elem.attrib["end"])
                    val = elem.attrib.get("value", "").strip().upper()
                    if val in VALID_NER:
                        entities.append({"start": b, "end": e, "text": raw_text[b:e], "label": VALID_NER[val]})


        return {"text": raw_text, "entities": entities}
