import gzip
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

    def _parse_tree(self, xmi_path: Path) -> ET.ElementTree:
        with xmi_path.open("rb") as fh:
            magic = fh.read(2)
            fh.seek(0)
            if magic == b"\x1f\x8b":
                with gzip.open(fh, "rb") as gfh:
                    return ET.parse(gfh)
            else:
                return ET.parse(fh)

    def parse(self, xmi_path: Path):
        tree = self._parse_tree(xmi_path)
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
