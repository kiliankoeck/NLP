"""
xmi into plain text

** input and output folders are for the test set
** need to change that for the full dataset
"""

import os
import xml.etree.ElementTree as ET

NS = {"cas": "http:///uima/cas.ecore"}

def _ns(tag, prefix):
    return f".//{{{NS[prefix]}}}{tag}"

def extract_plaintext(xmi_path: str) -> str:
    #get sofa string
    tree = ET.parse(xmi_path)
    root = tree.getroot()

    sofa_elem = None

    #fallback search
    for elem in root.findall(".//"):
        if elem.tag.endswith("Sofa") and "sofaString" in elem.attrib:
            sofa_elem = elem
            break

    #strict namespace search if above doesn't work
    if sofa_elem is None:
        sofa_elem = root.find(_ns("Sofa", "cas"))

    if sofa_elem is None or "sofaString" not in sofa_elem.attrib:
        raise ValueError(f"No sofaString found in {xmi_path}")

    return sofa_elem.attrib["sofaString"]


def main(input_path: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".xmi")
        ]

    for xmi_file in files:
        try:
            text = extract_plaintext(xmi_file)

            out_path = os.path.join(
                output_folder,
                os.path.basename(xmi_file).replace(".xmi", ".txt")
            )

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text)

            print(f"Extracted text: {xmi_file} → {out_path}")

        except Exception as e:
            print(f"Error parsing {xmi_file}: {e}")

#for sample
if __name__ == "__main__":
    import sys
    #testset
    input_path = "data/test_set/"
    output_folder = "data/test_set_txt/"

    ##full dataset
    #input_path = "data/raw_xmi/"
    #output_folder = "data/plain_text/"

    main(input_path, output_folder)

