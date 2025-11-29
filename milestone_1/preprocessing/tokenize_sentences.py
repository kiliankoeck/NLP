import re
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

input_dir = Path("../../data/plain_text")
output_dir = Path("../../data/sentence_tokenized")

# Test dirs (comment out)
#input_dir = Path("../testing_data/plain_text")
#output_dir = Path("../testing_data/sentence_tokenized")

output_dir.mkdir(parents=True, exist_ok=True)

for file_path in input_dir.glob("*.txt"):
    print("Processing {}".format(file_path))

    with open(file_path, "r", encoding="utf-8") as f:
        document = f.read()

    document = ' '.join(document.split())  # get it all in one line
    document = re.sub(r"\s+([.,!?])", r"\1", document)  # remove whitespaces before .,!?

    # start with "Beginn der Sitzung: x{x}.xx Uhr"
    # end with "Schluss der Sitzung: x{x}.xx Uhr"
    pattern = r"Beginn der Sitzung: \d{1,2}(?:\.\d{2})?\s* Uhr(.*?)Schlu(?:ss|ß) der Sitzung: \d{1,2}(?:\.\d{2})?\s* Uhr"
    match = re.search(pattern, document, re.DOTALL)

    protocol = match.group(1).strip() if match else document

    # Remove list of presidium if present
    if protocol.startswith("Vorsitzende"):
        m = re.search(r"^Vorsitzende[r]?:.*?\*{5}\s*(.*)$", protocol, re.DOTALL)
        protocol = m.group(1).strip() if m else protocol

    # remove additional remarks in parentheses (mostly referring to applause)
    protocol = re.sub(r"\([^()]*\)", "", protocol)

    # remove the new page headers e.g. (Bundesrat Stenographisches Protokoll 851. Sitzung / Seite 10)
    PAGE_HEADER_REGEX = re.compile(r'((Bundesrat\s)?Stenographisches\sProtokoll\s\d+\.\sSitzung\s\/\sSeite\s\d+)')
    protocol = re.sub(PAGE_HEADER_REGEX, '', protocol)
    protocol = protocol.replace("\xad", "")  # remove soft hyphens

    # protocol now looks like speaker:text ... time
    # in the following, the protocol will be split into speaker:text as key value pairs
    pattern = r"([A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]+)\s*:\s*(.*?) (?=\d{1,2}[.]\d{2}\d?|\bSchluss der Sitzung:|\Z)"
    matches = re.findall(pattern, protocol, re.DOTALL)
    results = [
        {'speaker': s.strip(), 'text': ' '.join(t.split())}
        for s, t in matches
    ]

    for line in results:
        v = line['text']
        v = re.sub(r"\s{2,}", " ", v).strip()  # remove double whitespaces
        # now we sentence tokenize the text
        line['tokenized'] = nltk.sent_tokenize(v, language="german")

    # write the resulting information to a file
    with open(output_dir / file_path.name, "w", encoding="utf-8") as f:
        for line in results:
            f.write(str(line['speaker']) + ": " + str(line["tokenized"]) + "\n")
