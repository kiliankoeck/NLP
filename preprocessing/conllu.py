import spacy
from pathlib import Path

# python -m spacy download de_core_news_sm)
nlp = spacy.load("de_core_news_sm")

input_dir = Path("../data/sentence_tokenized")
output_dir = Path("../data/conllu")

# Test dirs (comment out)
#input_dir = Path("../testing_data/sentence_tokenized")
#output_dir = Path("../testing_data/conllu")

output_dir.mkdir(parents=True, exist_ok=True)

HEADER_COLS = [
    "ID",
    "FORM",
    "LEMMA",
    "UPOS",
    "XPOS",
    "FEATS",
    "HEAD",
    "DEPREL",
    "DEPS",
    "MISC",
]
CONLLU_WIDTHS = [4, 20, 20, 8, 8, 6, 6, 10, 6, 6]

for file_path in input_dir.glob("*.txt"):
    print("Processing {}".format(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out_path = output_dir / (file_path.stem + ".conllu")
    with open(out_path, "w", encoding="utf-8") as out_f:
        id = 0
        for line in lines:
            if ":" not in line:
                continue
            # Split speaker and text
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip().replace("', '", " ").replace("['", "").replace("']", "")

            doc = nlp(text)
            for sentence in doc.sents:
                id += 1
                out_f.write(f"# file = {file_path.name}\n")
                out_f.write(f"# speaker = {speaker}\n")
                out_f.write(f"# id = {id}\n")
                out_f.write(f"# text = {sentence.text.strip()}\n")

                out_f.write("\t".join(HEADER_COLS) + "\n")

                for i, token in enumerate(sentence, start=1):
                    ID = str(i)
                    FORM = token.text
                    LEMMA = token.lemma_
                    UPOS = token.pos_
                    XPOS = token.tag_ if token.tag_ else "_"
                    FEATS = "_"
                    HEAD = token.head.i - sentence.start + 1 if token.head != token else 0
                    DEPREL = token.dep_
                    DEPS = "_"
                    MISC = f"start_char={token.idx}|end_char={token.idx + len(token)}"

                    cols = [
                        ID,
                        FORM,
                        LEMMA,
                        UPOS,
                        XPOS,
                        FEATS,
                        str(HEAD),
                        DEPREL,
                        DEPS,
                        MISC,
                    ]

                    out_f.write("\t".join(cols) + "\n")
                out_f.write("\n")

