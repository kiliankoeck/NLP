from flair.data import Sentence
from flair.models import SequenceTagger
from pathlib import Path
import json

#tagger = SequenceTagger.load("flair/ner-german")
tagger = SequenceTagger.load("flair/ner-german-large")
# tagger = SequenceTagger.load("flair/ner-multi-fast")

TARGETS = {"PER", "LOC", "ORG"}

input_folder = Path("./data/test_set_txt")
output_folder = Path("./flair_results")
output_folder.mkdir(exist_ok=True)

def read_full_text(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

def entity_recognition(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    entities = []

    for ent in sentence.get_spans("ner"):
        label = ent.get_label("ner").value
        if label in TARGETS:
            entities.append({
                "text": ent.text,
                "label": label,
            })

    return {
        "text": text,
        "entities": entities
    }

for file_path in input_folder.glob("*.txt"):
    out_file = output_folder / f"{file_path.stem}.json"
    full_text = read_full_text(file_path)
    struct_data = entity_recognition(full_text)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(struct_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(struct_data['entities'])} entities to {out_file}")
