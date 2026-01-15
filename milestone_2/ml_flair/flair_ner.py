import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from pathlib import Path
import json
import torch

from ..entities import Entity

if torch.cuda.is_available():
    flair.device = torch.device("cuda")
else:
    flair.device = torch.device("cpu")

tagger = SequenceTagger.load("flair/ner-german")
#tagger = SequenceTagger.load("flair/ner-german-large")
# tagger = SequenceTagger.load("flair/ner-multi-fast")

TARGETS = {"PER", "LOC", "ORG"}

class FlairNer:
    def __init__(self):
        return

    def annotate(self, text: str) -> list[Entity]:
        sentence = Sentence(text)
        tagger.predict(sentence)
        entities: list[Entity] = []

        for ent in sentence.get_spans("ner"):
            label = ent.get_label("ner").value
            if label in TARGETS:
                entities.append(Entity(ent.text, label, ent.start_position, ent.end_position))

        return entities