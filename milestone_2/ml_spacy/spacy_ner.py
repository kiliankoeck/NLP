import spacy

from ..entities import Entity
from spacy.language import Language

TARGETS = {"PER","ORG","LOC"}

class SpacyNer:
    def __init__(self):
        self._nlp = spacy.load("de_core_news_md")
        self._nlp.max_length = 3000000 # not a smart solution but it works
        return

    def annotate(self, text: str) -> list[Entity]:
        doc = self._nlp(text)
        results: list[Entity] = []
        for ent in doc.ents:
            if ent.label_ in TARGETS:
                results.append(Entity(ent.text, ent.label_, ent.start_char, ent.end_char))

        return results