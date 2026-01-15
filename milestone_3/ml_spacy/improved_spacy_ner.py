import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler #to use rul;es
from ..entities import Entity

import json
from pathlib import Path

TARGETS = {"PER", "ORG"}

# adding manual annotations file as json
#PATTERNS_FILE = Path("data/manual_patterns.json")

class ImprovedSpacyNer:
    def __init__(self):
        self._nlp = spacy.load("de_core_news_md")
        self._nlp.max_length = 3000000 
        
        # add the "entity_ruler" before statistical "ner" component
        # our rules take priority. If a rule matches, the model doesn't need to guess
        ruler = self._nlp.add_pipe("entity_ruler", before="ner")

        # adding manual annotation patters search
        # if PATTERNS_FILE.exists():
        #     print(f"IMPROVED_SPACY: Loading manual patterns from {PATTERNS_FILE}...")
        #     with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
        #         group_patterns = json.load(f)
        #         ruler.add_patterns(group_patterns)
        # else:
        #     print(f"IMPROVED_SPACY: Warning - {PATTERNS_FILE} not found. Skipping file load.")
        
        # specific rules 
        patterns = [
            # ORG
            {"label": "ORG", "pattern": "ÖVP"},
            {"label": "ORG", "pattern": "SPÖ"},
            {"label": "ORG", "pattern": "FPÖ"},
            {"label": "ORG", "pattern": "NEOS"},
            {"label": "ORG", "pattern": "KPÖ"},
            {"label": "ORG", "pattern": "Die Grünen"},
            {"label": "ORG", "pattern": "Grüne"},
            {"label": "ORG", "pattern": [{"LOWER": "liste"}, {"LOWER": "pilz"}]},

            {"label": "ORG", "pattern": "Nationalrat"},
            {"label": "ORG", "pattern": "Bundesrat"},
            {"label": "ORG", "pattern": "Bundesversammlung"},
            {"label": "ORG", "pattern": "Bundesregierung"},
            {"label": "ORG", "pattern": "Parlament"},
            {"label": "ORG", "pattern": "Verfassungsgerichtshof"},
            {"label": "ORG", "pattern": "Rechnungshof"},
            {"label": "ORG", "pattern": "Volksanwaltschaft"},
            {"label": "ORG", "pattern": "Bundeskanzleramt"},
        
            {"label": "ORG", "pattern": [{"TEXT": "Bundesministerium"}, {"TEXT": "für"}, {"POS": "NOUN"}]},

            {"label": "ORG", "pattern": [{"LOWER": "europäische"}, {"LOWER": "union"}]},
            {"label": "ORG", "pattern": [{"LOWER": "europäische"}, {"LOWER": "kommission"}]},
            {"label": "ORG", "pattern": [{"LOWER": "europäisches"}, {"LOWER": "parlament"}]},
        ]
        
        # add patterns
        ruler.add_patterns(patterns)

    def annotate(self, text: str) -> list[Entity]:
        doc = self._nlp(text)
        results: list[Entity] = []
        for ent in doc.ents:
            if ent.label_ in TARGETS:
                results.append(Entity(ent.text, ent.label_, ent.start_char, ent.end_char))

        return results