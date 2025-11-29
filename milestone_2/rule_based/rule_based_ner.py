from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any

import requests
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc

from ..entities import Entity


class RuleBasedNER:

    def __init__(self, geonames_dir: Path):
        self._verbose = True
        self.geonames_dir = geonames_dir
        self.gazetteers = self._build_gazetteers()
        self.nlp = self._build_nlp()

    def _fetch_parliament_persons(self) -> Set[str]:
        url = "https://www.parlament.gv.at/Filter/api/filter/data/409?1=1&showAll=true"

        if self._verbose:
            print("RULEBASED_NER: fetching parliament persons...")

        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        persons = self._parse_parliament_persons_from_json(data)

        return persons

    def _parse_parliament_persons_from_json(self, data: Dict[str, Any]) -> Set[str]:
        persons: Set[str] = set()

        headers: List[Dict[str, Any]] = data.get("header", [])
        rows: List[List[Any]] = data.get("rows", [])

        idx_name: Optional[int] = None
        idx_attr_json: Optional[int] = None
        idx_vorname: Optional[int] = None
        idx_nachname: Optional[int] = None

        for i, h in enumerate(headers):
            label = h.get("label")
            feld_name = h.get("feld_name")

            if feld_name == "PERSON_NAME" or label == "Name":
                idx_name = i
            elif feld_name == "ATTR_JSON" or label == "Attribute":
                idx_attr_json = i
            elif label == "vorname":
                idx_vorname = i
            elif label == "nachname":
                idx_nachname = i


        for row in rows:
            if idx_name is not None and idx_name < len(row):
                name_val = row[idx_name]
                if isinstance(name_val, str) and name_val.strip():
                    persons.add(name_val.strip())

            if idx_attr_json is not None and idx_attr_json < len(row):
                attr = row[idx_attr_json]
                if isinstance(attr, dict):
                    zit = attr.get("zit")
                    if isinstance(zit, str) and zit.strip():
                        persons.add(zit.strip())

                    name_nvg = attr.get("name_nvg")
                    if isinstance(name_nvg, str) and name_nvg.strip():
                        persons.add(name_nvg.strip())

            vorname = None
            nachname = None
            if idx_vorname is not None and idx_vorname < len(row):
                vorname = row[idx_vorname]
            if idx_nachname is not None and idx_nachname < len(row):
                nachname = row[idx_nachname]

            if isinstance(vorname, str) and vorname.strip():
                persons.add(vorname.strip())
            if isinstance(nachname, str) and nachname.strip():
                persons.add(nachname.strip())

            if (
                    isinstance(vorname, str)
                    and vorname.strip()
                    and isinstance(nachname, str)
                    and nachname.strip()
            ):
                full = f"{vorname.strip()} {nachname.strip()}"
                persons.add(full)

        return persons



    def _load_geonames_countries(self, country_info_path: Path) -> Set[str]:

        countries: Set[str] = set()

        if not country_info_path.exists():
            if self._verbose:
                print(f"RULEBASED_NER: Warning – countryInfo.txt not found at {country_info_path}")
            return countries

        with country_info_path.open(encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue

                country_name = parts[4].strip()  # "Country" column
                if country_name:
                    countries.add(country_name)

                capital = parts[5].strip()
                if capital:
                    countries.add(capital)

        return countries



    def _load_geonames_places(
            self,
            geonames_path: Path,

    ) -> Set[str]:
        feature_classes = {"P", "A"}
        locs: Set[str] = set()

        if not geonames_path.exists():
            if self._verbose:
                print(f"RULEBASED_NER: Warning – GeoNames file not found at {geonames_path}")
            return locs

        with geonames_path.open(encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 15:
                    continue

                (
                    geonameid,
                    name,
                    asciiname,
                    alternatenames,
                    lat,
                    lon,
                    feature_class,
                    feature_code,
                    country_code,
                    cc2,
                    admin1,
                    admin2,
                    admin3,
                    admin4,
                    population,
                    *rest,
                ) = parts + [""] * max(0, 15 - len(parts))

                if feature_class not in feature_classes:
                    continue

                if name:
                    locs.add(name.strip())
                if asciiname and asciiname != name:
                    locs.add(asciiname.strip())

        return locs



    def _non_overlapping_spans(self, doc: Doc) -> List[Tuple[int, int, str]]:

        spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        spans.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

        result: List[Tuple[int, int, str]] = []
        occupied: List[Tuple[int, int]] = []

        for start, end, label in spans:
            overlaps = any(not (end <= s or start >= e) for s, e in occupied)
            if overlaps:
                continue
            occupied.append((start, end))
            result.append((start, end, label))

        result.sort(key=lambda x: x[0])
        return result


    #TODO: store and load if already available
    #TODO: fix issue where normal words are tagged as person or locations
    def _build_gazetteers(self):
        if self._verbose:
            print("RULEBASED_NER: building gazetteers...")

        persons: Set[str] = set()
        locs: Set[str] = set()
        orgs: Set[str] = set()

        parliament_persons = self._fetch_parliament_persons()

        if self._verbose:
            print(f"RULEBASED_NER: fetched {len(parliament_persons)} parliament person entries")
        persons |= parliament_persons

        stop_parts = {
            "der", "die", "das", "den", "dem", "des",
            "ein", "eine", "einer", "einem", "eines",
            "und", "oder", "vom", "von", "zur", "zum",
            "im", "in", "am", "an"
        }

        parts: Set[str] = set()
        for full in parliament_persons:
            cleaned = full.replace(",", " ")
            for p in cleaned.split():
                p_stripped = p.strip()
                if len(p_stripped) < 3:
                    continue
                if p_stripped.lower() in stop_parts:
                    continue
                first = p_stripped[0]
                if not (first.isalpha() and first == first.upper()):
                    continue
                parts.add(p_stripped)

        persons |= parts
        if self._verbose:
            print(f"RULEBASED_NER: total person name variants (incl. parts): {len(persons)}")

        country_info_path = self.geonames_dir / "countryInfo.txt"
        country_names = self._load_geonames_countries(country_info_path)
        locs |= country_names

        at_file = self.geonames_dir / "AT.txt"
        at_places = self._load_geonames_places(
            at_file
        )
        locs |= at_places

        cities5000_file = self.geonames_dir / "cities5000.txt"
        cities_world = self._load_geonames_places(
            cities5000_file
        )
        locs |= cities_world

        if self._verbose:
            print(f"RULEBASED_NER: loaded {len(locs)} location names from GeoNames")

        orgs |= {
            "Nationalrat",
            "Bundesrat",
            "Bundesversammlung",
            "Österreichisches Parlament",
            "Parlament",
            "Österreichische Volkspartei",
            "ÖVP",
            "Sozialdemokratische Partei Österreichs",
            "SPÖ",
            "Freiheitliche Partei Österreichs",
            "FPÖ",
            "Die Grünen",
            "GRÜNE",
            "NEOS",
            "Kommunistische Partei Österreichs",
            "KPÖ",
            "Bundesregierung",
            "Bundeskanzleramt",
            "Bundespräsident",
            "Bundesministerium für Inneres",
            "Bundesministerium für Finanzen",
            "Bundesministerium für Justiz",
            "Bundesministerium für Bildung",
            "Bundesministerium für Landesverteidigung",
            "Österreichischer Gewerkschaftsbund",
            "Wirtschaftskammer Österreich",
            "Arbeiterkammer",
            "Europäische Union",
            "Europäische Kommission",
            "Europäischer Rat",
            "Europäisches Parlament",
        }

        if self._verbose:
            print(f"RULEBASED_NER: loaded {len(orgs)} organization names")

        return {"persons": persons, "locs": locs, "orgs": orgs}

    def _build_nlp(self) -> Language:
        if self._verbose:
            print("RULEBASED_NER: building spaCy pipeline")

        nlp = spacy.load("de_core_news_sm")

        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

        ruler: EntityRuler = nlp.add_pipe(
            "entity_ruler",
            config={
                "overwrite_ents": True,
                "phrase_matcher_attr": "LOWER",  # case-insensitive for phrase patterns
            },
        )

        patterns = [{
            "label": "ORG",
            "pattern": [
                {"LEMMA": "europäisch"},
                {"LEMMA": "Kommission"},
            ],
        }, {
            "label": "ORG",
            "pattern": [
                {"LEMMA": "europäisch"},
                {"LEMMA": "Union"},
            ],
        }, {
            "label": "ORG",
            "pattern": [
                {"LEMMA": "europäisch"},
                {"LEMMA": "Rat"},
            ],
        }, {
            "label": "ORG",
            "pattern": [
                {"LEMMA": "europäisch"},
                {"LEMMA": "Parlament"},
            ],
        }]

        org_lemma_targets = [
            "Nationalrat",
            "Bundesrat",
            "Parlament",
            "Bundesregierung",
            "Bundeskanzleramt",
            "Bundespräsident",
            "Bundesversammlung",
        ]

        for lemma in org_lemma_targets:
            patterns.append(
                {
                    "label": "ORG",
                    "pattern": [
                        {"LEMMA": lemma},
                    ],
                }
            )

        for name in sorted(self.gazetteers["persons"], key=len, reverse=True):
            patterns.append({"label": "PERS", "pattern": name})

        for name in sorted(self.gazetteers["locs"], key=len, reverse=True):
            patterns.append({"label": "LOC", "pattern": name})

        for name in sorted(self.gazetteers["orgs"], key=len, reverse=True):
            patterns.append({"label": "ORG", "pattern": name})

        ruler.add_patterns(patterns)
        if self._verbose:
            print(f"RULEBASED_NER: EntityRuler loaded with {len(patterns)} patterns")

        return nlp


    def annotate(self, text: str) -> list[Entity]:

        if self._verbose:
            print(f"RULEBASED_NER: annotating text of length {len(text)}")
        doc = self.nlp(text)
        spans = self._non_overlapping_spans(doc)
        if self._verbose:
            print(f"RULEBASED_NER: found {len(spans)} entities")

        entities: List[Entity] = []
        for start, end, label in spans:
            span_text = text[start:end]

            entities.append(Entity(span_text, label, start, end))

        return entities


if __name__ == "__main__":
    geonames_dir = Path("./location_data")

    ner = RuleBasedNER(geonames_dir)

    example_text = (
        "Ebenso schließe ich mich dem Dank an Karl Nehammer an. "
        "Die Sitzung des Nationalrats findet in Wien statt. "
        "Vertreterinnen der Europäischen Kommission sind anwesend."
    )

    ents = ner.annotate(example_text)
    for e in ents:
        print(e)
