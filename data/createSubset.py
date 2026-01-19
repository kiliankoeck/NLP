import json
from pathlib import Path

BASE_FILE = Path(r"data\testfiles_xmi")
OUT_DIR = Path(r"milestone_3\annotated_data\annotated_data_subset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP = {
    "01.02.2017_164._Sitzung.xmi.xmi",
    "05.11.2020_61._Sitzung.xmi.xmi",
    "14.03.2020_15._Sitzung.xmi.xmi",
    "20.09.2017_196._Sitzung.xmi.xmi",
    "20.10.2008_74._Sitzung.xmi.xmi",
    "20.10.2011_127._Sitzung.xmi.xmi",
    "20.11.2020_66._Sitzung.xmi.xmi",
    "21.12.2020_74._Sitzung.xmi.xmi",
    "23.05.2014_29._Sitzung.xmi.xmi",
    "26.02.2004_53._Sitzung.xmi.xmi",
    "27.02.2019_65._Sitzung.xmi.xmi",
    "29.05.2009_25._Sitzung.xmi.xmi",
    "30.01.2019_62._Sitzung.xmi.xmi",
    "Plenarprotokoll_955._Sitzung_22.03.2017.xmi.xmi",
    "Plenarprotokoll_987._Sitzung,_25.03.2020.txt",
    "Plenarprotokoll_913._Sitzung,_16.08.2013.txt"
}

def full_stem(filename):
    p = Path(filename)
    while p.suffix:
        p = p.with_suffix('')
    return p.name

KEEP_STEMS = {full_stem(f) for f in KEEP}

with BASE_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [
    entry for entry in data
    if full_stem(entry.get("meta", {}).get("filename", "")) in KEEP_STEMS
]

OUT_FILE = OUT_DIR / "filtered_subset.json"
with OUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"Saved {len(filtered)} entries to {OUT_FILE}")
