import os
import re
import unicodedata

def clean_text(text: str) -> str:
    """
    - Replace long-s (ſ) with s
    - Replace common ligatures (ﬂ, ﬁ, æ, œ, etc.)
    - Remove stray OCR symbols
    - Normalize whitespace
    - Remove long repeated noise sequences
    - Normalize unicode (NFC)
    """

    replacements = {
        "ſ": "s",
        "ﬂ": "fl",
        "ﬁ": "fi",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "æ": "ae",
        "œ": "oe",
        "Æ": "Ae",
        "Œ": "Oe",
        "„": "\"",
        "“": "\"",
        "”": "\"",
        "´": "'",
        "`": "'"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    #OCR
    text = re.sub(r"[|¬­¦~•·†‡¤]", " ", text)

    #overstretched punctuation
    text = re.sub(r"[=]{2,}", "=", text)
    text = re.sub(r"[-]{3,}", "--", text)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"[*]{2,}", "*", text)

    #OCR junk sequences
    text = re.sub(r"[a-z]{3,}n{2,}[a-z]{2,}", "", text)
    text = re.sub(
        r"[A-Za-z]{10,}",
        lambda m: "" if re.search(r"[aeiouAEIOU]", m.group(0)) is None else m.group(0),
        text
    )

    #norm whitespace
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    #norm unicode
    return unicodedata.normalize("NFC", text).strip()


def process_folder(folder: str):
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)

            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            cleaned = clean_text(raw)

            #overwrite
            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned)

            print(f"Cleaned (overwritten): {filename}")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/test_set_txt/"
    process_folder(folder)
