#!/usr/bin/env python3
"""
fetch_reccords.py

Goal:
- Fetch ALL stenographic records ("Stenographische Protokolle") of the Austrian Parliament
  (Nationalrat + Bundesrat) via the official Open Data API.
- Save each transcript as cleaned plain text for downstream NLP / NER.
- Also keep the raw HTML and a metadata file.

What you get on disk (default folder: ./data):
  data/
    metadata.json
    raw_html/   <GP>_<CHAMBER>_<SESSION>.html
    plain_text/ <GP>_<CHAMBER>_<SESSION>.txt

Requirements:
  pip install requests beautifulsoup4

Notes on the API:
  Endpoint:
    POST https://www.parlament.gv.at/Filter/api/filter/data/211?js=eval&showAll=true&export=true

  Body (JSON):
    {
      "NBVS": ["NRSITZ", "BRSITZ"]
    }

  Meaning:
    NBVS = session type, e.g. "NRSITZ" (Nationalrat), "BRSITZ" (Bundesrat)
    GP_CODE = legislative period ("XXVIII", ...)
    DATUM = ["YYYY-MM-DD","YYYY-MM-DD"] date range filter (optional)

  Response JSON keys we care about:
    "rows": [
      [
        0  - "05.04.2022" (date, human-readable)
        1  - "/gegenstand/XXVII/NRSITZ/152" (session detail URL, relative)
        2  - "XXVII" (legislative period)
        3  - "NRSITZ" (chamber tag)
        4  - 152 (session number)
        5  - "152. Sitzung (152/NRSITZ)" (title)
        6  - "152/NRSITZ" (citation)
        7  - "Plenarsitzung" (type label)
        ...
        9  - "2022-04-05T00:00:00" (ISO-ish date)
        10 - "<div>...<a href=\"/dokument/...PDF\">PDF</a> ... <a href=\"/dokument/...HTML\">HTML</a>...</div>"
             -> we parse this snippet to get the actual transcript URLs.
        ...
      ],
      ...
    ]

The HTML link points at the full stenographic protocol (verbatim spoken record).
We'll fetch that and strip tags to get plain text.

License:
- The Parliament publishes these stenographic protocols as Open Data (CC BY 4.0),
  and the transcripts themselves are "freie Werke" and can be reused freely for NR/BR
  stenographic protocols.
"""

import os
import time
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag


BASE_URL = "https://www.parlament.gv.at"
API_URL = (
    BASE_URL
    + "/Filter/api/filter/data/211?js=eval&showAll=true&export=true"
)

# polite-ish headers; some servers behave better with a UA
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; steno-scraper/1.0; +NER-prep)",
    "Accept": "application/json, text/plain, */*",
}


def fetch_index():
    """
    Call the API once and get the full list of stenographic protocols
    for Nationalrat ("NRSITZ") and Bundesrat ("BRSITZ").

    Returns:
        dict: parsed JSON from the API.
    """
    payload = {
        # both chambers, OR-combined by the API
        "NBVS": ["NRSITZ", "BRSITZ"]
        # You *could* also add filters like
        # "GP_CODE": ["XXVIII"],
        # "DATUM": ["2023-01-01", "2023-12-31"],
        # if you ever want to limit.
    }

    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_rows(api_json):
    """
    Extract metadata + transcript links from the API rows.

    Args:
        api_json (dict): JSON returned by fetch_index()

    Returns:
        list[dict]: one entry per session/protocol
    """
    rows = api_json.get("rows", [])
    protocols = []

    for row in rows:
        # Defensive: make sure expected indices exist
        if len(row) < 11:
            continue

        date_display = row[0]            # e.g. "05.04.2022"
        session_rel_url = row[1]         # e.g. "/gegenstand/XXVII/NRSITZ/152"
        gp_code = row[2]                 # e.g. "XXVII"
        chamber = row[3]                 # "NRSITZ" or "BRSITZ"
        session_number = row[4]          # e.g. 152
        title = row[5]                   # e.g. "152. Sitzung (152/NRSITZ)"
        citation = row[6]                # e.g. "152/NRSITZ"
        # row[7], row[8] are labels like "Plenarsitzung", not strictly needed
        iso_date = row[9]                # "2022-04-05T00:00:00"
        links_html_snippet = row[10]     # contains <a>PDF</a> and <a>HTML</a>

        # Parse the tiny HTML snippet in row[10] to get the actual transcript URLs
        soup = BeautifulSoup(links_html_snippet, "html.parser")
        pdf_url = None
        html_url = None

        # The snippet usually looks like:
        # <div><div class="link-list"...>
        #   <a href="/dokument/XXVII/NRSITZ/152/fname_1470282.pdf">PDF</a>
        #   <a href="/dokument/XXVII/NRSITZ/152/fnameorig_1470282.html">HTML</a>
        # </div></div>
        for a in soup.find_all("a"):
            label = a.get_text(strip=True)
            href = a.get("href")
            if not href:
                continue

            # Build absolute URL and strip off any "#Seite_00XX.html" fragments.
            abs_url = urljoin(BASE_URL, urldefrag(href)[0])

            if "PDF" in label.upper() and pdf_url is None:
                pdf_url = abs_url
            if "HTML" in label.upper() and html_url is None:
                html_url = abs_url

        session_detail_url = urljoin(BASE_URL, session_rel_url)

        protocols.append(
            {
                "session_date_display": date_display,
                "session_date_iso": iso_date,
                "gp_code": gp_code,
                "chamber": chamber,
                "session_number": session_number,
                "title": title,
                "citation": citation,
                "session_detail_url": session_detail_url,
                "protocol_pdf_url": pdf_url,
                "protocol_html_url": html_url,
            }
        )

    return protocols


def fetch_protocol_html(url):
    """
    Download the stenographic protocol HTML and return:
    - cleaned plain text (minimal preprocessing)
    - raw HTML (as string)

    Args:
        url (str): absolute URL to the protocol HTML

    Returns:
        (plain_text:str, raw_html:str)
    """
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()

    raw_html = resp.text
    soup = BeautifulSoup(raw_html, "html.parser")

    # get_text() will flatten <p>, <br>, etc. into text with separators
    text = soup.get_text(separator="\n")

    # Light cleanup:
    # - strip trailing/leading whitespace
    # - collapse multiple blank lines
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text, raw_html


def save_protocol(out_dir, item, plain_text, raw_html):
    """
    Persist plain text and HTML to disk using a stable filename scheme.

    out_dir/
        raw_html/<gp>_<chamber>_<session>.html
        plain_text/<gp>_<chamber>_<session>.txt
    """
    gp = item["gp_code"]
    chamber = item["chamber"]
    session_no = item["session_number"]

    uid = f"{gp}_{chamber}_{session_no}"

    raw_dir = out_dir / "raw_html"
    txt_dir = out_dir / "plain_text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{uid}.html"
    txt_path = txt_dir / f"{uid}.txt"

    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(raw_html)

    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(plain_text)

    return {
        "uid": uid,
        "raw_path": str(raw_path),
        "txt_path": str(txt_path),
    }


def main(output_folder="data", polite_delay=0.5):
    """
    End-to-end runner:
    - fetch index of all stenographic protocols
    - loop all sessions
    - download + clean + save
    - write metadata.json
    """
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Fetching index from Parliament API ...")
    index_json = fetch_index()

    print("[2/3] Parsing rows ...")
    protocol_items = parse_rows(index_json)
    print(f"   -> {len(protocol_items)} sessions found")

    all_metadata = []
    for i, item in enumerate(protocol_items, start=1):
        html_url = item.get("protocol_html_url")
        if not html_url:
            print(f"Skipping {item['citation']} (no HTML URL)")
            continue

        print(f"[{i}/{len(protocol_items)}] Download {item['citation']} -> {html_url}")

        try:
            plain_text, raw_html = fetch_protocol_html(html_url)
        except Exception as e:
            print(f"   !! FAILED for {item['citation']}: {e}")
            continue

        file_info = save_protocol(out_dir, item, plain_text, raw_html)

        # keep track for metadata.json
        all_metadata.append(
            {
                **item,
                **file_info,
            }
        )

        # be nice to the server
        time.sleep(polite_delay)

    # write metadata file
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(all_metadata, fh, ensure_ascii=False, indent=2)

    print("[3/3] Done.")
    print(f"Saved {len(all_metadata)} transcripts.")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
