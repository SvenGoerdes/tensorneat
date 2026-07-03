#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyzotero>=1.5.5"]
# ///
"""Add a PDF (with optional DOI-based metadata) to a Zotero library via the Web API.

Usage:
    uv run tools/add_to_zotero.py paper.pdf --doi 10.48550/arXiv.2106.13281
    uv run tools/add_to_zotero.py paper.pdf --title "Some Title" --collection "Thesis"

Credentials (either source):
    1. Env vars ZOTERO_USER_ID and ZOTERO_API_KEY
    2. ~/.config/zotero/credentials.json  -> {"user_id": "...", "api_key": "..."}

Create a key with write access at https://www.zotero.org/settings/keys
(the same page shows your numeric userID). The item lands in your cloud
library and syncs to the desktop app on the next sync.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from pyzotero import zotero

CREDENTIALS_PATH = Path.home() / ".config" / "zotero" / "credentials.json"
CROSSREF_URL = "https://api.crossref.org/works/{doi}"


def load_credentials() -> tuple[str, str]:
    user_id = os.environ.get("ZOTERO_USER_ID")
    api_key = os.environ.get("ZOTERO_API_KEY")
    if user_id and api_key:
        return user_id, api_key
    if CREDENTIALS_PATH.exists():
        data = json.loads(CREDENTIALS_PATH.read_text())
        return data["user_id"], data["api_key"]
    sys.exit(
        "No Zotero credentials found.\n"
        "Set ZOTERO_USER_ID / ZOTERO_API_KEY or create "
        f"{CREDENTIALS_PATH} with {{\"user_id\": ..., \"api_key\": ...}}.\n"
        "Keys: https://www.zotero.org/settings/keys (enable write access)."
    )


def crossref_metadata(doi: str) -> dict:
    """Fetch metadata for a DOI from CrossRef and map it to Zotero item fields."""
    request = urllib.request.Request(
        CROSSREF_URL.format(doi=doi),
        headers={"User-Agent": "add_to_zotero.py (mailto:sven@goerdes.com)"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        work = json.load(response)["message"]

    creators = [
        {
            "creatorType": "author",
            "firstName": author.get("given", ""),
            "lastName": author.get("family", ""),
        }
        for author in work.get("author", [])
    ]
    date_parts = (work.get("issued", {}).get("date-parts") or [[None]])[0]
    return {
        "title": " ".join(work.get("title", [""])),
        "creators": creators,
        "date": "-".join(str(p) for p in date_parts if p is not None),
        "DOI": doi,
        "publicationTitle": " ".join(work.get("container-title", [""])),
        "volume": work.get("volume", ""),
        "issue": work.get("issue", ""),
        "pages": work.get("page", ""),
        "url": work.get("URL", ""),
    }


def find_collection_key(zot: zotero.Zotero, name: str) -> str:
    matches = [
        c["key"]
        for c in zot.everything(zot.collections())
        if c["data"]["name"].lower() == name.lower()
    ]
    if not matches:
        sys.exit(f"Collection {name!r} not found in the library.")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path, help="Path to the PDF to upload")
    parser.add_argument("--doi", help="DOI to fetch metadata from CrossRef")
    parser.add_argument("--title", help="Item title (fallback: PDF filename)")
    parser.add_argument("--collection", help="Collection name to file the item under")
    parser.add_argument(
        "--item-type", default="journalArticle",
        help="Zotero item type (default: journalArticle; e.g. preprint, conferencePaper)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        sys.exit(f"PDF not found: {args.pdf}")

    user_id, api_key = load_credentials()
    zot = zotero.Zotero(user_id, "user", api_key)

    item = zot.item_template(args.item_type)
    if args.doi:
        print(f"Fetching CrossRef metadata for {args.doi} ...")
        fetched = crossref_metadata(args.doi)
        item.update({k: v for k, v in fetched.items() if k in item and v})
    if args.title:
        item["title"] = args.title
    if not item.get("title"):
        item["title"] = args.pdf.stem

    if args.collection:
        item["collections"] = [find_collection_key(zot, args.collection)]

    created = zot.create_items([item])
    if not created.get("successful"):
        sys.exit(f"Item creation failed: {created}")
    parent_key = created["successful"]["0"]["key"]
    print(f"Created item {parent_key}: {item['title']}")

    result = zot.attachment_simple([str(args.pdf)], parent_key)
    if result.get("success") or result.get("unchanged"):
        print(f"Attached PDF: {args.pdf.name}")
    else:
        sys.exit(f"PDF upload failed: {result}")

    print("Done. The item will appear in the desktop app on the next sync.")


if __name__ == "__main__":
    main()
