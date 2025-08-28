import json
import os

DATA_JSON = os.getenv("BOOK_JSON_PATH", "./data/book_summaries.json")

with open(DATA_JSON, encoding="utf-8") as f:
    BOOK_SUMMARIES = json.load(f)

def get_summary_by_title(title: str) -> str:
    """Returnează rezumatul complet pentru un titlu exact."""
    return BOOK_SUMMARIES.get(title) or "Nu am găsit un rezumat complet pentru acest titlu."
