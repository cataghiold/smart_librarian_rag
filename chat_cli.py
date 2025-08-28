import os
import json
import re
from typing import Dict, Any, List
from openai import OpenAI
from rag_utils import semantic_search
from tool_summaries import get_summary_by_title

# --- Config ---
MODEL = os.getenv("MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "Ești Smart Librarian, un asistent care recomandă cărți bazat pe căutare semantică (RAG). "
    "Ține cont de interesele utilizatorului și sugerează cea mai potrivită carte dintre rezultate. "
    "Dacă utilizatorul cere explicit o carte (ex: 'Ce este 1984?'), alege acel titlu."
)

# Limbaj nepotrivit
BAD_WORDS = {"prost", "idiot", "ură", "urăsc", "nașpa"}  # extensibil după nevoie

def is_offensive(text: str) -> bool:
    tokens = re.findall(r"[\w'-]+", text.lower())
    return any(tok in BAD_WORDS for tok in tokens)

# Tool schema pentru function-calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru titlul de carte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Titlul exact al cărții"}
                },
                "required": ["title"]
            }
        },
    }
]

def recommend_with_llm(query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = OpenAI()
    # Pregătim contextul RAG
    context = "CANDIDAȚI:\n" + "\n\n".join([f"- Title: {c['title']}\nSummary: {c['summary']}" for c in candidates])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "system", "content": context},
        {"role": "system", "content": "Alege cel mai potrivit titlu și răspunde conversațional."}
    ]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.4,
    )
    msg = resp.choices[0].message

    chosen_title = None
    assistant_text = msg.content or ""

    # Verificăm dacă modelul vrea să apeleze tool-ul
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments or "{}")
                chosen_title = args.get("title")

    # Dacă nu a ales explicit, ghicim primul candidat
    if not chosen_title and candidates:
        chosen_title = candidates[0]["title"]

    return {"assistant_text": assistant_text, "title": chosen_title}

def main():
    print("=== Smart Librarian (CLI) ===")
    print("Scrie cererea ta (ex: 'Vreau o carte despre prietenie și magie'). Ctrl+C pentru ieșire.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break

        if not query:
            continue

        # Filtru limbaj nepotrivit (opțional)
        if is_offensive(query):
            print("Prefer să păstrăm un limbaj politicos. Reformulează te rog.")
            continue

        # 1) căutăm semantic în Chroma
        candidates = semantic_search(query, n_results=3)
        if not candidates:
            print("Nu am găsit potriviri. Încearcă să reformulezi întrebarea.")
            continue

        # 2) LLM alege recomandarea și poate decide să cheme tool-ul
        rec = recommend_with_llm(query, candidates)
        title = rec["title"]
        assistant_text = rec["assistant_text"]

        # 3) Afișăm recomandarea
        if assistant_text:
            print(f"\nRecomandare: {assistant_text}\n")

        if title:
            # 4) Tool: rezumat complet
            full = get_summary_by_title(title)
            print(f"=== Rezumat complet pentru '{title}' ===")
            print(full)
            print("=" * 60)
        else:
            print("Nu am reușit să identific un titlu exact.")

if __name__ == "__main__":
    main()
