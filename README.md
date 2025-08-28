# Smart Librarian – AI cu RAG + Tool Completion

Acest proiect construiește un chatbot care recomandă cărți folosind **OpenAI GPT + RAG (ChromaDB)**. După recomandare, chatbotul apelează un **tool local** `get_summary_by_title(title)` ca să afișeze **rezumatul complet**.

## Conținut
- `data/book_summaries.md` – baza de date cu 10+ rezumate scurte (pentru indexare în vector store).
- `data/book_summaries.json` – rezumate complete pentru tool.
- `rag_utils.py` – inițializare ChromaDB, generare embeddinguri și căutare semantică.
- `tool_summaries.py` – implementarea tool-ului `get_summary_by_title(title)`.
- `chat_cli.py` – interfață CLI care integrează RAG + GPT + tool calling (și filtru simplu anti-limbaj nepotrivit).
- `streamlit_app.py` – interfață simplă în browser, include gTTS și Image Generation.
- `requirements.txt` – dependențe.

## Cerințe de mediu
- Python 3.10+
- Variabile de mediu:
  - `OPENAI_API_KEY` – cheia ta OpenAI
  - (opțional) `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
  - (opțional) `MODEL` (default: `gpt-4o-mini`)
  - (opțional) `CHROMA_DIR` (default: `./chroma_db`)

## Instalare
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set OPENAI_API_KEY=...
```

## Inițializare & rulare (CLI)
1. **Primul run** va crea automat colecția `book_summaries` în Chroma (persistată pe disc).
2. Rulează CLI:
```bash
python chat_cli.py
```
3. Exemple întrebări:
   - `Vreau o carte despre libertate și control social.`
   - `Ce-mi recomanzi dacă iubesc poveștile fantastice?`
   - `Vreau o carte despre prietenie și magie`
   - `Ce este 1984?`

## Interfață Web
```bash
streamlit run streamlit_app.py
```
Opțiuni în sidebar:
- **Text to Speech (gTTS, limba română)** – salvează `recommendation.mp3` local și îl redă folosind Google Text-to-Speech.
- **Generează imagine** – folosește `gpt-image-1` pentru o ilustrație reprezentativă.


> Notă: Pentru TTS în limba română, aplicația folosește gTTS (Google Text-to-Speech). Nu este nevoie de voci suplimentare instalate în Windows.

## Arhitectură & Flow
1. Utilizatorul scrie o cerere (ex: „Vreau o carte despre prietenie și magie”).  
2. `semantic_search()` caută în Chroma pe baza embeddingurilor OpenAI.  
3. Top 3 rezultate sunt trimise, ca **context RAG**, către modelul GPT.  
4. Modelul alege un titlu și **apelează tool-ul** `get_summary_by_title(title)` (function calling).  
5. Afișăm: recomandarea + **rezumatul complet**.  
6. (Opțional) gTTS salvează/reda audio. (Opțional) se generează o imagine.

## Filtru limbaj nepotrivit
În `chat_cli.py` există o listă minimală de cuvinte (configurabilă). Dacă sunt detectate, cererea nu e trimisă mai departe către LLM.
