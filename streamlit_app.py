import os
import json
import re
import io
import base64
import streamlit as st
from openai import OpenAI
from rag_utils import semantic_search
from tool_summaries import get_summary_by_title

MODEL = os.getenv("MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "Ești Smart Librarian, un asistent care recomandă cărți bazat pe căutare semantică (RAG). "
    "Ține cont de interesele utilizatorului și sugerează cea mai potrivită carte dintre rezultate."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru titlul de carte.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    }
]

def recommend(query, candidates):
    client = OpenAI()
    context = "CANDIDAȚI:\n" + "\n\n".join([f"- Title: {c['title']}\nSummary: {c['summary']}" for c in candidates])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "system", "content": context},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0.5
    )
    msg = resp.choices[0].message
    text = msg.content or ""
    title = None
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "get_summary_by_title":
                args = json.loads(tc.function.arguments or "{}")
                title = args.get("title")
    # Dacă nu există tool_call, încearcă să extragi titlul din textul recomandării
    if not title and candidates:
        # Caută titlul oricărei cărți candidate în textul generat
        for cand in candidates:
            if cand["title"] in text:
                title = cand["title"]
                break
        # Dacă nu găsește, folosește fallback-ul clasic
        if not title:
            title = candidates[0]["title"]
    return text, title

st.set_page_config(page_title="Smart Librarian", page_icon="📚")
st.title("📚 Smart Librarian")

# Inițializare variabile în session_state
if 'recomandare' not in st.session_state:
    st.session_state['recomandare'] = ''
if 'rezumat' not in st.session_state:
    st.session_state['rezumat'] = ''
if 'imagine' not in st.session_state:
    st.session_state['imagine'] = None
if 'audio' not in st.session_state:
    st.session_state['audio'] = None

# Funcție pentru resetare rezultate la fiecare căutare nouă
def reset_outputs():
    st.session_state['recomandare'] = ''
    st.session_state['rezumat'] = ''
    st.session_state['imagine'] = None
    st.session_state['audio'] = None



# Text box pe un rând, apoi sub el: butonul și opțiunile pe același rând
query = st.text_input("Spune-mi ce vrei să citești:", placeholder="Ex: Vreau o carte despre prietenie și magie")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    go = st.button("Caută", on_click=reset_outputs)
with col2:
    tts = st.checkbox("Audio", value=False)
with col3:
    img_gen = st.checkbox("Imagine", value=False)

if go and query:
    cands = semantic_search(query, n_results=3)
    if not cands:
        st.warning("Nu am găsit potriviri. Încearcă să reformulezi.")
    else:
        with st.spinner("Generez recomandarea..."):
            text, title = recommend(query, cands)
        if text:
            st.session_state['recomandare'] = f"**Recomandare**: {text}"
        if title and any(title == c['title'] for c in cands):
            full = get_summary_by_title(title)
            st.session_state['rezumat'] = f"### Rezumat complet: {title}\n{full}"

            if img_gen:
                try:
                    client = OpenAI()
                    prompt = f"Create a stylized, tasteful cover-like illustration capturing themes from the book '{title}'."
                    img = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
                    b64 = img.data[0].b64_json
                    st.session_state['imagine'] = base64.b64decode(b64)
                except Exception as e:
                    st.info(f"Nu am putut genera imaginea: {e}")

            if tts:
                try:
                    from gtts import gTTS
                    tts_text = f"Recomandare: {text}. Rezumat: {full}"
                    tts = gTTS(text=tts_text, lang='ro')
                    tts.save("recommendation.mp3")
                    st.session_state['audio'] = "recommendation.mp3"
                except Exception as e:
                    st.info(f"TTS indisponibil: {e}")
        else:
            st.session_state['rezumat'] = ''

# Afișare rezultate din session_state
if st.session_state['recomandare']:
    st.markdown(st.session_state['recomandare'])
if st.session_state['rezumat']:
    st.markdown(st.session_state['rezumat'])
if st.session_state['imagine']:
    st.image(st.session_state['imagine'], caption="Imagine generată")
if st.session_state['audio']:
    st.audio(st.session_state['audio'])
