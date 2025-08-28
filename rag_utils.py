import os
import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

def get_openai() -> OpenAI:
    # Requires OPENAI_API_KEY in env
    return OpenAI()

def load_summaries_md(md_path: str) -> List[Dict[str, str]]:
    items = []
    current_title = None
    current_lines = []
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("## Title:"):
                # flush previous
                if current_title:
                    items.append({"title": current_title, "summary": "\n".join(current_lines).strip()})
                current_title = line.replace("## Title:", "").strip()
                current_lines = []
            else:
                if line:
                    current_lines.append(line)
        if current_title:
            items.append({"title": current_title, "summary": "\n".join(current_lines).strip()})
    return items

def init_chroma(persist_directory: str = CHROMA_DIR):
    client = chromadb.PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    return client

def make_embeddings(texts: List[str]) -> List[List[float]]:
    client = get_openai()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_collection(client, collection_name: str, items: List[Dict[str, str]]):
    col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    # Upsert
    ids = []
    documents = []
    metadatas = []
    for idx, it in enumerate(items):
        ids.append(f"book-{idx}")
        documents.append(it["summary"])
        metadatas.append({"title": it["title"]})
    # Compute embeddings externally
    vecs = make_embeddings(documents)
    col.upsert(ids=ids, embeddings=vecs, documents=documents, metadatas=metadatas)
    return col

def get_or_create_book_collection() -> Tuple[Any, List[Dict[str, str]]]:
    client = init_chroma()
    md_path = os.getenv("BOOK_MD_PATH", "./data/book_summaries.md")
    items = load_summaries_md(md_path)
    try:
        col = client.get_collection("book_summaries")
    except Exception:
        col = build_collection(client, "book_summaries", items)
    return col, items

def semantic_search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    col, _ = get_or_create_book_collection()
    # Embed query
    q_emb = make_embeddings([query])[0]
    res = col.query(query_embeddings=[q_emb], n_results=n_results, include=["metadatas", "documents", "distances"])
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "title": res["metadatas"][0][i]["title"],
            "summary": res["documents"][0][i],
            "score": 1 - res["distances"][0][i] if res.get("distances") else None
        })
    return out
