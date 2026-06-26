import faiss
import json
import numpy as np
import os
from pathlib import Path
from typing import List
from groq import Groq
from functools import lru_cache
from app.config import get_settings
from app.services.embedding_service import embed_texts, embed_query
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_EMBED_DIM = 384

_index: faiss.IndexFlatIP = None
_metadata: List[dict] = []


def load_index() -> None:
    global _index, _metadata
    index_path = settings.faiss_index_path
    meta_path  = settings.faiss_meta_path

    if os.path.exists(index_path) and os.path.exists(meta_path):
        _index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        logger.info(f"FAISS index loaded from {index_path} ({_index.ntotal} vectors)")
    else:
        _index    = faiss.IndexFlatIP(_EMBED_DIM)
        _metadata = []
        logger.info("FAISS index initialised (empty)")


def save_index() -> None:
    index_path = Path(settings.faiss_index_path)
    meta_path  = Path(settings.faiss_meta_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(_index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_metadata, f, ensure_ascii=False)
    logger.info(f"FAISS index saved ({_index.ntotal} vectors)")


load_index()


@lru_cache()
def get_groq_client() -> Groq:
    client = Groq(api_key=settings.groq_api_key)
    logger.info("Groq client initialized")
    return client


def ingest_chunks(document_id: str, filename: str, chunks: list[dict]) -> int:
    global _index, _metadata

    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)
    vectors    = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    _index.add(vectors)

    for c in chunks:
        _metadata.append({
            "id":          f"{document_id}_chunk_{c['chunk_index']}",
            "document_id": document_id,
            "filename":    filename,
            "chunk_index": c["chunk_index"],
            "content":     c["content"],
        })

    save_index()
    logger.info(f"Ingested {len(chunks)} chunks for document_id={document_id}")
    return len(chunks)


def retrieve_relevant_chunks(query: str, document_id: str = None) -> list[dict]:
    global _index, _metadata

    if _index.ntotal == 0:
        logger.info("FAISS index is empty — no chunks to retrieve")
        return []

    query_vec = np.array([embed_query(query)], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    k = min(settings.top_k_results * 5 if document_id else settings.top_k_results,
            _index.ntotal)

    distances, indices = _index.search(query_vec, k)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = _metadata[idx]
        if document_id and meta["document_id"] != document_id:
            continue
        chunks.append({
            "content":  meta["content"],
            "metadata": {
                "document_id": meta["document_id"],
                "filename":    meta["filename"],
                "chunk_index": meta["chunk_index"],
            },
            "distance": float(dist),
        })
        if len(chunks) == settings.top_k_results:
            break

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
    return chunks


def generate_answer(question: str, context_chunks: list[dict], chat_history: list[dict]) -> str:
    groq = get_groq_client()

    context = "\n\n---\n\n".join(
        f"[Source {i+1} | {c['metadata']['filename']} | Chunk {c['metadata']['chunk_index']}]\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )

    system_prompt = """You are an intelligent AI Document Assistant. Your job is to understand documents and explain their content clearly to the user.

STRICT RULES:
1. NEVER copy-paste sentences directly from the source context. Always rephrase and explain in your own words.
2. Synthesize information from multiple chunks if needed — do not just repeat one chunk.
3. Answer like a knowledgeable human explaining to a friend — natural, clear, conversational.
4. If the question asks for a summary or explanation, give a structured response with your own framing.
5. If the answer is not present in the context at all, say: "I could not find relevant information about this in the uploaded documents."
6. You MAY quote a very short phrase (under 10 words) from the document only if it is a definition, name, or critical term — but always explain it in your own words after quoting.
7. Do NOT start your answer with phrases like "According to the document" or "The document says" — just answer directly.
8. Keep answers concise but complete. Use bullet points or numbered lists when explaining multiple points."""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Here is the relevant content retrieved from the user's documents:\n\n{context}\n\n---\nNow answer this question in your own words, do NOT copy text from above:\nQuestion: {question}"
    })

    response = groq.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.6,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    logger.info(f"Generated answer ({len(answer)} chars) for question: '{question[:60]}'")
    return answer


def delete_document_from_chroma(document_id: str) -> int:
    global _index, _metadata

    before  = len(_metadata)
    kept    = [m for m in _metadata if m["document_id"] != document_id]
    deleted = before - len(kept)

    if deleted == 0:
        return 0

    _index    = faiss.IndexFlatIP(_EMBED_DIM)
    _metadata = []

    if kept:
        texts      = [m["content"] for m in kept]
        embeddings = embed_texts(texts)
        vectors    = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        _index.add(vectors)
        _metadata = kept

    save_index()
    logger.info(f"Deleted {deleted} chunks for document_id={document_id}")
    return deleted


def list_documents_from_chroma() -> list[dict]:
    seen: dict[str, dict] = {}
    for meta in _metadata:
        doc_id = meta["document_id"]
        if doc_id not in seen:
            seen[doc_id] = {
                "document_id": doc_id,
                "filename":    meta["filename"],
                "chunk_count": 1,
            }
        else:
            seen[doc_id]["chunk_count"] += 1
    return list(seen.values())


async def generate_answer_stream(question: str, context_chunks: list[dict], chat_history: list[dict]):
    groq = get_groq_client()

    context = "\n\n---\n\n".join(
        f"[Source {i+1} | {c['metadata']['filename']} | Chunk {c['metadata']['chunk_index']}]\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )

    system_prompt = """You are an intelligent AI Document Assistant. Your job is to understand documents and explain their content clearly to the user.

STRICT RULES:
1. NEVER copy-paste sentences directly from the source context. Always rephrase and explain in your own words.
2. Synthesize information from multiple chunks if needed — do not just repeat one chunk.
3. Answer like a knowledgeable human explaining to a friend — natural, clear, conversational.
4. If the question asks for a summary or explanation, give a structured response with your own framing.
5. If the answer is not present in the context at all, say: "I could not find relevant information about this in the uploaded documents."
6. You MAY quote a very short phrase (under 10 words) from the document only if it is a definition, name, or critical term — but always explain it in your own words after quoting.
7. Do NOT start your answer with phrases like "According to the document" or "The document says" — just answer directly.
8. Keep answers concise but complete. Use bullet points or numbered lists when explaining multiple points."""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Here is the relevant content retrieved from the user's documents:\n\n{context}\n\n---\nNow answer this question in your own words, do NOT copy text from above:\nQuestion: {question}"
    })

    response = groq.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.6,
        max_tokens=1024,
        stream=True,
    )

    buffer = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            buffer += chunk.choices[0].delta.content
            if buffer.endswith((" ", "\n")):
                if buffer.strip():
                    yield buffer.strip()
                buffer = ""

    if buffer.strip():
        yield buffer.strip()
