import fitz
import uuid
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def save_upload(file_bytes: bytes, filename: str) -> str:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}_{filename}"
    file_path = upload_dir / safe_name

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info(f"Saved PDF: {safe_name} ({len(file_bytes)} bytes)")
    return doc_id, str(file_path)


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()
    extracted = "\n\n".join(full_text)
    logger.info(f"Extracted {len(extracted)} characters from {file_path}")
    return extracted


def chunk_text(text: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    result = [
        {"content": chunk, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]
    logger.info(f"Created {len(result)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return result
