from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from typing import List
from app.models.schemas import (
    MultiDocumentUploadResponse,
    SingleUploadResult,
    DocumentListResponse,
    DocumentInfo,
    DeleteDocumentResponse,
)
from app.services.pdf_service import save_upload, extract_text_from_pdf, chunk_text
from app.services.rag_service import ingest_chunks, delete_document_from_chroma, list_documents_from_chroma
from app.utils.logger import get_logger
from app.utils.auth import verify_api_key
from datetime import datetime

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)
logger = get_logger(__name__)

_doc_meta: dict[str, dict] = {}


@router.post("/upload", response_model=MultiDocumentUploadResponse)
async def upload_document(files: List[UploadFile] = File(...)):
    results: List[SingleUploadResult] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append(SingleUploadResult(
                filename=file.filename,
                status="failed",
                message="Only PDF files are supported.",
            ))
            continue

        logger.info(f"Uploading document: {file.filename}")
        file_bytes = await file.read()

        if len(file_bytes) == 0:
            results.append(SingleUploadResult(
                filename=file.filename,
                status="failed",
                message="Uploaded file is empty.",
            ))
            continue

        try:
            doc_id, file_path = save_upload(file_bytes, file.filename)
            text = extract_text_from_pdf(file_path)

            if not text.strip():
                results.append(SingleUploadResult(
                    filename=file.filename,
                    status="failed",
                    message="Could not extract text from PDF. It may be scanned/image-only.",
                ))
                continue

            chunks = chunk_text(text)
            total_chunks = ingest_chunks(doc_id, file.filename, chunks)

            _doc_meta[doc_id] = {
                "document_id": doc_id,
                "filename": file.filename,
                "total_chunks": total_chunks,
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Document uploaded successfully: doc_id={doc_id}, chunks={total_chunks}")
            results.append(SingleUploadResult(
                document_id=doc_id,
                filename=file.filename,
                total_chunks=total_chunks,
                status="success",
                message=f"Uploaded and indexed successfully with {total_chunks} chunks.",
            ))

        except Exception as e:
            logger.error(f"Upload failed for {file.filename}: {e}")
            results.append(SingleUploadResult(
                filename=file.filename,
                status="failed",
                message=f"Failed to process document: {str(e)}",
            ))

    total_uploaded = sum(1 for r in results if r.status == "success")
    total_failed = sum(1 for r in results if r.status == "failed")

    return MultiDocumentUploadResponse(
        results=results,
        total_uploaded=total_uploaded,
        total_failed=total_failed,
    )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    docs = list_documents_from_chroma()
    doc_infos = []
    for d in docs:
        meta = _doc_meta.get(d["document_id"], {})
        doc_infos.append(DocumentInfo(
            document_id=d["document_id"],
            filename=d["filename"],
            total_chunks=d["chunk_count"],
            uploaded_at=meta.get("uploaded_at", "unknown"),
        ))
    return DocumentListResponse(documents=doc_infos, total=len(doc_infos))


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    deleted = delete_document_from_chroma(document_id)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    _doc_meta.pop(document_id, None)
    return DeleteDocumentResponse(
        document_id=document_id,
        message=f"Document deleted successfully ({deleted} chunks removed).",
    )
