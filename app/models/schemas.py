from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    message: str


class SingleUploadResult(BaseModel):
    document_id: Optional[str] = None
    filename: str
    total_chunks: int = 0
    status: str
    message: str


class MultiDocumentUploadResponse(BaseModel):
    results: List[SingleUploadResult]
    total_uploaded: int
    total_failed: int


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteDocumentResponse(BaseModel):
    document_id: str
    message: str


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    content_preview: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    document_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What are the main topics covered in this document?",
                "document_id": None,
                "session_id": "user-session-001"
            }
        }
    }


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str
    question: str


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryMessage]
    total: int


class ClearHistoryResponse(BaseModel):
    session_id: str
    message: str
