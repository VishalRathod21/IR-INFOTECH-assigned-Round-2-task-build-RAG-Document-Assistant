from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time

from app.routers import documents, chat
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="IR INFOTECH — RAG Document Assistant",
    description="AI-powered Document Assistant that answers questions from uploaded PDF documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} [{duration_ms:.1f}ms]"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "RAG Document Assistant", "version": "1.0.0"}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "IR INFOTECH RAG Document Assistant",
        "docs": "/docs",
        "endpoints": {
            "documents": [
                "POST /documents/upload",
                "GET /documents/",
                "DELETE /documents/{document_id}",
            ],
            "chat": [
                "POST /chat/ask",
                "GET /chat/history/{session_id}",
                "DELETE /chat/history/{session_id}",
            ],
        },
    }
