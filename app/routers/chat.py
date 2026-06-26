from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    ChatHistoryResponse,
    ChatHistoryMessage,
    ClearHistoryResponse,
)
from app.services.rag_service import retrieve_relevant_chunks, generate_answer, generate_answer_stream
from app.services.chat_service import (
    get_or_create_session,
    add_message,
    get_history,
    clear_history,
)
from app.utils.logger import get_logger
from app.utils.auth import verify_api_key

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)
logger = get_logger(__name__)


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest) -> ChatResponse:
    logger.info(
        f"POST /chat/ask | session={request.session_id} "
        f"doc_filter={request.document_id} question='{request.question[:60]}'"
    )

    session_id = get_or_create_session(request.session_id)

    try:
        chunks = retrieve_relevant_chunks(request.question, request.document_id)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector search failed: {str(e)}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found. Please upload a document first.",
        )

    history = get_history(session_id)

    try:
        answer = generate_answer(request.question, chunks, history)
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Answer generation failed: {str(e)}",
        )

    add_message(session_id, "user", request.question)
    add_message(session_id, "assistant", answer)

    sources = [
        Source(
            document_id=c["metadata"]["document_id"],
            filename=c["metadata"]["filename"],
            chunk_index=c["metadata"]["chunk_index"],
            content_preview=c["content"][:200],
        )
        for c in chunks
    ]

    logger.info(f"Answer generated | session={session_id} sources={len(sources)}")
    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
        question=request.question,
    )


@router.post("/ask/stream")
async def ask_question_stream(request: ChatRequest):
    logger.info(
        f"POST /chat/ask/stream | session={request.session_id} "
        f"doc_filter={request.document_id} question='{request.question[:60]}'"
    )

    session_id = get_or_create_session(request.session_id)

    try:
        chunks = retrieve_relevant_chunks(request.question, request.document_id)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector search failed: {str(e)}",
        )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found. Please upload a document first.",
        )

    history = get_history(session_id)

    async def event_generator():
        full_answer_parts = []
        try:
            async for chunk_text in generate_answer_stream(request.question, chunks, history):
                full_answer_parts.append(chunk_text)
                yield f"data: {chunk_text}\n\n"

            full_answer = "".join(full_answer_parts)
            add_message(session_id, "user", request.question)
            add_message(session_id, "assistant", full_answer)
            logger.info(
                f"Stream complete | session={session_id} "
                f"answer_len={len(full_answer)}"
            )
        except Exception as e:
            logger.error(f"Streaming LLM generation failed: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str) -> ChatHistoryResponse:
    history = get_history(session_id)
    messages = [
        ChatHistoryMessage(
            role=m["role"],
            content=m["content"],
            timestamp=m["timestamp"],
        )
        for m in history
    ]
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
    )


@router.delete("/history/{session_id}", response_model=ClearHistoryResponse)
async def clear_chat_history(session_id: str) -> ClearHistoryResponse:
    cleared = clear_history(session_id)
    if not cleared:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return ClearHistoryResponse(
        session_id=session_id,
        message="Chat history cleared successfully.",
    )
