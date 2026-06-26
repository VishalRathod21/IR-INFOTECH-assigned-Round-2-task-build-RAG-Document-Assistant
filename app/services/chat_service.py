import uuid
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger(__name__)

_sessions: dict[str, list[dict]] = {}


def get_or_create_session(session_id: str = None) -> str:
    if not session_id or session_id not in _sessions:
        session_id = session_id or str(uuid.uuid4())
        _sessions[session_id] = []
        logger.info(f"New session created: {session_id}")
    return session_id


def add_message(session_id: str, role: str, content: str):
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    })


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def clear_history(session_id: str) -> bool:
    if session_id in _sessions:
        _sessions[session_id] = []
        logger.info(f"Cleared history for session: {session_id}")
        return True
    return False
