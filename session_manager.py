"""
Session creation, history, and isolation.
In-memory store with TTL expiration.
"""
import uuid
import time
import logging
import threading
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# Thread-safe session store
_sessions: Dict[str, Dict] = {}
_lock = threading.Lock()


def create_session() -> str:
    """Create a new isolated conversation session. Returns session_id."""
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "turns": [],
        }
    logger.info("Created session %s", session_id)
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Get session data, or None if expired/missing."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        # Check TTL
        if time.time() - session["last_active"] > config.SESSION_TTL_MINUTES * 60:
            del _sessions[session_id]
            logger.info("Session %s expired", session_id)
            return None
        return session


def add_turn(session_id: str, role: str, content: str, citations: Optional[List] = None):
    """Append a turn to the session history."""
    with _lock:
        session = _sessions.get(session_id)
        if session is None:
            return
        turn = {"role": role, "content": content}
        if citations is not None:
            turn["citations"] = citations
        session["turns"].append(turn)
        session["last_active"] = time.time()


def get_history(session_id: str) -> List[Dict]:
    """Get conversation turns for a session."""
    session = get_session(session_id)
    if session is None:
        return []
    return session["turns"]


def get_history_for_llm(session_id: str) -> List[Dict]:
    """
    Get conversation history formatted for the LLM.
    Returns list of {"role": "user"|"assistant", "content": str}.
    """
    turns = get_history(session_id)
    llm_history = []
    for t in turns:
        llm_history.append({
            "role": t["role"],
            "content": t["content"],
        })
    return llm_history


def get_session_count() -> int:
    """Return number of active sessions."""
    _cleanup_expired()
    with _lock:
        return len(_sessions)


def _cleanup_expired():
    """Remove expired sessions."""
    now = time.time()
    ttl = config.SESSION_TTL_MINUTES * 60
    with _lock:
        expired = [
            sid for sid, s in _sessions.items()
            if now - s["last_active"] > ttl
        ]
        for sid in expired:
            del _sessions[sid]
    if expired:
        logger.info("Cleaned up %d expired sessions", len(expired))
