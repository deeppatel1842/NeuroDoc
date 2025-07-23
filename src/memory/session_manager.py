"""
Session Management System

Handles user sessions, conversation history, and memory management
for the NeuroDoc RAG system.
"""

import logging
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from ..config import MEMORY_CONFIG, DATA_PATHS
from ..utils.file_utils import ensure_directory, save_json, load_json, delete_file_safe

logger = logging.getLogger(__name__)


class ConversationTurn:
    """Single conversation turn between user and AI."""
    
    def __init__(
        self, 
        question: str, 
        answer: str, 
        citations: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ):
        self.turn_id = str(uuid.uuid4())
        self.question = question
        self.answer = answer
        self.citations = citations
        self.metadata = metadata
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        turn = cls(
            question=data["question"],
            answer=data["answer"],
            citations=data["citations"],
            metadata=data["metadata"]
        )
        turn.turn_id = data["turn_id"]
        turn.timestamp = datetime.fromisoformat(data["timestamp"])
        return turn


class Session:
    """User session with conversation history."""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.last_activity = self.created_at
        self.conversation_history: List[ConversationTurn] = []
        self.document_ids: List[str] = []
        self.is_active = True
    
    def add_conversation_turn(
        self, 
        question: str, 
        answer: str, 
        citations: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ):
        """Add new conversation turn."""
        turn = ConversationTurn(question, answer, citations, metadata)
        self.conversation_history.append(turn)
        self.last_activity = datetime.utcnow()
        
        # Keep conversation history manageable
        max_history = MEMORY_CONFIG.get("max_conversation_turns", 50)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def add_document(self, document_id: str):
        """Add document to session."""
        if document_id not in self.document_ids:
            self.document_ids.append(document_id)
            self.last_activity = datetime.utcnow()
    
    def remove_document(self, document_id: str):
        """Remove document from session."""
        if document_id in self.document_ids:
            self.document_ids.remove(document_id)
            self.last_activity = datetime.utcnow()
    
    def get_recent_conversation(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation turns."""
        recent_turns = self.conversation_history[-limit:] if limit > 0 else self.conversation_history
        return [turn.to_dict() for turn in recent_turns]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary."""
        return {
            "total_turns": len(self.conversation_history),
            "session_duration": (self.last_activity - self.created_at).total_seconds(),
            "document_count": len(self.document_ids),
            "last_activity": self.last_activity.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "document_ids": self.document_ids,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {})
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.document_ids = data.get("document_ids", [])
        session.is_active = data.get("is_active", True)
        
        # Rebuild conversation history
        for turn_data in data.get("conversation_history", []):
            turn = ConversationTurn.from_dict(turn_data)
            session.conversation_history.append(turn)
        
        return session


class SessionManager:
    """Manages user sessions and conversation memory."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.sessions_path = Path(DATA_PATHS["sessions"])
        ensure_directory(self.sessions_path)
        
        # Config settings
        self.session_timeout = timedelta(hours=MEMORY_CONFIG.get("session_timeout_hours", 24))
        self.max_sessions_per_user = MEMORY_CONFIG.get("max_sessions_per_user", 10)
        self.auto_save_interval = MEMORY_CONFIG.get("auto_save_interval_minutes", 5)
        
        # Will start background tasks when needed
        self._background_tasks_started = False
    
    def _start_background_tasks(self):
        """Start background tasks."""
        if self._background_tasks_started:
            return
        
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._periodic_cleanup())
            asyncio.create_task(self._periodic_save())
            self._background_tasks_started = True
        except RuntimeError:
            # No event loop yet, will start later
            pass
    
    async def create_session(
        self, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional specific session ID to use
            user_id: Optional user identifier
            metadata: Optional session metadata
            
        Returns:
            New session ID
        """
        try:
            # Start background tasks if not already started
            if not self._background_tasks_started:
                self._start_background_tasks()
                
            if session_id is None:
                session_id = str(uuid.uuid4())
            session = Session(session_id, user_id, metadata)
            
            # Check user session limit
            if user_id:
                await self._enforce_user_session_limit(user_id)
            
            self.sessions[session_id] = session
            
            # Save immediately
            await self._save_session(session)
            
            logger.info(f"Created new session {session_id} for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        # Check in-memory first
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try to load from disk
        session = await self._load_session(session_id)
        if session:
            self.sessions[session_id] = session
            return session
        
        return None
    
    async def add_to_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        citations: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ):
        """Add a conversation turn to a session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.add_conversation_turn(question, answer, citations, metadata)
        
        # Auto-save if needed
        if len(session.conversation_history) % 5 == 0:  # Save every 5 turns
            await self._save_session(session)
    
    async def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        return session.get_recent_conversation(limit)
    
    async def add_document_to_session(self, session_id: str, document_id: str):
        """Add a document to a session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.add_document(document_id)
        await self._save_session(session)
    
    async def remove_document_from_session(self, session_id: str, document_id: str):
        """Remove a document from a session."""
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.remove_document(document_id)
        await self._save_session(session)
    
    async def delete_session(self, session_id: str):
        """Delete a session and its data."""
        try:
            # Remove from memory
            if session_id in self.sessions:
                del self.sessions[session_id]
            
            # Delete from disk
            session_file = self.sessions_path / f"{session_id}.json"
            delete_file_safe(session_file)
            
            logger.info(f"Deleted session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise
    
    async def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a user."""
        user_sessions = []
        
        # Check in-memory sessions
        for session in self.sessions.values():
            if session.user_id == user_id:
                user_sessions.append({
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "document_count": len(session.document_ids),
                    "conversation_turns": len(session.conversation_history)
                })
        
        # Check disk for sessions not in memory
        for session_file in self.sessions_path.glob("*.json"):
            session_id = session_file.stem
            if session_id not in self.sessions:
                try:
                    session_data = load_json(session_file)
                    if session_data.get("user_id") == user_id:
                        user_sessions.append({
                            "session_id": session_id,
                            "created_at": session_data["created_at"],
                            "last_activity": session_data["last_activity"],
                            "document_count": len(session_data.get("document_ids", [])),
                            "conversation_turns": len(session_data.get("conversation_history", []))
                        })
                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
                    continue
        
        # Sort by last activity
        user_sessions.sort(key=lambda x: x["last_activity"], reverse=True)
        return user_sessions
    
    async def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session."""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        stats = session.get_conversation_summary()
        stats.update({
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active
        })
        
        return stats
    
    async def _save_session(self, session: Session):
        """Save a session to disk."""
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"
            save_json(session.to_dict(), session_file)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    async def _load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk."""
        try:
            session_file = self.sessions_path / f"{session_id}.json"
            if not session_file.exists():
                return None
            
            session_data = load_json(session_file)
            return Session.from_dict(session_data)
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    async def _enforce_user_session_limit(self, user_id: str):
        """Enforce maximum sessions per user."""
        user_sessions = await self.list_user_sessions(user_id)
        
        if len(user_sessions) >= self.max_sessions_per_user:
            # Delete oldest sessions
            sessions_to_delete = user_sessions[self.max_sessions_per_user-1:]
            for session_info in sessions_to_delete:
                await self.delete_session(session_info["session_id"])
                logger.info(f"Deleted old session {session_info['session_id']} for user {user_id}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Session cleanup failed: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        # Check in-memory sessions
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        # Check disk sessions
        for session_file in self.sessions_path.glob("*.json"):
            session_id = session_file.stem
            if session_id not in self.sessions:
                try:
                    session_data = load_json(session_file)
                    last_activity = datetime.fromisoformat(session_data["last_activity"])
                    if current_time - last_activity > self.session_timeout:
                        expired_sessions.append(session_id)
                except Exception as e:
                    logger.error(f"Failed to check session {session_id}: {e}")
                    continue
        
        # Delete expired sessions
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _periodic_save(self):
        """Periodically save active sessions."""
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval * 60)  # Convert minutes to seconds
                
                # Save all in-memory sessions
                for session in self.sessions.values():
                    await self._save_session(session)
                
                logger.debug(f"Auto-saved {len(self.sessions)} sessions")
                
            except Exception as e:
                logger.error(f"Periodic save failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory and session statistics."""
        return {
            "active_sessions": len(self.sessions),
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            "max_sessions_per_user": self.max_sessions_per_user,
            "auto_save_interval_minutes": self.auto_save_interval
        }
