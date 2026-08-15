from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from app.controllers.chat_controller import ChatController
from app.models.chat_history import ChatHistoryEntry
from app.rag_services.rag_service import RAGService
from app.repositories.chat_history_repository import ChatHistoryRepository
from app.schemas.chat import ChatRequest, ChatResponse, ResponseMetadata


class _FakeRAGService:
    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def answer(self, question: str, top_k: int = 5) -> ChatResponse:
        self.calls.append({"question": question, "top_k": top_k})
        return self._response


class _FakeChatHistoryRepository:
    def __init__(self, *, fail_create: bool = False) -> None:
        self._fail_create = fail_create
        self.created: list[dict[str, object]] = []
        self._entries: list[ChatHistoryEntry] = []

    def create(
        self,
        username: str,
        question: str,
        answer: str,
        sources: list[str],
        confidence: float,
    ) -> ChatHistoryEntry:
        if self._fail_create:
            raise ConnectionError("db unavailable")
        entry = ChatHistoryEntry(
            id=len(self._entries) + 1,
            username=username,
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            created_at=datetime.now(UTC),
        )
        self._entries.append(entry)
        self.created.append(
            {"username": username, "question": question, "answer": answer, "sources": sources}
        )
        return entry

    def list_by_user(self, username: str, limit: int, offset: int) -> list[ChatHistoryEntry]:
        matching = [e for e in self._entries if e.username == username]
        return matching[offset : offset + limit]


def _response(answer: str = "hello") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["a.pdf"],
        confidence=0.9,
        metadata=ResponseMetadata(route="rag", retrieved_chunks=[]),
    )


def test_chat_returns_rag_response_unchanged() -> None:
    rag_service = _FakeRAGService(_response("the answer"))
    history_repo = _FakeChatHistoryRepository()
    controller = ChatController(
        cast(RAGService, rag_service), cast(ChatHistoryRepository, history_repo)
    )

    result = controller.chat("alice", ChatRequest(question="what is the policy?"))

    assert result.answer == "the answer"


def test_chat_persists_a_history_entry_for_the_caller() -> None:
    rag_service = _FakeRAGService(_response("the answer"))
    history_repo = _FakeChatHistoryRepository()
    controller = ChatController(
        cast(RAGService, rag_service), cast(ChatHistoryRepository, history_repo)
    )

    controller.chat("alice", ChatRequest(question="what is the policy?"))

    assert len(history_repo.created) == 1
    assert history_repo.created[0]["username"] == "alice"
    assert history_repo.created[0]["question"] == "what is the policy?"
    assert history_repo.created[0]["answer"] == "the answer"


def test_chat_history_write_failure_does_not_crash_the_request() -> None:
    rag_service = _FakeRAGService(_response("the answer"))
    history_repo = _FakeChatHistoryRepository(fail_create=True)
    controller = ChatController(
        cast(RAGService, rag_service), cast(ChatHistoryRepository, history_repo)
    )

    result = controller.chat("alice", ChatRequest(question="q"))

    assert result.answer == "the answer"


def test_get_history_returns_only_the_caller_own_entries_most_recent_first() -> None:
    rag_service = _FakeRAGService(_response())
    history_repo = _FakeChatHistoryRepository()
    controller = ChatController(
        cast(RAGService, rag_service), cast(ChatHistoryRepository, history_repo)
    )
    controller.chat("alice", ChatRequest(question="alice q1"))
    controller.chat("bob", ChatRequest(question="bob q1"))
    controller.chat("alice", ChatRequest(question="alice q2"))

    result = controller.get_history("alice", limit=20, offset=0)

    assert [item.question for item in result.items] == ["alice q1", "alice q2"]


def test_get_history_respects_limit_and_offset() -> None:
    rag_service = _FakeRAGService(_response())
    history_repo = _FakeChatHistoryRepository()
    controller = ChatController(
        cast(RAGService, rag_service), cast(ChatHistoryRepository, history_repo)
    )
    for i in range(5):
        controller.chat("alice", ChatRequest(question=f"q{i}"))

    result = controller.get_history("alice", limit=2, offset=1)

    assert [item.question for item in result.items] == ["q1", "q2"]
