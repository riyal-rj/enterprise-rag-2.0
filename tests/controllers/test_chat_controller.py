from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from app.controllers.chat_controller import ChatController
from app.core.exceptions import ConversationNotFoundError
from app.models.chat_history import ChatHistoryEntry
from app.models.conversation import ConversationEntry
from app.rag_services.rag_service import RAGService
from app.repositories.chat_history_repository import ChatHistoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.chat import ChatRequest, ChatResponse, RerankingMetadata, ResponseMetadata


class _FakeRAGService:
    def __init__(self, response: ChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def answer(
        self, question: str, top_k: int = 5, retrieval_mode: str | None = None
    ) -> ChatResponse:
        self.calls.append({"question": question, "top_k": top_k, "retrieval_mode": retrieval_mode})
        return self._response


class _FakeChatHistoryRepository:
    def __init__(self, *, fail_create: bool = False) -> None:
        self._fail_create = fail_create
        self.created: list[dict[str, object]] = []
        self._entries: list[ChatHistoryEntry] = []

    def create(
        self,
        username: str,
        conversation_id: int,
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
            {
                "username": username,
                "conversation_id": conversation_id,
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )
        return entry

    def list_by_user(self, username: str, limit: int, offset: int) -> list[ChatHistoryEntry]:
        matching = [e for e in self._entries if e.username == username]
        return matching[offset : offset + limit]

    def list_by_conversation(
        self, conversation_id: int, limit: int = 200
    ) -> list[ChatHistoryEntry]:
        matching_ids = [
            i for i, c in enumerate(self.created) if c["conversation_id"] == conversation_id
        ]
        return [self._entries[i] for i in matching_ids][:limit]


class _FakeConversationRepository:
    def __init__(self) -> None:
        self._conversations: dict[int, ConversationEntry] = {}
        self._next_id = 1
        self.touched: list[int] = []

    def create(self, username: str, title: str) -> ConversationEntry:
        conversation = ConversationEntry(
            id=self._next_id,
            username=username,
            title=title,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._conversations[conversation.id] = conversation
        self._next_id += 1
        return conversation

    def get_owned(self, conversation_id: int, username: str) -> ConversationEntry | None:
        conversation = self._conversations.get(conversation_id)
        if conversation is None or conversation.username != username:
            return None
        return conversation

    def list_by_user(
        self, username: str, search: str | None, limit: int, offset: int
    ) -> list[ConversationEntry]:
        matching = [c for c in self._conversations.values() if c.username == username]
        if search:
            matching = [c for c in matching if search.lower() in c.title.lower()]
        matching.sort(key=lambda c: c.updated_at, reverse=True)
        return matching[offset : offset + limit]

    def touch(self, conversation_id: int) -> None:
        self.touched.append(conversation_id)


def _response(answer: str = "hello") -> ChatResponse:
    return ChatResponse(
        answer=answer,
        sources=["a.pdf"],
        confidence=0.9,
        metadata=ResponseMetadata(
            route="rag",
            reranking=RerankingMetadata(enabled=False, backend="none"),
            retrieved_chunks=[],
        ),
    )


def _controller(
    *,
    answer: str = "the answer",
    fail_create: bool = False,
) -> tuple[
    ChatController, _FakeRAGService, _FakeChatHistoryRepository, _FakeConversationRepository
]:
    rag_service = _FakeRAGService(_response(answer))
    history_repo = _FakeChatHistoryRepository(fail_create=fail_create)
    conversation_repo = _FakeConversationRepository()
    controller = ChatController(
        cast(RAGService, rag_service),
        cast(ChatHistoryRepository, history_repo),
        cast(ConversationRepository, conversation_repo),
    )
    return controller, rag_service, history_repo, conversation_repo


def test_chat_returns_rag_response_unchanged() -> None:
    controller, *_ = _controller(answer="the answer")

    result = controller.chat("alice", ChatRequest(question="what is the policy?"))

    assert result.answer == "the answer"


def test_chat_forwards_retrieval_mode_override_to_rag_service() -> None:
    controller, rag_service, _, _ = _controller()

    controller.chat("alice", ChatRequest(question="what is the policy?", retrieval_mode="hybrid"))

    assert rag_service.calls[0]["retrieval_mode"] == "hybrid"


def test_chat_persists_a_history_entry_for_the_caller() -> None:
    controller, _, history_repo, _ = _controller()

    controller.chat("alice", ChatRequest(question="what is the policy?"))

    assert len(history_repo.created) == 1
    assert history_repo.created[0]["username"] == "alice"
    assert history_repo.created[0]["question"] == "what is the policy?"
    assert history_repo.created[0]["answer"] == "the answer"


def test_chat_history_write_failure_does_not_crash_the_request() -> None:
    controller, *_ = _controller(fail_create=True)

    result = controller.chat("alice", ChatRequest(question="q"))

    assert result.answer == "the answer"


def test_get_history_returns_only_the_caller_own_entries_most_recent_first() -> None:
    controller, *_ = _controller()
    controller.chat("alice", ChatRequest(question="alice q1"))
    controller.chat("bob", ChatRequest(question="bob q1"))
    controller.chat("alice", ChatRequest(question="alice q2"))

    result = controller.get_history("alice", limit=20, offset=0)

    assert [item.question for item in result.items] == ["alice q1", "alice q2"]


def test_get_history_respects_limit_and_offset() -> None:
    controller, *_ = _controller()
    for i in range(5):
        controller.chat("alice", ChatRequest(question=f"q{i}"))

    result = controller.get_history("alice", limit=2, offset=1)

    assert [item.question for item in result.items] == ["q1", "q2"]


def test_chat_without_conversation_id_creates_a_new_conversation() -> None:
    controller, _, _, conversation_repo = _controller()

    result = controller.chat("alice", ChatRequest(question="what is the refund policy?"))

    assert result.conversation_id != 0
    conversations = conversation_repo.list_by_user("alice", None, limit=20, offset=0)
    assert len(conversations) == 1
    assert conversations[0].id == result.conversation_id
    assert conversations[0].title == "what is the refund policy?"


def test_chat_title_is_truncated_at_a_word_boundary() -> None:
    controller, _, _, conversation_repo = _controller()
    long_question = "why " * 30  # far longer than the 60-char title cap

    controller.chat("alice", ChatRequest(question=long_question))

    title = conversation_repo.list_by_user("alice", None, limit=20, offset=0)[0].title
    assert len(title) <= 61  # 60 chars + the truncation ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")


def test_chat_with_conversation_id_reuses_it_instead_of_creating_a_new_one() -> None:
    controller, _, history_repo, conversation_repo = _controller()
    first = controller.chat("alice", ChatRequest(question="first question"))

    second = controller.chat(
        "alice", ChatRequest(question="second question", conversation_id=first.conversation_id)
    )

    assert second.conversation_id == first.conversation_id
    assert len(conversation_repo.list_by_user("alice", None, limit=20, offset=0)) == 1
    assert len(history_repo.created) == 2


def test_chat_touches_the_conversation_after_saving_a_turn() -> None:
    controller, _, _, conversation_repo = _controller()

    result = controller.chat("alice", ChatRequest(question="q"))

    assert conversation_repo.touched == [result.conversation_id]


def test_chat_with_conversation_id_owned_by_another_user_raises_not_found() -> None:
    controller, _, _, _ = _controller()
    someone_elses = controller.chat("bob", ChatRequest(question="bob's question"))

    with pytest.raises(ConversationNotFoundError):
        controller.chat(
            "alice",
            ChatRequest(question="trying to hijack", conversation_id=someone_elses.conversation_id),
        )


def test_chat_with_nonexistent_conversation_id_raises_not_found() -> None:
    controller, *_ = _controller()

    with pytest.raises(ConversationNotFoundError):
        controller.chat("alice", ChatRequest(question="q", conversation_id=999))


def test_list_conversations_delegates_to_repository() -> None:
    controller, _, _, _ = _controller()
    controller.chat("alice", ChatRequest(question="alice q1"))
    controller.chat("bob", ChatRequest(question="bob q1"))

    result = controller.list_conversations("alice", None, limit=20, offset=0)

    assert [c.title for c in result.items] == ["alice q1"]


def test_list_conversations_filters_by_search() -> None:
    controller, *_ = _controller()
    controller.chat("alice", ChatRequest(question="refund policy question"))
    controller.chat("alice", ChatRequest(question="branch hours question"))

    result = controller.list_conversations("alice", "refund", limit=20, offset=0)

    assert len(result.items) == 1
    assert "refund" in result.items[0].title.lower()


def test_get_conversation_messages_returns_the_full_thread() -> None:
    controller, *_ = _controller()
    first = controller.chat("alice", ChatRequest(question="first question"))
    controller.chat(
        "alice", ChatRequest(question="second question", conversation_id=first.conversation_id)
    )

    result = controller.get_conversation_messages("alice", first.conversation_id)

    assert [item.question for item in result.items] == ["first question", "second question"]


def test_get_conversation_messages_owned_by_another_user_raises_not_found() -> None:
    controller, *_ = _controller()
    someone_elses = controller.chat("bob", ChatRequest(question="bob's question"))

    with pytest.raises(ConversationNotFoundError):
        controller.get_conversation_messages("alice", someone_elses.conversation_id)
