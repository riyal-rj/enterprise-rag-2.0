from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.core.llm import sparse_embedding_client as module
from app.core.llm.sparse_embedding_client import FastEmbedSparseClient


class _Arr(list):
    """Minimal stand-in for the numpy-backed arrays FastEmbed returns -
    only ``tolist()`` is exercised by ``FastEmbedSparseClient``."""

    def tolist(self) -> list:
        return list(self)


def test_construction_does_not_load_the_model(monkeypatch) -> None:
    """The model must not be loaded at __init__ time - this client is
    always constructed by app.api.deps.get_retrieval_strategies()
    regardless of HYBRID_SEARCH_ENABLED, so a dense-only deployment must
    not pay for a local model load it will never use."""
    load_calls: list[str] = []
    monkeypatch.setattr(
        module, "SparseTextEmbedding", lambda model_name: load_calls.append(model_name)
    )

    FastEmbedSparseClient()

    assert load_calls == []


def test_model_loads_lazily_on_first_embed_documents_call(monkeypatch) -> None:
    load_calls: list[str] = []
    fake_model = MagicMock()
    fake_model.embed.return_value = [SimpleNamespace(indices=_Arr([1, 2]), values=_Arr([0.1, 0.2]))]

    def _fake_ctor(model_name: str) -> MagicMock:
        load_calls.append(model_name)
        return fake_model

    monkeypatch.setattr(module, "SparseTextEmbedding", _fake_ctor)
    client = FastEmbedSparseClient()

    client.embed_documents(["hello"])

    assert load_calls == ["Qdrant/bm25"]
    fake_model.embed.assert_called_once_with(["hello"])


def test_model_loads_lazily_on_first_embed_query_call(monkeypatch) -> None:
    load_calls: list[str] = []
    fake_model = MagicMock()
    fake_model.query_embed.return_value = iter(
        [SimpleNamespace(indices=_Arr([1]), values=_Arr([1.0]))]
    )

    def _fake_ctor(model_name: str) -> MagicMock:
        load_calls.append(model_name)
        return fake_model

    monkeypatch.setattr(module, "SparseTextEmbedding", _fake_ctor)
    client = FastEmbedSparseClient()

    client.embed_query("refund policy")

    assert load_calls == ["Qdrant/bm25"]


def test_model_is_loaded_only_once_across_multiple_calls(monkeypatch) -> None:
    load_calls: list[str] = []
    fake_model = MagicMock()
    fake_model.embed.return_value = []

    def _fake_ctor(model_name: str) -> MagicMock:
        load_calls.append(model_name)
        return fake_model

    monkeypatch.setattr(module, "SparseTextEmbedding", _fake_ctor)
    client = FastEmbedSparseClient()

    client.embed_documents(["a"])
    client.embed_documents(["b"])

    assert len(load_calls) == 1


def test_embed_documents_short_circuits_on_empty_input_without_loading_model(monkeypatch) -> None:
    load_calls: list[str] = []
    monkeypatch.setattr(
        module, "SparseTextEmbedding", lambda model_name: load_calls.append(model_name)
    )
    client = FastEmbedSparseClient()

    result = client.embed_documents([])

    assert result == []
    assert load_calls == []
