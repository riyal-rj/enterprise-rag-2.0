from __future__ import annotations

import uuid
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)

from app.core.ingestion.document_processor import DocumentChunk
from app.models.retrieved_chunk import RetrievedChunk

_VECTOR_SIZE = 1536

def build_qdrant_client(url: str, 
                        timeout_seconds: float) -> QdrantClient:
    return QdrantClient(url,timeout=int(timeout_seconds))

class VectorRepository(Protocol):
    def upsert_chunks(self, 
                    chunks: list[DocumentChunk], 
                    embeddings: list[list[float]]) -> None:
        ...

    def search(self,
               query_embedding: list[float],
               top_k: int = 5) -> list[RetrievedChunk]:
        ...

    def scroll_all_chunks(self, limit: int = 10_000) -> list[dict[str, str]]:
        ...


class QdrantVectorRepository:

    def __init__(self,
                 client: QdrantClient,
                 collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    def upsert_chunks(self,
                      chunks: list[DocumentChunk],
                      embeddings: list[list[float]]) -> None:

        points=[
            PointStruct(
                id = str(uuid.uuid4()),
                vector = embedding,
                payload={
                    "text":chunk.text,
                    "source":chunk.source
                }
            )
            for chunk, embedding in zip(chunks, embeddings, strict= True)
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points
        )

    def search(self,
               query_embedding: list[float],
               top_k:int = 5) -> list[RetrievedChunk]:
        
        results = self._client.query_points(
            collection_name = self._collection_name,
            query = query_embedding,
            limit = top_k,
            with_payload = True,
        ).points

        return [
            RetrievedChunk(
                text = str(point.payload.get("text","")) if point.payload else "",
                source = str(point.payload.get("source","")) if point.payload else "",
                score = float(point.score)            
                )
            for point in results
        ]

    def scroll_all_chunks(self,limit: int = 10_000) -> list[dict[str, str]]:

        points, _next_page = self._client.scroll(
            collection_name = self._collection_name,
            limit = limit,
            with_payload = True,
            with_vectors = False
        ) 
        return [
            {
                "id":str(point.id),
                "text":str(point.payload.get("text","")) if point.payload else "",
                "source":str(point.payload.get("source","")) if point.payload else "",
            }
            for point in points
        ]   

    def _ensure_collection(self) -> None :
        existing = {collection.name for collection in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )