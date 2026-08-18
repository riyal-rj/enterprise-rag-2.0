"""Corrective-RAG package: contracts, graders, refiners, web retrieval, rollout."""

from __future__ import annotations

from app.rag_services.crag.crag import (
    ChunkGrade,
    CorrectiveRetriever,
    CRAGCohort,
    CRAGDecision,
    CRAGOutcome,
    CRAGPlan,
    EvidenceChunk,
    EvidenceOrigin,
    FailSafeCorrectiveRetriever,
    KnowledgeRefiner,
    PlannedCorrectiveRetriever,
    PlannedNoOpCorrectiveRetriever,
    RetrievalGrade,
    RetrievalGrader,
    SourceScopePolicy,
    StaticPlannedCorrectiveRetriever,
    WebEvidence,
    WebRetriever,
    local_evidence,
)
from app.rag_services.crag.telemetry import CRAGTelemetry, NoOpCRAGTelemetry

__all__ = [
    "CRAGCohort",
    "CRAGDecision",
    "CRAGOutcome",
    "CRAGPlan",
    "CRAGTelemetry",
    "ChunkGrade",
    "CorrectiveRetriever",
    "EvidenceChunk",
    "EvidenceOrigin",
    "FailSafeCorrectiveRetriever",
    "KnowledgeRefiner",
    "NoOpCRAGTelemetry",
    "PlannedCorrectiveRetriever",
    "PlannedNoOpCorrectiveRetriever",
    "RetrievalGrade",
    "RetrievalGrader",
    "SourceScopePolicy",
    "StaticPlannedCorrectiveRetriever",
    "WebEvidence",
    "WebRetriever",
    "local_evidence",
]
