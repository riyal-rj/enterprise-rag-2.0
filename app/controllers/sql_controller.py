"""SQL proposal approval controller: schema in, schema out.

Same "thin controller" shape as ``ChatController``/``RagOpsController`` -
all approval/execution logic lives in ``app.sql.sql_service.SQLService``;
this only translates between the HTTP layer and it.
"""

from __future__ import annotations

from uuid import UUID

from app.models.auth_user import AuthenticatedUser
from app.schemas.chat import ChatResponse as ChatResponseSchema
from app.schemas.chat import (
    CRAGMetadata,
    HyDEMetadata,
    RerankingMetadata,
    ResponseMetadata,
    SelfReflectionMetadata,
    SQLMetadata,
)
from app.sql.models import to_sql_principal
from app.sql.sql_service import SQLService


class SQLController:
    def __init__(self, sql_service: SQLService) -> None:
        self._sql_service = sql_service

    def approve(self, user: AuthenticatedUser, proposal_id: UUID) -> ChatResponseSchema:
        completed = self._sql_service.approve_and_execute(
            principal=to_sql_principal(user), proposal_id=proposal_id
        )
        sql_metadata = SQLMetadata(
            enabled=True,
            proposal_id=str(completed.proposal.id),
            proposal_status=completed.proposal.status.value,
            normalized_sql=completed.validated.sql,
            referenced_tables=list(completed.validated.referenced_tables),
            row_count=completed.result.row_count,
            truncated=completed.result.truncated,
            catalog_version=completed.proposal.catalog_version,
            policy_version=completed.proposal.policy_version,
        )
        return ChatResponseSchema(
            status="completed",
            answer=completed.answer,
            sources=[],
            confidence=1.0,
            metadata=ResponseMetadata(
                route="sql",
                hyde=HyDEMetadata(enabled=False),
                reranking=RerankingMetadata(enabled=False, backend="none"),
                crag=CRAGMetadata(enabled=False),
                self_reflection=SelfReflectionMetadata(enabled=False),
                retrieved_chunks=[],
                sql=sql_metadata,
            ),
        )

    def reject(self, user: AuthenticatedUser, proposal_id: UUID) -> None:
        self._sql_service.reject(principal=to_sql_principal(user), proposal_id=proposal_id)
