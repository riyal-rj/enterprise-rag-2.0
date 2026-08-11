import pytest

from app.eval.invokers import ServiceInvoker, SkippedIntent
from app.eval.profiles import PROFILES
from app.eval.schemas import Intent


def test_service_invoker_raises_skipped_intent_when_not_wired() -> None:
    invoker = ServiceInvoker()

    with pytest.raises(SkippedIntent):
        invoker.invoke("What is a pod?", PROFILES["naive"], Intent.RAG)
