"""Thin adapter around the Tavily web-search API."""

from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from app.core.config import get_settings


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a web search via Tavily.

    Raises:
        ValueError: if no Tavily API key is configured.
    """
    api_key = get_settings().external_apis.tavily_api_key.get_secret_value()
    if not api_key:
        raise ValueError("Tavily API key is not configured")

    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    return response.get("results", [])
