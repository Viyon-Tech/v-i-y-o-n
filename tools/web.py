"""Web search and URL fetching for research, with a session-lifetime memory cache.

`search` uses Brave (``BRAVE_API_KEY``) or Serper (``SERPER_API_KEY``); `fetch`
returns cleaned page text; `fetch_pdf` extracts PDF text (via pypdf if present).
All network I/O runs in threads (stdlib urllib — no extra deps) and results are
cached in-process for the session. With no API key, `search` returns ``[]`` and
logs a warning so PULSE can degrade gracefully.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
import urllib.request

from core import config

logger = logging.getLogger("viyon.web")

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
SERPER_URL = "https://google.serper.dev/search"
_USER_AGENT = "VIYON/0.1 (+research agent)"

# Session-lifetime cache: key -> result.
_CACHE: dict = {}


def clear_cache() -> None:
    """Drop all cached search/fetch results (e.g. at the start of a session)."""
    _CACHE.clear()


# -- HTTP (run in threads) ---------------------------------------------------

def _http_get(url: str, headers: dict | None = None, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_post(url: str, body: bytes, headers: dict, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# -- public API --------------------------------------------------------------

async def search(query: str, count: int = 5) -> list[dict]:
    """Search the web; return ``[{title, url, description}]`` (cached)."""
    cache_key = f"search::{query}::{count}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    config.load_env()
    brave = os.getenv("BRAVE_API_KEY")
    serper = os.getenv("SERPER_API_KEY")
    results: list[dict] = []

    try:
        if brave:
            qs = urllib.parse.urlencode({"q": query, "count": count})
            raw = await asyncio.to_thread(
                _http_get,
                f"{BRAVE_URL}?{qs}",
                {"X-Subscription-Token": brave, "Accept": "application/json"},
            )
            data = json.loads(raw)
            for r in (data.get("web", {}).get("results") or [])[:count]:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "description": re.sub(r"<[^>]+>", "", r.get("description", "")),
                    }
                )
        elif serper:
            body = json.dumps({"q": query, "num": count}).encode()
            raw = await asyncio.to_thread(
                _http_post,
                SERPER_URL,
                body,
                {"X-API-KEY": serper, "Content-Type": "application/json"},
            )
            data = json.loads(raw)
            for r in (data.get("organic") or [])[:count]:
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "description": r.get("snippet", ""),
                    }
                )
        else:
            logger.warning("No BRAVE_API_KEY / SERPER_API_KEY set — web search unavailable.")
    except Exception as exc:
        logger.warning("Web search failed for %r: %s", query, exc)

    _CACHE[cache_key] = results
    return results


async def fetch(url: str) -> str:
    """Fetch a URL and return cleaned, whitespace-collapsed text (cached)."""
    cache_key = f"fetch::{url}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    text = ""
    try:
        raw = await asyncio.to_thread(_http_get, url, {"User-Agent": _USER_AGENT})
        text = _html_to_text(raw.decode(errors="replace"))
    except Exception as exc:
        logger.warning("fetch failed for %s: %s", url, exc)
    _CACHE[cache_key] = text
    return text


async def fetch_pdf(url: str) -> str:
    """Download a PDF and return its extracted text (cached)."""
    cache_key = f"pdf::{url}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    text = ""
    try:
        raw = await asyncio.to_thread(_http_get, url, {"User-Agent": _USER_AGENT})
        text = await asyncio.to_thread(_pdf_to_text, raw)
    except Exception as exc:
        logger.warning("fetch_pdf failed for %s: %s", url, exc)
    _CACHE[cache_key] = text
    return text


# -- helpers -----------------------------------------------------------------

_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(html: str) -> str:
    """Strip scripts/styles/tags and collapse whitespace."""
    html = _SCRIPT_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pdf_to_text(raw: bytes) -> str:
    """Extract text from PDF bytes; degrade if pypdf isn't installed."""
    try:
        import io

        import pypdf
    except ImportError:
        return "(PDF text extraction unavailable: install pypdf)"
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:
        return f"(could not parse PDF: {exc})"
