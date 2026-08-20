"""Deterministic text normalization, ahead of any scanner.

Pure function, no I/O, no ``llm_guard`` dependency - trivially unit-testable
and always runs, regardless of whether ML scanners are available in this
deployment (see ``app.guardrails.factory``).

Strips the same class of invisible-character tricks
``app.rag_services.crag.web_retriever.TavilyRegulatoryWebRetriever`` already
strips from *web* output, but applied here to untrusted *user input*: a
prompt-injection payload hidden behind zero-width joiners or bidi override
characters would otherwise reach the ML scanners (and the LLM) unnormalized.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

# Zero-width and bidi-control characters, built from explicit code points
# (never literal glyphs in source - those are unreviewable in a diff) -
# same character classes TavilyRegulatoryWebRetriever's own control-
# character regex strips from web output, applied here to input instead:
# zero-width space/joiners/LRM/RLM, bidi embedding/override, word joiner,
# bidi isolates, and the BOM / zero-width no-break space.
_INVISIBLE_CODEPOINTS = (
    list(range(0x200B, 0x2010))  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    + list(range(0x202A, 0x202F))  # LRE, RLE, PDF, LRO, RLO
    + [0x2060]  # word joiner
    + list(range(0x2066, 0x206A))  # bidi isolates
    + [0xFEFF]  # BOM / zero-width no-break space
)
_INVISIBLE_CHARS = re.compile("[" + "".join(chr(cp) for cp in _INVISIBLE_CODEPOINTS) + "]")

# C0/C1 control characters other than the whitespace we intentionally keep
# (tab/newline), same discipline as the web retriever's stripping.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# A long-enough run of base64 alphabet to be worth decoding and rescanning -
# short runs are common in ordinary text (acronyms, hex-ish tokens) and would
# just add noise.
# No trailing \b: "=" is a non-word character, so a boundary assertion
# right after base64 padding at end-of-string never matches - greedy
# matching of the optional "={0,2}" already bounds the match correctly.
_BASE64_CANDIDATE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}")
_HEX_CANDIDATE = re.compile(r"\b(?:[0-9a-fA-F]{2}){12,}\b")


def canonicalize(text: str, *, max_decode_depth: int = 2) -> str:
    """Normalize ``text`` for scanning: fold Unicode confusables via NFKC,
    strip invisible/control characters, collapse whitespace, and append
    (not replace - the original text must still be scanned/echoed) any
    base64/hex-decoded payloads found inside it, up to ``max_decode_depth``
    layers, so a smuggled instruction like an encoded
    "ignore previous instructions" is visible to the scanners that run
    after this.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE_CHARS.sub("", normalized)
    normalized = _CONTROL_CHARS.sub("", normalized)
    normalized = _collapse_whitespace(normalized)

    decoded_layers = _decode_smuggled_payloads(normalized, max_depth=max_decode_depth)
    if decoded_layers:
        normalized = normalized + "\n" + "\n".join(decoded_layers)
    return normalized


def _collapse_whitespace(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def _decode_smuggled_payloads(text: str, *, max_depth: int) -> list[str]:
    decoded: list[str] = []
    _decode_layer(text, depth=0, max_depth=max_depth, out=decoded)
    return decoded


def _decode_layer(text: str, *, depth: int, max_depth: int, out: list[str]) -> None:
    """Decode one layer of base64/hex payloads found in ``text``.

    Recursion is deliberately independent of :func:`_looks_meaningful`:
    a doubly-encoded payload's *intermediate* layer is itself just a
    base64 blob (no spaces, so not "meaningful" on its own) and would
    otherwise terminate the descent before the real instruction inside it
    was ever surfaced. Meaningfulness gates only what gets appended for
    scanning, never whether to keep descending.
    """
    if depth >= max_depth:
        return
    for match in _BASE64_CANDIDATE.finditer(text):
        candidate = match.group(0)
        try:
            raw = base64.b64decode(candidate, validate=True)
            decoded_text = raw.decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if _looks_meaningful(decoded_text):
            out.append(decoded_text)
        _decode_layer(decoded_text, depth=depth + 1, max_depth=max_depth, out=out)
    for match in _HEX_CANDIDATE.finditer(text):
        candidate = match.group(0)
        try:
            raw = bytes.fromhex(candidate)
            decoded_text = raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _looks_meaningful(decoded_text):
            out.append(decoded_text)
        _decode_layer(decoded_text, depth=depth + 1, max_depth=max_depth, out=out)


def _looks_meaningful(decoded_text: str) -> bool:
    """Cheap filter against decoding noise (binary garbage that happens to
    be valid UTF-8) - require mostly printable characters and at least one
    space-separated word, so a random base64-looking product code doesn't
    get treated as a smuggled instruction."""
    if not decoded_text.strip():
        return False
    printable = sum(1 for ch in decoded_text if ch.isprintable() or ch in "\n\t")
    if printable / len(decoded_text) < 0.9:
        return False
    return " " in decoded_text.strip()
