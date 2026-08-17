from __future__ import annotations

import hashlib


def sampled_in(subject: str, percentage: int, *, salt: str) -> bool:

    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    if percentage == 0:
        return False
    if percentage == 100:
        return True

    normalized = " ".join(subject.split()).casefold()
    digest = hashlib.sha256(f"{salt}\x00{normalized}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 10_000
    return bucket < percentage * 100
