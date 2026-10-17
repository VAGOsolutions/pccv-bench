"""Stable random-seed derivation for reproducible injection experiments."""

from __future__ import annotations

import json
from hashlib import sha256


def stable_seed(*components: int | str) -> int:
    """Return a process-independent NumPy seed for experiment components.

    Python's built-in :func:`hash` is intentionally salted for every
    interpreter process. It must not be used to derive persisted experiment
    randomness because identical commands would then produce different data.
    This canonical JSON encoding and SHA-256 digest are stable across Python
    processes and platforms for the integer and string experiment identifiers
    used by the providers.
    """
    encoded = json.dumps(components, ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(sha256(encoded.encode("utf-8")).digest()[:8], "big")
