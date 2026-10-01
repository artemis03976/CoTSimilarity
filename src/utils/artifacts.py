"""Content fingerprints for cached inputs and experiment outputs."""

import hashlib
import json
from pathlib import Path


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def response_hash(response: str) -> str:
    return hashlib.sha256(response.encode("utf-8")).hexdigest()


def file_identity(path) -> dict:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


