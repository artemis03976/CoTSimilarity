"""UTF-8 JSON/JSONL readers and atomic writers."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected an object")
            except ValueError as exc:
                raise ValueError(f"Invalid JSONL in {path}, line {line_number}: {exc}") from exc
            yield record


@contextmanager
def atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yield stream
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value) -> None:
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, records) -> None:
    with atomic_text(path) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def read_manifest(path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
