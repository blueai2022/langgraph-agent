"""Internal helpers for reading JSON/CSV record sources."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional


class SourceError(Exception):
    """Raised when a JSON/CSV source cannot be read or is malformed."""


def optional_string(value: Any) -> Optional[str]:
    """Coerce a value to a trimmed string or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_records(source: str | Path, kind: str) -> list[dict[str, Any]]:
    """Read a list of record mappings from a JSON or CSV file."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(text, kind)
    if suffix == ".csv":
        return _load_csv_records(text, kind)
    raise SourceError(
        f"unsupported {kind} source file extension {suffix!r}"
    )


def _load_json_records(text: str, kind: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceError(f"invalid JSON in {kind} file: {exc}") from exc
    if not isinstance(data, list):
        raise SourceError(
			f"{kind} JSON must contain list of records, got {type(data).__name__}"
		)
    records: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise SourceError(
                f"{kind} JSON source must contain objects, got {type(item).__name__}"
            )
        records.append(item)
    return records


def _load_csv_records(text: str, kind: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise SourceError(f"{kind} CSV source does not have a header row")
    return list(reader)