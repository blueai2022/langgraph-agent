"""Define, validate, and load aliases over corpus codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._source import load_records, optional_string
from .icd10_corpus import Corpus


class AliasError(Exception):
    """Raised when an alias record is invalid or cannot be loaded."""


class UnknownAliasTargetError(AliasError):
    """Raised when an alias targets an unknown code."""


@dataclass(frozen=True)
class Alias:
    """A code alias with a short form and an expanded name."""

    short_form: str
    target_code: str
    expanded_name: str

    @property
    def display_text(self) -> str:
        """Return the display text for the alias."""
        return f"{self.expanded_name} - {self.short_form}"


@dataclass(frozen=True)
class SearchableAlias:
    """An alias projected as a searchable entry."""

    match_text: str
    target_code: str
    display_text: str

    @classmethod
    def from_alias(cls, alias: Alias) -> "SearchableAlias":
        return cls(
            match_text=alias.short_form,
            target_code=alias.target_code,
            display_text=alias.display_text,
        )


def load_aliases(source: str | Path) -> list[Alias]:
    """Load alias records from a source."""
    return [
        _mapping_to_alias(mapping, index)
        for index, mapping in enumerate(load_records(source, "alias"), start=1)
    ]


def validate_aliases(aliases: Iterable[Alias], corpus: Corpus) -> None:
    """Validate that all aliases target known codes."""
    for alias in aliases:
        if corpus.lookup(alias.target_code) is None:
            raise UnknownAliasTargetError(
                f"alias '{alias.short_form}' targets unknown code '{alias.target_code}'"
            )


def _mapping_to_alias(mapping: Mapping[str, Any], index: int) -> Alias:
    short_form = optional_string(mapping.get("acronym"))
    target_code = optional_string(mapping.get("code"))
    expanded_name = optional_string(mapping.get("name"))
    missing = [
        name
        for name, value in (
            ("acronym", short_form),
            ("code", target_code),
            ("name", expanded_name),
        )
        if not value
    ]
    if missing:
        raise AliasError(f"record {index}: missing required field(s) {', '.join(missing)}")
    assert short_form and target_code and expanded_name
    return Alias(short_form=short_form, target_code=target_code, expanded_name=expanded_name)
