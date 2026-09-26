"""Bounded memo groups for exact reusable semantic materializations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class MemoKey:
    """One physical materialization inside a logical memo group."""

    group_identity: str
    alternative_identity: str

    def __post_init__(self) -> None:
        if not self.group_identity:
            raise ValueError("memo group identity must be non-empty")
        if not self.alternative_identity:
            raise ValueError("memo alternative identity must be non-empty")


@dataclass(frozen=True, slots=True)
class MemoStats:
    hits: int
    misses: int
    writes: int
    evictions: int
    entries: int


class SemanticMemo(Generic[T]):
    """Small explicit LRU for exact semantic materializations.

    Callers own identity construction. The memo never decides that two computations are
    equivalent; it only reuses a value after the caller presents the exact same key.
    """

    def __init__(self, *, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("memo max_entries must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[MemoKey, T] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0

    def get(self, key: MemoKey) -> T | None:
        value = self._entries.get(key)
        if value is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: MemoKey, value: T) -> None:
        if key in self._entries:
            self._entries[key] = value
            self._entries.move_to_end(key)
        else:
            self._entries[key] = value
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
        self._writes += 1

    def alternatives(self, group_identity: str) -> tuple[str, ...]:
        if not group_identity:
            raise ValueError("memo group identity must be non-empty")
        return tuple(
            key.alternative_identity
            for key in self._entries
            if key.group_identity == group_identity
        )

    @property
    def stats(self) -> MemoStats:
        return MemoStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            evictions=self._evictions,
            entries=len(self._entries),
        )

    def clear(self) -> None:
        self._entries.clear()
