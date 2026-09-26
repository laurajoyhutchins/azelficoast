from __future__ import annotations

from azelficoast.core.memo import MemoKey, SemanticMemo


def test_memo_groups_keep_physical_alternatives_separate() -> None:
    memo = SemanticMemo[str](max_entries=4)
    group = "sha256:logical"
    python_key = MemoKey(group, "python-evaluation-frontier")
    compiled_key = MemoKey(group, "compiled-topology")

    assert memo.get(python_key) is None
    memo.put(python_key, "frontier")
    memo.put(compiled_key, "topology")

    assert memo.get(python_key) == "frontier"
    assert memo.get(compiled_key) == "topology"
    assert memo.alternatives(group) == (
        "python-evaluation-frontier",
        "compiled-topology",
    )
    assert memo.stats.hits == 2
    assert memo.stats.misses == 1


def test_memo_is_bounded_lru_not_unbounded_history() -> None:
    memo = SemanticMemo[int](max_entries=2)
    first = MemoKey("g1", "python")
    second = MemoKey("g2", "python")
    third = MemoKey("g3", "python")

    memo.put(first, 1)
    memo.put(second, 2)
    assert memo.get(first) == 1
    memo.put(third, 3)

    assert memo.get(first) == 1
    assert memo.get(second) is None
    assert memo.get(third) == 3
    assert memo.stats.evictions == 1
    assert memo.stats.entries == 2


def test_memo_rejects_empty_authority_identity() -> None:
    try:
        MemoKey("", "python")
    except ValueError as error:
        assert "group identity" in str(error)
    else:
        raise AssertionError("empty memo group identity was accepted")
