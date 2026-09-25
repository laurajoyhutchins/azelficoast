from __future__ import annotations

from azelficoast.belief.status_move_prior import compare_item_samples


def _sample(*, offset: int, matched: int, weights: dict[str, float]) -> dict[str, object]:
    return {
        "schema": "azelficoast.showdown-world-sample",
        "schema_version": 1,
        "requested_species": "articuno",
        "observed_moves": ["roost"],
        "showdown_commit": "pinned",
        "seed_offset": offset,
        "rounds": 8192,
        "matched": matched,
        "item_weights": weights,
    }


def test_compare_item_samples_accepts_stable_close_prior() -> None:
    result = compare_item_samples(
        _sample(offset=0, matched=100, weights={"Leftovers": 0.6, "Heavy-Duty Boots": 0.4}),
        _sample(offset=32768, matched=120, weights={"Leftovers": 0.55, "Heavy-Duty Boots": 0.45}),
        minimum_matches=32,
        maximum_total_variation=0.1,
    )

    assert result["support_stable"] is True
    assert result["item_support_size"] == 2
    assert abs(result["total_variation"] - 0.05) < 1e-12
    assert result["passed"] is True


def test_compare_item_samples_rejects_unstable_support() -> None:
    result = compare_item_samples(
        _sample(offset=0, matched=100, weights={"Leftovers": 1.0}),
        _sample(offset=32768, matched=100, weights={"Leftovers": 0.9, "Rocky Helmet": 0.1}),
        minimum_matches=32,
        maximum_total_variation=0.1,
    )

    assert result["support_stable"] is False
    assert result["passed"] is False


def test_compare_item_samples_rejects_insufficient_matches() -> None:
    result = compare_item_samples(
        _sample(offset=0, matched=31, weights={"Leftovers": 1.0}),
        _sample(offset=32768, matched=100, weights={"Leftovers": 1.0}),
        minimum_matches=32,
        maximum_total_variation=0.1,
    )

    assert result["sufficient_matches"] is False
    assert result["passed"] is False
