from __future__ import annotations

import hashlib
import json

import pytest

from azelficoast.belief.showdown_packing import (
    ShowdownPackingError,
    ShowdownVocabulary,
    pack_joint_posterior,
)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vocabulary_record() -> dict[str, object]:
    material: dict[str, object] = {
        "schema": "azelficoast.showdown-vocabulary",
        "schema_version": 1,
        "showdown_commit": "a" * 40,
        "generation": 9,
        "identity": {},
        "species": [
            {
                "id": "rotom",
                "num": 479,
                "forme_index": 0,
                "base_species_id": "rotom",
                "forme": "",
            },
            {
                "id": "rotomwash",
                "num": 479,
                "forme_index": 1,
                "base_species_id": "rotom",
                "forme": "wash",
            },
        ],
        "moves": [
            {"id": "hydropump", "num": 56},
            {"id": "voltswitch", "num": 521},
            {"id": "willowisp", "num": 261},
        ],
        "items": [
            {"id": "choicescarf", "num": 287},
            {"id": "leftovers", "num": 234},
        ],
        "abilities": [{"id": "levitate", "num": 26}],
        "types": [{"id": "water", "index": 1}],
        "natures": [{"id": "timid", "index": 1}],
        "roles": [{"id": "fastpivot", "index": 1}],
    }
    return {**material, "vocabulary_sha256": _digest(material)}


def _member(*, item: str, moves: list[str]) -> dict[str, object]:
    return {
        "species": "Rotom-Wash",
        "level": 80,
        "gender": "N",
        "ability": "Levitate",
        "item": item,
        "moves": moves,
        "tera_type": "Water",
        "role": "Fast Pivot",
        "nature": "Timid",
        "evs": {"hp": 84, "spa": 84, "spe": 84},
        "ivs": {},
        "was_lead": True,
    }


def _posterior() -> dict[str, object]:
    return {
        "schema": "azelficoast.joint-random-battle-posterior",
        "schema_version": 1,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "support_status": "sufficient",
        "showdown_commit": "a" * 40,
        "construction": {
            "kind": "full-team-generator-rejection-particles",
            "posterior_treatment": "generator_faithful_joint_empirical",
            "preserves_joint_team_set_correlations": True,
        },
        "worlds": [
            {
                "world_id": "scarf",
                "weight": 0.4,
                "hidden": {
                    "team": [
                        _member(
                            item="Choice Scarf",
                            moves=["Hydro Pump", "Volt Switch"],
                        )
                    ]
                },
            },
            {
                "world_id": "leftovers",
                "weight": 0.6,
                "hidden": {
                    "team": [
                        _member(
                            item="Leftovers",
                            moves=["Hydro Pump", "Will-O-Wisp"],
                        )
                    ]
                },
            },
        ],
        "public_evidence": {
            "opponent_team_size": 1,
            "revealed": [{"species": "Rotom-Wash", "was_lead": True}],
        },
    }


def test_pack_reuses_showdown_native_integer_ids() -> None:
    vocabulary = ShowdownVocabulary.from_record(_vocabulary_record())
    packed = pack_joint_posterior(_posterior(), vocabulary)

    assert packed.world_ids == ("scarf", "leftovers")
    assert packed.weights == (0.4, 0.6)
    assert packed.species_num == ((479,), (479,))
    assert packed.species_forme == ((1,), (1,))
    assert packed.ability_num == ((26,), (26,))
    assert packed.item_num == ((287,), (234,))
    assert packed.move_num == (
        ((56, 521, 0, 0),),
        ((56, 261, 0, 0),),
    )
    assert packed.move_mask == (
        ((True, True, False, False),),
        ((True, True, False, False),),
    )
    assert packed.evs[0][0] == (84, 0, 0, 84, 0, 84)
    assert packed.ivs[0][0] == (31, 31, 31, 31, 31, 31)


def test_pack_fails_closed_on_unknown_semantic_identity() -> None:
    vocabulary = ShowdownVocabulary.from_record(_vocabulary_record())
    posterior = _posterior()
    worlds = posterior["worlds"]
    assert isinstance(worlds, list)
    first = worlds[0]
    assert isinstance(first, dict)
    hidden = first["hidden"]
    assert isinstance(hidden, dict)
    team = hidden["team"]
    assert isinstance(team, list)
    member = team[0]
    assert isinstance(member, dict)
    member["moves"] = ["Earthquake"]

    with pytest.raises(ShowdownPackingError, match="unknown Showdown move"):
        pack_joint_posterior(posterior, vocabulary)


def test_vocabulary_digest_rejects_manual_drift() -> None:
    record = _vocabulary_record()
    moves = record["moves"]
    assert isinstance(moves, list)
    first = moves[0]
    assert isinstance(first, dict)
    first["num"] = 999

    with pytest.raises(ShowdownPackingError, match="content digest mismatch"):
        ShowdownVocabulary.from_record(record)


def test_numpy_materialization_has_accelerator_friendly_shapes() -> None:
    np = pytest.importorskip("numpy")
    vocabulary = ShowdownVocabulary.from_record(_vocabulary_record())
    arrays = pack_joint_posterior(_posterior(), vocabulary).as_numpy()

    assert arrays["species_num"].shape == (2, 1)
    assert arrays["move_num"].shape == (2, 1, 4)
    assert arrays["evs"].shape == (2, 1, 6)
    assert arrays["weights"].dtype == np.float32
    assert arrays["move_num"].dtype == np.int16
