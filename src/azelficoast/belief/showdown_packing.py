"""Dense, revision-bound packing for joint Random Battle posteriors.

The semantic source remains the validated posterior plus pinned Pokémon Showdown.
This module only lowers already-authorized categorical data into compact integer
coordinates suitable for NumPy/JAX batches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from azelficoast.belief.joint_posterior import validate_joint_posterior

VOCABULARY_SCHEMA = "azelficoast.showdown-vocabulary"
VOCABULARY_SCHEMA_VERSION = 1
PACK_SCHEMA = "azelficoast.packed-joint-posterior"
PACK_SCHEMA_VERSION = 1

_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_GENDERS = {"": 0, "M": 1, "F": 2, "N": 3}


class ShowdownPackingError(ValueError):
    """Raised when source-bound categorical data cannot be packed losslessly."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _to_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShowdownPackingError(f"{label} must be an object")
    return value


def _integer_table(rows: Any, *, label: str, field: str) -> dict[str, int]:
    if not isinstance(rows, list) or not rows:
        raise ShowdownPackingError(f"vocabulary {label} must be a non-empty list")
    result: dict[str, int] = {}
    used: dict[int, str] = {}
    for row in rows:
        entry = _require_mapping(row, f"vocabulary {label} entry")
        key = _to_id(entry.get("id"))
        value = entry.get(field)
        if not key or not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ShowdownPackingError(f"invalid {label} entry {entry!r}")
        previous = result.setdefault(key, value)
        if previous != value:
            raise ShowdownPackingError(f"duplicate {label} id {key!r}")
        other = used.setdefault(value, key)
        if other != key:
            raise ShowdownPackingError(
                f"{label} native integer {value} aliases {other!r} and {key!r}"
            )
    return result


def _derived_table(rows: Any, *, label: str) -> dict[str, int]:
    if not isinstance(rows, list) or not rows:
        raise ShowdownPackingError(f"vocabulary {label} must be a non-empty list")
    result: dict[str, int] = {}
    used: dict[int, str] = {}
    for row in rows:
        entry = _require_mapping(row, f"vocabulary {label} entry")
        key = _to_id(entry.get("id"))
        index = entry.get("index")
        if not key or not isinstance(index, int) or isinstance(index, bool) or index <= 0:
            raise ShowdownPackingError(f"invalid {label} entry {entry!r}")
        previous = result.setdefault(key, index)
        if previous != index:
            raise ShowdownPackingError(f"duplicate {label} id {key!r}")
        other = used.setdefault(index, key)
        if other != key:
            raise ShowdownPackingError(
                f"{label} index {index} aliases {other!r} and {key!r}"
            )
    return result


@dataclass(frozen=True)
class ShowdownVocabulary:
    """Validated categorical coordinates derived from one exact Showdown revision."""

    showdown_commit: str
    vocabulary_sha256: str
    species: Mapping[str, tuple[int, int]]
    moves: Mapping[str, int]
    items: Mapping[str, int]
    abilities: Mapping[str, int]
    types: Mapping[str, int]
    natures: Mapping[str, int]
    roles: Mapping[str, int]

    @classmethod
    def from_record(cls, document: Mapping[str, Any]) -> "ShowdownVocabulary":
        if (
            document.get("schema") != VOCABULARY_SCHEMA
            or document.get("schema_version") != VOCABULARY_SCHEMA_VERSION
        ):
            raise ShowdownPackingError("unexpected Showdown vocabulary schema")
        if document.get("generation") != 9:
            raise ShowdownPackingError("Showdown vocabulary must target generation 9")
        showdown_commit = document.get("showdown_commit")
        if not (
            isinstance(showdown_commit, str)
            and len(showdown_commit) == 40
            and all(ch in "0123456789abcdef" for ch in showdown_commit)
        ):
            raise ShowdownPackingError("Showdown vocabulary has an invalid revision")

        expected_digest = document.get("vocabulary_sha256")
        if not isinstance(expected_digest, str):
            raise ShowdownPackingError("Showdown vocabulary lacks a content digest")
        material = dict(document)
        material.pop("vocabulary_sha256", None)
        actual_digest = _sha256(material)
        if expected_digest != actual_digest:
            raise ShowdownPackingError("Showdown vocabulary content digest mismatch")

        raw_species = document.get("species")
        if not isinstance(raw_species, list) or not raw_species:
            raise ShowdownPackingError("vocabulary species must be a non-empty list")
        species: dict[str, tuple[int, int]] = {}
        for row in raw_species:
            entry = _require_mapping(row, "vocabulary species entry")
            key = _to_id(entry.get("id"))
            num = entry.get("num")
            forme_index = entry.get("forme_index")
            if (
                not key
                or not isinstance(num, int)
                or isinstance(num, bool)
                or num <= 0
                or not isinstance(forme_index, int)
                or isinstance(forme_index, bool)
                or forme_index < 0
            ):
                raise ShowdownPackingError(f"invalid species entry {entry!r}")
            if key in species:
                raise ShowdownPackingError(f"duplicate species id {key!r}")
            species[key] = (num, forme_index)

        return cls(
            showdown_commit=showdown_commit,
            vocabulary_sha256=expected_digest,
            species=species,
            moves=_integer_table(document.get("moves"), label="moves", field="num"),
            items=_integer_table(document.get("items"), label="items", field="num"),
            abilities=_integer_table(
                document.get("abilities"), label="abilities", field="num"
            ),
            types=_derived_table(document.get("types"), label="types"),
            natures=_derived_table(document.get("natures"), label="natures"),
            roles=_derived_table(document.get("roles"), label="roles"),
        )

    def _lookup(self, table: Mapping[str, int], value: Any, *, label: str) -> int:
        key = _to_id(value)
        try:
            return int(table[key])
        except KeyError as error:
            raise ShowdownPackingError(f"unknown Showdown {label} {value!r}") from error

    def species_id(self, value: Any) -> tuple[int, int]:
        key = _to_id(value)
        try:
            return self.species[key]
        except KeyError as error:
            raise ShowdownPackingError(f"unknown Showdown species {value!r}") from error


@dataclass(frozen=True)
class PackedJointPosterior:
    """Pure-Python dense pack that can be copied into NumPy/JAX without reparsing strings."""

    source_digest: str
    showdown_commit: str
    vocabulary_sha256: str
    world_ids: tuple[str, ...]
    weights: tuple[float, ...]
    species_num: tuple[tuple[int, ...], ...]
    species_forme: tuple[tuple[int, ...], ...]
    level: tuple[tuple[int, ...], ...]
    ability_num: tuple[tuple[int, ...], ...]
    item_num: tuple[tuple[int, ...], ...]
    move_num: tuple[tuple[tuple[int, ...], ...], ...]
    move_mask: tuple[tuple[tuple[bool, ...], ...], ...]
    tera_type: tuple[tuple[int, ...], ...]
    nature: tuple[tuple[int, ...], ...]
    role: tuple[tuple[int, ...], ...]
    gender: tuple[tuple[int, ...], ...]
    evs: tuple[tuple[tuple[int, ...], ...], ...]
    ivs: tuple[tuple[tuple[int, ...], ...], ...]
    was_lead: tuple[tuple[bool, ...], ...]

    @property
    def world_count(self) -> int:
        return len(self.world_ids)

    @property
    def team_size(self) -> int:
        return len(self.species_num[0]) if self.species_num else 0

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PACK_SCHEMA,
            "schema_version": PACK_SCHEMA_VERSION,
            "source_digest": self.source_digest,
            "showdown_commit": self.showdown_commit,
            "vocabulary_sha256": self.vocabulary_sha256,
            "world_count": self.world_count,
            "team_size": self.team_size,
            "max_moves": 4,
            "claim": (
                "This is a deterministic dense lowering of one validated joint posterior; "
                "it adds no mechanics or posterior authority."
            ),
        }

    def as_numpy(self) -> dict[str, Any]:
        """Materialize fixed-dtype arrays suitable for direct JAX device transfer."""

        try:
            import numpy as np
        except ImportError as error:
            raise ShowdownPackingError("NumPy is required to materialize packed arrays") from error

        return {
            "weights": np.asarray(self.weights, dtype=np.float32),
            "species_num": np.asarray(self.species_num, dtype=np.int32),
            "species_forme": np.asarray(self.species_forme, dtype=np.int16),
            "level": np.asarray(self.level, dtype=np.int16),
            "ability_num": np.asarray(self.ability_num, dtype=np.int16),
            "item_num": np.asarray(self.item_num, dtype=np.int16),
            "move_num": np.asarray(self.move_num, dtype=np.int16),
            "move_mask": np.asarray(self.move_mask, dtype=np.bool_),
            "tera_type": np.asarray(self.tera_type, dtype=np.int8),
            "nature": np.asarray(self.nature, dtype=np.int8),
            "role": np.asarray(self.role, dtype=np.int16),
            "gender": np.asarray(self.gender, dtype=np.int8),
            "evs": np.asarray(self.evs, dtype=np.int16),
            "ivs": np.asarray(self.ivs, dtype=np.int8),
            "was_lead": np.asarray(self.was_lead, dtype=np.bool_),
        }


def _stats(value: Any, *, defaults: Mapping[str, int], label: str) -> tuple[int, ...]:
    row = _require_mapping(value, label)
    output: list[int] = []
    for stat in _STATS:
        raw = row.get(stat, defaults[stat])
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ShowdownPackingError(f"{label}.{stat} must be an integer")
        output.append(raw)
    return tuple(output)


def pack_joint_posterior(
    document: Mapping[str, Any],
    vocabulary: ShowdownVocabulary,
    *,
    require_sufficient_support: bool = True,
) -> PackedJointPosterior:
    """Pack a validated posterior while preserving every joint team particle."""

    checked = validate_joint_posterior(
        document,
        require_sufficient_support=require_sufficient_support,
    )
    posterior_commit = checked.get("showdown_commit")
    if posterior_commit is not None and posterior_commit != vocabulary.showdown_commit:
        raise ShowdownPackingError("posterior and vocabulary use different Showdown revisions")

    world_ids: list[str] = []
    weights: list[float] = []
    species_num: list[tuple[int, ...]] = []
    species_forme: list[tuple[int, ...]] = []
    level: list[tuple[int, ...]] = []
    ability_num: list[tuple[int, ...]] = []
    item_num: list[tuple[int, ...]] = []
    move_num: list[tuple[tuple[int, ...], ...]] = []
    move_mask: list[tuple[tuple[bool, ...], ...]] = []
    tera_type: list[tuple[int, ...]] = []
    nature: list[tuple[int, ...]] = []
    role: list[tuple[int, ...]] = []
    gender: list[tuple[int, ...]] = []
    evs: list[tuple[tuple[int, ...], ...]] = []
    ivs: list[tuple[tuple[int, ...], ...]] = []
    was_lead: list[tuple[bool, ...]] = []

    for world in checked["worlds"]:
        team = world["hidden"]["team"]
        species_row: list[int] = []
        forme_row: list[int] = []
        level_row: list[int] = []
        ability_row: list[int] = []
        item_row: list[int] = []
        moves_row: list[tuple[int, ...]] = []
        move_mask_row: list[tuple[bool, ...]] = []
        tera_row: list[int] = []
        nature_row: list[int] = []
        role_row: list[int] = []
        gender_row: list[int] = []
        evs_row: list[tuple[int, ...]] = []
        ivs_row: list[tuple[int, ...]] = []
        lead_row: list[bool] = []

        for member in team:
            native_species, forme_index = vocabulary.species_id(member["species"])
            raw_moves = list(member["moves"])
            if len(raw_moves) > 4:
                raise ShowdownPackingError(
                    f"{world['world_id']}/{member['species']}: more than four moves"
                )
            encoded_moves = [
                vocabulary._lookup(vocabulary.moves, move, label="move")
                for move in raw_moves
            ]
            moves_row.append(tuple(encoded_moves + [0] * (4 - len(encoded_moves))))
            move_mask_row.append(
                tuple([True] * len(encoded_moves) + [False] * (4 - len(encoded_moves)))
            )

            raw_gender = str(member["gender"])
            if raw_gender not in _GENDERS:
                raise ShowdownPackingError(f"unknown Showdown gender {raw_gender!r}")

            species_row.append(native_species)
            forme_row.append(forme_index)
            raw_level = member["level"]
            if not isinstance(raw_level, int) or isinstance(raw_level, bool):
                raise ShowdownPackingError("posterior level must be an integer")
            level_row.append(raw_level)
            ability_row.append(
                vocabulary._lookup(vocabulary.abilities, member["ability"], label="ability")
            )
            item_row.append(
                vocabulary._lookup(vocabulary.items, member["item"], label="item")
            )
            tera_row.append(
                vocabulary._lookup(vocabulary.types, member["tera_type"], label="type")
            )
            nature_row.append(
                vocabulary._lookup(vocabulary.natures, member["nature"], label="nature")
            )
            role_row.append(
                vocabulary._lookup(vocabulary.roles, member["role"], label="role")
            )
            gender_row.append(_GENDERS[raw_gender])
            evs_row.append(
                _stats(
                    member["evs"],
                    defaults={stat: 0 for stat in _STATS},
                    label="evs",
                )
            )
            ivs_row.append(
                _stats(
                    member["ivs"],
                    defaults={stat: 31 for stat in _STATS},
                    label="ivs",
                )
            )
            lead_row.append(member["was_lead"] is True)

        world_ids.append(str(world["world_id"]))
        weights.append(float(world["weight"]))
        species_num.append(tuple(species_row))
        species_forme.append(tuple(forme_row))
        level.append(tuple(level_row))
        ability_num.append(tuple(ability_row))
        item_num.append(tuple(item_row))
        move_num.append(tuple(moves_row))
        move_mask.append(tuple(move_mask_row))
        tera_type.append(tuple(tera_row))
        nature.append(tuple(nature_row))
        role.append(tuple(role_row))
        gender.append(tuple(gender_row))
        evs.append(tuple(evs_row))
        ivs.append(tuple(ivs_row))
        was_lead.append(tuple(lead_row))

    return PackedJointPosterior(
        source_digest=_sha256(checked),
        showdown_commit=vocabulary.showdown_commit,
        vocabulary_sha256=vocabulary.vocabulary_sha256,
        world_ids=tuple(world_ids),
        weights=tuple(weights),
        species_num=tuple(species_num),
        species_forme=tuple(species_forme),
        level=tuple(level),
        ability_num=tuple(ability_num),
        item_num=tuple(item_num),
        move_num=tuple(move_num),
        move_mask=tuple(move_mask),
        tera_type=tuple(tera_type),
        nature=tuple(nature),
        role=tuple(role),
        gender=tuple(gender),
        evs=tuple(evs),
        ivs=tuple(ivs),
        was_lead=tuple(was_lead),
    )
