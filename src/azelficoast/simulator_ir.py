"""Reference dependency IR for a future compiled Pokémon battle simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping, Sequence


class StateField(IntEnum):
    OWN_HP = 0
    OPPONENT_HP = 1
    OWN_SPEED = 2
    OPPONENT_SPEED = 3
    OWN_ITEM = 4
    OPPONENT_ITEM = 5
    OPPONENT_MOVE = 6
    OWN_PROTECTED = 7
    WEATHER = 8
    OWN_STATUS = 9
    OPPONENT_STATUS = 10
    BENCH_SIGNATURE = 11


class RandomField(IntEnum):
    DAMAGE_ROLL = 0


class EffectOp(IntEnum):
    PROTECT_BLOCK = 1
    SPECIAL_DAMAGE = 2


class Observation(IntEnum):
    NONE = 0
    BLOCKED = 1
    DAMAGE = 2


ITEM_NONE = 0
ITEM_CHOICE_SCARF = 1
ITEM_CHOICE_SPECS = 2

MOVE_MOONBLAST = 1


class DependencyViolation(ValueError):
    """Raised when declared dependencies do not explain observed behavior."""


def field_mask(*fields: StateField) -> int:
    mask = 0
    for field in fields:
        mask |= 1 << int(field)
    return mask


def random_mask(*fields: RandomField) -> int:
    mask = 0
    for field in fields:
        mask |= 1 << int(field)
    return mask


def _members(mask: int, enum_type: type[IntEnum]) -> tuple[IntEnum, ...]:
    return tuple(field for field in enum_type if mask & (1 << int(field)))


@dataclass(frozen=True)
class World:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(StateField):
            raise ValueError(f"expected {len(StateField)} state fields, got {len(self.values)}")

    @classmethod
    def from_values(cls, values: Mapping[StateField, int]) -> World:
        packed = [0] * len(StateField)
        for field, value in values.items():
            packed[int(field)] = int(value)
        return cls(tuple(packed))

    def get(self, field: StateField) -> int:
        return self.values[int(field)]

    def project(self, mask: int) -> tuple[int, ...]:
        return tuple(self.get(field) for field in _members(mask, StateField))

    def apply(self, delta: TransitionDelta) -> World:
        packed = list(self.values)
        for field, value in delta.writes:
            packed[int(field)] = value
        return World(tuple(packed))


@dataclass(frozen=True)
class RandomInput:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(RandomField):
            raise ValueError(
                f"expected {len(RandomField)} random fields, got {len(self.values)}"
            )

    @classmethod
    def from_values(cls, values: Mapping[RandomField, int]) -> RandomInput:
        packed = [0] * len(RandomField)
        for field, value in values.items():
            packed[int(field)] = int(value)
        return cls(tuple(packed))

    def get(self, field: RandomField) -> int:
        return self.values[int(field)]

    def project(self, mask: int) -> tuple[int, ...]:
        return tuple(self.get(field) for field in _members(mask, RandomField))


@dataclass(frozen=True)
class EffectSpec:
    name: str
    op: EffectOp
    reads: int
    writes: int
    random: int


@dataclass(frozen=True)
class TransitionDelta:
    writes: tuple[tuple[StateField, int], ...]
    observation: Observation


@dataclass(frozen=True)
class CollapsedClass:
    dependency_key: tuple[int, ...]
    representative_index: int
    member_indices: tuple[int, ...]
    delta: TransitionDelta


@dataclass(frozen=True)
class CollapsedBatch:
    effect: EffectSpec
    world_count: int
    classes: tuple[CollapsedClass, ...]

    @property
    def transition_count(self) -> int:
        return len(self.classes)

    @property
    def reduction_factor(self) -> float:
        if not self.classes:
            return 1.0
        return self.world_count / len(self.classes)


PROTECT_BLOCK = EffectSpec(
    name="protect-block",
    op=EffectOp.PROTECT_BLOCK,
    reads=field_mask(StateField.OPPONENT_MOVE),
    writes=field_mask(StateField.OWN_PROTECTED),
    random=0,
)

SPECIAL_DAMAGE = EffectSpec(
    name="choice-special-damage",
    op=EffectOp.SPECIAL_DAMAGE,
    reads=field_mask(StateField.OWN_HP, StateField.OPPONENT_ITEM),
    writes=field_mask(StateField.OWN_HP),
    random=random_mask(RandomField.DAMAGE_ROLL),
)


def transition(spec: EffectSpec, world: World, random: RandomInput) -> TransitionDelta:
    if spec.op is EffectOp.PROTECT_BLOCK:
        delta = TransitionDelta(
            writes=((StateField.OWN_PROTECTED, 1),),
            observation=Observation.BLOCKED,
        )
    elif spec.op is EffectOp.SPECIAL_DAMAGE:
        roll = random.get(RandomField.DAMAGE_ROLL)
        if not 0 <= roll <= 15:
            raise ValueError(f"damage roll must be in [0, 15], got {roll}")
        base_damage = 90 if world.get(StateField.OPPONENT_ITEM) == ITEM_CHOICE_SPECS else 60
        next_hp = max(0, world.get(StateField.OWN_HP) - base_damage - roll)
        delta = TransitionDelta(
            writes=((StateField.OWN_HP, next_hp),),
            observation=Observation.DAMAGE,
        )
    else:
        raise ValueError(f"unsupported effect op {spec.op!r}")

    actual = field_mask(*(field for field, _ in delta.writes))
    if actual & ~spec.writes:
        raise DependencyViolation(
            f"{spec.name} wrote undeclared fields: actual={actual:#x} declared={spec.writes:#x}"
        )
    return delta


def _group_indices(
    worlds: Sequence[World],
    reads: int,
) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, world in enumerate(worlds):
        groups.setdefault(world.project(reads), []).append(index)
    return groups


def verify_dependency_corpus(
    spec: EffectSpec,
    worlds: Sequence[World],
    random_inputs: Sequence[RandomInput],
) -> None:
    if not random_inputs:
        raise ValueError("dependency verification requires at least one random input")

    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[TransitionDelta]] = {}
    for world in worlds:
        for random in random_inputs:
            key = (world.project(spec.reads), random.project(spec.random))
            groups.setdefault(key, []).append(transition(spec, world, random))

    for key, deltas in groups.items():
        unique = set(deltas)
        if len(unique) != 1:
            raise DependencyViolation(
                f"{spec.name} produced {len(unique)} deltas for one declared dependency class "
                f"{key!r}; a dependency is missing"
            )


def verify_declared_dependencies(
    spec: EffectSpec,
    worlds: Sequence[World],
    random: RandomInput,
) -> None:
    verify_dependency_corpus(spec, worlds, (random,))


def collapse_worlds(
    spec: EffectSpec,
    worlds: Sequence[World],
    random: RandomInput,
) -> CollapsedBatch:
    verify_declared_dependencies(spec, worlds, random)
    classes = []
    for key, indices in _group_indices(worlds, spec.reads).items():
        representative = indices[0]
        classes.append(
            CollapsedClass(
                dependency_key=key,
                representative_index=representative,
                member_indices=tuple(indices),
                delta=transition(spec, worlds[representative], random),
            )
        )
    return CollapsedBatch(
        effect=spec,
        world_count=len(worlds),
        classes=tuple(classes),
    )


def expand_collapsed(
    worlds: Sequence[World],
    batch: CollapsedBatch,
) -> tuple[World, ...]:
    expanded: list[World | None] = [None] * len(worlds)
    for group in batch.classes:
        for index in group.member_indices:
            expanded[index] = worlds[index].apply(group.delta)
    if any(world is None for world in expanded):
        raise DependencyViolation("collapsed batch did not cover every input world")
    return tuple(world for world in expanded if world is not None)


def execute_direct(
    spec: EffectSpec,
    worlds: Iterable[World],
    random: RandomInput,
) -> tuple[World, ...]:
    return tuple(world.apply(transition(spec, world, random)) for world in worlds)
