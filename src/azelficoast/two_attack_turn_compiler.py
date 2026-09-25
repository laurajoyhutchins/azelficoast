"""Native compiler surface for the bounded two-attack turn."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from azelficoast.native_damage_compiler import (
    KERNEL_FUNCTIONS,
    NativeBuild,
    NativeKernelCompileError,
    _Emitter,
    _build_shared_library,
    _emit_native_c,
    _selected_functions,
)

TWO_ATTACK_TURN_FUNCTIONS = (
    "_turn_stage_stat_numeric",
    "_turn_damage_numeric",
    "_turn_p1_first_numeric",
    "two_attack_turn_numeric",
)


def emit_two_attack_turn_c(
    damage_source: str,
    turn_source: str,
    *,
    context_width: int = 53,
) -> str:
    if context_width <= 0:
        raise NativeKernelCompileError("context width must be positive")

    names = set(KERNEL_FUNCTIONS + TWO_ATTACK_TURN_FUNCTIONS)
    emitter = _Emitter(names)
    functions = (
        *_selected_functions(damage_source, KERNEL_FUNCTIONS),
        *_selected_functions(turn_source, TWO_ATTACK_TURN_FUNCTIONS),
    )
    body = "".join(emitter.function(function) for function in functions)

    return _emit_native_c(body, f"""int32_t az_two_attack_turn_one(
    const int32_t *params,
    int32_t order_tie_roll,
    int32_t p1_accuracy_roll,
    int32_t p1_damage_roll,
    int32_t p1_secondary_roll,
    int32_t p2_accuracy_roll,
    int32_t p2_damage_roll
) {{
    return (int32_t)two_attack_turn_numeric(
        params,
        order_tie_roll,
        p1_accuracy_roll,
        p1_damage_roll,
        p1_secondary_roll,
        p2_accuracy_roll,
        p2_damage_roll
    );
}}

void az_two_attack_turn_batch(
    const int32_t *params,
    const int32_t *order_tie_rolls,
    const int32_t *p1_accuracy_rolls,
    const int32_t *p1_damage_rolls,
    const int32_t *p1_secondary_rolls,
    const int32_t *p2_accuracy_rolls,
    const int32_t *p2_damage_rolls,
    int32_t *out,
    int64_t count
) {{
    for (int64_t index = 0; index < count; ++index) {{
        out[index] = (int32_t)two_attack_turn_numeric(
            params + index * {context_width},
            order_tie_rolls[index],
            p1_accuracy_rolls[index],
            p1_damage_rolls[index],
            p1_secondary_rolls[index],
            p2_accuracy_rolls[index],
            p2_damage_rolls[index]
        );
    }}
}}
""")


def build_two_attack_turn_library(
    damage_source_path: str | Path,
    turn_source_path: str | Path,
    output_path: str | Path,
    *,
    context_width: int = 53,
    cc: str = "cc",
    extra_cflags: Iterable[str] = (),
) -> NativeBuild:
    damage_source_path = Path(damage_source_path)
    turn_source_path = Path(turn_source_path)
    output_path = Path(output_path)
    c_source = emit_two_attack_turn_c(
        damage_source_path.read_text(encoding="utf-8"),
        turn_source_path.read_text(encoding="utf-8"),
        context_width=context_width,
    )

    return _build_shared_library(
        c_source,
        output_path,
        source_name="two_attack_turn_kernel.c",
        cc=cc,
        extra_cflags=extra_cflags,
    )
