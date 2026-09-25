"""Native compiler surface for the bounded ordered-attack transition.

It reuses the restricted AST emitter from the proven damage compiler without
changing that compiler's public attack surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from azelficoast.research.mechanics.native_damage_compiler import (
    ATTACK_KERNEL_FUNCTIONS,
    KERNEL_FUNCTIONS,
    NativeBuild,
    NativeKernelCompileError,
    _Emitter,
    _build_shared_library,
    _emit_native_c,
    _selected_functions,
)

ORDERED_ATTACK_KERNEL_FUNCTIONS = (
    "_attacker_acts_first_numeric",
    "ordered_attack_transition_numeric",
)


def emit_ordered_attack_c(
    damage_source: str,
    attack_source: str,
    ordered_source: str,
    *,
    context_width: int = 29,
) -> str:
    if context_width <= 0:
        raise NativeKernelCompileError("context width must be positive")

    names = set(KERNEL_FUNCTIONS + ATTACK_KERNEL_FUNCTIONS + ORDERED_ATTACK_KERNEL_FUNCTIONS)
    emitter = _Emitter(names)
    functions = (
        *_selected_functions(damage_source, KERNEL_FUNCTIONS),
        *_selected_functions(attack_source, ATTACK_KERNEL_FUNCTIONS),
        *_selected_functions(ordered_source, ORDERED_ATTACK_KERNEL_FUNCTIONS),
    )
    body = "".join(emitter.function(function) for function in functions)

    return _emit_native_c(body, f"""int32_t az_ordered_attack_one(
    const int32_t *params,
    int32_t order_tie_roll,
    int32_t accuracy_roll,
    int32_t damage_roll,
    int32_t secondary_roll
) {{
    return (int32_t)ordered_attack_transition_numeric(
        params, order_tie_roll, accuracy_roll, damage_roll, secondary_roll
    );
}}

void az_ordered_attack_batch(
    const int32_t *params,
    const int32_t *order_tie_rolls,
    const int32_t *accuracy_rolls,
    const int32_t *damage_rolls,
    const int32_t *secondary_rolls,
    int32_t *out,
    int64_t count
) {{
    for (int64_t index = 0; index < count; ++index) {{
        out[index] = (int32_t)ordered_attack_transition_numeric(
            params + index * {context_width},
            order_tie_rolls[index],
            accuracy_rolls[index],
            damage_rolls[index],
            secondary_rolls[index]
        );
    }}
}}
""")


def build_ordered_attack_library(
    damage_source_path: str | Path,
    attack_source_path: str | Path,
    ordered_source_path: str | Path,
    output_path: str | Path,
    *,
    context_width: int = 29,
    cc: str = "cc",
    extra_cflags: Iterable[str] = (),
) -> NativeBuild:
    damage_source_path = Path(damage_source_path)
    attack_source_path = Path(attack_source_path)
    ordered_source_path = Path(ordered_source_path)
    output_path = Path(output_path)
    c_source = emit_ordered_attack_c(
        damage_source_path.read_text(encoding="utf-8"),
        attack_source_path.read_text(encoding="utf-8"),
        ordered_source_path.read_text(encoding="utf-8"),
        context_width=context_width,
    )

    return _build_shared_library(
        c_source,
        output_path,
        source_name="ordered_attack_kernel.c",
        cc=cc,
        extra_cflags=extra_cflags,
    )
