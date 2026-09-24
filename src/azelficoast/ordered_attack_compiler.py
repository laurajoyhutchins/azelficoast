"""Native compiler surface for the bounded ordered-attack transition.

It reuses the restricted AST emitter from the proven damage compiler without
changing that compiler's public attack surface.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from azelficoast.native_damage_compiler import (
    ATTACK_KERNEL_FUNCTIONS,
    KERNEL_FUNCTIONS,
    NativeBuild,
    NativeKernelCompileError,
    _Emitter,
    _selected_functions,
)

ORDERED_ATTACK_KERNEL_FUNCTIONS = (
    "_attacker_acts_first_numeric",
    "ordered_attack_transition_numeric",
)


def _link_flags() -> tuple[str, ...]:
    if platform.system() == "Darwin":
        return ("-dynamiclib",)
    if platform.system() == "Windows":
        raise NativeKernelCompileError("the research compiler currently supports Unix C toolchains")
    return ("-shared", "-fPIC")


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

    return f"""#include <stdint.h>
#include <stddef.h>

static int64_t az_floor_div(int64_t a, int64_t b) {{
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) {{
        q -= 1;
    }}
    return q;
}}

{body}
int32_t az_ordered_attack_one(
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
"""


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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="azelficoast-native-") as temp_dir:
        c_path = Path(temp_dir) / "ordered_attack_kernel.c"
        c_path.write_text(c_source, encoding="utf-8")
        command = [
            cc, "-std=c99", "-O3", *extra_cflags, *_link_flags(),
            str(c_path), "-o", str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise NativeKernelCompileError(
                "C compiler failed:\n"
                + completed.stderr
                + ("\n" + completed.stdout if completed.stdout else "")
            )
    return NativeBuild(library=output_path, c_source=c_source)
