"""Tiny clean-room compiler for Azelficoast's numeric mechanics kernels.

This is deliberately not a general Python compiler. It accepts only the AST forms
used by explicitly selected mechanics functions and emits C99 with 64-bit
intermediates plus Python-compatible floor division. Unsupported syntax is rejected
rather than guessed.
"""

from __future__ import annotations

import ast
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KERNEL_FUNCTIONS = (
    "_kernel_modify",
    "_kernel_ordinary_stat",
    "_kernel_type_effectiveness",
    "damage_numeric",
)
ATTACK_KERNEL_FUNCTIONS = ("attack_transition_numeric",)


class NativeKernelCompileError(ValueError):
    """Raised when the restricted mechanics source leaves the supported subset."""


@dataclass(frozen=True)
class NativeBuild:
    library: Path
    c_source: str


def _annotation_is_array(annotation: ast.expr | None) -> bool:
    if not isinstance(annotation, ast.Subscript):
        return False
    return isinstance(annotation.value, ast.Name) and annotation.value.id in {"tuple", "list"}


class _Emitter:
    def __init__(self, allowed_calls: set[str]) -> None:
        self.allowed_calls = allowed_calls
        self.declared: set[str] = set()
        self.indent = 0

    def _line(self, text: str) -> str:
        return "    " * self.indent + text + "\n"

    def expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name):
                raise NativeKernelCompileError("only direct array-name subscripts are supported")
            return f"{node.value.id}[{self.expr(node.slice)}]"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self.expr(node.operand)})"
        if isinstance(node, ast.BinOp):
            left = self.expr(node.left)
            right = self.expr(node.right)
            if isinstance(node.op, ast.FloorDiv):
                return f"az_floor_div({left}, {right})"
            operators = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
            }
            symbol = next(
                (value for kind, value in operators.items() if isinstance(node.op, kind)),
                None,
            )
            if symbol is None:
                raise NativeKernelCompileError(
                    f"unsupported binary operator {type(node.op).__name__}"
                )
            return f"({left} {symbol} {right})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            operators = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            symbol = next(
                (value for kind, value in operators.items() if isinstance(node.ops[0], kind)),
                None,
            )
            if symbol is None:
                raise NativeKernelCompileError(
                    f"unsupported comparison {type(node.ops[0]).__name__}"
                )
            return f"({self.expr(node.left)} {symbol} {self.expr(node.comparators[0])})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in self.allowed_calls:
                raise NativeKernelCompileError(f"unsupported call {node.func.id!r}")
            if node.keywords:
                raise NativeKernelCompileError("keyword arguments are outside the native subset")
            return f"{node.func.id}({', '.join(self.expr(arg) for arg in node.args)})"
        raise NativeKernelCompileError(f"unsupported expression {type(node).__name__}")

    def statement(self, node: ast.stmt) -> str:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                return ""
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise NativeKernelCompileError("only single-name assignment is supported")
            name = node.targets[0].id
            value = self.expr(node.value)
            if name in self.declared:
                return self._line(f"{name} = {value};")
            self.declared.add(name)
            return self._line(f"int64_t {name} = {value};")
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            operators = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.FloorDiv: None}
            if isinstance(node.op, ast.FloorDiv):
                return self._line(
                    f"{node.target.id} = az_floor_div({node.target.id}, {self.expr(node.value)});"
                )
            symbol = next(
                (value for kind, value in operators.items() if isinstance(node.op, kind)),
                None,
            )
            if symbol is None:
                raise NativeKernelCompileError(
                    f"unsupported augmented operator {type(node.op).__name__}"
                )
            return self._line(f"{node.target.id} {symbol} {self.expr(node.value)};")
        if isinstance(node, ast.If):
            result = self._line(f"if {self.expr(node.test)} {{")
            self.indent += 1
            result += "".join(self.statement(statement) for statement in node.body)
            self.indent -= 1
            if node.orelse:
                result += self._line("} else {")
                self.indent += 1
                result += "".join(self.statement(statement) for statement in node.orelse)
                self.indent -= 1
            result += self._line("}")
            return result
        if isinstance(node, ast.For):
            if not isinstance(node.target, ast.Name):
                raise NativeKernelCompileError("for-loop target must be a name")
            if (
                not isinstance(node.iter, ast.Call)
                or not isinstance(node.iter.func, ast.Name)
                or node.iter.func.id != "range"
                or len(node.iter.args) != 1
                or node.iter.keywords
            ):
                raise NativeKernelCompileError("only range(stop) loops are supported")
            stop = self.expr(node.iter.args[0])
            target = node.target.id
            result = self._line(
                f"for (int64_t {target} = 0; {target} < {stop}; ++{target}) {{"
            )
            self.indent += 1
            result += "".join(self.statement(statement) for statement in node.body)
            self.indent -= 1
            result += self._line("}")
            return result
        if isinstance(node, ast.Return):
            if node.value is None:
                raise NativeKernelCompileError("native kernel functions must return a value")
            return self._line(f"return {self.expr(node.value)};")
        raise NativeKernelCompileError(f"unsupported statement {type(node).__name__}")

    def function(self, node: ast.FunctionDef) -> str:
        if node.decorator_list:
            raise NativeKernelCompileError(f"decorators are unsupported on {node.name}")
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs:
            raise NativeKernelCompileError(f"variadic arguments are unsupported on {node.name}")
        if node.args.defaults or node.args.kw_defaults:
            raise NativeKernelCompileError(f"default arguments are unsupported on {node.name}")

        params: list[str] = []
        self.declared = set()
        for arg in node.args.args:
            self.declared.add(arg.arg)
            if _annotation_is_array(arg.annotation):
                params.append(f"const int32_t *{arg.arg}")
            else:
                params.append(f"int64_t {arg.arg}")

        result = f"static int64_t {node.name}({', '.join(params)}) {{\n"
        self.indent = 1
        result += "".join(self.statement(statement) for statement in node.body)
        self.indent = 0
        result += "}\n\n"
        return result


def _selected_functions(
    source: str,
    names: tuple[str, ...] = KERNEL_FUNCTIONS,
) -> tuple[ast.FunctionDef, ...]:
    module = ast.parse(source)
    by_name = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise NativeKernelCompileError(
            "numeric damage source is missing required functions: " + ", ".join(missing)
        )
    return tuple(by_name[name] for name in names)


def emit_damage_c(source: str, *, context_width: int = 18) -> str:
    """Emit standalone C99 for the selected numeric damage functions."""
    if context_width <= 0:
        raise NativeKernelCompileError("context width must be positive")

    functions = _selected_functions(source)
    emitter = _Emitter(set(KERNEL_FUNCTIONS))
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
int32_t az_damage_one(const int32_t *params, int32_t roll) {{
    return (int32_t)damage_numeric(params, roll);
}}

void az_damage_batch(
    const int32_t *params,
    const int32_t *rolls,
    int32_t *out,
    int64_t count
) {{
    for (int64_t index = 0; index < count; ++index) {{
        out[index] = (int32_t)damage_numeric(
            params + index * {context_width},
            rolls[index]
        );
    }}
}}

int64_t az_damage_score(
    const int32_t *params,
    const int32_t *rolls,
    int64_t count
) {{
    int64_t total = 0;
    for (int64_t index = 0; index < count; ++index) {{
        total += damage_numeric(params + index * {context_width}, rolls[index]);
    }}
    return total;
}}

int64_t az_weighted_damage_score(
    const int32_t *params,
    const int32_t *rolls,
    const int32_t *weights,
    int64_t count
) {{
    int64_t total = 0;
    for (int64_t index = 0; index < count; ++index) {{
        total += damage_numeric(params + index * {context_width}, rolls[index])
            * (int64_t)weights[index];
    }}
    return total;
}}
"""


def _link_flags() -> tuple[str, ...]:
    if platform.system() == "Darwin":
        return ("-dynamiclib",)
    if platform.system() == "Windows":
        raise NativeKernelCompileError("the research compiler currently supports Unix C toolchains")
    return ("-shared", "-fPIC")


def build_damage_library(
    source_path: str | Path,
    output_path: str | Path,
    *,
    context_width: int = 18,
    cc: str = "cc",
    extra_cflags: Iterable[str] = (),
) -> NativeBuild:
    """Compile the selected Python kernel functions into one native shared library."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    c_source = emit_damage_c(
        source_path.read_text(encoding="utf-8"),
        context_width=context_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="azelficoast-native-") as temp_dir:
        c_path = Path(temp_dir) / "damage_kernel.c"
        c_path.write_text(c_source, encoding="utf-8")
        command = [
            cc,
            "-std=c99",
            "-O3",
            *extra_cflags,
            *_link_flags(),
            str(c_path),
            "-o",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise NativeKernelCompileError(
                "C compiler failed:\n"
                + completed.stderr
                + ("\n" + completed.stdout if completed.stdout else "")
            )

    return NativeBuild(library=output_path, c_source=c_source)


def emit_attack_c(
    damage_source: str,
    attack_source: str,
    *,
    context_width: int = 23,
) -> str:
    """Emit standalone C99 for damage plus one whole-attack transition."""
    if context_width <= 0:
        raise NativeKernelCompileError("context width must be positive")

    names = set(KERNEL_FUNCTIONS + ATTACK_KERNEL_FUNCTIONS)
    emitter = _Emitter(names)
    damage_functions = _selected_functions(damage_source, KERNEL_FUNCTIONS)
    attack_functions = _selected_functions(attack_source, ATTACK_KERNEL_FUNCTIONS)
    body = "".join(emitter.function(function) for function in (*damage_functions, *attack_functions))

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
int32_t az_attack_one(
    const int32_t *params,
    int32_t accuracy_roll,
    int32_t damage_roll
) {{
    return (int32_t)attack_transition_numeric(params, accuracy_roll, damage_roll);
}}

void az_attack_batch(
    const int32_t *params,
    const int32_t *accuracy_rolls,
    const int32_t *damage_rolls,
    int32_t *out,
    int64_t count
) {{
    for (int64_t index = 0; index < count; ++index) {{
        out[index] = (int32_t)attack_transition_numeric(
            params + index * {context_width},
            accuracy_rolls[index],
            damage_rolls[index]
        );
    }}
}}
"""


def build_attack_library(
    damage_source_path: str | Path,
    attack_source_path: str | Path,
    output_path: str | Path,
    *,
    context_width: int = 23,
    cc: str = "cc",
    extra_cflags: Iterable[str] = (),
) -> NativeBuild:
    """Compile damage plus the whole-attack transition into one shared library."""
    damage_source_path = Path(damage_source_path)
    attack_source_path = Path(attack_source_path)
    output_path = Path(output_path)
    c_source = emit_attack_c(
        damage_source_path.read_text(encoding="utf-8"),
        attack_source_path.read_text(encoding="utf-8"),
        context_width=context_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="azelficoast-native-") as temp_dir:
        c_path = Path(temp_dir) / "attack_kernel.c"
        c_path.write_text(c_source, encoding="utf-8")
        command = [
            cc,
            "-std=c99",
            "-O3",
            *extra_cflags,
            *_link_flags(),
            str(c_path),
            "-o",
            str(output_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise NativeKernelCompileError(
                "C compiler failed:\n"
                + completed.stderr
                + ("\n" + completed.stdout if completed.stdout else "")
            )

    return NativeBuild(library=output_path, c_source=c_source)
