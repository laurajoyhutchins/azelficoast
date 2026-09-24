from __future__ import annotations

import ctypes
import shutil
from pathlib import Path

import pytest

import azelficoast.gen9_attack as gen9_attack
import azelficoast.gen9_damage as gen9_damage
from azelficoast.gen9_attack import (
    AttackTransitionContext,
    attack_transition,
    compile_attack_context,
)
from azelficoast.gen9_damage import DamageContext, compile_numeric_context, damage
from azelficoast.native_damage_compiler import (
    NativeKernelCompileError,
    build_attack_library,
    build_damage_library,
    emit_attack_c,
    emit_damage_c,
)


def _context() -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=80,
        category="Special",
        move_id="aurasphere",
        move_type="Fighting",
        attacker_types=("Fighting", "Steel"),
        tera_type=None,
        attacker_base_stat=115,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=110,
        defender_base_stat=95,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item="Choice Specs",
        type_mod=1,
    )


def _damage_source() -> str:
    source_path = Path(gen9_damage.__file__)
    return source_path.read_text(encoding="utf-8")


def test_emitter_is_narrow_and_exports_damage_wrappers() -> None:
    emitted = emit_damage_c(_damage_source())

    assert "static int64_t damage_numeric(" in emitted
    assert "static int64_t _kernel_modify(" in emitted
    assert "int32_t az_damage_one(" in emitted
    assert "void az_damage_batch(" in emitted
    assert "int64_t az_weighted_damage_score(" in emitted
    assert "az_floor_div" in emitted


def test_emitter_rejects_unknown_calls() -> None:
    hostile = _damage_source().replace(
        "def _kernel_modify(value: int, modifier: int) -> int:\n"
        "    return (value * modifier + 2047) // 4096",
        "def _kernel_modify(value: int, modifier: int) -> int:\n"
        "    return abs(value)",
        1,
    )

    with pytest.raises(NativeKernelCompileError, match="unsupported call 'abs'"):
        emit_damage_c(hostile)


def test_emitter_rejects_unsupported_control_flow() -> None:
    hostile = _damage_source().replace(
        "    attacker_level = params[0]\n",
        "    while roll > 0:\n        roll -= 1\n    attacker_level = params[0]\n",
        1,
    )

    with pytest.raises(NativeKernelCompileError, match="unsupported statement While"):
        emit_damage_c(hostile)


@pytest.mark.skipif(shutil.which("cc") is None, reason="system C compiler unavailable")
def test_compiled_damage_matches_python_numeric_kernel(tmp_path: Path) -> None:
    source_path = Path(gen9_damage.__file__)
    library_path = tmp_path / "damage.so"
    build_damage_library(source_path, library_path)

    library = ctypes.CDLL(str(library_path))
    library.az_damage_one.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    library.az_damage_one.restype = ctypes.c_int32

    context = _context()
    numeric = compile_numeric_context(context)
    params = (ctypes.c_int32 * len(numeric))(*numeric)

    assert library.az_damage_one(params, 7) == damage(context, 7)



def test_attack_emitter_contains_whole_transition_wrapper() -> None:
    emitted = emit_attack_c(
        _damage_source(),
        Path(gen9_attack.__file__).read_text(encoding="utf-8"),
    )

    assert "static int64_t attack_transition_numeric(" in emitted
    assert "int32_t az_attack_one(" in emitted
    assert "void az_attack_batch(" in emitted


@pytest.mark.skipif(shutil.which("cc") is None, reason="system C compiler unavailable")
def test_compiled_attack_matches_python_whole_transition(tmp_path: Path) -> None:
    library_path = tmp_path / "attack.so"
    build_attack_library(
        Path(gen9_damage.__file__),
        Path(gen9_attack.__file__),
        library_path,
    )

    library = ctypes.CDLL(str(library_path))
    library.az_attack_one.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.az_attack_one.restype = ctypes.c_int32

    attack_context = AttackTransitionContext(
        damage=_context(),
        accuracy=80,
        attacker_hp=341,
        attacker_max_hp=341,
        defender_hp=400,
        move_pp=8,
    )
    numeric = compile_attack_context(attack_context)
    params = (ctypes.c_int32 * len(numeric))(*numeric)

    expected = attack_transition(attack_context, 79, 7).packed
    assert library.az_attack_one(params, 79, 7) == expected
