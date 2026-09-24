from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from azelficoast.gen9_damage import DamageContext, damage
from azelficoast.jax_gen9_damage import contexts_to_array, damage_batch


def _contexts() -> tuple[DamageContext, ...]:
    return (
        DamageContext(
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
            burned=False,
        ),
        DamageContext(
            attacker_level=100,
            defender_level=100,
            base_power=120,
            category="Physical",
            move_id="closecombat",
            move_type="Fighting",
            attacker_types=("Fighting", "Steel"),
            tera_type="Fighting",
            attacker_base_stat=110,
            attacker_iv=31,
            attacker_ev=252,
            attacker_nature_percent=110,
            defender_base_stat=80,
            defender_iv=31,
            defender_ev=252,
            defender_nature_percent=110,
            attacker_item="",
            type_mod=1,
            burned=True,
        ),
    )


def test_jax_real_damage_kernel_matches_scalar_all_rolls() -> None:
    contexts = _contexts()
    expanded = tuple(context for context in contexts for _ in range(16))
    rolls = np.tile(np.arange(16, dtype=np.int32), len(contexts))

    actual = np.asarray(
        damage_batch(contexts_to_array(expanded), jnp.asarray(rolls)),
        dtype=np.int32,
    )
    expected = np.asarray(
        [damage(context, roll) for context in contexts for roll in range(16)],
        dtype=np.int32,
    )
    assert np.array_equal(actual, expected)
