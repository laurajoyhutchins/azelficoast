from __future__ import annotations

from types import SimpleNamespace

from azelficoast.live.opponents import dirty_tricks_move


def _move(move_id: str, *, base_power: int = 0, priority: int = 0):
    return SimpleNamespace(id=move_id, base_power=base_power, priority=priority)


def _battle(
    moves,
    *,
    opponent_boosts=None,
    opponent_hp: float = 1.0,
    opponent_status=None,
):
    return SimpleNamespace(
        active_pokemon=SimpleNamespace(),
        opponent_active_pokemon=SimpleNamespace(
            boosts=opponent_boosts or {},
            current_hp_fraction=opponent_hp,
            status=opponent_status,
        ),
        available_moves=list(moves),
    )


def test_dirty_player_punishes_setup_before_generic_damage() -> None:
    battle = _battle(
        [_move("earthquake", base_power=100), _move("haze")],
        opponent_boosts={"atk": 2},
    )
    assert dirty_tricks_move(battle).id == "haze"


def test_dirty_player_uses_priority_to_finish_low_hp_target() -> None:
    battle = _battle(
        [
            _move("earthquake", base_power=100),
            _move("suckerpunch", base_power=70, priority=1),
        ],
        opponent_hp=0.2,
        opponent_status="PAR",
    )
    assert dirty_tricks_move(battle).id == "suckerpunch"


def test_dirty_player_prefers_denial_when_no_higher_priority_dirty_line_exists() -> None:
    battle = _battle(
        [_move("earthquake", base_power=100), _move("knockoff", base_power=65)],
        opponent_status="PAR",
    )
    assert dirty_tricks_move(battle).id == "knockoff"


def test_dirty_player_delegates_when_no_disruptive_move_is_available() -> None:
    battle = _battle([_move("earthquake", base_power=100)], opponent_status="PAR")
    assert dirty_tricks_move(battle) is None
