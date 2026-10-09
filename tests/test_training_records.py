from __future__ import annotations

import json

import pytest

from azelficoast.live.corpus import build_fixtures
from azelficoast.live.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from azelficoast.research.matched_comparison import freeze_packet
from azelficoast.research.training_records import TrainingRecordError, build_training_records


def _record(
    *,
    run: str = "run",
    event: int,
    kind: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run,
        "event_index": event,
        "observed_at": "2026-09-25T00:00:00+00:00",
        "kind": kind,
        **extra,
    }


def _state(battle_tag: str, turn: int = 4) -> dict[str, object]:
    return {
        "battle_tag": battle_tag,
        "turn": turn,
        "player": "azelficoast",
        "opponent": "opponent",
        "active": {"species": "Azelf"},
        "opponent_active": {"species": "Rotom-Wash", "item": None},
        "team": {},
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "available_moves": ["psychic", "thunderbolt"],
        "available_switches": [],
        "legal_actions": ["/choose move psychic", "/choose move thunderbolt"],
        "force_switch": False,
        "trapped": False,
        "can_tera": False,
    }


def _plan() -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 2,
        "posterior_treatments": ["generator_faithful"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 4096,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1],
        "confirmatory_predictors": ["entropy", "branch_count"],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "evaluator": {
            "schema": "azelficoast.belief-policy-value-evaluator",
            "schema_version": 1,
            "checkpoint_digest": "sha256:" + "a" * 64,
            "observability": "public_belief_only",
            "architecture": "weighted_deep_sets_policy_value",
            "spec": {"hidden_width": 256},
        },
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def _posterior() -> dict[str, object]:
    return {
        "treatment": "generator_faithful",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {
                "world_id": "band",
                "weight": 0.6,
                "hidden": {"opponent.active.item": "Choice Band"},
            },
            {
                "world_id": "scarf",
                "weight": 0.4,
                "hidden": {"opponent.active.item": "Choice Scarf"},
            },
        ],
    }


def _search_evidence(
    fixture,
    battle_tag: str,
    *,
    run_id: str | None = None,
    posterior: dict[str, object] | None = None,
    info_values: dict[str, float] | None = None,
):
    posterior = posterior or _posterior()
    info_values = info_values or {
        "/choose move psychic": 0.9,
        "/choose move thunderbolt": 0.2,
    }
    state = {
        "fixture_id": fixture.fixture_id,
        "battle_tag": battle_tag,
        **({"run_id": run_id} if run_id is not None else {}),
        "public_state": dict(fixture.state),
        "legal_actions": list(fixture.legal_actions),
        "predictors": {"entropy": 1.0, "branch_count": 2},
    }
    packet = freeze_packet(
        plan=_plan(),
        state=state,
        posterior=posterior,
        posterior_treatment="generator_faithful",
        depth=1,
    )

    def receipt(method: str, values: dict[str, float]) -> dict[str, object]:
        best = max(values.values())
        chosen = min(action for action, value in values.items() if value == best)
        return {
            "schema": "azelficoast.matched-search-receipt",
            "schema_version": 4,
            "packet_digest": packet["packet_digest"],
            "matched_spec_digest": packet["matched_spec_digest"],
            "method": method,
            "input_digest": packet["input_digest"],
            "posterior_digest": packet["posterior_digest"],
            "posterior_semantic_digest": packet["posterior_semantic_digest"],
            "evaluator_digest": packet["evaluator_digest"],
            "evaluator_checkpoint_digest": packet["evaluator"]["checkpoint_digest"],
            "mechanics_identity_digest": packet["mechanics_identity_digest"],
            "mechanics_evidence_digest": "e" * 64,
            "showdown_commit": packet["showdown_commit"],
            "chance_treatment": packet["chance_treatment"],
            "transition_oracle_digest": "f" * 64,
            "budget_unit_definition": "one transition_evaluation per verified whole-turn execution class consumed",
            "evaluator_call_unit_definition": "one learned value prediction per successor public information set",
            "evaluator_calls": 7,
            "transition_program_digest": "program-sha256",
            "transition_artifact_digest": "f" * 64,
            "transition_program_source": "verified-transition-program",
            "compute_budget": packet["compute_budget"],
            "consumed": 16,
            "chosen_action": chosen,
            "root_values": values,
        }

    det_values = {
        "/choose move psychic": 0.8,
        "/choose move thunderbolt": 0.7,
    }
    return (
        packet,
        receipt("determinization", det_values),
        receipt("information_set", info_values),
        posterior,
    )


def _write_json(path, document: object) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _write_trace(path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_evidence(tmp_path, evidence_rows):
    packet_paths = []
    receipt_paths = []
    posterior_paths = []
    seen_posteriors = set()
    for index, (packet, det, info, posterior) in enumerate(evidence_rows):
        packet_path = tmp_path / f"packet-{index}.json"
        det_path = tmp_path / f"det-{index}.json"
        info_path = tmp_path / f"info-{index}.json"
        _write_json(packet_path, packet)
        _write_json(det_path, det)
        _write_json(info_path, info)
        packet_paths.append(packet_path)
        receipt_paths.extend((det_path, info_path))

        encoded = json.dumps(posterior, sort_keys=True)
        if encoded not in seen_posteriors:
            posterior_path = tmp_path / f"posterior-{len(posterior_paths)}.json"
            _write_json(posterior_path, posterior)
            posterior_paths.append(posterior_path)
            seen_posteriors.add(encoded)
    return packet_paths, receipt_paths, posterior_paths


def test_training_record_is_joined_from_real_outcome_and_settled_search(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                event=0,
                kind="protocol",
                room="battle-one",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                event=1,
                kind="decision",
                battle_tag="battle-one",
                decision_index=0,
                state=_state("battle-one"),
                chosen_action="/choose move thunderbolt",
            ),
            _record(
                event=2,
                kind="terminal",
                battle_tag="battle-one",
                won=False,
                lost=True,
                tied=False,
                final_state={},
            ),
        ],
    )
    [fixture] = build_fixtures([trace])
    paths = _write_evidence(
        tmp_path,
        [_search_evidence(fixture, "battle-one")],
    )

    records, summary = build_training_records(
        [trace],
        search_packet_paths=paths[0],
        search_receipt_paths=paths[1],
        posterior_paths=paths[2],
    )

    [record] = records
    assert record["targets"]["value"] == {
        "eventual_battle_outcome": -1.0,
        "public_belief_search_return": 0.9,
    }
    assert record["targets"]["policy"]["selected_action"] == "/choose move psychic"
    assert record["provenance"]["behavior_action"] == "/choose move thunderbolt"
    assert record["provenance"]["behavior_matches_policy_target"] is False
    assert record["provenance"]["search"]["kind"] == (
        "settled-matched-information-set-search"
    )
    assert record["provenance"]["search"]["evaluator_checkpoint_digest"] == (
        "sha256:" + "a" * 64
    )
    assert record["input"]["posterior"] == _posterior()
    assert summary["search_target_source"] == (
        "settled-matched-information-set-search"
    )


def test_same_battle_never_crosses_splits(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                event=0,
                kind="protocol",
                room="battle-one",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                event=1,
                kind="decision",
                battle_tag="battle-one",
                decision_index=0,
                state=_state("battle-one", 4),
                chosen_action="/choose move psychic",
            ),
            _record(
                event=2,
                kind="protocol",
                room="battle-one",
                protocol_index=1,
                messages=[["", "turn", "5"]],
            ),
            _record(
                event=3,
                kind="decision",
                battle_tag="battle-one",
                decision_index=1,
                state=_state("battle-one", 5),
                chosen_action="/choose move psychic",
            ),
            _record(
                event=4,
                kind="terminal",
                battle_tag="battle-one",
                won=True,
                lost=False,
                tied=False,
                final_state={},
            ),
        ],
    )
    fixtures = build_fixtures([trace])
    evidence = []
    for fixture in fixtures:
        battle_tag = fixture.control_decisions[0]["battle_tag"]
        evidence.append(_search_evidence(fixture, battle_tag))
    paths = _write_evidence(tmp_path, evidence)

    records, _ = build_training_records(
        [trace],
        search_packet_paths=paths[0],
        search_receipt_paths=paths[1],
        posterior_paths=paths[2],
    )

    assert len(records) == 2
    assert len({row["battle_id"] for row in records}) == 1
    assert len({row["split"] for row in records}) == 1
    assert len({row["split_group_id"] for row in records}) == 1


def test_repeated_fixture_across_battles_is_grouped_into_one_split(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = []
    for run, battle_tag, offset, won in (
        ("run-a", "battle-a", 0, True),
        ("run-b", "battle-b", 10, False),
    ):
        rows.extend(
            [
                _record(
                    run=run,
                    event=offset,
                    kind="protocol",
                    room=battle_tag,
                    protocol_index=0,
                    messages=[["", "turn", "4"]],
                ),
                _record(
                    run=run,
                    event=offset + 1,
                    kind="decision",
                    battle_tag=battle_tag,
                    decision_index=0,
                    state=_state(battle_tag),
                    chosen_action="/choose move psychic",
                ),
                _record(
                    run=run,
                    event=offset + 2,
                    kind="terminal",
                    battle_tag=battle_tag,
                    won=won,
                    lost=not won,
                    tied=False,
                    final_state={},
                ),
            ]
        )
    _write_trace(trace, rows)
    [fixture] = build_fixtures([trace])
    evidence = [
        _search_evidence(fixture, control["battle_tag"])
        for control in fixture.control_decisions
    ]
    paths = _write_evidence(tmp_path, evidence)

    records, summary = build_training_records(
        [trace],
        search_packet_paths=paths[0],
        search_receipt_paths=paths[1],
        posterior_paths=paths[2],
    )

    assert len(records) == 2
    assert len({row["input"]["fixture_id"] for row in records}) == 1
    assert len({row["split"] for row in records}) == 1
    assert len({row["split_group_id"] for row in records}) == 1
    assert summary["split_group_count"] == 1


def test_training_rejects_receipt_from_another_packet(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                event=0,
                kind="protocol",
                room="battle-one",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                event=1,
                kind="decision",
                battle_tag="battle-one",
                decision_index=0,
                state=_state("battle-one"),
                chosen_action="/choose move psychic",
            ),
            _record(
                event=2,
                kind="terminal",
                battle_tag="battle-one",
                won=True,
                lost=False,
                tied=False,
                final_state={},
            ),
        ],
    )
    [fixture] = build_fixtures([trace])
    packet, det, info, posterior = _search_evidence(fixture, "battle-one")
    info["packet_digest"] = "wrong-packet"
    paths = _write_evidence(tmp_path, [(packet, det, info, posterior)])

    with pytest.raises(TrainingRecordError, match="another frozen packet|not supplied"):
        build_training_records(
            [trace],
            search_packet_paths=paths[0],
            search_receipt_paths=paths[1],
            posterior_paths=paths[2],
        )



def test_same_battle_tag_in_distinct_runs_keeps_distinct_teacher_targets(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = []
    for run, won in (("run-a", True), ("run-b", False)):
        rows.extend(
            [
                _record(
                    run=run,
                    event=0,
                    kind="protocol",
                    room="battle-repeat",
                    protocol_index=0,
                    messages=[["", "turn", "4"]],
                ),
                _record(
                    run=run,
                    event=1,
                    kind="decision",
                    battle_tag="battle-repeat",
                    decision_index=0,
                    state=_state("battle-repeat"),
                    chosen_action="/choose move psychic",
                ),
                _record(
                    run=run,
                    event=2,
                    kind="terminal",
                    battle_tag="battle-repeat",
                    won=won,
                    lost=not won,
                    tied=False,
                    final_state={},
                ),
            ]
        )
    _write_trace(trace, rows)
    [fixture] = build_fixtures([trace])
    evidence = [
        _search_evidence(
            fixture,
            str(control["battle_tag"]),
            run_id=str(control["run_id"]),
        )
        for control in fixture.control_decisions
    ]
    paths = _write_evidence(tmp_path, evidence)

    records, summary = build_training_records(
        [trace],
        search_packet_paths=paths[0],
        search_receipt_paths=paths[1],
        posterior_paths=paths[2],
    )

    assert len(records) == 2
    assert len({row["battle_id"] for row in records}) == 2
    assert {
        row["provenance"]["run_id"] for row in records
    } == {"run-a", "run-b"}
    assert summary["battle_count"] == 2
