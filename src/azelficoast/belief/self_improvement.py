"""Turn completed battle traces into deterministic teacher evidence and model updates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from azelficoast.belief.evaluator import BeliefEvaluatorRuntime
from azelficoast.belief.improvement import AdmissionPolicy, ImprovementError, improve_checkpoint
from azelficoast.corpus import DecisionFixture, build_fixtures
from azelficoast.live.belief import PinnedShowdownBeliefPolicy, build_probe_source
from azelficoast.research.matched_comparison import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    freeze_packet,
    settle_packet,
)
from azelficoast.research.matched_search import execute_method
from azelficoast.research.training_records import (
    build_training_records,
    write_training_records,
)
from azelficoast.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT

TEACHER_MANIFEST_SCHEMA = "azelficoast.training-teacher-manifest"
TEACHER_MANIFEST_SCHEMA_VERSION = 2
CYCLE_RECEIPT_SCHEMA = "azelficoast.self-improvement-cycle"
CYCLE_RECEIPT_SCHEMA_VERSION = 1


class TeacherEvidenceError(ValueError):
    """Raised when teacher evidence cannot be derived faithfully."""


@dataclass(frozen=True)
class TeacherArtifacts:
    posterior: Mapping[str, Any]
    transition_program: Mapping[str, Any]


@dataclass(frozen=True)
class TeacherExclusion:
    reason: str
    detail: str | None = None


class TeacherSource(Protocol):
    showdown_commit: str

    def artifacts(self, fixture: DecisionFixture) -> TeacherArtifacts | TeacherExclusion:
        """Return exact teacher mechanics for one frozen public decision state."""


@dataclass(frozen=True)
class TeacherEvidence:
    root: Path
    manifest_path: Path
    manifest_digest: str
    trace_paths: tuple[Path, ...]
    packet_paths: tuple[Path, ...]
    receipt_paths: tuple[Path, ...]
    posterior_paths: tuple[Path, ...]
    admitted_decision_count: int
    excluded_decision_count: int


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise TeacherEvidenceError(f"{path} already contains different evidence")
        return
    path.write_text(payload, encoding="utf-8")


def _normalized_trace_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    if not paths:
        raise TeacherEvidenceError("at least one completed trace is required")
    by_digest: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise TeacherEvidenceError(f"trace does not exist: {path}")
        digest = _file_digest(path)
        by_digest.setdefault(digest, path)
    return tuple(by_digest[digest] for digest in sorted(by_digest))


def _fixture_mining_signals(fixture: DecisionFixture) -> dict[str, Any]:
    """Extract deterministic curriculum signals from public live-decision evidence."""

    fallback = 0
    searched = 0
    margins: list[float] = []
    entropies: list[float] = []
    battle_tags: set[str] = set()

    for control in fixture.control_decisions:
        battle_tag = control.get("battle_tag")
        if isinstance(battle_tag, str) and battle_tag:
            battle_tags.add(battle_tag)
        metadata = control.get("decision_metadata")
        if not isinstance(metadata, Mapping):
            continue
        belief = metadata.get("belief")
        if not isinstance(belief, Mapping):
            continue
        status = belief.get("status")
        fallback += int(status == "fallback")
        diagnostics = belief.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        learned_route = diagnostics.get("learned_route")
        searched += int(
            status == "search"
            or belief.get("reason") == "learned-policy-uncertain"
            or learned_route in {
                "exact-public-belief-search",
                "transition-program-search",
                "search-after-evaluator-error",
                "search-after-posterior-probe-error",
            }
        )
        prediction = diagnostics.get("learned_prediction")
        if not isinstance(prediction, Mapping):
            continue
        margin = prediction.get("policy_margin")
        entropy = prediction.get("policy_entropy_bits")
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            margins.append(float(margin))
        if isinstance(entropy, (int, float)) and not isinstance(entropy, bool):
            entropies.append(float(entropy))

    min_margin = min(margins) if margins else 1.0
    max_entropy = max(entropies) if entropies else 0.0
    return {
        "fallback_count": fallback,
        "search_count": searched,
        "uncertainty": max(0.0, min(1.0, 1.0 - min_margin)),
        "policy_entropy_bits": max_entropy,
        "legal_action_count": len(fixture.legal_actions),
        "battle_tags": sorted(battle_tags),
    }


def _mine_informative_fixtures(
    fixtures: Sequence[DecisionFixture],
    *,
    max_fixtures: int | None,
) -> tuple[list[DecisionFixture], dict[str, Any]]:
    """Select at most one informative public decision from each battle."""

    if max_fixtures is not None and (
        not isinstance(max_fixtures, int)
        or isinstance(max_fixtures, bool)
        or max_fixtures <= 0
    ):
        raise TeacherEvidenceError("max teacher fixtures must be a positive integer")

    if max_fixtures is None:
        selected_rows = [
            {
                "fixture_id": fixture.fixture_id,
                "run_id": str(control.get("run_id")),
                "battle_tag": str(control.get("battle_tag")),
                "event_index": control.get("event_index"),
            }
            for fixture in fixtures
            for control in fixture.control_decisions
        ]
        return list(fixtures), {
            "kind": "all-public-decisions",
            "candidate_fixture_count": len(fixtures),
            "candidate_decision_count": len(selected_rows),
            "max_fixtures": None,
            "selected_fixture_count": len(fixtures),
            "selected_decision_count": len(selected_rows),
            "one_decision_per_battle": False,
            "selected": selected_rows,
        }

    candidates: list[tuple[DecisionFixture, Mapping[str, Any], dict[str, Any]]] = []
    for fixture in fixtures:
        for control in fixture.control_decisions:
            run_id = control.get("run_id")
            battle_tag = control.get("battle_tag")
            if (
                not isinstance(run_id, str)
                or not run_id
                or not isinstance(battle_tag, str)
                or not battle_tag
            ):
                continue
            scoped = DecisionFixture(
                fixture_id=fixture.fixture_id,
                state=fixture.state,
                protocol_prefix=fixture.protocol_prefix,
                control_decisions=(control,),
            )
            candidates.append((fixture, control, _fixture_mining_signals(scoped)))

    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(item[2]["search_count"]),
            -float(item[2]["uncertainty"]),
            -float(item[2]["policy_entropy_bits"]),
            -int(item[2]["legal_action_count"]),
            -int(item[2]["fallback_count"]),
            str(item[1].get("run_id")),
            str(item[1].get("battle_tag")),
            int(item[1].get("event_index", -1)),
            item[0].fixture_id,
        ),
    )

    selected: list[tuple[DecisionFixture, Mapping[str, Any], dict[str, Any]]] = []
    represented_battles: set[tuple[str, str]] = set()
    for fixture, control, signals in ranked:
        battle_key = (str(control["run_id"]), str(control["battle_tag"]))
        if battle_key in represented_battles:
            continue
        selected.append((fixture, control, signals))
        represented_battles.add(battle_key)
        if max_fixtures is not None and len(selected) >= max_fixtures:
            break

    controls_by_fixture: dict[str, list[Mapping[str, Any]]] = {}
    fixture_by_id: dict[str, DecisionFixture] = {}
    fixture_order: list[str] = []
    for fixture, control, _ in selected:
        if fixture.fixture_id not in controls_by_fixture:
            fixture_order.append(fixture.fixture_id)
            controls_by_fixture[fixture.fixture_id] = []
            fixture_by_id[fixture.fixture_id] = fixture
        controls_by_fixture[fixture.fixture_id].append(control)

    selected_fixtures = [
        DecisionFixture(
            fixture_id=fixture_id,
            state=fixture_by_id[fixture_id].state,
            protocol_prefix=fixture_by_id[fixture_id].protocol_prefix,
            control_decisions=tuple(controls_by_fixture[fixture_id]),
        )
        for fixture_id in fixture_order
    ]

    selection = {
        "kind": "public-evidence-debt-curriculum",
        "candidate_fixture_count": len(fixtures),
        "candidate_decision_count": len(candidates),
        "max_fixtures": max_fixtures,
        "selected_fixture_count": len(selected_fixtures),
        "selected_decision_count": len(selected),
        "one_decision_per_battle": True,
        "selected": [
            {
                "fixture_id": fixture.fixture_id,
                "run_id": str(control["run_id"]),
                "battle_tag": str(control["battle_tag"]),
                "event_index": control.get("event_index"),
                "signals": signals,
            }
            for fixture, control, signals in selected
        ],
    }
    return selected_fixtures, selection


class PinnedShowdownTeacherSource:
    """Reuse the live pinned-Showdown reconstruction boundary for offline teaching."""

    showdown_commit = PINNED_SHOWDOWN_COMMIT

    def __init__(self, showdown_root: str | Path, *, timeout_seconds: float = 20.0) -> None:
        self.engine = PinnedShowdownBeliefPolicy(
            showdown_root,
            timeout_seconds=timeout_seconds,
        )
        if not self.engine.configured:
            raise TeacherEvidenceError("pinned Showdown teacher is not configured")

    def artifacts(self, fixture: DecisionFixture) -> TeacherArtifacts | TeacherExclusion:
        probe_fixture = DecisionFixture(
            fixture_id=fixture.fixture_id,
            state=fixture.state,
            protocol_prefix=fixture.protocol_prefix,
            control_decisions=(),
        )
        source, admission = build_probe_source(probe_fixture)
        if source is None:
            return TeacherExclusion(admission)
        if not isinstance(source.get("opponent_policy"), Mapping):
            return TeacherExclusion("opponent-model-unavailable")
        try:
            posterior = self.engine._probe_posterior(source)
            transition_program = self.engine._probe_transition_program(source)
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            detail = str(error)
            if isinstance(error, subprocess.CalledProcessError):
                detail = (error.stderr or error.stdout or detail).strip()
            return TeacherExclusion("pinned-showdown-probe-failed", detail[-1000:])

        for document, kind in (
            (posterior, "posterior"),
            (transition_program, "transition program"),
        ):
            if document.get("source_fixture_id") != fixture.fixture_id:
                raise TeacherEvidenceError(f"{kind} fixture identity drifted")
            if document.get("showdown_commit") != self.showdown_commit:
                raise TeacherEvidenceError(f"{kind} Showdown revision drifted")
            if document.get("legal_actions") != list(fixture.legal_actions):
                raise TeacherEvidenceError(f"{kind} legal actions drifted")
        return TeacherArtifacts(
            posterior=dict(posterior),
            transition_program=dict(transition_program),
        )


def _teacher_plan(
    *,
    evaluator_identity: Mapping[str, Any],
    showdown_commit: str,
    compute_budget: int,
) -> dict[str, Any]:
    if not isinstance(compute_budget, int) or isinstance(compute_budget, bool) or compute_budget <= 0:
        raise TeacherEvidenceError("teacher compute budget must be a positive integer")
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "posterior_treatments": ["generator_faithful"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": compute_budget,
        },
        "opponent_model": "repeat-last-or-uniform-legal-moves",
        "depths": [1],
        "confirmatory_predictors": [
            "posterior_world_count",
            "legal_action_count",
        ],
        "cluster_unit": "battle_tag",
        "showdown_commit": showdown_commit,
        "evaluator": dict(evaluator_identity),
        "inference": {
            "bootstrap_replicates": 1,
            "bootstrap_seed": 0,
        },
    }


def generate_teacher_evidence(
    trace_paths: Sequence[str | Path],
    *,
    evaluator: Any,
    source: TeacherSource,
    output_root: str | Path,
    compute_budget: int = 4096,
    max_teacher_fixtures: int | None = None,
) -> TeacherEvidence:
    """Generate settled search teacher artifacts from completed real traces."""

    traces = _normalized_trace_paths(trace_paths)
    all_fixtures = build_fixtures(traces)
    fixtures, selection = _mine_informative_fixtures(
        all_fixtures,
        max_fixtures=max_teacher_fixtures,
    )
    evaluator_identity = getattr(evaluator, "identity", None)
    if not isinstance(evaluator_identity, Mapping):
        raise TeacherEvidenceError("teacher evaluator lacks an immutable identity")
    plan = _teacher_plan(
        evaluator_identity=evaluator_identity,
        showdown_commit=source.showdown_commit,
        compute_budget=compute_budget,
    )
    trace_digests = [_file_digest(path) for path in traces]
    run_digest = _digest(
        {
            "trace_digests": trace_digests,
            "evaluator": dict(evaluator_identity),
            "showdown_commit": source.showdown_commit,
            "compute_budget": compute_budget,
            "selection": selection,
        }
    )
    root = Path(output_root) / run_digest.removeprefix("sha256:")
    packets_dir = root / "packets"
    artifacts_dir = root / "artifacts"

    packet_paths: list[Path] = []
    receipt_paths: list[Path] = []
    posterior_paths_by_digest: dict[str, Path] = {}
    admitted_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for fixture in fixtures:
        result = source.artifacts(fixture)
        if isinstance(result, TeacherExclusion):
            excluded_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "decision_count": len(fixture.control_decisions),
                    "reason": result.reason,
                    **({"detail": result.detail} if result.detail else {}),
                }
            )
            continue

        posterior = dict(result.posterior)
        transition_program = dict(result.transition_program)
        worlds = posterior.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise TeacherEvidenceError("teacher posterior contains no hidden-world support")
        if posterior.get("treatment") != "generator_faithful":
            raise TeacherEvidenceError("teacher posterior must be generator_faithful")

        posterior_digest = _digest(posterior)
        posterior_path = artifacts_dir / f"posterior-{posterior_digest.removeprefix('sha256:')}.json"
        _write_immutable_json(posterior_path, posterior)
        posterior_paths_by_digest[posterior_digest] = posterior_path

        program_digest = _digest(transition_program)
        program_path = artifacts_dir / f"program-{program_digest.removeprefix('sha256:')}.json"
        _write_immutable_json(program_path, transition_program)

        for control in fixture.control_decisions:
            run_id = control.get("run_id")
            battle_tag = control.get("battle_tag")
            if (
                not isinstance(run_id, str)
                or not run_id
                or not isinstance(battle_tag, str)
                or not battle_tag
            ):
                raise TeacherEvidenceError("fixture control lacks battle identity")
            state = {
                "fixture_id": fixture.fixture_id,
                "run_id": run_id,
                "battle_tag": battle_tag,
                "public_state": dict(fixture.state),
                "legal_actions": list(fixture.legal_actions),
                "predictors": {
                    "posterior_world_count": len(worlds),
                    "legal_action_count": len(fixture.legal_actions),
                },
            }
            packet = freeze_packet(
                plan=plan,
                state=state,
                posterior=posterior,
                posterior_treatment="generator_faithful",
                depth=1,
            )
            receipts: list[dict[str, Any]] = []
            for method in ("determinization", "information_set"):
                receipt = execute_method(
                    packet=packet,
                    posterior=posterior,
                    transition_program=transition_program,
                    method=method,
                    evaluator=evaluator,
                )
                # Wall-clock telemetry is useful operationally but is not scientific
                # authority and would make identical teacher evidence non-reproducible.
                receipt = dict(receipt)
                receipt.pop("resource_accounting", None)
                receipts.append(receipt)
            settled = settle_packet(packet=packet, receipts=receipts)

            packet_digest = str(packet["packet_digest"])
            packet_root = packets_dir / packet_digest
            packet_path = packet_root / "packet.json"
            det_path = packet_root / "determinization.json"
            info_path = packet_root / "information-set.json"
            settled_path = packet_root / "settled.json"
            _write_immutable_json(packet_path, packet)
            _write_immutable_json(det_path, receipts[0])
            _write_immutable_json(info_path, receipts[1])
            _write_immutable_json(settled_path, settled)
            packet_paths.append(packet_path)
            receipt_paths.extend((det_path, info_path))
            admitted_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "run_id": run_id,
                    "battle_tag": battle_tag,
                    "packet_digest": packet_digest,
                    "posterior_digest": posterior_digest,
                    "program_digest": program_digest,
                    "packet": str(packet_path.relative_to(root)),
                    "settled": str(settled_path.relative_to(root)),
                }
            )

    exclusion_counts = Counter(
        row["reason"]
        for row in excluded_rows
        for _ in range(int(row["decision_count"]))
    )
    manifest_unsigned = {
        "schema": TEACHER_MANIFEST_SCHEMA,
        "schema_version": TEACHER_MANIFEST_SCHEMA_VERSION,
        "run_digest": run_digest,
        "trace_digests": trace_digests,
        "showdown_commit": source.showdown_commit,
        "evaluator": dict(evaluator_identity),
        "plan": plan,
        "selection": selection,
        "admitted_decision_count": len(admitted_rows),
        "excluded_decision_count": sum(
            int(row["decision_count"]) for row in excluded_rows
        ),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "admitted": admitted_rows,
        "excluded": excluded_rows,
    }
    manifest_digest = _digest(manifest_unsigned)
    manifest = {**manifest_unsigned, "manifest_digest": manifest_digest}
    manifest_path = root / "manifest.json"
    _write_immutable_json(manifest_path, manifest)

    return TeacherEvidence(
        root=root,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        trace_paths=traces,
        packet_paths=tuple(packet_paths),
        receipt_paths=tuple(receipt_paths),
        posterior_paths=tuple(
            posterior_paths_by_digest[digest] for digest in sorted(posterior_paths_by_digest)
        ),
        admitted_decision_count=len(admitted_rows),
        excluded_decision_count=sum(int(row["decision_count"]) for row in excluded_rows),
    )


def run_self_improvement_cycle(
    trace_paths: Sequence[str | Path],
    *,
    showdown_root: str | Path,
    incumbent_checkpoint: str | Path,
    workspace: str | Path,
    models_dir: str | Path,
    receipts_dir: str | Path,
    promotion_file: str | Path,
    teacher_compute_budget: int = 4096,
    max_teacher_fixtures: int | None = None,
    teacher_timeout_seconds: float = 20.0,
    split_seed: str = "azelficoast.training-records",
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    value_target_source: str = "public_belief_search_return",
    admission_policy: AdmissionPolicy = AdmissionPolicy(),
) -> dict[str, Any]:
    """Run trace -> teacher -> dataset -> candidate -> admission -> promotion."""

    evaluator = BeliefEvaluatorRuntime.from_checkpoint(incumbent_checkpoint)
    incumbent_digest = str(evaluator.identity["checkpoint_digest"])
    teacher = generate_teacher_evidence(
        trace_paths,
        evaluator=evaluator,
        source=PinnedShowdownTeacherSource(
            showdown_root,
            timeout_seconds=teacher_timeout_seconds,
        ),
        output_root=Path(workspace) / "teachers",
        compute_budget=teacher_compute_budget,
        max_teacher_fixtures=max_teacher_fixtures,
    )
    cycle_inputs = {
        "teacher_manifest_digest": teacher.manifest_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
        "max_teacher_fixtures": max_teacher_fixtures,
        "split_seed": split_seed,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "policy_weight": policy_weight,
        "value_target_source": value_target_source,
        "admission_policy": admission_policy.as_record(),
    }
    cycle_id = _digest(cycle_inputs)
    cycle_dir = Path(workspace) / "cycles" / cycle_id.removeprefix("sha256:")

    if not teacher.packet_paths:
        receipt = {
            "schema": CYCLE_RECEIPT_SCHEMA,
            "schema_version": CYCLE_RECEIPT_SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "status": "not-ready",
            "reason": "no-admitted-teacher-targets",
            "inputs": cycle_inputs,
            "teacher_manifest": str(teacher.manifest_path),
        }
        _write_immutable_json(cycle_dir / "receipt.json", receipt)
        return receipt

    records, build_summary = build_training_records(
        teacher.trace_paths,
        search_packet_paths=teacher.packet_paths,
        search_receipt_paths=teacher.receipt_paths,
        posterior_paths=teacher.posterior_paths,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    dataset_path = cycle_dir / "training.jsonl"
    dataset_digest = write_training_records(records, dataset_path)
    split_counts = Counter(str(row["split"]) for row in records)
    missing_splits = [
        split for split in ("train", "validation", "test") if split_counts.get(split, 0) == 0
    ]
    if missing_splits:
        receipt = {
            "schema": CYCLE_RECEIPT_SCHEMA,
            "schema_version": CYCLE_RECEIPT_SCHEMA_VERSION,
            "cycle_id": cycle_id,
            "status": "not-ready",
            "reason": "held-out-splits-incomplete",
            "missing_splits": missing_splits,
            "inputs": cycle_inputs,
            "teacher_manifest": str(teacher.manifest_path),
            "dataset": str(dataset_path),
            "dataset_digest": dataset_digest,
            "dataset_summary": build_summary,
        }
        _write_immutable_json(cycle_dir / "receipt.json", receipt)
        return receipt

    try:
        improvement = improve_checkpoint(
            dataset_path,
            incumbent_checkpoint=incumbent_checkpoint,
            expected_incumbent_digest=incumbent_digest,
            models_dir=models_dir,
            receipts_dir=receipts_dir,
            promotion_file=promotion_file,
            epochs=epochs,
            learning_rate=learning_rate,
            policy_weight=policy_weight,
            value_target_source=value_target_source,
            admission_policy=admission_policy,
        )
    except ImprovementError as error:
        raise TeacherEvidenceError(str(error)) from error

    receipt = {
        "schema": CYCLE_RECEIPT_SCHEMA,
        "schema_version": CYCLE_RECEIPT_SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "promoted" if improvement["admitted"] else "rejected",
        "inputs": cycle_inputs,
        "teacher_manifest": str(teacher.manifest_path),
        "dataset": str(dataset_path),
        "dataset_digest": dataset_digest,
        "dataset_summary": build_summary,
        "improvement": improvement,
    }
    _write_immutable_json(cycle_dir / "receipt.json", receipt)
    return receipt
