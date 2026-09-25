"""Bootstrap the learned evaluator from public human Random Battle decisions.

This path is deliberately separate from the settled-search teacher. Human actions are
useful initialization targets, but they are not scientific authority for claims about
public-belief search. Once an evaluator exists, normal self-improvement returns to
search-generated policy targets.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorSpec,
    checkpoint_digest,
    init_params,
    write_checkpoint,
)
from azelficoast.belief.improvement import (
    AdmissionPolicy,
    ImprovementError,
    improve_checkpoint,
)
from azelficoast.corpus import DecisionFixture, build_fixtures
from azelficoast.live.belief import PinnedShowdownBeliefPolicy, build_probe_source
from azelficoast.research.matched_comparison import _sha256 as matched_digest
from azelficoast.research.training_records import (
    DEFAULT_SPLIT_SEED,
    TRAINING_SCHEMA,
    TRAINING_SCHEMA_VERSION,
    TrainingRecordError,
    _assign_leakage_safe_splits,
    _battle_id,
    _sha256,
    _terminal_outcomes,
    write_training_records,
)
from azelficoast.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT

PUBLIC_PRETRAINING_BUILD_SCHEMA = "azelficoast.public-pretraining-build"
PUBLIC_PRETRAINING_BUILD_SCHEMA_VERSION = 1


class PublicPretrainingError(ValueError):
    """Raised when public imitation evidence cannot be constructed faithfully."""


@dataclass(frozen=True)
class PosteriorExclusion:
    reason: str
    detail: str | None = None


class PublicPosteriorSource(Protocol):
    showdown_commit: str

    def posterior(
        self, fixture: DecisionFixture
    ) -> Mapping[str, Any] | PosteriorExclusion:
        """Return a public-history posterior without consulting realized hidden state."""


class PinnedShowdownPublicPosteriorSource:
    """Generate the same generator-faithful posterior used by live belief reasoning."""

    showdown_commit = PINNED_SHOWDOWN_COMMIT

    def __init__(self, showdown_root: str | Path, *, timeout_seconds: float = 20.0) -> None:
        self.engine = PinnedShowdownBeliefPolicy(
            showdown_root,
            timeout_seconds=timeout_seconds,
        )
        if not self.engine.configured:
            raise PublicPretrainingError("pinned Showdown posterior source is not configured")

    def posterior(
        self, fixture: DecisionFixture
    ) -> Mapping[str, Any] | PosteriorExclusion:
        probe_fixture = DecisionFixture(
            fixture_id=fixture.fixture_id,
            state=fixture.state,
            protocol_prefix=fixture.protocol_prefix,
            control_decisions=(),
        )
        source, admission = build_probe_source(probe_fixture)
        if source is None:
            return PosteriorExclusion(admission)
        try:
            posterior = self.engine._probe_posterior(source)
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
            return PosteriorExclusion("pinned-showdown-posterior-failed", detail[-1000:])

        if posterior.get("source_fixture_id") != fixture.fixture_id:
            raise PublicPretrainingError("posterior fixture identity drifted")
        if posterior.get("showdown_commit") != self.showdown_commit:
            raise PublicPretrainingError("posterior Showdown revision drifted")
        if posterior.get("legal_actions") != list(fixture.legal_actions):
            raise PublicPretrainingError("posterior legal actions drifted")
        if posterior.get("conditioned_on_public_history") is not True:
            raise PublicPretrainingError("posterior is not conditioned on public history")
        if posterior.get("realized_hidden_state_revealed") is not False:
            raise PublicPretrainingError("public pretraining may not reveal hidden state")
        if posterior.get("treatment") != "generator_faithful":
            raise PublicPretrainingError("public pretraining requires generator_faithful posterior")
        return dict(posterior)


def _human_metadata(control: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = control.get("decision_metadata")
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("selected_policy") != "recorded-human":
        return None
    if metadata.get("training_policy_authority") is not False:
        return None
    replay_id = metadata.get("source_replay_id")
    side = metadata.get("source_side")
    if not isinstance(replay_id, str) or not replay_id:
        return None
    if side not in {"p1", "p2"}:
        return None
    return metadata


def build_public_pretraining_records(
    trace_paths: Sequence[str | Path],
    *,
    posterior_source: PublicPosteriorSource,
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build leakage-safe imitation/value records from public replay traces."""

    try:
        fixtures = build_fixtures(trace_paths)
        terminals = _terminal_outcomes(trace_paths)
    except TrainingRecordError as error:
        raise PublicPretrainingError(str(error)) from error

    rows: list[dict[str, Any]] = []
    source_decisions = 0
    excluded = Counter()

    for fixture in fixtures:
        human_controls = [
            control
            for control in fixture.control_decisions
            if _human_metadata(control) is not None
        ]
        source_decisions += len(fixture.control_decisions)
        excluded["non_public_human_decision"] += (
            len(fixture.control_decisions) - len(human_controls)
        )
        if not human_controls:
            continue

        posterior_result = posterior_source.posterior(fixture)
        if isinstance(posterior_result, PosteriorExclusion):
            excluded[posterior_result.reason] += len(human_controls)
            continue
        posterior = dict(posterior_result)
        posterior_digest = matched_digest(posterior)

        for control in human_controls:
            metadata = _human_metadata(control)
            assert metadata is not None
            run_id = control.get("run_id")
            battle_tag = control.get("battle_tag")
            decision_index = control.get("decision_index")
            chosen_action = control.get("chosen_action")
            if (
                not isinstance(run_id, str)
                or not isinstance(battle_tag, str)
                or not isinstance(decision_index, int)
                or not isinstance(chosen_action, str)
            ):
                raise PublicPretrainingError(
                    f"fixture {fixture.fixture_id}: malformed public decision provenance"
                )
            if chosen_action not in fixture.legal_actions:
                raise PublicPretrainingError(
                    f"fixture {fixture.fixture_id}: recorded human action is not legal"
                )
            terminal = terminals.get((run_id, battle_tag))
            if terminal is None:
                raise PublicPretrainingError(
                    f"public battle {(run_id, battle_tag)!r} has no terminal outcome"
                )

            battle_id = _battle_id(run_id, battle_tag, metadata)
            policy_distribution = {
                action: 1.0 if action == chosen_action else 0.0
                for action in fixture.legal_actions
            }
            record_material = {
                "kind": "public-human-imitation",
                "battle_id": battle_id,
                "fixture_id": fixture.fixture_id,
                "decision_index": decision_index,
                "posterior_digest": posterior_digest,
                "chosen_action": chosen_action,
            }
            rows.append(
                {
                    "schema": TRAINING_SCHEMA,
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "record_id": _sha256(record_material),
                    "battle_id": battle_id,
                    "decision_index": decision_index,
                    "_battle_id": battle_id,
                    "_fixture_id": fixture.fixture_id,
                    "input": {
                        "fixture_id": fixture.fixture_id,
                        "public_state": fixture.state,
                        "protocol_prefix": [
                            [list(message) for message in batch]
                            for batch in fixture.protocol_prefix
                        ],
                        "posterior": posterior,
                        "posterior_digest": posterior_digest,
                        "posterior_treatment": "generator_faithful",
                        "legal_actions": list(fixture.legal_actions),
                    },
                    "targets": {
                        "value": {
                            "eventual_battle_outcome": terminal["outcome"],
                        },
                        "policy": {
                            "selected_action": chosen_action,
                            "action_probabilities": policy_distribution,
                        },
                    },
                    "provenance": {
                        "run_id": run_id,
                        "battle_tag": battle_tag,
                        "terminal_event_index": terminal["event_index"],
                        "policy_target": {
                            "kind": "public-human-imitation",
                            "replay_id": metadata["source_replay_id"],
                            "side": metadata["source_side"],
                            "rating": metadata.get("source_rating"),
                            "scientific_search_teacher": False,
                        },
                        "value_target": {
                            "kind": "eventual-battle-outcome",
                        },
                        "posterior": {
                            "kind": "pinned-showdown-generator-faithful",
                            "showdown_commit": posterior_source.showdown_commit,
                            "digest": posterior_digest,
                            "realized_hidden_state_revealed": False,
                        },
                    },
                }
            )

    if not rows:
        raise PublicPretrainingError(
            "no public human decisions were admitted for pretraining"
        )

    _assign_leakage_safe_splits(
        rows,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    rows.sort(
        key=lambda row: (
            row["battle_id"],
            row["decision_index"],
            row["record_id"],
        )
    )
    summary = {
        "schema": PUBLIC_PRETRAINING_BUILD_SCHEMA,
        "schema_version": PUBLIC_PRETRAINING_BUILD_SCHEMA_VERSION,
        "source_decision_count": source_decisions,
        "record_count": len(rows),
        "excluded_decision_count": sum(excluded.values()),
        "exclusion_counts": dict(sorted(excluded.items())),
        "battle_count": len({row["battle_id"] for row in rows}),
        "fixture_count": len({row["input"]["fixture_id"] for row in rows}),
        "split_group_count": len({row["split_group_id"] for row in rows}),
        "split_record_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "policy_target_source": "public-human-imitation",
        "value_target_source": "eventual-battle-outcome",
        "posterior_treatment": "generator_faithful",
        "showdown_commit": posterior_source.showdown_commit,
    }
    return rows, summary


def build_public_pretraining_dataset(
    trace_paths: Sequence[str | Path],
    output: str | Path,
    *,
    posterior_source: PublicPosteriorSource,
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    records, summary = build_public_pretraining_records(
        trace_paths,
        posterior_source=posterior_source,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    digest = write_training_records(records, output)
    return {**summary, "output": str(output), "sha256": digest}


def _create_untrained_baseline(
    *,
    models_dir: Path,
    seed: int,
) -> tuple[Path, str]:
    spec = BeliefEvaluatorSpec()
    params = init_params(spec, seed=seed)
    digest = checkpoint_digest(params, spec)
    destination = models_dir / digest.removeprefix("sha256:")
    if not destination.exists():
        write_checkpoint(
            destination,
            params,
            spec,
            metadata={
                "kind": "deterministic-untrained-public-bootstrap-baseline",
                "seed": seed,
            },
        )
    return destination, digest


def run_public_pretraining(
    trace_paths: Sequence[str | Path],
    *,
    showdown_root: str | Path,
    dataset_path: str | Path,
    models_dir: str | Path,
    receipts_dir: str | Path,
    promotion_file: str | Path,
    posterior_timeout_seconds: float = 20.0,
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    admission_policy: AdmissionPolicy = AdmissionPolicy(),
) -> dict[str, Any]:
    """Build public imitation records and admit a first evaluator against a fixed baseline."""

    promotion = Path(promotion_file)
    if promotion.exists():
        raise PublicPretrainingError(
            "public pretraining is a cold-start path and will not replace an existing "
            "promoted evaluator; use training cycle for subsequent improvements"
        )

    source = PinnedShowdownPublicPosteriorSource(
        showdown_root,
        timeout_seconds=posterior_timeout_seconds,
    )
    dataset_summary = build_public_pretraining_dataset(
        trace_paths,
        dataset_path,
        posterior_source=source,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    baseline_path, baseline_digest = _create_untrained_baseline(
        models_dir=Path(models_dir),
        seed=seed,
    )
    try:
        improvement = improve_checkpoint(
            dataset_path,
            incumbent_checkpoint=baseline_path,
            expected_incumbent_digest=baseline_digest,
            models_dir=models_dir,
            receipts_dir=receipts_dir,
            promotion_file=promotion,
            epochs=epochs,
            learning_rate=learning_rate,
            policy_weight=policy_weight,
            value_target_source="eventual_battle_outcome",
            admission_policy=admission_policy,
        )
    except ImprovementError as error:
        raise PublicPretrainingError(str(error)) from error

    return {
        "schema": "azelficoast.public-pretraining-run",
        "schema_version": 1,
        "dataset": dataset_summary,
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_digest": baseline_digest,
        "improvement": improvement,
        "promoted": bool(improvement["admitted"]),
        "next_step": (
            "training cycle"
            if improvement["admitted"]
            else "increase public corpus or adjust explicit bootstrap training parameters"
        ),
    }
