from __future__ import annotations

import json
from pathlib import Path

from azelficoast.research.hosted.common import (
    EVIDENCE_ROOT,
    SHOWDOWN_ROOT,
    SOURCE_ARTIFACT,
    HostedResearchError,
    as_dict,
    find_jsonl,
    load_json,
    node,
    print_json,
    pytest,
    python_module,
    showdown_revision,
    start_showdown_server,
    write_json,
)


EXHAUSTED_FIXTURES = {
    "regigigas-bruxish-turn17": "7a60eceebb54d4213afa90a51d6120e58f26bbcfd159d537087a6cd832a7b9ad",
    "regigigas-bruxish-turn18": "e806414b1c6f2afa650d7c15f466aafc032c4c1fb0adae94f94e152d483d30ad",
}

PUBLIC_BELIEF_FIXTURES = {
    "jirachi-staraptor": "2b29e00e9babc864a5a1cfc32ba43be00ee587d02d3c3e0bfe3d0539deaaa898",
    "regigigas-staraptor-turn12": "6314b9a24e28265bb3b36c490c0297357ece57e7963ac653c3d798f4d3b82821",
    "regigigas-bruxish-turn17": "7a60eceebb54d4213afa90a51d6120e58f26bbcfd159d537087a6cd832a7b9ad",
    "tinkaton-gardevoir": "bbe2078827062b7e32e0efcd28a77619b5d1e4edec4fd490ea71a5005e9e49d4",
    "regigigas-staraptor-turn13": "ce6f0239a59d5d64bd303bce884ea063f9c1305bcbe6b2b55b18c51b349d5b46",
    "regigigas-bruxish-turn18": "e806414b1c6f2afa650d7c15f466aafc032c4c1fb0adae94f94e152d483d30ad",
}

FACTORED_FIXTURE = "6314b9a24e28265bb3b36c490c0297357ece57e7963ac653c3d798f4d3b82821"
SURVIVAL_FIXTURE = "65261d732db208427215265f3171af6607a21c9b36606054604883cb5e4654f2"


def _candidate(fixture_id: str) -> dict[str, object]:
    candidates = as_dict(
        load_json(EVIDENCE_ROOT / "discovery" / "candidates.json"),
        label="candidate manifest",
    )
    rows = candidates.get("candidates")
    if not isinstance(rows, list):
        raise HostedResearchError("candidate manifest lacks candidates")
    return next(
        row
        for row in rows
        if isinstance(row, dict) and row.get("fixture_id") == fixture_id
    )


def conditional_team_prior() -> None:
    output = Path("/tmp/conditional-team-prior.json")
    node(
        "scripts/evaluate_conditional_team_prior.cjs",
        str(SHOWDOWN_ROOT),
        "80000",
        "10000",
        "10000",
        stdout=output,
    )
    result = as_dict(load_json(output), label="conditional team prior")
    test = as_dict(result["test"], label="conditional prior test")
    uniform = as_dict(test["uniform_compatible"], label="uniform prior")
    unigram = as_dict(test["unigram_compatible"], label="unigram prior")
    pairwise = as_dict(test["pairwise_compatible"], label="pairwise prior")
    assert float(result["selected_lambda"]) > 0
    assert uniform["support_misses"] == 0
    assert unigram["support_misses"] == 0
    assert pairwise["support_misses"] == 0
    nll_gain = float(unigram["mean_nll"]) - float(pairwise["mean_nll"])
    top20_gain = float(pairwise["top20_recall"]) - float(unigram["top20_recall"])
    assert float(pairwise["mean_nll"]) < float(uniform["mean_nll"])
    assert nll_gain >= 0.05
    assert top20_gain >= 0.02
    real = as_dict(result["real_staraptor_fixture"], label="real Staraptor fixture")
    print_json(
        {
            "selected_lambda": result["selected_lambda"],
            "uniform_nll": uniform["mean_nll"],
            "unigram_nll": unigram["mean_nll"],
            "pairwise_nll": pairwise["mean_nll"],
            "nll_gain_over_unigram": nll_gain,
            "unigram_top20": unigram["top20_recall"],
            "pairwise_top20": pairwise["top20_recall"],
            "top20_gain": top20_gain,
            "pairwise_top5": pairwise["top5_recall"],
            "pairwise_top1": pairwise["top1_recall"],
            "real_fixture_support_count": real["support_count"],
            "real_fixture_top_20": real["top_20"],
            "evidence_sha256": result["evidence_sha256"],
        }
    )


def _assert_same_policy(full: dict[str, object], quotient: dict[str, object]) -> None:
    qtrace = as_dict(quotient["quotient_trace"], label="quotient trace")
    for policy in ("public_belief", "determinization"):
        full_policy = as_dict(full[policy], label=f"full {policy}")
        quotient_policy = as_dict(qtrace[policy], label=f"quotient {policy}")
        assert quotient_policy["chosen_action"] == full_policy["chosen_action"]
        left = as_dict(full_policy["root_values"], label="full root values")
        right = as_dict(quotient_policy["root_values"], label="quotient root values")
        assert left.keys() == right.keys()
        assert all(abs(float(left[action]) - float(right[action])) <= 1e-12 for action in left)
    assert qtrace["policy_disagreement"] == full["policy_disagreement"]
    assert qtrace["strategy_fusion_observation_count"] == full["strategy_fusion_observation_count"]


def decision_relevance_quotient() -> None:
    sources = {
        "status": EVIDENCE_ROOT / "status-source.json",
        "fusion": Path("experiments/real-belief-source-tinkaton-zapdosgalar-mid.json"),
    }
    for name, source in sources.items():
        oracle = Path(f"/tmp/{name}-oracle.json")
        node(
            "scripts/probe_real_belief_trace.cjs",
            str(SHOWDOWN_ROOT),
            str(source),
            stdout=oracle,
        )
        python_module(
            "azelficoast.research.verification.real_belief_trace",
            str(oracle),
            "--quotient",
            stdout=f"/tmp/{name}-quotient.json",
        )
        python_module(
            "azelficoast.research.verification.real_belief_trace",
            str(oracle),
            stdout=f"/tmp/{name}-full.json",
        )

    status = as_dict(load_json("/tmp/status-quotient.json"), label="status quotient")
    status_full = as_dict(load_json("/tmp/status-full.json"), label="status full")
    fusion = as_dict(load_json("/tmp/fusion-quotient.json"), label="fusion quotient")
    fusion_full = as_dict(load_json("/tmp/fusion-full.json"), label="fusion full")
    _assert_same_policy(status_full, status)
    _assert_same_policy(fusion_full, fusion)

    status_cert = as_dict(status["certificate"], label="status certificate")
    fusion_cert = as_dict(fusion["certificate"], label="fusion certificate")
    assert status_cert["worlds_in"] == 18
    assert status_cert["classes_out"] == 3
    assert status_cert["decision_fields"] == ["opponent.active.exact_hp"]
    assert status_cert["belief_branching_required"] is True
    assert fusion_cert["worlds_in"] == 18
    assert fusion_cert["classes_out"] == 6
    assert fusion_cert["decision_fields"] == [
        "opponent.active.item",
        "opponent.active.exact_hp",
    ]
    assert fusion_cert["belief_branching_required"] is True
    assert int(fusion_full["strategy_fusion_observation_count"]) > 0
    assert "opponent.active.item" in fusion_cert["decision_fields"]
    assert "opponent.active.item" not in status_cert["decision_fields"]

    summary = {
        "status_move": {
            "source_worlds": status_cert["worlds_in"],
            "decision_classes": status_cert["classes_out"],
            "reduction_fraction": status_cert["reduction_fraction"],
            "decision_fields": status_cert["decision_fields"],
            "public_action": as_dict(status_full["public_belief"], label="status public")["chosen_action"],
        },
        "strategy_fusion": {
            "source_worlds": fusion_cert["worlds_in"],
            "decision_classes": fusion_cert["classes_out"],
            "reduction_fraction": fusion_cert["reduction_fraction"],
            "decision_fields": fusion_cert["decision_fields"],
            "strategy_fusion_observation_count": fusion_full["strategy_fusion_observation_count"],
            "public_action": as_dict(fusion_full["public_belief"], label="fusion public")["chosen_action"],
        },
    }
    write_json("/tmp/decision-relevance-summary.json", summary, pretty=True)
    print_json(summary)


def _latest_tera_and_team_size(fixture: dict[str, object]) -> tuple[object | None, int | None]:
    latest_tera = None
    public_team_size = None
    protocol = fixture.get("protocol_prefix")
    if not isinstance(protocol, list):
        raise HostedResearchError("fixture lacks protocol prefix")
    for batch in protocol:
        if not isinstance(batch, list):
            continue
        for message in batch:
            if not isinstance(message, list) or len(message) < 2 or message[0] != "":
                continue
            if message[1] == "teamsize" and len(message) >= 4 and message[2] == "p2":
                public_team_size = int(message[3])
            if message[1] == "request" and len(message) >= 3:
                try:
                    request = json.loads(message[2])
                except json.JSONDecodeError:
                    continue
                active = request.get("active") or []
                if active and active[0].get("canTerastallize"):
                    latest_tera = active[0]["canTerastallize"]
    return latest_tera, public_team_size


def exhausted_bench(name: str) -> None:
    fixture_id = EXHAUSTED_FIXTURES[name]
    fixture = find_jsonl(
        EVIDENCE_ROOT / "discovery" / "fixtures.jsonl",
        "fixture_id",
        fixture_id,
    )
    candidate = _candidate(fixture_id)
    latest_tera, public_team_size = _latest_tera_and_team_size(fixture)
    state = as_dict(fixture["state"], label="fixture state")
    team = as_dict(state["opponent_team"], label="opponent team")
    public_team = list(team.values())
    known_species = {
        as_dict(view, label="opponent view").get("species")
        for view in public_team
        if as_dict(view, label="opponent view").get("species")
    }
    active_species = as_dict(state["opponent_active"], label="opponent active")["species"]
    survivors = [
        as_dict(view, label="opponent view")["species"]
        for view in public_team
        if not as_dict(view, label="opponent view").get("fainted")
    ]
    assert public_team_size == 6
    assert len(known_species) == public_team_size
    assert survivors == [active_species]
    assert latest_tera is not None

    source = {
        "schema": "azelficoast.real-belief-source-fixture",
        "schema_version": 1,
        "fixture_id": fixture_id,
        "showdown_commit": candidate["showdown_commit"],
        "source_artifact": SOURCE_ARTIFACT,
        "source_projection": "exact frozen speed-fork fixture with publicly exhausted opponent bench",
        "state": state,
        "protocol_prefix": fixture["protocol_prefix"],
        "control_decisions": fixture.get("control_decisions", []),
        "plausible_items": sorted(as_dict(candidate["item_counts"], label="item counts")),
        "observed_opponent_moves": candidate["revealed_moves"],
        "opponent_response_move": candidate["locked_move"],
        "own_active_tera_type": latest_tera,
    }
    write_json("/tmp/source.json", source)

    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        "/tmp/source.json",
        stdout="/tmp/oracle.json",
    )
    python_module("azelficoast.research.verification.real_belief_trace", "/tmp/oracle.json", stdout="/tmp/trace.json")
    python_module("azelficoast.research.verification.real_belief_miner", "/tmp/oracle.json", stdout="/tmp/mining.json")

    trace = as_dict(load_json("/tmp/trace.json"), label="exhausted trace")
    mining = as_dict(load_json("/tmp/mining.json"), label="exhausted mining")
    assert trace["source_fixture_id"] == fixture_id
    assert trace["experiment_valid"] is True
    assert int(trace["world_count"]) >= 2
    assert int(trace["legal_action_count"]) >= 1
    assert mining["ranking_uses_policy_result"] is False
    summary = {
        "fixture_id": fixture_id,
        "world_count": trace["world_count"],
        "legal_action_count": trace["legal_action_count"],
        "strategy_fusion_observation_count": trace["strategy_fusion_observation_count"],
        "policy_disagreement": trace["policy_disagreement"],
        "determinization_action": as_dict(trace["determinization"], label="determinization")["chosen_action"],
        "public_belief_action": as_dict(trace["public_belief"], label="public belief")["chosen_action"],
    }
    write_json("/tmp/result.json", summary)
    print_json(summary)


def factored_hidden_bench_prior() -> None:
    fixture = find_jsonl(
        EVIDENCE_ROOT / "discovery" / "fixtures.jsonl",
        "fixture_id",
        FACTORED_FIXTURE,
    )
    candidate = _candidate(FACTORED_FIXTURE)
    source = {
        "schema": "azelficoast.real-belief-source-fixture",
        "schema_version": 1,
        "fixture_id": FACTORED_FIXTURE,
        "showdown_commit": candidate["showdown_commit"],
        "fixture": fixture,
        "observed_opponent_moves": ["doubleedge"],
        "opponent_response_move": "doubleedge",
        "plausible_items": sorted(as_dict(candidate["item_counts"], label="item counts")),
        "source_artifact": SOURCE_ARTIFACT,
    }
    write_json("/tmp/source.json", source)

    node(
        "scripts/evaluate_conditional_team_prior.cjs",
        str(SHOWDOWN_ROOT),
        "80000",
        "10000",
        "10000",
        stdout="/tmp/conditional-team-prior.json",
    )
    prior = as_dict(load_json("/tmp/conditional-team-prior.json"), label="conditional prior")
    test = as_dict(prior["test"], label="prior test")
    unigram = as_dict(test["unigram_compatible"], label="unigram")
    pairwise = as_dict(test["pairwise_compatible"], label="pairwise")
    real = as_dict(prior["real_staraptor_fixture"], label="real Staraptor fixture")
    assert pairwise["support_misses"] == 0
    assert float(unigram["mean_nll"]) - float(pairwise["mean_nll"]) >= 0.05
    assert float(pairwise["top20_recall"]) - float(unigram["top20_recall"]) >= 0.02
    species_prior = real["species_prior"]
    assert isinstance(species_prior, list)
    assert len(species_prior) == real["support_count"] == 504
    assert abs(sum(float(as_dict(row, label="species prior row")["probability"]) for row in species_prior) - 1) <= 1e-9

    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        "/tmp/source.json",
        "--bench-prior",
        "/tmp/conditional-team-prior.json",
        stdout="/tmp/factored-bench-oracle.json",
    )
    python_module(
        "azelficoast.research.verification.real_belief_trace",
        "/tmp/factored-bench-oracle.json",
        stdout="/tmp/factored-bench-trace.json",
    )
    oracle = as_dict(load_json("/tmp/factored-bench-oracle.json"), label="factored oracle")
    trace = as_dict(load_json("/tmp/factored-bench-trace.json"), label="factored trace")
    field = "opponent.bench.species"
    factor = as_dict(as_dict(oracle["factored_hidden"], label="factored hidden")[field], label="bench factor")
    evidence = as_dict(factor["evidence"], label="factor evidence")
    legal_actions = oracle["legal_actions"]
    assert isinstance(legal_actions, list)
    distribution = factor["distribution"]
    assert isinstance(distribution, list) and len(distribution) == 504
    assert evidence["audited_support_count"] == 504
    assert int(evidence["audited_active_variant_count"]) >= 1
    assert set(factor["unread_actions"]) == set(legal_actions)
    assert all(as_dict(evidence["immediate_equivalence_by_action"], label="equivalence").values())
    assert not any(as_dict(evidence["continuation_read_by_action"], label="continuation reads").values())
    assert evidence["first_divergence_by_action"] == {}
    root_survival = as_dict(evidence["root_survival_by_action"], label="root survival")
    assert root_survival["/choose move bodyslam"] is False
    assert root_survival["/choose move bodyslam terastallize"] is False
    assert trace["materialized_world_count"] == trace["world_count"]
    assert trace["factoring_ratio"] == 504
    assert int(trace["latent_world_count"]) == int(trace["materialized_world_count"]) * 504
    assert as_dict(as_dict(trace["factored_hidden"], label="trace factored hidden")[field], label="trace factor")["support_count"] == 504
    actions = trace["actions"]
    assert isinstance(actions, list)
    assert all(
        as_dict(as_dict(row, label="action")["dependency_signature"], label="signature")["marginalized_hidden_factors"] == [field]
        for row in actions
    )
    assert trace["experiment_valid"] is True
    print_json(
        {
            "materialized_world_count": trace["materialized_world_count"],
            "latent_world_count": trace["latent_world_count"],
            "factoring_ratio": trace["factoring_ratio"],
            "bench_support_count": 504,
            "audited_active_variant_count": evidence["audited_active_variant_count"],
            "legal_action_count": len(legal_actions),
            "root_switch_frontier_actions": sorted(action for action, survives in root_survival.items() if not survives),
            "dynamic_bench_read_actions": sorted(
                action
                for action, read in as_dict(evidence["continuation_read_by_action"], label="continuation reads").items()
                if read
            ),
            "policy_disagreement": trace["policy_disagreement"],
            "determinization_action": as_dict(trace["determinization"], label="determinization")["chosen_action"],
            "public_belief_action": as_dict(trace["public_belief"], label="public belief")["chosen_action"],
        }
    )


def joint_random_battle_posterior() -> None:
    candidates: list[tuple[int, str]] = []
    for path in EVIDENCE_ROOT.rglob("*.json"):
        try:
            document = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        fixture = document.get("fixture", document)
        if not isinstance(fixture, dict):
            continue
        state = fixture.get("state", {})
        protocol = fixture.get("protocol_prefix", [])
        opponent_team = state.get("opponent_team", {}) if isinstance(state, dict) else {}
        if fixture.get("fixture_id") and isinstance(protocol, list) and protocol and isinstance(opponent_team, dict) and opponent_team:
            candidates.append((len(opponent_team), str(path)))
    if not candidates:
        raise HostedResearchError("canonical evidence has no joint-posterior fixture")
    candidates.sort()
    source = candidates[0][1]
    print(f"fixture={source}")

    node(
        "scripts/sample_joint_random_battle_posterior.cjs",
        str(SHOWDOWN_ROOT),
        source,
        "--target-particles",
        "16",
        "--minimum-particles",
        "8",
        "--max-rounds",
        "65536",
        stdout="/tmp/joint-posterior.json",
        allowed_returncodes=(0, 3),
    )

    from azelficoast.belief.joint_posterior import (
        evaluator_posterior,
        posterior_validity_record,
        validate_joint_posterior,
    )

    document = as_dict(load_json("/tmp/joint-posterior.json"), label="joint posterior")
    checked = validate_joint_posterior(document)
    projected = evaluator_posterior(checked)
    validity = posterior_validity_record(checked)
    write_json("/tmp/joint-posterior-validity.json", validity, pretty=True)
    construction = as_dict(checked["construction"], label="construction")
    assert int(construction["accepted_team_count"]) >= 8
    assert construction["preserves_joint_team_set_correlations"] is True
    worlds = projected["worlds"]
    assert isinstance(worlds, list) and worlds
    assert all(len(as_dict(as_dict(world, label="world")["hidden"], label="hidden")["team"]) == 6 for world in worlds)
    print_json(
        {
            "attempted": construction["attempted_team_count"],
            "accepted": construction["accepted_team_count"],
            "unique": construction["unique_particle_count"],
            "support_ess": as_dict(validity["support"], label="support")["effective_sample_size"],
            "support_entropy_bits": as_dict(validity["support"], label="support")["entropy_bits"],
        }
    )


def live_belief_coverage() -> None:
    process = start_showdown_server()
    try:
        python_module(
            "azelficoast.live.harness",
            "--results",
            "/tmp/results.jsonl",
            "--decisions",
            "/tmp/decisions.jsonl",
            "--replays",
            "/tmp/replays",
            "local",
            "--battles",
            "256",
            "--concurrency",
            "8",
        )
        python_module(
            "azelficoast.live.harness",
            "belief-coverage",
            "/tmp/decisions.jsonl",
            stdout="/tmp/belief-coverage.json",
        )
    finally:
        process.terminate()

    report = as_dict(load_json("/tmp/belief-coverage.json"), label="belief coverage")
    observed = as_dict(report["observed_routing"], label="observed routing")
    admission = as_dict(report["static_admission"], label="static admission")
    assert report["schema"] == "azelficoast.live-belief-coverage"
    assert report["schema_version"] == 1
    assert int(observed["decision_count"]) >= 256
    assert admission["decision_count"] == observed["decision_count"]
    assert int(admission["admitted_decision_count"]) >= 1
    fallback_count = int(admission["fallback_decision_count"])
    fallback_reasons = admission["fallback_reason_counts"]
    assert isinstance(fallback_reasons, list)
    assert fallback_count == (
        int(admission["decision_count"]) - int(admission["admitted_decision_count"])
    )
    assert sum(
        int(as_dict(row, label="fallback reason")["count"])
        for row in fallback_reasons
    ) == fallback_count
    print_json(
        {
            "decision_count": admission["decision_count"],
            "admitted_decision_count": admission["admitted_decision_count"],
            "admission_rate": admission["admission_rate"],
            "top_fallback_reasons": admission["fallback_reason_counts"][:8],
        }
    )


def natural_status_move() -> None:
    source_path = EVIDENCE_ROOT / "status-source.json"
    treatment_path = Path("experiments/natural-status-move-treatment.json")
    source = as_dict(load_json(source_path), label="status source")
    treatment = as_dict(load_json(treatment_path), label="status treatment")
    selection = as_dict(treatment["selection"], label="selection")
    assert source["fixture_id"] == selection["expected_fixture_id"]
    assert source["plausible_items"] == selection["plausible_items"]
    assert str(source["opponent_response_move"]).lower() == "roost"

    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        str(source_path),
        stdout="/tmp/status-oracle.json",
    )
    python_module(
        "azelficoast.research.verification.real_belief_trace",
        "/tmp/status-oracle.json",
        stdout="/tmp/status-trace.json",
    )

    oracle = as_dict(load_json("/tmp/status-oracle.json"), label="status oracle")
    trace = as_dict(load_json("/tmp/status-trace.json"), label="status trace")
    acceptance = as_dict(treatment["acceptance"], label="acceptance")
    expected_fixture = selection["expected_fixture_id"]
    expected_items = selection["plausible_items"]
    assert oracle["source_fixture_id"] == expected_fixture
    assert oracle["showdown_commit"] == treatment["showdown_commit"]
    reconstruction = as_dict(oracle["reconstruction"], label="reconstruction")
    assert reconstruction["observed_opponent_moves"] == ["roost"]
    assert int(reconstruction["hidden_world_count"]) >= int(acceptance["minimum_hidden_worlds"])
    legal_actions = oracle["legal_actions"]
    assert isinstance(legal_actions, list)
    assert len(legal_actions) == acceptance["expected_legal_action_count"]
    mechanics = as_dict(oracle["mechanics"], label="mechanics")
    assert mechanics["opponent_policy"] == {
        "kind": "repeat-last-or-uniform-legal-moves",
        "preferred_move": "roost",
        "voluntary_switches": False,
    }
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    assert {
        as_dict(as_dict(world, label="world")["hidden"], label="hidden")["opponent.active.item"]
        for world in worlds
    } == set(expected_items)
    assert trace["experiment_valid"] is True
    assert trace["source_fixture_id"] == expected_fixture
    assert trace["world_count"] == reconstruction["hidden_world_count"]
    assert trace["legal_action_count"] == acceptance["expected_legal_action_count"]
    assert as_dict(trace["public_belief"], label="public belief")["chosen_action"] in legal_actions
    assert as_dict(trace["determinization"], label="determinization")["chosen_action"] in legal_actions
    if acceptance["require_policy_disagreement"]:
        assert trace["policy_disagreement"] is True
    print_json(
        {
            "fixture_id": expected_fixture,
            "world_count": trace["world_count"],
            "legal_action_count": trace["legal_action_count"],
            "generator_matches": reconstruction["generator_matches"],
            "strategy_fusion_observation_count": trace["strategy_fusion_observation_count"],
            "policy_disagreement": trace["policy_disagreement"],
            "determinization_action": as_dict(trace["determinization"], label="determinization")["chosen_action"],
            "public_belief_action": as_dict(trace["public_belief"], label="public belief")["chosen_action"],
            "determinization_value": as_dict(trace["determinization"], label="determinization")["value"],
            "public_belief_value": as_dict(trace["public_belief"], label="public belief")["value"],
        }
    )


def _public_candidate_source(name: str) -> bool:
    fixture_id = PUBLIC_BELIEF_FIXTURES[name]
    fixture = find_jsonl(
        EVIDENCE_ROOT / "discovery" / "fixtures.jsonl",
        "fixture_id",
        fixture_id,
    )
    candidate = _candidate(fixture_id)
    assert candidate["current_speed_fork"] is True
    assert candidate["showdown_commit"] == showdown_revision()

    latest_tera = None
    p2_active = None
    seen: list[str] = []
    fainted: set[str] = set()
    protocol = fixture["protocol_prefix"]
    assert isinstance(protocol, list)
    for batch in protocol:
        if not isinstance(batch, list):
            continue
        for message in batch:
            if not isinstance(message, list) or len(message) < 2 or message[0] != "":
                continue
            if message[1] == "request" and len(message) >= 3:
                try:
                    request = json.loads(message[2])
                except json.JSONDecodeError:
                    continue
                active = request.get("active") or []
                if active and active[0].get("canTerastallize"):
                    latest_tera = active[0]["canTerastallize"]
            if message[1] in {"switch", "drag"} and str(message[2]).startswith("p2"):
                p2_active = str(message[3]).split(",", 1)[0]
                if p2_active not in seen:
                    seen.append(p2_active)
            if message[1] == "faint" and str(message[2]).startswith("p2") and p2_active:
                fainted.add(p2_active.lower())

    state = as_dict(fixture["state"], label="fixture state")
    current = str(as_dict(state["opponent_active"], label="opponent active")["species"]).lower()
    bench = next(
        (
            species
            for species in reversed(seen)
            if species.lower().replace("-", "") != current.replace("-", "")
            and species.lower() not in fainted
        ),
        None,
    )
    if latest_tera is None or bench is None:
        write_json(
            "/tmp/candidate-status.json",
            {
                "schema": "azelficoast.real-belief-exact-candidate-status",
                "schema_version": 1,
                "fixture_id": fixture_id,
                "supported": False,
                "reason": "missing-public-own-tera" if latest_tera is None else "missing-public-opponent-bench",
            },
        )
        return False

    source = {
        "schema": "azelficoast.real-belief-source-fixture",
        "schema_version": 1,
        "fixture_id": fixture_id,
        "showdown_commit": candidate["showdown_commit"],
        "source_artifact": SOURCE_ARTIFACT,
        "source_projection": "exact frozen speed-fork decision selected before policy evaluation",
        "state": state,
        "protocol_prefix": fixture["protocol_prefix"],
        "control_decisions": fixture.get("control_decisions", []),
        "plausible_items": sorted(as_dict(candidate["item_counts"], label="item counts")),
        "observed_opponent_moves": candidate["revealed_moves"],
        "opponent_response_move": candidate["locked_move"],
        "own_active_tera_type": latest_tera,
        "opponent_bench_species": bench,
    }
    write_json("/tmp/candidate-source.json", source)
    write_json(
        "/tmp/candidate-status.json",
        {
            "schema": "azelficoast.real-belief-exact-candidate-status",
            "schema_version": 1,
            "fixture_id": fixture_id,
            "supported": True,
            "candidate": candidate,
            "inferred": {
                "own_active_tera_type": latest_tera,
                "opponent_bench_species": bench,
            },
        },
    )
    return True


def public_belief_exact(name: str) -> None:
    fixture_id = PUBLIC_BELIEF_FIXTURES[name]
    if not _public_candidate_source(name):
        return
    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        "/tmp/candidate-source.json",
        stdout="/tmp/candidate-oracle.json",
    )
    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        "/tmp/candidate-source.json",
        "--transition-program-only",
        stdout="/tmp/candidate-lazy-program.json",
    )
    pytest(
        "tests/hostile/test_transition_program_hostile.py",
        "-m",
        "hostile_showdown",
        env={
            "AZELFICOAST_HOSTILE_ORACLE_PATH": "/tmp/candidate-oracle.json",
            "AZELFICOAST_HOSTILE_LAZY_PROGRAM_PATH": "/tmp/candidate-lazy-program.json",
        },
    )
    python_module(
        "azelficoast.research.verification.real_belief_trace",
        "/tmp/candidate-oracle.json",
        stdout="/tmp/candidate-trace.json",
    )
    python_module(
        "azelficoast.research.verification.real_belief_miner",
        "/tmp/candidate-oracle.json",
        stdout="/tmp/candidate-mining.json",
    )
    trace = as_dict(load_json("/tmp/candidate-trace.json"), label="candidate trace")
    mining = as_dict(load_json("/tmp/candidate-mining.json"), label="candidate mining")
    assert trace["source_fixture_id"] == fixture_id
    assert trace["experiment_valid"] is True
    assert int(trace["world_count"]) >= 2
    assert int(trace["legal_action_count"]) >= 1
    actions = trace["actions"]
    assert isinstance(actions, list)
    protect = next(as_dict(row, label="action") for row in actions if as_dict(row, label="action")["action"] == "/choose move protect")
    assert int(as_dict(protect["dependency_signature"], label="protect signature")["classes_out"]) < int(trace["world_count"])
    assert mining["ranking_uses_policy_result"] is False
    assert mining["evaluated_count"] == 1
    summary = {
        "schema": "azelficoast.real-belief-exact-candidate-result",
        "schema_version": 1,
        "fixture_id": fixture_id,
        "world_count": trace["world_count"],
        "legal_action_count": trace["legal_action_count"],
        "strategy_fusion_observation_count": trace["strategy_fusion_observation_count"],
        "max_world_aware_choices_per_observation": trace["max_world_aware_choices_per_observation"],
        "policy_disagreement": trace["policy_disagreement"],
        "determinization_action": as_dict(trace["determinization"], label="determinization")["chosen_action"],
        "public_belief_action": as_dict(trace["public_belief"], label="public belief")["chosen_action"],
    }
    write_json("/tmp/candidate-result.json", summary)
    print_json(summary)


def real_belief_decision_trace() -> None:
    source = "experiments/real-belief-source-gliscor-urshifu.json"
    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        source,
        stdout="/tmp/real-belief-transition-oracle.json",
    )
    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        source,
        "--transition-program-only",
        stdout="/tmp/whole-turn-transition-program.json",
    )

    from azelficoast.core.whole_turn_program import verify_whole_turn_program_set

    program = as_dict(load_json("/tmp/whole-turn-transition-program.json"), label="whole-turn program")
    oracle = as_dict(load_json("/tmp/real-belief-transition-oracle.json"), label="real-belief oracle")
    certificate = verify_whole_turn_program_set(program, oracle)
    producer = as_dict(program.get("producer", {}), label="producer")
    assert producer.get("strategy") == "counterfactual-causal-refinement"
    unique = int(producer.get("unique_world_action_executions", -1))
    exhaustive = int(producer.get("exhaustive_world_action_product", -1))
    saved = int(producer.get("saved_world_action_executions", -1))
    assert 0 < unique < exhaustive
    assert saved == exhaustive - unique and saved > 0
    certificate["producer"] = producer
    write_json("/tmp/whole-turn-transition-verification.json", certificate)
    print_json(certificate)

    python_module(
        "azelficoast.research.verification.real_belief_trace",
        "/tmp/real-belief-transition-oracle.json",
        stdout="/tmp/real-belief-decision-trace.json",
    )
    Path("/tmp/real-belief-analysis-status").write_text("0\n", encoding="utf-8")
    python_module(
        "azelficoast.research.verification.real_belief_miner",
        "/tmp/real-belief-transition-oracle.json",
        stdout="/tmp/real-belief-trace-mining.json",
    )

    trace = as_dict(load_json("/tmp/real-belief-decision-trace.json"), label="real-belief trace")
    mining = as_dict(load_json("/tmp/real-belief-trace-mining.json"), label="real-belief mining")
    corpus = as_dict(load_json("experiments/real-belief-negative-corpus.json"), label="negative corpus")
    cases = corpus["cases"]
    assert isinstance(cases, list) and cases
    case = as_dict(cases[0], label="negative case")
    expected = as_dict(case["expected"], label="expected")
    assert oracle["showdown_commit"] == case["showdown_commit"]
    assert oracle["source_fixture_id"] == case["source_fixture_id"]
    reconstruction = as_dict(oracle["reconstruction"], label="reconstruction")
    assert reconstruction["generator_variant_count"] == 11
    assert reconstruction["mechanics_projection_variant_count"] == 2
    assert reconstruction["hidden_world_count"] == expected["world_count"]
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    assert all(
        "opponent.active.moves" in as_dict(as_dict(world, label="world")["hidden"], label="hidden")
        and "opponent.active.tera_type" in as_dict(as_dict(world, label="world")["hidden"], label="hidden")
        for world in worlds
    )
    legal_actions = oracle["legal_actions"]
    transitions = oracle["transitions"]
    assert isinstance(legal_actions, list) and isinstance(transitions, list)
    assert len(legal_actions) == expected["legal_action_count"]
    assert len(transitions) == len(worlds) * int(expected["legal_action_count"])
    assert trace["world_count"] == expected["world_count"]
    assert trace["legal_action_count"] == expected["legal_action_count"]
    actions = trace["actions"]
    assert isinstance(actions, list)
    assert any(
        int(as_dict(as_dict(row, label="action")["dependency_signature"], label="signature")["classes_out"]) < int(trace["world_count"])
        for row in actions
    )
    protect = next(as_dict(row, label="action") for row in actions if as_dict(row, label="action")["action"] == "/choose move protect")
    assert int(as_dict(protect["dependency_signature"], label="signature")["classes_out"]) < int(trace["world_count"])
    assert trace["experiment_valid"] is True
    assert trace["policy_disagreement"] is expected["policy_disagreement"]
    assert as_dict(trace["determinization"], label="determinization")["chosen_action"] == expected["determinization_action"]
    assert as_dict(trace["public_belief"], label="public belief")["chosen_action"] == expected["public_belief_action"]
    assert trace["strategy_fusion_observation_count"] == expected["strategy_fusion_observation_count"]
    assert mining["ranking_uses_policy_result"] is False
    assert mining["evaluated_count"] == 1
    assert mining["strategy_fusion_candidate_count"] == 0
    assert mining["disagreement_count"] == 0
    assert mining["first_strategy_fusion_candidate"] is None
    assert mining["first_disagreement"] is None
    ranked = mining["ranked_candidates"]
    assert isinstance(ranked, list) and as_dict(ranked[0], label="ranked candidate")["source_fixture_id"] == case["source_fixture_id"]


def real_belief_survival_witness() -> None:
    source_path = EVIDENCE_ROOT / "survival-source.json"
    source = as_dict(load_json(source_path), label="survival source")
    assert source["fixture_id"] == SURVIVAL_FIXTURE
    state = as_dict(source["state"], label="survival state")
    assert state["turn"] == 42
    assert as_dict(state["active"], label="active")["species"] == "tropius"
    assert as_dict(state["opponent_active"], label="opponent active")["species"] == "chiyu"
    assert len(state["legal_actions"]) == 11
    assert source["own_active_tera_type"] == "Steel"
    assert source["opponent_bench_species"] == "Hippowdon"

    node(
        "scripts/probe_real_belief_trace.cjs",
        str(SHOWDOWN_ROOT),
        str(source_path),
        stdout="/tmp/tropius-chiyu-oracle.json",
    )
    python_module(
        "azelficoast.research.verification.real_belief_trace",
        "/tmp/tropius-chiyu-oracle.json",
        stdout="/tmp/tropius-chiyu-trace.json",
    )
    python_module(
        "azelficoast.research.verification.real_belief_miner",
        "/tmp/tropius-chiyu-oracle.json",
        stdout="/tmp/tropius-chiyu-mining.json",
    )
    trace = as_dict(load_json("/tmp/tropius-chiyu-trace.json"), label="survival trace")
    mining = as_dict(load_json("/tmp/tropius-chiyu-mining.json"), label="survival mining")
    assert trace["source_fixture_id"] == SURVIVAL_FIXTURE
    assert trace["showdown_commit"] == showdown_revision()
    assert trace["experiment_valid"] is True
    assert int(trace["world_count"]) >= 2
    assert trace["legal_action_count"] == 11
    actions = trace["actions"]
    assert isinstance(actions, list)
    protect = next(as_dict(row, label="action") for row in actions if as_dict(row, label="action")["action"] == "/choose move protect")
    assert int(as_dict(protect["dependency_signature"], label="signature")["classes_out"]) < int(trace["world_count"])
    assert mining["ranking_uses_policy_result"] is False
    assert mining["evaluated_count"] == 1


def status_move_prior() -> None:
    python_module(
        "azelficoast.belief.status_move_prior",
        "experiments/status-move-prior-selection.json",
        "--showdown-root",
        str(SHOWDOWN_ROOT),
        stdout="/tmp/status-move-prior-results.json",
    )
    result = as_dict(load_json("/tmp/status-move-prior-results.json"), label="status prior")
    assert result["schema"] == "azelficoast.status-move-prior-results"
    assert result["schema_version"] == 1
    assert result["case_count"] == 6
    cases = result["cases"]
    assert isinstance(cases, list)
    assert all(
        as_dict(as_dict(case, label="case")["left"], label="left")["seed_offset"]
        != as_dict(as_dict(case, label="case")["right"], label="right")["seed_offset"]
        for case in cases
    )
    assert all(
        min(
            int(as_dict(as_dict(case, label="case")["left"], label="left")["matched"]),
            int(as_dict(as_dict(case, label="case")["right"], label="right")["matched"]),
        ) > 0
        for case in cases
    )
    print_json(
        {
            "passed": result["passed"],
            "passed_case_count": result["passed_case_count"],
            "cases": [
                {
                    "move": as_dict(case, label="case")["move"],
                    "species": as_dict(case, label="case")["species"],
                    "matched": [
                        as_dict(as_dict(case, label="case")["left"], label="left")["matched"],
                        as_dict(as_dict(case, label="case")["right"], label="right")["matched"],
                    ],
                    "item_support": as_dict(case, label="case")["item_support"],
                    "support_stable": as_dict(case, label="case")["support_stable"],
                    "total_variation": as_dict(case, label="case")["total_variation"],
                    "passed": as_dict(case, label="case")["passed"],
                }
                for case in cases
            ],
        }
    )
