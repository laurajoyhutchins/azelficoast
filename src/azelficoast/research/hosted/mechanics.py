from __future__ import annotations

from azelficoast.research.hosted.common import (
    EVIDENCE_ROOT,
    SHOWDOWN_ROOT,
    HostedResearchError,
    as_dict,
    checkout_showdown_in_place,
    find_jsonl,
    load_json,
    node,
    print_json,
    python_module,
    python_script,
    start_showdown_server,
    write_json,
)


TEAM_COMPLETION_FIXTURES = {
    "turn12": "6314b9a24e28265bb3b36c490c0297357ece57e7963ac653c3d798f4d3b82821",
    "turn13": "ce6f0239a59d5d64bd303bce884ea063f9c1305bcbe6b2b55b18c51b349d5b46",
}
URSHIFU_PROTECT_FIXTURE = "0f5fd1d00371410b75e722768b9bbddd0fdacd24725e6a60c142bcc87783c92a"
TYPHLOSION_PROTECT_FIXTURE = "90322c79fa3101b1dc66bcd7554888d37b1e1d9da112e09746e2082278525664"
HISTORICAL_REPLAY_SHOWDOWN = "6397bfddb3db4e916dd792e03c43355f7366e8ab"


def opponent_team_completion_support() -> None:
    for name, fixture_id in TEAM_COMPLETION_FIXTURES.items():
        row = find_jsonl(
            EVIDENCE_ROOT / "discovery" / "fixtures.jsonl",
            "fixture_id",
            fixture_id,
        )
        write_json(f"/tmp/{name}.json", row)

    for name in ("turn12", "turn13"):
        node(
            "scripts/enumerate_team_completion_support.cjs",
            str(SHOWDOWN_ROOT),
            f"/tmp/{name}.json",
            stdout=f"/tmp/{name}-support.json",
        )

    first = as_dict(load_json("/tmp/turn12-support.json"), label="turn12 support")
    second = as_dict(load_json("/tmp/turn13-support.json"), label="turn13 support")
    expected_known = sorted(
        ["camerupt", "clawitzer", "hitmonchan", "ironthorns", "staraptor"]
    )
    for result in (first, second):
        assert result["public_team_size"] == 6
        assert result["unknown_slots"] == 1
        assert result["known_species"] == expected_known
        assert int(result["support_count"]) > 0
        assert int(result["support_count"]) < int(result["random_set_species_count"])
        assert "support-is-not-a-probability-distribution" in result["caveats"]
    assert first["support_set_sha256"] == second["support_set_sha256"]
    first_support = first["support"]
    second_support = second["support"]
    assert isinstance(first_support, list) and isinstance(second_support, list)
    assert [
        as_dict(row, label="support row")["species_id"] for row in first_support
    ] == [
        as_dict(row, label="support row")["species_id"] for row in second_support
    ]
    print_json(
        {
            "support_count": first["support_count"],
            "random_set_species_count": first["random_set_species_count"],
            "support_set_sha256": first["support_set_sha256"],
            "rejection_counts": first["rejection_counts"],
        }
    )


def policy_boundary_refinement() -> None:
    inputs = [
        str(EVIDENCE_ROOT / "policy-boundary" / f"case-{index:02d}.json")
        for index in range(1, 10)
    ]
    python_script(
        "scripts/policy_boundary_refinement_experiment.py",
        *inputs,
        stdout="/tmp/policy-boundary-refinement-results.json",
    )
    actual = load_json("/tmp/policy-boundary-refinement-results.json")
    expected = load_json("experiments/policy-boundary-refinement-results.json")
    assert actual == expected
    result = as_dict(actual, label="policy boundary result")
    assert result["case_count"] == 9
    assert result["certification_errors"] == 0
    assert result["headline_pass"] is False
    acceptance = as_dict(result["acceptance"], label="acceptance")
    assert float(result["median_fraction"]) > float(acceptance["median_fraction_at_most"])
    assert float(result["p90_fraction"]) > float(acceptance["p90_fraction_at_most"])


def protect_action_survival() -> None:
    python_module(
        "azelficoast.research.action_survival",
        "experiments/protect-action-survival.json",
        stdout="/tmp/protect-action-survival.json",
    )
    result = as_dict(
        load_json("/tmp/protect-action-survival.json"),
        label="protect action survival",
    )
    assert result["passed"] is True
    damage = as_dict(result["damage"], label="damage")
    assert damage["Choice Scarf"] == {"min": 153, "max": 180}
    assert damage["Choice Specs"] == {"min": 229, "max": 270}
    assert result["defender_full_hp"] == 229
    assert result["full_hp_survival_split"] is True
    assert result["observed_hp_survival_split"] is True
    assert result["negative_control_collapses_full_hp_split"] is True


def protect_continuation() -> None:
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
            "512",
            "--concurrency",
            "8",
        )
    finally:
        process.terminate()

    python_module(
        "azelficoast.live.harness",
        "corpus",
        "build",
        "/tmp/decisions.jsonl",
        "--output",
        "/tmp/corpus.jsonl",
        stdout="/tmp/corpus-summary.json",
    )
    python_module(
        "azelficoast.research.protect_continuations",
        "/tmp/corpus.jsonl",
        "--showdown-root",
        str(SHOWDOWN_ROOT),
        "--rounds",
        "2048",
        stdout="/tmp/protect-candidates.json",
    )
    result = as_dict(load_json("/tmp/protect-candidates.json"), label="protect candidates")
    assert result["schema"] == "azelficoast.protect-continuation-candidates"
    assert int(result["source_fixture_count"]) > 0
    assert int(result["protect_fixture_count"]) > 0
    assert result["persistent_only"] is True
    assert result["candidate_count"] == result["persistent_candidate_count"]
    print_json(
        {
            "protect_fixture_count": result["protect_fixture_count"],
            "candidate_count": result["candidate_count"],
            "current_speed_fork_count": result["current_speed_fork_count"],
            "sampled_world_queries": result["sampled_world_queries"],
        }
    )


def protect_speed_fork() -> None:
    node(
        "scripts/probe_protect_speed_forks.cjs",
        str(SHOWDOWN_ROOT),
        "experiments/protect-speed-forks.json",
        stdout="/tmp/protect-speed-fork-mechanics.json",
    )
    result = as_dict(
        load_json("/tmp/protect-speed-fork-mechanics.json"),
        label="protect speed fork",
    )
    cases = result["cases"]
    if not isinstance(cases, list):
        raise HostedResearchError("protect speed fork result lacks cases")
    by_fixture = {
        str(as_dict(candidate, label="candidate")["fixture_id"]): as_dict(
            candidate, label="candidate"
        )
        for candidate in cases
    }
    urshifu = by_fixture.get(URSHIFU_PROTECT_FIXTURE)
    typhlosion = by_fixture.get(TYPHLOSION_PROTECT_FIXTURE)
    if urshifu is None or typhlosion is None:
        raise HostedResearchError("missing frozen Protect fixture")

    urshifu_worlds = urshifu["worlds"]
    typhlosion_worlds = typhlosion["worlds"]
    assert isinstance(urshifu_worlds, list) and isinstance(typhlosion_worlds, list)
    assert not any(
        as_dict(as_dict(world, label="world")["protect"], label="protect")[
            "blocked_without_damage"
        ]
        for world in urshifu_worlds
    )
    assert all(
        as_dict(as_dict(world, label="world")["protect"], label="protect")[
            "blocked_without_damage"
        ]
        for world in typhlosion_worlds
    )
    urshifu_orders = {
        (float(as_dict(world, label="world")["own_speed"]) > float(as_dict(world, label="world")["opponent_speed"]))
        - (float(as_dict(world, label="world")["own_speed"]) < float(as_dict(world, label="world")["opponent_speed"]))
        for world in urshifu_worlds
    }
    typhlosion_orders = {
        (float(as_dict(world, label="world")["own_speed"]) > float(as_dict(world, label="world")["opponent_speed"]))
        - (float(as_dict(world, label="world")["own_speed"]) < float(as_dict(world, label="world")["opponent_speed"]))
        for world in typhlosion_worlds
    }
    assert len(urshifu_orders) == 2
    assert len(typhlosion_orders) == 2
    print_json({"urshifu": urshifu_worlds, "typhlosion": typhlosion_worlds})


def replay_world() -> None:
    checkout_showdown_in_place(HISTORICAL_REPLAY_SHOWDOWN, build=False)
    node(
        "scripts/sample_showdown_worlds.cjs",
        str(SHOWDOWN_ROOT),
        "infernape",
        "closecombat",
        "65536",
        stdout="/tmp/infernape-closecombat.json",
    )
    python_module(
        "azelficoast.research.verification.replay_worlds",
        "--world-sample",
        "/tmp/infernape-closecombat.json",
        stdout="/tmp/infernape-belief.json",
    )
    result = as_dict(load_json("/tmp/infernape-belief.json"), label="replay belief")
    damage = as_dict(result["damage_observation"], label="damage observation")
    assert damage["damage"] == 54
    assert damage["target_tera_type"] == "Flying"
    assert result["generator_prior"] == {
        "Choice Band": 16086 / 65536,
        "Choice Scarf": 7944 / 65536,
        "Life Orb": 41506 / 65536,
    }
    assert result["public_history_updates"] == [
        {
            "turn": 18,
            "kind": "life-orb-recoil",
            "observed": False,
            "authority": "complete-public-turn",
            "remaining_items": ["Choice Band", "Choice Scarf"],
        }
    ]
    assert result["prior"] == {
        "Choice Band": 16086 / (16086 + 7944),
        "Choice Scarf": 7944 / (16086 + 7944),
    }
    compatible = as_dict(result["compatible_worlds"], label="compatible worlds")
    assert as_dict(compatible["Choice Scarf"], label="Choice Scarf worlds")[
        "matching_rolls"
    ] == 3
    assert result["posterior"] == {"Choice Scarf": 1.0}
    assert result["showdown_commit"] == HISTORICAL_REPLAY_SHOWDOWN
    assert as_dict(result["sample"], label="sample")["showdown_commit"] == result["showdown_commit"]
    assert len(str(result["belief_sha256"])) == 64

    node(
        "scripts/sample_showdown_worlds.cjs",
        str(SHOWDOWN_ROOT),
        "infernape",
        "closecombat,flareblitz",
        "65536",
        stdout="/tmp/infernape-closecombat-flareblitz.json",
    )
    sample = as_dict(
        load_json("/tmp/infernape-closecombat-flareblitz.json"),
        label="later move sample",
    )
    assert int(sample["matched"]) > 0
    assert set(as_dict(sample["item_counts"], label="item counts")) == {
        "Choice Band",
        "Choice Scarf",
        "Life Orb",
    }
