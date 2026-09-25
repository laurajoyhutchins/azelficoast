from __future__ import annotations

import json
from pathlib import Path

import pytest

from azelficoast import public_replays
from azelficoast.public_replays import (
    PublicReplay,
    PublicReplayError,
    _input_choices,
    _input_players,
    _recorded_action,
    discover_public_replays,
    fetch_public_replay,
    freeze_public_replay,
)


def test_discovery_uses_51st_row_only_as_pagination_signal(monkeypatch) -> None:
    first = [
        {
            "id": f"gen9randombattle-{index}",
            "rating": 1500 if index % 2 == 0 else 1100,
            "uploadtime": 10_000 - index,
        }
        for index in range(51)
    ]
    second = [
        {
            "id": f"gen9randombattle-next-{index}",
            "rating": 1600,
            "uploadtime": 9_000 - index,
        }
        for index in range(3)
    ]
    urls: list[str] = []

    def fake_get(url: str):
        urls.append(url)
        return first if len(urls) == 1 else second

    monkeypatch.setattr(public_replays, "_get_json", fake_get)
    rows = discover_public_replays(max_battles=30, min_rating=1200)

    assert len(rows) == 28
    assert all(int(row["rating"]) >= 1200 for row in rows)
    assert len(urls) == 2
    assert "before=9950" in urls[1]
    assert "gen9randombattle-50" not in {row["id"] for row in rows}


def test_fetch_replay_requires_reconstructible_random_battle_inputlog(monkeypatch) -> None:
    monkeypatch.setattr(
        public_replays,
        "_get_json",
        lambda _url: {
            "id": "gen9randombattle-1",
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
        },
    )
    with pytest.raises(PublicReplayError, match="inputlog"):
        fetch_public_replay(
            {"id": "gen9randombattle-1", "rating": 1500, "uploadtime": 1}
        )


def test_fetch_replay_preserves_source_revision_for_later_reconstruction(
    monkeypatch,
) -> None:
    source_revision = "d" * 40
    monkeypatch.setattr(
        public_replays,
        "_get_json",
        lambda _url: {
            "id": "gen9randombattle-1",
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
            "inputlog": f">version {source_revision}\n>start {}",
        },
    )
    replay = fetch_public_replay(
        {"id": "gen9randombattle-1", "rating": "1500", "uploadtime": 1}
    )

    assert replay.source_showdown_version == source_revision
    assert replay.rating == 1500


def test_input_players_preserve_side_specific_ratings() -> None:
    inputlog = "\n".join(
        (
            '>player p1 {"name":"Alice","rating":1801}',
            '>player p2 {"name":"Bob","rating":"1664"}',
        )
    )

    assert _input_players(inputlog) == {
        "p1": {"name": "Alice", "rating": 1801},
        "p2": {"name": "Bob", "rating": 1664},
    }


def test_input_choices_remove_cancelled_choice_on_undo() -> None:
    inputlog = "\n".join(
        (
            ">p1 move 1",
            ">p1 undo",
            ">p1 move 2",
            ">p2 move 1",
            ">p1 switch 3",
        )
    )

    assert _input_choices(inputlog, "p1") == ["move 2", "switch 3"]


def test_input_choices_reject_orphan_undo() -> None:
    with pytest.raises(PublicReplayError, match="undo has no prior choice"):
        _input_choices(">p1 undo", "p1")


def test_freeze_public_replay_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    replay = PublicReplay(
        replay_id="gen9randombattle-1",
        payload={
            "id": "gen9randombattle-1",
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
            "inputlog": (
                ">version "
                + public_replays.PINNED_SHOWDOWN_COMMIT
                + "\n>start {}"
            ),
        },
        rating=1500,
        uploadtime=1,
    )
    first = freeze_public_replay(replay, tmp_path)
    second = freeze_public_replay(replay, tmp_path)

    assert first["raw_sha256"] == second["raw_sha256"]
    frozen = tmp_path / "raw" / "gen9randombattle-1.json"
    assert json.loads(frozen.read_text(encoding="utf-8"))["id"] == replay.replay_id


def test_recorded_numeric_move_maps_to_exact_legal_action() -> None:
    request = {
        "active": [
            {
                "moves": [
                    {"id": "earthquake", "move": "Earthquake"},
                    {"id": "protect", "move": "Protect"},
                ]
            }
        ],
        "side": {"pokemon": []},
    }
    legal = [
        "/choose move earthquake",
        "/choose move protect",
        "/choose move earthquake terastallize",
    ]

    assert _recorded_action("move 1", request, legal) == "/choose move earthquake"
    assert (
        _recorded_action("move 1 terastallize", request, legal)
        == "/choose move earthquake terastallize"
    )


def test_recorded_numeric_switch_maps_through_request_identity() -> None:
    request = {
        "forceSwitch": [True],
        "side": {
            "pokemon": [
                {"ident": "p1: Rotom-Wash"},
                {"ident": "p1: Great Tusk"},
            ]
        },
    }
    legal = ["/choose switch Rotom-Wash", "/choose switch Great Tusk"]

    assert _recorded_action("switch 2", request, legal) == "/choose switch Great Tusk"


def test_public_import_freezes_revision_mismatch_for_later_reconstruction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay = PublicReplay(
        replay_id="gen9randombattle-old",
        payload={
            "id": "gen9randombattle-old",
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
            "inputlog": f">version {'d' * 40}\n>start {}",
        },
        rating=1700,
        uploadtime=1,
    )
    metadata = [{"id": replay.replay_id, "rating": 1700, "uploadtime": 1}]
    monkeypatch.setattr(
        public_replays,
        "_showdown_revision",
        lambda _root: public_replays._input_version(
            f">version {'a' * 40}"
        ),
    )
    monkeypatch.setattr(public_replays, "discover_public_replays", lambda **_kwargs: metadata)
    monkeypatch.setattr(public_replays, "fetch_public_replay", lambda _metadata: replay)

    result = public_replays.import_public_replays(
        showdown_root=tmp_path / "showdown",
        output_root=tmp_path / "corpus",
        max_battles=1,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert result["admitted_count"] == 0
    assert result["excluded_count"] == 1
    assert manifest["excluded"][0]["reason"] == "PublicReplayRevisionMismatch"
    assert manifest["excluded"][0]["source_showdown_version"] == "d" * 40
    assert Path(manifest["excluded"][0]["raw_path"]).is_file()


def test_public_import_reuses_frozen_raw_replay(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay_id = "gen9randombattle-cached"
    revision = "a" * 40
    root = tmp_path / "corpus"
    replay = PublicReplay(
        replay_id=replay_id,
        payload={
            "id": replay_id,
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
            "inputlog": f">version {revision}\n>start {}",
        },
        rating=1700,
        uploadtime=1,
    )
    public_replays.freeze_public_replay(replay, root)
    metadata = [{"id": replay_id, "rating": "1700", "uploadtime": 1}]

    monkeypatch.setattr(public_replays, "_showdown_revision", lambda _root: revision)
    monkeypatch.setattr(public_replays, "discover_public_replays", lambda **_kwargs: metadata)
    monkeypatch.setattr(
        public_replays,
        "fetch_public_replay",
        lambda _metadata: (_ for _ in ()).throw(AssertionError("network fetch should be skipped")),
    )

    async def fake_reconstruct(_replay, *, showdown_root):
        return []

    monkeypatch.setattr(public_replays, "_reconstruct_replay_trace", fake_reconstruct)

    result = public_replays.import_public_replays(
        showdown_root=tmp_path / "showdown",
        output_root=root,
        max_battles=1,
    )

    assert result["admitted_count"] == 1
    assert result["excluded_count"] == 0


def test_public_import_manifest_marks_identifiers_and_human_actions_non_authoritative(
    monkeypatch,
    tmp_path: Path,
) -> None:
    replay = PublicReplay(
        replay_id="gen9randombattle-1",
        payload={
            "id": "gen9randombattle-1",
            "format": "[Gen 9] Random Battle",
            "log": "|win|Alice",
            "inputlog": (
                ">version "
                + public_replays.PINNED_SHOWDOWN_COMMIT
                + "\n>start {}"
            ),
        },
        rating=1500,
        uploadtime=1,
    )
    metadata = [{"id": replay.replay_id, "rating": 1500, "uploadtime": 1}]
    rows = [
        {
            "schema": "azelficoast.decision-trace",
            "schema_version": 1,
            "run_id": "public-replay:gen9randombattle-1:p1",
            "event_index": 0,
            "observed_at": "1970-01-01T00:00:01+00:00",
            "kind": "decision",
            "battle_tag": "battle-gen9randombattle-1",
            "decision_index": 0,
            "state": {"legal_actions": ["/choose move earthquake"]},
            "chosen_action": "/choose move earthquake",
            "decision_metadata": {
                "selected_policy": "recorded-human",
                "training_policy_authority": False,
            },
        }
    ]

    monkeypatch.setattr(
        public_replays, "_showdown_revision", lambda _root: public_replays.PINNED_SHOWDOWN_COMMIT
    )
    monkeypatch.setattr(public_replays, "discover_public_replays", lambda **_kwargs: metadata)
    monkeypatch.setattr(public_replays, "fetch_public_replay", lambda _metadata: replay)

    async def fake_reconstruct(_replay, *, showdown_root):
        return rows

    monkeypatch.setattr(public_replays, "_reconstruct_replay_trace", fake_reconstruct)

    result = public_replays.import_public_replays(
        showdown_root=tmp_path / "showdown",
        output_root=tmp_path / "corpus",
        max_battles=1,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    trace = [
        json.loads(line)
        for line in Path(result["trace"]).read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["provenance"]["contains_user_identifiers"] is True
    assert manifest["provenance"]["license_or_terms"] == "not asserted"
    assert trace[0]["decision_metadata"]["training_policy_authority"] is False
