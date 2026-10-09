"""Study resources remain anchored to repository-owned Showdown authority."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from azelficoast.core.showdown import SHOWDOWN_REVISION_PATH
from azelficoast.research.studies.natural_disagreements import _sample_worlds


def test_world_sampler_resolves_script_from_pinned_showdown_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(
        "azelficoast.research.studies.natural_disagreements.subprocess.run",
        fake_run,
    )
    result = _sample_worlds(
        showdown_root=tmp_path,
        species="crabominable",
        observed_moves=("Earthquake",),
        rounds=4,
        is_lead=False,
        public_level=90,
        public_ability="",
    )

    script = (
        SHOWDOWN_REVISION_PATH.parent / "research" / "belief" / "sample_showdown_worlds.cjs"
    )
    assert script.is_file()
    assert commands == [
        ["node", str(script), str(tmp_path), "crabominable", "Earthquake", "4", "false", "0", "90", ""]
    ]
    assert result == {}
