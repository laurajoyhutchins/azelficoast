from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_joint_random_battle_sampler_has_valid_node_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable")

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sample_joint_random_battle_posterior.cjs"
    )
    completed = subprocess.run(
        [node, "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
