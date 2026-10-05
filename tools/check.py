from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> None:
    print(f"==> {label}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    run("Python type check", ["uv", "run", "mypy"])
    run("JavaScript lint", ["npm", "run", "lint:showdown"])
    run("JavaScript type check", ["npm", "run", "typecheck:showdown"])

    scripts = sorted((ROOT / "showdown").rglob("*.cjs"))
    if not scripts:
        raise SystemExit("no CJS scripts found")
    for script in scripts:
        run(
            f"JavaScript syntax: {script.relative_to(ROOT).as_posix()}",
            ["node", "--check", str(script)],
        )

    run(
        "Canonical research evidence",
        ["uv", "run", "python", "-m", "azelficoast.research.evidence", "unpack"],
    )
    run("Python lint", ["uv", "run", "ruff", "check", "."])
    run("Tests", ["uv", "run", "pytest"])
    run("Fast hostile correctness gate", ["uv", "run", "pytest", "tests/hostile", "-m", "hostile_fast"])
    run(
        "Imperfect-information reference experiment",
        ["uv", "run", "python", "-m", "azelficoast.search.imperfect_information"],
    )
    run(
        "Pokemon-shaped counterexample",
        [
            "uv",
            "run",
            "python",
            "-m",
            "azelficoast.research.verification.pokemon_counterexample",
        ],
    )
    run(
        "Simulator dependency experiment",
        ["uv", "run", "python", "-m", "azelficoast.research.experiments.simulator_experiment"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
