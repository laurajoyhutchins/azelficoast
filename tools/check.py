from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> None:
    print(f"==> {label}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def run_static() -> None:
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


def run_tests() -> None:
    run(
        "Canonical research evidence",
        ["uv", "run", "python", "-m", "azelficoast.research.evidence", "unpack"],
    )
    run("Python lint", ["uv", "run", "ruff", "check", "."])
    run("Tests", ["uv", "run", "pytest"])
    run(
        "Fast hostile correctness gate",
        ["uv", "run", "pytest", "tests/hostile", "-m", "hostile_fast"],
    )
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Azelficoast repository verification.")
    parser.add_argument("group", nargs="?", choices=("all", "static", "test"), default="all")
    args = parser.parse_args()

    if args.group in {"all", "static"}:
        run_static()
    if args.group in {"all", "test"}:
        run_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
