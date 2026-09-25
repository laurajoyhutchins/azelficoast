from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REMOVED_PATHS = (
    ROOT / "src/azelficoast/transition_oracle.py",
    ROOT / "src/azelficoast/search/transition_program.py",
    ROOT / "src/azelficoast/search/decision_relevance.py",
)

BANNED_TEXT = (
    "azelficoast.real-belief-transition-oracle",
    "azelficoast.whole-turn-transition-program-set",
    "azelficoast.weighted-whole-turn-outcomes",
    "azelficoast.whole-turn-program-verification",
    "azelficoast.decision-relevance-certificate",
    "azelficoast.transition-program-search",
    "azelficoast.partial-information-search",
    "azelficoast.transition-execution-certificate",
    "azelficoast.transition_oracle",
    "azelficoast.search.transition_program",
    "azelficoast.search.decision_relevance",
    "validate_oracle_core",
)

SCANNED_SUFFIXES = {".py", ".cjs", ".json", ".yml", ".yaml", ".md"}
SCANNED_ROOTS = (
    ROOT / "src",
    ROOT / "tests",
    ROOT / "scripts",
    ROOT / ".github",
    ROOT / "experiments",
    ROOT / "docs",
)


def test_removed_schema_shims_stay_removed() -> None:
    assert not [path for path in REMOVED_PATHS if path.exists()]


def test_legacy_core_schema_names_and_imports_do_not_reappear() -> None:
    violations: list[str] = []
    this_file = Path(__file__).resolve()

    for root in SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path == this_file
                or not path.is_file()
                or path.suffix not in SCANNED_SUFFIXES
            ):
                continue
            text = path.read_text(encoding="utf-8")
            for banned in BANNED_TEXT:
                if banned in text:
                    violations.append(
                        f"{path.relative_to(ROOT)} contains legacy contract {banned!r}"
                    )

    assert not violations, "\n".join(violations)
