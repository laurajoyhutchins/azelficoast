from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "azelficoast"


def test_package_root_contains_no_unowned_modules() -> None:
    modules = sorted(
        path.name
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name != "__init__.py"
    )
    assert not modules, (
        "Move package-root modules into core, belief, search, research, or live: "
        + ", ".join(modules)
    )


RESEARCH_ROOT = PACKAGE_ROOT / "research"


def test_research_experiments_have_an_explicit_owner() -> None:
    modules = sorted(path.name for path in RESEARCH_ROOT.glob("*_experiment.py"))
    assert not modules, (
        "Move executable experiment modules into azelficoast.research.experiments: "
        + ", ".join(modules)
    )


ALLOWED_RESEARCH_ROOT_MODULES = {
    "__init__.py",
    "adaptive_execution.py",
    "ci.py",
    "contracts.py",
    "public_replays.py",
    "training_records.py",
    "training_service.py",
}


def test_research_root_contains_only_shared_infrastructure() -> None:
    modules = {path.name for path in RESEARCH_ROOT.glob("*.py")}
    assert modules == ALLOWED_RESEARCH_ROOT_MODULES, (
        "Research package-root modules must be explicitly shared infrastructure; "
        f"unexpected={sorted(modules - ALLOWED_RESEARCH_ROOT_MODULES)}, "
        f"missing={sorted(ALLOWED_RESEARCH_ROOT_MODULES - modules)}"
    )
