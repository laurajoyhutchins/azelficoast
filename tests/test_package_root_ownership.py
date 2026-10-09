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


ALLOWED_RESEARCH_ROOT_MODULES = {
    "__init__.py",
    "adaptive_execution.py",
    "ci.py",
    "decision_contracts.py",
    "evidence.py",
    "experiment_contracts.py",
    "matched_contracts.py",
    "posterior_evidence.py",
    "posterior_population_contract.py",
    "posterior_population_selection.py",
    "public_replays.py",
    "training_records.py",
    "training_service.py",
}


def test_research_root_contains_only_shared_infrastructure() -> None:
    modules = {path.name for path in RESEARCH_ROOT.glob("*.py")}
    assert modules == ALLOWED_RESEARCH_ROOT_MODULES, (
        "Move retained scientific studies to azelficoast.research.studies: "
        + ", ".join(sorted(modules ^ ALLOWED_RESEARCH_ROOT_MODULES))
    )


def test_research_experiments_have_an_explicit_owner() -> None:
    modules = sorted(path.name for path in RESEARCH_ROOT.glob("*_experiment.py"))
    assert not modules, (
        "Move executable experiment modules into azelficoast.research.experiments: "
        + ", ".join(modules)
    )


REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
SHOWDOWN_ROOT = REPOSITORY_ROOT / "showdown"


def test_repository_has_no_generic_scripts_owner() -> None:
    assert not (REPOSITORY_ROOT / "scripts").exists()


def test_showdown_code_has_explicit_owners() -> None:
    owners = {path.name for path in SHOWDOWN_ROOT.iterdir() if path.is_dir()}
    assert owners == {"research", "runtime", "shared", "verification"}
