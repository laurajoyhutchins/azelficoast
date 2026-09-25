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
