from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "experiments" / "evidence" / "canonical-evidence.json"
)
DEFAULT_DESTINATION = Path("/tmp/azelficoast-evidence")


class EvidenceBundleError(RuntimeError):
    """Raised when canonical repository evidence cannot be admitted."""


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceBundleError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bundle_path(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[Path, str]:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceBundleError(
            f"cannot read canonical evidence manifest: {error}"
        ) from error

    root = _object(document, label="canonical evidence manifest")
    if root.get("schema") != "azelficoast.canonical-research-evidence":
        raise EvidenceBundleError("unsupported canonical evidence schema")
    bundle = _object(root.get("bundle"), label="canonical evidence bundle")
    relative_path = bundle.get("path")
    expected = bundle.get("sha256")
    if not isinstance(relative_path, str) or not relative_path:
        raise EvidenceBundleError("canonical evidence bundle path is invalid")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise EvidenceBundleError("canonical evidence SHA-256 is invalid")

    path = repository_root / relative_path
    try:
        path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise EvidenceBundleError(
            "canonical evidence bundle escapes the repository"
        ) from error
    return path, expected


def verify_canonical_evidence(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    bundle_path, expected = canonical_bundle_path(
        manifest_path,
        repository_root=repository_root,
    )
    if not bundle_path.is_file():
        raise EvidenceBundleError(
            f"canonical evidence bundle is missing: {bundle_path}"
        )
    actual = _sha256(bundle_path)
    if actual != expected:
        raise EvidenceBundleError(
            f"canonical evidence digest mismatch: expected {expected}, got {actual}"
        )
    return bundle_path


def _safe_members(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, ...]:
    members = tuple(archive.getmembers())
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise EvidenceBundleError(
                f"canonical evidence contains unsafe path {member.name!r}"
            )
        if not (member.isdir() or member.isfile()):
            raise EvidenceBundleError(
                f"canonical evidence contains unsupported entry {member.name!r}"
            )
    return members


def unpack_canonical_evidence(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    destination: Path = DEFAULT_DESTINATION,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    bundle_path = verify_canonical_evidence(
        manifest_path,
        repository_root=repository_root,
    )
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = _safe_members(archive)
            for member in members:
                archive.extract(member, path=destination)
    except (tarfile.TarError, OSError) as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise EvidenceBundleError(
            f"cannot extract canonical evidence: {error}"
        ) from error
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and unpack Git-authoritative research evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    unpack = commands.add_parser("unpack")
    unpack.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        print(verify_canonical_evidence())
    else:
        print(unpack_canonical_evidence(destination=args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
