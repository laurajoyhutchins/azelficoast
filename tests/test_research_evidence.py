from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from azelficoast.research.evidence import (
    EvidenceBundleError,
    unpack_canonical_evidence,
    verify_canonical_evidence,
)


def _fixture_bundle(
    root: Path,
    members: dict[str, bytes],
) -> tuple[Path, Path]:
    evidence = root / "experiments" / "evidence"
    evidence.mkdir(parents=True)
    bundle = evidence / "canonical-evidence.tar.gz"
    with tarfile.open(bundle, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest = evidence / "canonical-evidence.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "azelficoast.canonical-research-evidence",
                "bundle": {
                    "path": "experiments/evidence/canonical-evidence.tar.gz",
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, bundle


def test_canonical_evidence_verifies_and_extracts(tmp_path: Path) -> None:
    manifest, bundle = _fixture_bundle(
        tmp_path,
        {"population/manifest.json": b'{"selected_count": 1}\n'},
    )
    destination = tmp_path / "unpacked"

    assert verify_canonical_evidence(
        manifest,
        repository_root=tmp_path,
    ) == bundle
    assert unpack_canonical_evidence(
        manifest_path=manifest,
        destination=destination,
        repository_root=tmp_path,
    ) == destination
    assert (destination / "population" / "manifest.json").read_text(
        encoding="utf-8"
    ) == '{"selected_count": 1}\n'


def test_canonical_evidence_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest, bundle = _fixture_bundle(
        tmp_path,
        {"fixture.json": b"{}\n"},
    )
    bundle.write_bytes(bundle.read_bytes() + b"corrupt")

    with pytest.raises(EvidenceBundleError, match="digest mismatch"):
        verify_canonical_evidence(
            manifest,
            repository_root=tmp_path,
        )


def test_canonical_evidence_rejects_archive_escape(tmp_path: Path) -> None:
    manifest, _ = _fixture_bundle(
        tmp_path,
        {"../escape.json": b"{}\n"},
    )
    destination = tmp_path / "unpacked"

    with pytest.raises(EvidenceBundleError, match="unsafe path"):
        unpack_canonical_evidence(
            manifest_path=manifest,
            destination=destination,
            repository_root=tmp_path,
        )

    assert not (tmp_path / "escape.json").exists()
