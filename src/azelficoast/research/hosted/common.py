from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SHOWDOWN_ROOT = Path("/tmp/pokemon-showdown")
EVIDENCE_ROOT = Path("/tmp/azelficoast-evidence")


def _canonical_discovery_source_artifact() -> dict[str, object]:
    manifest_path = (
        REPOSITORY_ROOT
        / "experiments"
        / "evidence"
        / "canonical-evidence.json"
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        source = document["contents"]["discovery"]["source_artifact"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            "canonical evidence manifest lacks discovery source artifact"
        ) from error
    if not isinstance(source, dict):
        raise RuntimeError("canonical discovery source artifact must be an object")
    artifact_id = source.get("id")
    digest = source.get("digest")
    if (
        not isinstance(artifact_id, int)
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
    ):
        raise RuntimeError("canonical discovery source artifact is malformed")
    return {"id": artifact_id, "digest": digest}


SOURCE_ARTIFACT = _canonical_discovery_source_artifact()


class HostedResearchError(RuntimeError):
    """Raised when hosted research evidence fails its executable contract."""


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: object, *, pretty: bool = False) -> None:
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
    )


def run(
    command: Sequence[str],
    *,
    stdout: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Iterable[int] = (0,),
) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("+", " ".join(command), flush=True)
    target = None if stdout is None else Path(stdout)
    if target is None:
        completed = subprocess.run(command, env=merged_env, check=False)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                env=merged_env,
                check=False,
                text=True,
                stdout=handle,
            )
        if target.is_file():
            print(target.read_text(encoding="utf-8"), end="", flush=True)
    allowed = set(allowed_returncodes)
    if completed.returncode not in allowed:
        raise HostedResearchError(
            f"command failed with {completed.returncode}: {' '.join(command)}"
        )
    return completed.returncode


def python_module(
    module: str,
    *args: str,
    stdout: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Iterable[int] = (0,),
) -> int:
    return run(
        (sys.executable, "-m", module, *args),
        stdout=stdout,
        env=env,
        allowed_returncodes=allowed_returncodes,
    )


def node(
    script: str,
    *args: str,
    stdout: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Iterable[int] = (0,),
) -> int:
    return run(
        ("node", script, *args),
        stdout=stdout,
        env=env,
        allowed_returncodes=allowed_returncodes,
    )


def python_script(
    script: str,
    *args: str,
    stdout: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Iterable[int] = (0,),
) -> int:
    return run(
        (sys.executable, script, *args),
        stdout=stdout,
        env=env,
        allowed_returncodes=allowed_returncodes,
    )


def pytest(*tests: str, env: Mapping[str, str] | None = None) -> None:
    run((sys.executable, "-m", "pytest", *tests), env=env)


def start_showdown_server() -> subprocess.Popen[bytes]:
    config = SHOWDOWN_ROOT / "config" / "config.js"
    if not config.exists():
        shutil.copyfile(
            SHOWDOWN_ROOT / "config" / "config-example.js",
            config,
        )
    log = Path("/tmp/showdown.log").open("wb")
    process = subprocess.Popen(
        ("node", "pokemon-showdown", "start", "--no-security"),
        cwd=SHOWDOWN_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                return process
        except OSError:
            if process.poll() is not None:
                break
            time.sleep(0.25)
    tail = ""
    try:
        tail = Path("/tmp/showdown.log").read_text(encoding="utf-8")[-4000:]
    except OSError:
        pass
    process.terminate()
    raise HostedResearchError(f"Showdown did not open port 8000\n{tail}")


def checkout_showdown_in_place(revision: str, *, build: bool) -> None:
    shutil.rmtree(SHOWDOWN_ROOT, ignore_errors=True)
    run(("git", "init", str(SHOWDOWN_ROOT)))
    run(("git", "-C", str(SHOWDOWN_ROOT), "remote", "add", "origin", "https://github.com/smogon/pokemon-showdown.git"))
    run(("git", "-C", str(SHOWDOWN_ROOT), "fetch", "--depth", "1", "origin", revision))
    run(("git", "-C", str(SHOWDOWN_ROOT), "checkout", "--detach", "FETCH_HEAD"))
    print("+ npm ci --omit=optional --no-audit --no-fund", flush=True)
    subprocess.run(
        ("npm", "ci", "--omit=optional", "--no-audit", "--no-fund"),
        cwd=SHOWDOWN_ROOT,
        check=True,
    )
    if build:
        print("+ npm run build", flush=True)
        subprocess.run(("npm", "run", "build"), cwd=SHOWDOWN_ROOT, check=True)


def find_jsonl(path: str | Path, key: str, value: object) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            if isinstance(row, dict) and row.get(key) == value:
                return row
    raise HostedResearchError(f"{value!r} not found in {path}")


def as_dict(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise HostedResearchError(f"{label} must be a JSON object")
    return value


def showdown_revision() -> str:
    from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT

    return PINNED_SHOWDOWN_COMMIT


def print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)
