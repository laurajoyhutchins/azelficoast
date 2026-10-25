"""Draft 2020-12 validation of repo-owned research orchestration contracts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class ContractShapeError(ValueError):
    """A research contract or its committed schema is malformed."""


def _reject_constant(value: str) -> None:
    raise ContractShapeError(f"nonfinite JSON constant is forbidden: {value}")


@lru_cache(maxsize=2)
def _validator(schema_path: Path) -> Draft202012Validator:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)
    except (OSError, ValueError, SchemaError) as error:
        raise ContractShapeError(f"invalid committed JSON Schema {schema_path}: {error}") from error


def _location(parts: list[Any]) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def load_contract_document(path: Path, schema_path: Path) -> dict[str, Any]:
    """Validate raw JSON before projecting it into the existing typed contract."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        raise ContractShapeError(f"cannot read research contract {path}: {error}") from error
    validator = _validator(schema_path)
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = list(error.absolute_path)
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = next(
                (field for field in error.validator_value if field not in error.instance),
                None,
            )
            if missing is not None:
                location.append(missing)
        raise ContractShapeError(
            f"{path}: JSON Schema {_location(location)}: {error.message}"
        )
    return cast(dict[str, Any], document)
