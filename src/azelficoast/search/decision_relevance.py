"""Azelficoast adapter for domain-neutral exact decision-relevance quotienting."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from azelficoast.core.decision_relevance import (
    CERTIFICATE_SCHEMA,
    CERTIFICATE_SCHEMA_VERSION,
    DecisionRelevanceError,
    decision_relevance_quotient,
)
from azelficoast.real_belief_trace import analyze_oracle


def analyze_quotiented_oracle(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze the exact quotient with Azelficoast's battle trace analyzer."""

    certificate, quotient = decision_relevance_quotient(document)
    trace = analyze_oracle(quotient)
    trace["source_world_count"] = certificate["worlds_in"]
    trace["decision_relevance"] = {
        key: copy.deepcopy(value)
        for key, value in certificate.items()
        if key != "classes"
    }
    return trace, certificate


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Certify and analyze the exact decision-relevance quotient."
    )
    parser.add_argument("oracle", type=Path)
    args = parser.parse_args(argv)

    document = json.loads(args.oracle.read_text(encoding="utf-8"))
    trace, certificate = analyze_quotiented_oracle(document)
    print(
        json.dumps(
            {
                "schema": "azelficoast.decision-relevance-analysis",
                "schema_version": 1,
                "certificate": certificate,
                "quotient_trace": trace,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CERTIFICATE_SCHEMA",
    "CERTIFICATE_SCHEMA_VERSION",
    "DecisionRelevanceError",
    "analyze_quotiented_oracle",
    "decision_relevance_quotient",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
