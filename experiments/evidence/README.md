# Canonical evidence

This directory holds the canonical frozen evidence bundle used by hosted and candidate research that needs larger immutable inputs.

[`canonical-evidence.json`](canonical-evidence.json) is the manifest and digest authority for [`canonical-evidence.tar.gz`](canonical-evidence.tar.gz). Repository code verifies the digest and extracts the archive fail-closed before experiments consume it.

Small experiment records belong in [`../data/`](../data/). Execution and settlement authority belongs in [`../contracts/`](../contracts/).
