# Experiment contracts

These files are the repository-owned orchestration authority for research.

- [`candidate.json`](candidate.json) declares cheap exact-head candidate experiments: affected paths, required capabilities, generators, tests, callable experiment entrypoints, produced artifacts, and settlement checks.
- [`hosted.json`](hosted.json) declares hosted studies: affected paths, capabilities, execution units, artifact transport, and aggregation requirements.

The contracts say **what must run and what evidence must settle**. Scientific implementation remains in repository Python and frozen scientific inputs remain under [`../data/`](../data/). GitHub Actions should compile and execute these contracts, not restate their scientific parameters in YAML.

Changing either contract is therefore an authority change and must select the consumers of that contract for validation.

The structural authority for these JSON documents is **JSON Schema Draft 2020-12**:
[`schemas/candidate.schema.json`](schemas/candidate.schema.json) and
[`schemas/hosted.schema.json`](schemas/hosted.schema.json). The pinned
`jsonschema` validator rejects malformed and unexpected fields before the
existing typed Python projections run. Schemas are local and never fetched
at runtime. Distinct artifact names, affected-path selection, capability
execution, settlement evidence, and scientific admissibility remain in Python.
Schema changes select all dependent candidate/hosted studies for verification.
