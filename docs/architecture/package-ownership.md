# Package ownership

Azelficoast has five package-level owners. The package root contains only its
initializer; every implementation file belongs to one of these areas.

| Package | Role |
| --- | --- |
| `core` | Authoritative transition, program, and admission machinery used by runtime search. |
| `belief` | Runtime belief state, posterior, evaluator, and training interfaces. |
| `search` | Runtime decision algorithms over public information. |
| `live` | Battle integration, operator harness, and live trace/corpus capture. |
| `research` | Independent verification and retained experiments that produce evidence. |

Within `research`, `experiments` contains executable benchmark implementations,
`studies` contains retained scientific analyses and their study-owned resources,
`mechanics` contains bounded reference mechanics and simulator models, and
`verification` contains Showdown comparisons, replay reconstruction, and
counterexamples that independently check those models. Research package-root
modules are shared infrastructure: contracts, orchestration, evidence acquisition,
or training plumbing that is not owned by one study.

A source module stays only when a current runtime consumer, independent
verification claim, or retained experiment uses it. Otherwise it is removed.
Frozen results and provenance records under `experiments/` remain unchanged;
moving current source does not rewrite historical evidence.

`tests/test_package_root_ownership.py` enforces the package-root boundary so
new modules must choose an owner before they are added.
