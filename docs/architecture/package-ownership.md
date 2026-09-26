# Package ownership

Azelficoast has five Python package-level owners plus one explicit Node-side Showdown boundary. The Python package root contains only its initializer; every Python implementation file belongs to one of these areas.

| Package | Role |
| --- | --- |
| `core` | Authoritative transition, program, and admission machinery used by runtime search. |
| `belief` | Runtime belief state, posterior, evaluator, and training interfaces. |
| `search` | Runtime decision algorithms over public information. |
| `live` | Battle integration, operator harness, and live trace/corpus capture. |
| `research` | Independent verification and retained experiments that produce evidence. |

Within `research`, `experiments` contains experiment-specific scientific semantics
invoked through shared research execution contracts, `mechanics` contains bounded reference
mechanics and simulator models, and
`verification` contains Showdown comparisons, replay reconstruction, and
counterexamples that independently check those models. Research package-root
modules are reusable contracts, orchestration, evidence acquisition, or
analysis utilities shared by more than one experiment.

A source module stays only when a current runtime consumer, independent
verification claim, or retained experiment uses it. Otherwise it is removed.
Frozen results and provenance records under `experiments/` remain unchanged;
moving current source does not rewrite historical evidence.\n\nNode code that directly depends on the pinned Pokémon Showdown checkout lives under `showdown/`, not a generic `scripts/` directory. `showdown/runtime` is the live mechanics bridge, `showdown/verification` produces independent reference evidence, `showdown/research` contains Showdown-backed research utilities, and `showdown/shared` contains cross-cutting serialization/identity helpers. Python callers may depend on those paths only through their owning live or research boundary.

`tests/test_package_root_ownership.py` enforces the package-root boundary so
new modules must choose an owner before they are added.
