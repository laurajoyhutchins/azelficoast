# Research boundaries

The population and matched-comparison path is organized as:

`public evidence -> belief -> mechanics/search -> matched experiment -> inference`

Flexible JSON is accepted at artifact and CLI edges. It is validated once into
immutable contracts before entering the corresponding research boundary.

## Contracts and forbidden flows

| Boundary | Allowed information | Explicitly forbidden |
| --- | --- | --- |
| Public evidence -> `PublicDecisionInput` | Canonical public state, public history identity, legal root actions, fixture identity, mechanics identity | Realized hidden state, opponent-private fields, transport/world IDs, and information revealed after the decision point |
| Belief evidence -> `BeliefInput` | A public-history-conditioned posterior over semantic hidden worlds, with finite positive normalized weights | Transport IDs as features; future-only fields; world ordering as meaning. Equivalent support is coalesced and semantic identities are canonical |
| Mechanics -> search | Admitted exact transitions for a pinned mechanics revision, public observation/successor, and dependency evidence | Search code deciding mechanics truth, filling unsupported transitions, or inferring missing read evidence. Incomplete or unsupported mechanics fail closed |
| Search/evaluator -> matched experiment | Valuations and selected actions from the same typed public input, posterior, mechanics surface, and evaluator | Realized hidden-world IDs/state, mutable shared input, or one arm receiving a different public or posterior input |
| Matched experiment -> inference | A settled result whose typed receipts agree on the frozen specification, transition evidence, root actions, evaluator, and measured compute use | Caller-declared budget equality, unmatched arms, incomplete receipts, or policy results feeding back into frozen cohort selection |

`BeliefTransportIndex` and opaque transport IDs exist only at the artifact
adapter boundary so transition-program records can be joined to semantic worlds.
`MechanicsExecutor` is the sole transition interface used by matched search.
Search owns valuation and planning; the mechanics provider owns admission,
exact transition behavior, revision identity, and dependency/refinement evidence.
The direction is one-way: search values cannot revise mechanics evidence or
posterior support, and settled results cannot alter public evidence, treatment
definitions, or frozen cohort selection.
Realized-world scoring in `posterior_validity.py` is post-decision diagnostics
only; realized-world fields are never accepted by the decision-time belief or
search contract. Router ablations are analysis outputs, not a source of new
thresholds or treatment assignments in this refactor.

## Matched comparison

`MatchedExperimentSpec` freezes the source and public-state identities, legal
action set, canonical public state, semantic posterior digest, mechanics
identity, evaluator identity, chance treatment, opponent model, depth, and
authorized budget. Both treatment arms must cite that same specification.
`ComputeReceipt` is parsed from each
generated arm result; settlement checks measured execution-class counts for
equality and against the authorization limit, in addition to the shared
transition program, mechanics artifact, and dependency-evidence digests.

The current budget unit is one verified whole-turn execution class consumed.
Evaluator-call counts are recorded separately and are not substituted for the
transition budget. Transport IDs are used only to join mechanics records and
never enter evaluator features. The cohort selector remains outcome-blind and
retains the existing admission reasons, hash sampling, and CLI. Its extraction
into `studies/population_cohort.py` does not change cohort policy.
Receipts also retain typed resource accounting for evaluator calls and executor,
preparation, and search wall-clock times; these measurements do not redefine the
counted class budget.

## Evidence compatibility

Existing plan and cohort schema versions remain unchanged. Matched packet,
receipt, and result schemas are versioned for the new required contract evidence;
loaders continue to validate serialized inputs at the edge. Frozen research
evidence is not rewritten by this refactor. A mismatch, unrecognized schema,
missing transition, incomplete dependency declaration, or unsupported mechanics
condition is an explicit error rather than an approximation admitted into the
evidence set.
