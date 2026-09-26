# Execution backend architecture

Azelficoast has three different execution problems. They should not be solved by one
runtime merely because that runtime is fast in one benchmark.

## Ownership

```text
public state + posterior + verified TransitionProgram
                    |
                    v
          Python search topology
                    |
                    v
          EvaluationFrontier
       /                    \
      v                      v
exact mechanics         learned values
native kernel              JAX batch
      \                      /
       +------> root reduction
                    |
                    v
              chosen action
```

### Python owns judgment and topology

Python owns validation, information-flow semantics, construction of successor
information sets, search topology, deterministic reduction, fail-closed behavior, and
evidence accounting.

A numerical backend must never decide which hidden worlds may share an information set,
which observations are public, which actions are legal, or how a leaf contributes to a
root action. Those decisions are represented explicitly by
`azelficoast.core.evaluation.EvaluationFrontier`.

### Pokémon Showdown owns semantic authority

Pinned Pokémon Showdown is the independent mechanics authority. It produces adversarial
verification evidence and, where required, representative transition executions.

Showdown execution is not itself a performance target. Repeated world-by-world Showdown
construction in a live decision is evidence debt to remove by proving narrower reusable
machinery, not work to accelerate with a larger JAX kernel.

For the learned live policy, process and reconstruction reuse are explicitly transport
optimizations. One persistent Node worker keeps the pinned Showdown module graph loaded
and caches exact Random Battle generator populations. A posterior request leaves its
content-addressed VM context resident until routing is known. If the learned policy is
uncertain, TransitionProgram construction resumes from that exact context instead of
reconstructing the posterior in a second process. A high-confidence learned decision
releases the context without compiling mechanics.

This reuse grants no semantic authority. The session identity is derived from the complete
probe source, the Showdown revision is still checked against the project pin, and the
ordinary one-shot full oracle remains the fail-closed recovery path. Reusing a context may
remove process startup, module loading, generator sweeps, and duplicate posterior
construction; it may not change hidden-world support, opponent-policy identity, or
transition semantics.

Transition execution reuse is narrower still. The persistent worker keeps a bounded
LRU of representative world-action executions. A cached execution is eligible only when
the complete serialized Showdown root snapshot, hidden-world mechanics material, root
action, opponent-policy identity, chance-sample configuration, and action seed position
all match exactly. The new fixture still receives a newly bound TransitionProgram and
effect signature. A cache hit therefore reuses deterministic execution work, not an old
fixture's authority.

Within one compilation, root snapshots are also memoized per hidden world so multiple
actions do not rebuild the same battle. Cache hit, miss, fresh-turn, reused-turn, and
root-snapshot counts are emitted as producer diagnostics so latency improvements remain
separate from scientific transition/evaluator counts.

The worker also maintains a second, independently bounded public-projection cache.
Before an execution is admitted there, the probe wraps a conservative surface of mutable
Battle and Pokemon fields and records every read made by the opponent policy, Showdown
turn execution, and public successor construction. The cache key retains a normalized
post-restart Showdown state with only those traceable fields masked. Non-traceable state,
derived requests, queues, effect state, move slots, and every other simulator byte remain
part of the static identity.

A projection entry is eligible only when its read trace was complete and every field read
by the original execution has the same value in the new root. If a property cannot be
instrumented, the execution is ineligible for projection reuse. Showdown's wall-clock
`|t:|` log metadata is normalized before exact-root hashing because it is explicitly
excluded from battle semantics and from emitted observations.

Projection reuse stores mechanics output as a successor delta rather than freezing the
old public successor. Observation construction and legal-action generation remain inside
the public read trace because they can expose semantically relevant simulator state.
Successor rendering runs after that trace. The worker diffs the fresh successor against
the root public state and stores only changed leaves. On a later projection hit it applies
those changes to the current root public state, carries unrelated current leaves through,
and recomputes the semantic hash from the rehydrated outcomes.

This is deliberately not a generic patch of serialized Showdown internals. Structural
simulator state still participates in the static identity. The delta crosses only the
already-public successor contract, and the ordinary direct oracle remains the independent
authority capable of rejecting an invalid reuse proposal.

The projection cache also separates hidden exact HP from its base identity because active
HP is already one of the masked mutable root fields. This does not make HP irrelevant:
each entry records the hidden fields actually read by the execution, and a later root must
match that hidden read projection. A move that reads exact HP therefore still invalidates
on an HP change; an execution that never reads it can reuse its mechanics delta while the
current HP simply flows through successor rehydration.

The two caches have separate limits:
`AZELFICOAST_SHOWDOWN_TRANSITION_CACHE_ENTRIES` controls exact normalized-root reuse,
while `AZELFICOAST_SHOWDOWN_PUBLIC_PROJECTION_CACHE_ENTRIES` controls traced projection
reuse. Both default to 512 entries.

### Physical planning is per logical operator

The core planner does not compare heterogeneous work merely because every path is fast.
It chooses among semantically equivalent implementations of one logical operator at a
time. A transition implementation competes with other exact transition implementations;
a learned evaluator implementation competes with other evaluator implementations.

For live Showdown transitions, the persistent policy now feeds measured locality and
latency back into the planner. The first uncertain search turn exercises the complete
exact-cache -> projected-delta -> fresh-Showdown route to obtain measurements. Later
turns compare three physical routes:

```text
fresh Showdown

exact cache
  -> fresh Showdown on miss

exact cache
  -> projected-delta cache on miss
  -> fresh Showdown on miss
```

Hit probabilities use smoothed observed hit/miss counts. Hit, miss, and fresh execution
latencies are measured separately inside the persistent worker. If the planner selects a
route that stops consulting a cache, the full route is sampled again every sixteen
program compilations so locality changes can be detected rather than freezing an old
cost decision forever.

Compiled mechanics can enter the same `TRANSITION` candidate set only when an adapter
has an independently verified binding for the current effect signature. JAX belongs to
the `EVALUATE` operator and is compared only with semantically equivalent evaluator
implementations. An unavailable implementation is not assigned an optimistic cost; it
simply is not a candidate.

The planner's prediction changes execution machinery, never semantic authority. Every
selected transition route still terminates in exact work, cached entries retain their
existing dependency/effect fences, and fresh pinned Showdown remains the terminal
fallback.

### The owned compiler owns small exact integer kernels

The custom compiler is deliberately a tiny lowering from an explicitly supported Python
subset to native C. It is appropriate when all of the following hold:

- the operation is exact, branchy integer mechanics rather than learned numerical work;
- the Python source is already the semantic implementation being tested;
- the compiled result is independently checked against Python, JAX where applicable,
  and pinned Showdown evidence;
- measured end-to-end latency beats the alternative on the target execution profile.

The compiler must remain narrow. Unsupported Python syntax fails compilation rather than
quietly expanding the trusted language surface. It is not a second Python implementation
and should not acquire search, posterior, model, or orchestration semantics.

### JAX owns dense batched numerical work

JAX owns learned policy/value evaluation, training, and other sufficiently large,
shape-stable numerical batches. Search does not call the learned model leaf by leaf.

Instead, search first constructs the complete successor `EvaluationFrontier`. The
evaluator then packs that frontier into power-of-two shape buckets and evaluates all
successor values through one cached JIT dispatch. Padding has zero belief mass, so it
cannot alter the pooled public-belief value.

JAX must not own information-flow branching. An XLA program may calculate values for an
already-authorized batch, but it does not decide which worlds are observationally
equivalent.

Immutable evidence may be lowered before this boundary into revision-bound dense arrays.
The [compiled data plane](compiled-data-plane.md) reuses pinned Showdown numeric identities
for categorical Pokémon data and preserves joint posterior particles without granting the
packed representation any semantic authority.

## Cost units

Three counts describe different work and must remain separate:

- `transition_evaluations`: verified whole-turn execution classes consumed;
- `evaluator_calls`: semantic successor leaves whose learned values were required;
- `evaluator_batches`: physical evaluator backend dispatches.

A batched evaluator therefore reduces `evaluator_batches` without pretending that the
scientific treatment required fewer `evaluator_calls`. Wall-clock measurements remain
separate from both counts.

## Optimization rule

Move work only after identifying its authority and shape:

1. remove repeated semantic proof from the hot path when it can be certified offline;
2. collapse hidden worlds into verified execution classes;
3. construct the full search evaluation frontier in Python;
4. batch learned numerical evaluation through JAX;
5. lower exact integer mechanics to the owned native compiler only when target-specific
   evidence demonstrates a benefit.

This keeps optimization subordinate to the correctness boundary: generated output is
not verified output, and a faster backend does not gain authority by being faster.
