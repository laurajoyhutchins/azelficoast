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
