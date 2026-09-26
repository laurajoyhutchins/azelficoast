# Compiled data plane

Azelficoast keeps semantic authority separate from numerical representation.

Pinned Pokémon Showdown and validated Azelficoast evidence answer what a state means.
The compiled data plane lowers those already-authorized objects into dense arrays so
JAX can process large corpora without repeatedly parsing strings, traversing Python
mappings, or rebuilding categorical features.

## First boundary: Showdown-native categorical IDs

The vocabulary exporter reads one exact built Pokémon Showdown checkout and emits a
content-addressed vocabulary bound to that revision.

For moves, items, abilities, and species, Azelficoast reuses Showdown's own numeric
`BasicEffect.num` values instead of inventing a second integer namespace. Species
formes share a National Dex number, so the exporter adds only a revision-bound
`forme_index` within each shared number. Types, natures, and Random Battle roles do
not have equivalent native numeric identities; they receive deterministic 1-based
indices derived from canonical Showdown IDs, with zero reserved for padding.

The resulting identity is therefore:

```text
pinned Showdown revision
        |
        +-- native nums: species / moves / items / abilities
        |
        +-- derived coordinates: forme / type / nature / randbats role
        |
        v
content-addressed ShowdownVocabulary
```

A vocabulary from another Showdown revision is not interchangeable. A posterior bound
to one revision fails closed if packed with a vocabulary from another.

## Joint posterior packing

`azelficoast.belief.showdown_packing` lowers a validated joint Random Battle posterior
without factorizing it. Every world remains one atomic team particle with its original
weight. The pack contains fixed-shape categorical and scalar tensors such as:

```text
weights          [world]
species_num      [world, team]
species_forme    [world, team]
ability_num      [world, team]
item_num         [world, team]
move_num         [world, team, 4]
move_mask        [world, team, 4]
tera_type        [world, team]
nature           [world, team]
role             [world, team]
evs / ivs        [world, team, 6]
```

The pure-Python pack is dependency-light. `as_numpy()` performs the final fixed-dtype
materialization for NumPy/JAX consumers. That means an immutable posterior can be
validated and encoded once, then reused across search, evaluator training, posterior
ablations, and corpus mining.

The pack is a cache, not authority. Unknown categorical identities, stale vocabulary
digests, Showdown-revision mismatches, and impossible move widths fail rather than being
silently mapped to an unknown token.

## Why this is upstream of JAX

The expensive pattern to remove is not merely scalar arithmetic. It is repeated
representation work:

```text
structured evidence
 -> string normalization
 -> dictionary lookup
 -> feature hashing
 -> Python tuples
 -> NumPy
 -> JAX
```

For immutable derived data, the first four steps can happen once:

```text
validated evidence
 -> ShowdownVocabulary
 -> dense immutable pack
 -> JAX device batch
```

This is the beginning of a shared computational representation, not a JAX rewrite of
Pokémon Showdown.

## Information-flow boundary

The current execution architecture still owns information-set topology outside JAX.
Packing a posterior does not authorize hidden worlds to merge, determine which
observations are public, or decide legal actions. Those judgments remain explicit and
auditable.

A later compiled `DecisionBatch` may carry already-authorized observation partitions,
transition classes, and legal-action masks into a JAX kernel. The kernel may reduce and
evaluate that structure at scale, but it does not get to invent the structure.

## Prior art used as design input

This design deliberately learns from public Pokémon AI work without inheriting another
project's semantic registry:

- Jaxcalibur demonstrates the throughput available when battle computation has static,
  accelerator-friendly structure and large batches.
- ps-ppo uses compact categorical Pokémon IDs and learned embeddings rather than
  repeatedly treating domain entities as arbitrary strings.
- Metamon separates replay/dex preprocessing from model execution and maintains
  structured Pokémon tokenization for large replay corpora.
- pkmn/randbats continuously derives large Random Battle population statistics from
  Pokémon Showdown and is useful as an independent distributional reference.

No external model ontology becomes Azelficoast authority. Source-native identity comes
from the exact pinned Showdown revision and is independently bound into Azelficoast
evidence.

## Next measurements

The next useful experiment is matched:

1. compile a fixed corpus of validated posteriors once;
2. compare current hashed evaluator feature construction with Showdown-native packed
   categorical inputs;
3. report encoding wall time, bytes per world, JAX dispatch throughput, training
   examples per second, validation loss, and policy/value quality;
4. retain identical source decisions, posterior support, splits, targets, and promotion
   criteria.

If the packed representation only improves throughput, it is still useful. If learned
embeddings over source-native IDs also improve sample efficiency or playing strength,
that becomes a separate measured result rather than an assumption.


## Packed evaluator treatment

The first consumer of the compiled data plane is a research-only evaluator treatment in
`azelficoast.belief.packed_evaluator`.

It deliberately holds two surfaces constant with the existing evaluator:

- public state still uses the existing stable hashed feature contract;
- legal actions still use the existing stable hashed feature contract.

Only hidden-world encoding changes. Instead of flattening each hidden world into hashed
feature buckets, the treatment consumes the Showdown-bound joint posterior pack directly:

```text
joint posterior particle
        |
        v
species / forme / ability / item / moves / tera / nature / role embeddings
        |
        v
per-Pokémon encoder
        |
        v
permutation-invariant team mean + variance
        |
        v
world representation
        |
        v
posterior-weighted mean + variance + entropy
        |
        v
shared public-belief policy/value trunk
```

Move order and team order are treated as incidental representation. Move embeddings are
mask-pooled and team members are pooled as a set. Posterior particles retain their
scientific weights, and duplicate equivalent support atoms therefore preserve predictions
when their total mass is unchanged.

The packed treatment shares the existing deterministic Adam implementation rather than
introducing a second optimizer path.

## Matched representation experiment

`azelficoast.research.experiments.evaluator_representation_experiment` compares the existing hashed
hidden-world evaluator against the packed treatment over the same frozen training records.
It fixes split membership, targets, public/action feature widths, optimizer settings,
epochs, and random seed.

The receipt reports separately:

- representation encoding wall time;
- dense input bytes;
- model parameter count;
- training wall time and examples/second;
- validation inference examples/second;
- value MSE;
- policy cross-entropy;
- policy accuracy;
- total policy/value loss before and after training.

The experiment is descriptive. It does not promote either model and does not infer battle
strength from validation loss. A later battle panel can test playing strength after a
representation treatment has earned further attention.


## Search topology joins the compiled data plane

The packed posterior is now paired with a separate compiled search-topology artifact.
The two objects have different authority and reuse properties:

```text
Showdown-bound hidden worlds        verified TransitionProgram
            |                                |
            v                                v
   PackedJointPosterior              CompiledSearchTopology
            |                                |
            +--------------+-----------------+
                           v
                    JAX search batch
```

`PackedJointPosterior` owns dense hidden-world coordinates and prior weights.
`CompiledSearchTopology` owns only the already-authorized incidence between worlds,
execution classes, chance outcomes, public observations, public successors, legal
successor actions, and evaluation leaves.

Because posterior weights are not part of topology identity, prior stress treatments can
reuse the same compiled search object. Because hidden-world arrays are not duplicated per
leaf, a successor frontier can be represented by one shared packed tensor plus a
`[leaf, world]` conditional-weight matrix.

That gives the accelerator a much better unit of work:

```text
shared worlds           [W, T, ...]
leaf public features    [L, P]
leaf/world weights      [L, W]
leaf legal mask         [L, A]
root incidence          [L]
              |
              v
         one JAX frontier
```

The legal mask is carried even though the current depth-one value head does not consume
it. It belongs in the compiled representation because deeper policy/value evaluation can
use it without reconstructing successor action sets from Python strings.

The current implementation remains a research path. Candidate evidence must establish
semantic equivalence and mass conservation before any live routing change.



## SQL lowering into the packed/JAX path

The read-only SQL decision DSL now has one reviewed accelerator semantic class. Multiple
SQL spellings may enter that class only through the core's explicit relational rewrite
rules; other admitted SQL remains explainable but does not silently acquire execution
semantics.

The logical SQL operators bind to existing machinery:

```text
SCAN + FILTER
    -> pack_joint_posterior

PROJECT + PARTITION + TRANSITION + OBSERVE
    -> compile_search_topology

UPDATE_BELIEF
    -> transport_posterior_mass

EVALUATE
    -> predict_packed_shared_world_values

AGGREGATE
    -> reduce_compiled_root_values
```

The physical order is allowed to differ from the relational source order. The current
packed path hoists topology compilation ahead of posterior packing because topology
identity is independent of posterior weights:

```text
compile_search_topology
        |
        v
pack_joint_posterior
        |
        v
transport_posterior_mass
        |
        v
predict_packed_shared_world_values
        |
        v
reduce_compiled_root_values
```

The SQL-generated plan is content-addressed by semantic identity and physical bindings,
not by source spelling. A CTE form and an equivalent inline/reordered form therefore
produce different source hashes but the same physical-plan hash. The exact source hash
and the rewrite evidence remain in the receipt.

At execution time the generated plan is compared with the physical-stage declaration
emitted by the existing hand-built packed/JAX path. Any drift fails closed rather than
letting the SQL compiler and executable path quietly disagree.

This remains an intentionally narrow compiler, not a promise that arbitrary SQLite
syntax can be lowered to JAX. New relational laws must be added to the reviewed rewrite
system, and semantic changes receive no logical identity until they have independent
evidence.
