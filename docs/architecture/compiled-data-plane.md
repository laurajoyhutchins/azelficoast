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
