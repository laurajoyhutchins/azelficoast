# Showdown boundary

This directory owns Node-side code that directly depends on the pinned Pokémon Showdown checkout.

- `runtime/` is the live mechanics bridge used by Azelficoast decisions.
- `verification/` produces independent Showdown reference evidence and replay reconstruction.
- `research/` contains Showdown-backed research utilities that are not live runtime authority.
- `shared/` contains cross-cutting serialization and identity helpers.

Generic executable source does not belong in a root-level `scripts/` directory. Put code under the owner whose authority it exercises, and keep frozen historical outputs under `experiments/` unchanged.
