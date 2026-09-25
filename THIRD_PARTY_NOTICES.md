# Third-party software and materials

Azelficoast's original source code is licensed under the repository's
[MIT License](LICENSE). That license does not relicense third-party software,
data, trademarks, artwork, battle records, or other materials used by or
alongside the project.

## Software dependencies

The complete resolved Python environment is recorded in `uv.lock`. Those
packages retain their upstream licenses. In particular:

- **poke-env 0.16.1** is distributed under the MIT License.
- **JAX 0.11.2** is distributed under the Apache License 2.0.
- **NumPy 2.5.3** uses the BSD 3-Clause license for the main NumPy project;
  particular distributions may include additional components under additional
  licenses.

Azelficoast normally consumes these as separately installed dependencies. If
third-party source or binary material is ever vendored or redistributed with
Azelficoast, its applicable upstream license and notices must be preserved.

## Pokémon Showdown

Pokémon Showdown is a separate project distributed under the MIT License.
Azelficoast uses pinned Showdown revisions as an external mechanics oracle and
may build revision-keyed artifacts for reproducible experiments. Azelficoast's
MIT License does not replace the Showdown license.

Any redistributed Showdown source or build artifact must retain the applicable
upstream license and provenance, including the exact Showdown revision used.

## Pokémon and related material

Pokémon names, characters, artwork, logos, game assets, and related
intellectual property belong to their respective owners. Azelficoast's MIT
License grants no rights in those materials.

This repository should not vendor proprietary Pokémon artwork, sprites, audio,
or other game assets merely for presentation. Names and mechanics terminology
may appear where needed to describe interoperability and research behavior.

Azelficoast is an unofficial research project and is not affiliated with or
endorsed by Nintendo, Game Freak, Creatures, The Pokémon Company, Pokémon
Showdown, or Smogon.

## Research data

A dataset, replay, fixture, trace, or derived corpus is not automatically
MIT-licensed merely because it is stored in this repository. Its provenance
and any applicable third-party terms remain authoritative.

See [docs/data-provenance.md](docs/data-provenance.md) for the repository's
data boundary.
