# Data provenance

Azelficoast's root MIT License covers original software. It does not
automatically license third-party-derived research data.

The rule is deliberately simple:

> Record where evidence came from before deciding what rights attach to it.

## Provenance classes

| Material | Treatment |
| --- | --- |
| Synthetic data produced solely by Azelficoast | May be distributed as project-generated research data unless a generator input imposes additional terms. |
| Locally generated battle traces | Record the generating software/revision and whether the trace contains third-party or user-supplied material. |
| Pokémon Showdown-derived fixtures | Record the exact Showdown revision and transformation. Showdown's own license remains applicable to copied Showdown material. |
| Public replays or battle records | Record the source identifier and acquisition path. Do not assume the repository's MIT License applies to the underlying record. |
| Smogon or other community-derived data | Record the exact source and any stated terms before committing or redistributing it. |
| Pokémon artwork, sprites, audio, logos, or other proprietary assets | Do not add them to the MIT-licensed source tree merely for presentation. |

## Minimum record

Durable experimental evidence derived from an external source should carry
enough metadata, either inside the artifact or in an adjacent manifest, to
recover:

```yaml
source: <project, dataset, replay service, or generator>
source_locator: <stable identifier or path>
source_revision: <commit/version when applicable>
acquisition: <generated, downloaded, observed, transformed>
transformation: <what Azelficoast changed or extracted>
contains_third_party_assets: <true|false>
contains_user_identifiers: <true|false>
license_or_terms: <known license/terms, or "not asserted">
```

Exact experiment-specific fields may be added when they improve
reproducibility. The point is not a universal metadata schema; it is to prevent
generated output from silently acquiring provenance or licensing claims that
were never established.

## Repository boundary

The following statements are intentionally distinct:

1. **Azelficoast source code is MIT-licensed.**
2. **A frozen experiment can be reproducible.**
3. **A third-party-derived dataset can be redistributable.**

One does not prove either of the others. When rights or provenance are
uncertain, preserve the source reference and make no broader licensing claim.
