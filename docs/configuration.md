# Configuration

Azelficoast keeps configuration at the process boundary. Runtime modules receive typed
configuration objects and do not read environment variables themselves.

## Precedence

For options that support environment variables, precedence is:

1. explicit CLI flag;
2. environment variable;
3. documented code default.

Invalid CLI or environment values fail during argument parsing before a battle or
experiment starts.

The repository ignores `.env`, but Azelficoast does **not** load dotenv files
automatically. Export variables in the shell, inject them with a process supervisor, or
use the secret/configuration mechanism of the execution environment.

## Live timing

Live decisions use Pokémon Showdown's own player-specific timer observation when one has
been received:

```text
Time left: <turn> sec this turn | <total> sec total | <grace> sec grace
```

Azelficoast subtracts a submission reserve, creates one absolute monotonic deadline for
the decision, and shares that deadline across posterior construction and exact search.
Each subprocess also has its own operation ceiling so a hung probe cannot consume an
entire long battle clock.

When no authoritative timer observation is available, such as a local battle with the
timer disabled, Azelficoast uses an explicit fallback decision budget. The fallback is
not presented as a Showdown clock measurement.

| Purpose | CLI | Environment | Default |
| --- | --- | --- | ---: |
| Per-subprocess live ceiling | `--live-operation-timeout` | `AZELFICOAST_LIVE_OPERATION_TIMEOUT_SECONDS` | 20 s |
| Move-submission reserve | `--live-clock-reserve` | `AZELFICOAST_LIVE_CLOCK_RESERVE_SECONDS` | 5 s |
| No-clock decision budget | `--live-fallback-budget` | `AZELFICOAST_LIVE_FALLBACK_BUDGET_SECONDS` | 20 s |
| Exact-search routing margin | `--search-policy-margin` | `AZELFICOAST_SEARCH_POLICY_MARGIN` | 1.0 |

The clock is runtime authority. These settings do not duplicate Showdown's starting time,
per-turn cap, grace period, or increment.

Example:

```bash
export AZELFICOAST_LIVE_CLOCK_RESERVE_SECONDS=5
export AZELFICOAST_LIVE_OPERATION_TIMEOUT_SECONDS=20
export AZELFICOAST_LIVE_FALLBACK_BUDGET_SECONDS=20

uv run azelficoast \
  --showdown-root /path/to/pinned/pokemon-showdown \
  --evaluator-checkpoint artifacts/evaluators/current.json \
  ladder --battles 10
```

Decision traces record the budget source, observed clock values when available, reserve,
usable decision budget, clock-observation age, and measured decision latency.

## Timeout migration

The previous `--belief-timeout` / `AZELFICOAST_BELIEF_TIMEOUT_SECONDS`
setting mixed two different concerns and is intentionally removed.

- live subprocess ceilings use `--live-operation-timeout` /
  `AZELFICOAST_LIVE_OPERATION_TIMEOUT_SECONDS`;
- offline teacher and posterior probes use `--teacher-timeout` /
  `AZELFICOAST_TEACHER_TIMEOUT_SECONDS`.

There is no compatibility alias. A stale deployment therefore fails on the removed CLI
flag instead of silently applying one timeout to the wrong timing domain.

## Offline teacher timing

Scientific target generation is not constrained by the live battle clock. Its subprocess
timeout is configured separately:

| Purpose | CLI | Environment | Default |
| --- | --- | --- | ---: |
| Offline teacher/posterior probe ceiling | `--teacher-timeout` | `AZELFICOAST_TEACHER_TIMEOUT_SECONDS` | 20 s |

`--teacher-timeout` applies to `training cycle`, `training auto`, and
`training bootstrap-public`. It is intentionally independent of live timing.

## Paths and evaluator routing

| Purpose | CLI | Environment |
| --- | --- | --- |
| Pinned Showdown checkout | `--showdown-root` | `AZELFICOAST_SHOWDOWN_ROOT` |
| Learned evaluator checkpoint | `--evaluator-checkpoint` | `AZELFICOAST_EVALUATOR_CHECKPOINT` |

CLI paths override environment values.

## Showdown credentials

Official-server commands use:

- `SHOWDOWN_USERNAME`, unless `--username` is supplied;
- `SHOWDOWN_PASSWORD`.

The password intentionally has no CLI flag, avoiding routine exposure through shell
history and process listings. Supply it through the execution environment or a secret
manager. Do not commit credentials or a populated `.env` file.
