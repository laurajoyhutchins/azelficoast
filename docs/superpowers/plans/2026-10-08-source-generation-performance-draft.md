# Source generation performance draft

Stacked on PR #218 at `e1244fa8c30361882cbe6d602abb283fe2bd6076`.
This does not change or restart active acquisition run `37815648932`.

## Change

The unlearned live policy previously restarted Node and reconstructed the
Showdown module graph for every admitted decision. Extend the existing
line-protocol worker with a full-oracle operation that calls the same
`runProbe` function with fresh request-local state. Reuse the worker's
content-addressed generator cache. Keep the existing 20-second probe deadline,
fixture/revision/legal-action checks, source reconstruction, full oracle,
analyzer, and fallback reasons. A timeout or response identity violation
terminates the worker and resets its response queue before another request.
Python owns each worker's temporary generator cache, so timeout/SIGKILL and
normal shutdown clean it up without depending on a Node exit handler.

Add an independent acquisition observer that tails complete append-only
trace lines once per minute. It logs decisions, unique completed battles,
policy reason counts, activity age, bytes, and interval throughput. It emits
a best-effort final snapshot on success or exception, and retains its separate progress
JSONL in the immutable source artifact. It never reads win/loss values,
modifies source traces, supplies actions, changes budgets, or votes on
admission. An observer failure reports to stderr without failing generation.
Shutdown waits at most five seconds for telemetry and reports an incomplete
final snapshot if that wait expires; telemetry never delays source execution
indefinitely.

## Evidence and limits

- The 43-test focused baseline passed before editing.
- Final repository verification: 690 passed, 50 skipped; Ruff passed; strict
  mypy passed across 63 source files; Node syntax and diff-whitespace checks
  passed. The explicitly enabled pinned-Showdown probe test run passed all
  eight tests. One pre-existing tar-extraction deprecation warning remained.
- New tests were observed failing before their implementations.
- A real pinned-Showdown integration test compares complete oracle documents
  and analyzed action results for two bounded one-root-action queries from the
  frozen low-HP Tinkaton/Zapdos state at production generator/chance budgets.
  The worker handles A, B, A, a wrong-revision request, then A; every valid
  result equals the fresh-process result.
- Process tests exercise timeout shutdown/restart, stale response identity
  rejection, and recovery. Telemetry tests exercise split writes, no double
  counting, unique terminals, source bytes unchanged, output without outcomes,
  final snapshots on generation errors, and observer output failure isolation.
- On this machine, the bounded parity queries took 2.585/2.814 seconds in fresh
  processes, 2.082 seconds on the first worker call, and 1.468/1.529 seconds on
  warm calls. These observations are not a population throughput estimate.
- Run the integration test with
  `AZELFICOAST_TEST_SHOWDOWN_ROOT=/path/to/pinned/build uv run pytest tests/test_showdown_probe.py -q -s`.
  Without an explicit build, this integration test skips rather than silently
  using another mechanics revision.
- The original full 13-root-action fixture timed out at the existing 20-second
  limit during investigation. Thus startup reuse is not evidence that broad
  exhaustive searches now meet that deadline. Synchronous computation still
  blocks the battle event loop; network concurrency is not CPU parallelism.

Keep this PR draft until a broader same-fixture comparison covers the natural
source boundaries, fallback/error behavior, and actual throughput. Timing
improvements can allow a previously timed-out search to finish; do not claim
timeout-limited action distributions are identical. Any future corpus uses
its own frozen generation head and exact provenance. Do not restart or splice
the active 3,072-battle source to incorporate this draft.

The 100-state minimum, 128-state cap, outcome-blind selector, pinned Showdown
authority, unchanged Python mechanics verifier, treatment settings, and
positive exact-head hosted-study merge requirement remain unchanged.
