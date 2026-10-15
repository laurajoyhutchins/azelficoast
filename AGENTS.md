# Agent development workflow

Read [README](README.md), [repository checks](tools/check.py), and [Overcenter governance](.github/workflows/overcenter.yml). Inspect current GitHub issues/PRs, exact head/base and authoritative project state before changing work. Preserve the governance workflow's explicit command authorization and trusted pinned runtime; a proposal is not an admitted candidate.

Use native GitHub connector operations where sufficient. [Laura Dev Tools agent operations](https://github.com/laurajoyhutchins/laura-dev-tools/blob/main/docs/agent-operations.md) documents bounded readiness/portfolio inspection and fenced operations. Readiness never grants merge authorization, and similar PR titles do not establish supersession.

Install the pinned Python/Node toolchains and dependencies used by [CI](.github/workflows/ci.yml). Run `python tools/check.py static` and `python tools/check.py test` locally before expensive verification. Ordinary PR verification uses the established exact-job GCP runner lane from [engineering-infra](https://github.com/laurajoyhutchins/engineering-infra). Do not replace it with substantive usage-limited GitHub-hosted execution. Required evidence must match the current source/base and provider attempt; missing, skipped, failed, stale or unknown is not passing.

Finish ordinary authorized PRs after required verification and independent review, preserving proposal/source and owner-protected admission. Use the expected head for integration and read GitHub back. Never retry an uncertain mutation blindly. Report implemented, verified, merged, deployed and independently observed separately, and observe acceptance before opening dependent work.
