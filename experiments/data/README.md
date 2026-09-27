# Frozen experiment data

This directory contains the small, reviewable machine-readable artifacts used by the research story in [`../README.md`](../README.md).

It intentionally mixes several *artifact kinds* while keeping them separate from orchestration authority:

- **source fixtures**: frozen public or replay-derived inputs;
- **selections and treatments**: the exact cohort or intervention chosen for a study;
- **plans and scientific contracts**: preregistered comparisons that may not yet have completed results;
- **results**: checked-in outcomes that are useful as durable reference evidence;
- **negative controls/corpora**: cases that did not support the attractive hypothesis and therefore protect the project from selection bias;
- **semantic records**: frozen definitions such as opponent-policy semantics.

Files keep their historical names so citations, receipts, and older research notes remain traceable. New execution authority does not belong here. Candidate and hosted orchestration live in [`../contracts/`](../contracts/), while larger canonical evidence lives in [`../evidence/`](../evidence/).

A file named `plan` or `contract` is not evidence that the planned result occurred. Code and documentation should preserve that distinction.
