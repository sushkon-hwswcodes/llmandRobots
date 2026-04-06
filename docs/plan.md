# Project Plan

Last updated: 2026-04-06

## Immediate next steps

1. Keep Panda as the only active benchmark target until the local regression is understood.
2. Compare the current Panda control path against the earlier `20/20` milestone and isolate the regression surface.
3. Use oracle / deterministic smoke tests before relying on model benchmarks to judge control changes.
4. Keep hand configuration support at the simulation-smoke level only.
5. Start any future hand work from a fresh direction after Panda is stable again.

## Short-term goals

1. Recover a trustworthy local Panda cube-lift baseline.
2. Re-check clutter once Panda behavior is credible again.
3. Preserve shape / clutter / YCB infrastructure that is not tied to the removed Inspire prompt experiments.
4. Keep benchmark artifacts easy to audit: configs, code, logs, summaries, and videos.

## Decision rule for the next iteration

- If a change touches shared Panda control behavior, test it with oracle or deterministic smoke code first.
- If Panda local results stay low even with oracle code, prioritize low-level control and simulator drift over prompt tuning.
- If a hand idea requires prompt-heavy workaround logic before Panda is stable, defer it.
- If a hand profile can instantiate and step but not grasp reliably, treat that only as simulation readiness, not as an active benchmark path.
