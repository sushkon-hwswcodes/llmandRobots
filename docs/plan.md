# Project Plan

Last updated: 2026-04-06

## Immediate next steps

1. Keep the recovered Panda / shape / clutter path stable while extending the same grasp semantics into YCB.
2. Use oracle / deterministic smoke tests before relying on model benchmarks to judge shared control changes.
3. Re-run YCB single-object and YCB clutter at larger trial counts.
4. Keep hand configuration support at the simulation-smoke level only.
5. Start any future hand work from a fresh direction after the maintained benchmark path is stable again.
6. Defer any new dexterous-hand implementation until the restart interface is narrowed to constrained primitives instead of vague prompt control.

## Short-term goals

1. Preserve the recovered local Panda cube-lift baseline.
2. Preserve the recovered local shape-generalization and clutter baselines.
3. Improve the maintained YCB path with the smallest task-specific changes necessary.
4. Keep benchmark artifacts easy to audit: configs, code, logs, summaries, and videos.

## Decision rule for the next iteration

- If a change touches shared grasp targeting, test it with deterministic probe code before relying on benchmark outcomes.
- If a maintained benchmark family is alive but low, compare center-vs-offset grasp targets before widening the prompt/API surface.
- If a hand idea requires prompt-heavy workaround logic before the maintained Panda / YCB path is stable, defer it.
- If a hand profile can instantiate and step but not grasp reliably, treat that only as simulation readiness, not as an active benchmark path.

## Hand restart plan when resumed later

1. Keep Panda benchmark behavior fixed as the calibration reference.
2. Treat the current Inspire path as a simulation-ready hand mount, not a completed grasp path.
3. Replace hand-only preset naming with full grasp primitives that include wrist pose and approach geometry.
4. Constrain the model-facing API to a small object-class-conditioned menu instead of exposing vague dexterous control.
5. Use the hand preset harness only for articulation sanity checks, then add a second grasp-primitive harness around canonical objects.
6. Use PPO, if needed, as an offline primitive-discovery tool per object class rather than the runtime controller.
