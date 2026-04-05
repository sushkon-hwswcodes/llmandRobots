# Project Plan

Last updated: 2026-04-05

## Immediate next steps

1. Review the six failures in `benchmark_green_target_clutter_30.log`.
2. Categorize each failure as one of: wrong-object selection, grasp failure, lift failure, premature finish, or simulator/task-spec issue.
3. Generate montage videos from successful clutter trials using `scripts/make_success_video_grids.py`.
4. Decide whether the next iteration should be a prompt-only change or a simulator / reward adjustment.
5. Preserve the intended Phase 3 scope: move from randomly generated shapes to real-world object assets once Phase 2.75 is stable enough.
6. Benchmark the new YCB-backed Phase 3 lift environment and identify the dominant grasp / orientation failures.
7. Review the first YCB target-clutter benchmark (`9/30`, average reward `0.394`) and verify whether failures come from target selection, clutter collisions, or the simple top-down grasp policy.

## Short-term goals

1. Raise the green-target clutter task from `24/30` toward the shape-generalization level of `25/30` or better.
2. Preserve the already-solved `20/20` privileged baseline while making clutter-specific changes.
3. Add a lightweight regression or smoke check for the clutter environment so future prompt or API edits do not silently break it.
4. Define the first real-world object set for Phase 3 so the transition is concrete instead of implicit.
5. Improve the first YCB bridge by replacing the temporary OBJ-for-collision fallback, which currently exists because MuJoCo would not load the provided YCB `collision.ply` files in this setup.
6. Keep the first real-object clutter benchmark focused on graspable YCB objects so the main variable is cluttered target selection rather than impossible grasps.
7. Improve the target-clutter grasp policy or success shaping so the current `9/30` result becomes a more stable real-object clutter baseline.

## Medium-term goals

1. Promote the shape-generalization and green-target clutter tasks into a more formal benchmark suite once failure modes are understood.
2. Reuse the same evaluation harness to compare single-turn, multi-turn, and prompt-variant behavior on harder visual-grounding tasks.
3. Keep collecting benchmark artifacts that are easy to audit: logs, saved code, overview videos, and montage summaries.
4. Implement Phase 3 by replacing synthetic/generated shapes with real-world objects while keeping the evaluation protocol comparable.
5. Extend Phase 3 from single-object lift into harder real-object variants after the base YCB lift is benchmarked.
6. Extend Phase 3 from single-object lift into specific-target selection from real-object clutter once the base YCB lift and first clutter benchmark are characterized.

## Planned phase sequence

1. Phase 2: shape generalization over generated shapes.
2. Phase 2.75: green-target selection in clutter.
3. Phase 3: real-world objects instead of randomly generated shapes.

## Decision rule for the next iteration

- If most failures are wrong-object selection, prioritize prompt/task-language changes.
- If most failures are near-success grasps or unstable lifts, prioritize simulator reward/success shaping and grasp-pose affordances.
- If failures are mixed, preserve the current environment and run a small A/B prompt experiment before changing simulator logic.
