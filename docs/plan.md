# Project Plan

Last updated: 2026-04-05

## Immediate next steps

1. Review the six failures in `benchmark_green_target_clutter_30.log`.
2. Categorize each failure as one of: wrong-object selection, grasp failure, lift failure, premature finish, or simulator/task-spec issue.
3. Generate montage videos from successful clutter trials using `scripts/make_success_video_grids.py`.
4. Decide whether the next iteration should be a prompt-only change or a simulator / reward adjustment.

## Short-term goals

1. Raise the green-target clutter task from `24/30` toward the shape-generalization level of `25/30` or better.
2. Preserve the already-solved `20/20` privileged baseline while making clutter-specific changes.
3. Add a lightweight regression or smoke check for the clutter environment so future prompt or API edits do not silently break it.

## Medium-term goals

1. Promote the shape-generalization and green-target clutter tasks into a more formal benchmark suite once failure modes are understood.
2. Reuse the same evaluation harness to compare single-turn, multi-turn, and prompt-variant behavior on harder visual-grounding tasks.
3. Keep collecting benchmark artifacts that are easy to audit: logs, saved code, overview videos, and montage summaries.

## Decision rule for the next iteration

- If most failures are wrong-object selection, prioritize prompt/task-language changes.
- If most failures are near-success grasps or unstable lifts, prioritize simulator reward/success shaping and grasp-pose affordances.
- If failures are mixed, preserve the current environment and run a small A/B prompt experiment before changing simulator logic.
