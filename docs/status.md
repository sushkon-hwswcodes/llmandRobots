# Project Status

Last updated: 2026-04-05

## Current branch state

- Branch: `main`
- Remote tracking: `origin/main`
- Latest committed milestone: `eae9fc8` "Add clutter sim with green target among red distractors (Phase 2.75)"

## Recent progress

### Baseline and prompt iterations

- `bda3afb`: initial 20-trial Qwen2.5-Coder-7B cube-lifting benchmark
- `0d7ac23`: enabled multi-turn recovery for privileged Qwen config
- `9cf0c1e`: multi-turn benchmark improved to `6/20` from `2/20` single-turn
- `b7b89b2`: prompt v2 forbade `open_gripper`, added a few-shot example, and clarified task completion
- `ab933df`: prompt v2 benchmark reached `17/20`
- `56c359e`: tuned temperature to `0.3`, reaching `20/20`

### Shape generalization

- `97ced36`: added the shape-generalization environment, `get_object_shape()` API, and shape-conditioned prompt
- `2c7f47d`: fixed the `lift_shape` robosuite import path
- `641a299`: corrected success / reward logic to require lift-from-reset plus grasp contact
- `1da0479`: fixed the prompt to use `sample_grasp_pose(object)` and support the `object` alias
- `4038b0f`: prompt-fix benchmark results reached `25/30`

### Green-target clutter task

- `eae9fc8`: added the Phase 2.75 clutter simulator with one green target and red distractors
- `benchmark_green_target_clutter_30.log`: completed 30 trials with `24/30` successes
- `scripts/make_success_video_grids.py`: added local tooling to build montage videos for successful clutter trials

## Current local artifacts

- `benchmark_green_target_clutter_30.log`: full 30-trial clutter benchmark output
- `scripts/make_success_video_grids.py`: helper for assembling 4x2 success-video grids

## What is done

- Robosuite privileged baseline is stable at `20/20`
- Shape generalization is implemented and benchmarked at `25/30`
- Green-target clutter is implemented and benchmarked at `24/30`

## What still needs attention

- Inspect the six clutter-task failures to separate prompt issues from simulator/task-design issues
- Decide whether the clutter task needs another prompt pass before expanding it into a broader benchmark tier
- Turn representative successful trials into montage videos for easier review and sharing
