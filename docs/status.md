# Project Status

Last updated: 2026-04-05

## Current branch state

- Branch: `main`
- Remote tracking: `origin/main`
- Latest committed milestone: `8dcc9aa` "Add configurable hand parameters for Franka lift benchmarks"

## Recent progress

### Baseline and prompt iterations

- `bda3afb`: initial 20-trial Qwen2.5-Coder-7B cube-lifting benchmark
- `0d7ac23`: enabled multi-turn recovery for privileged Qwen config
- `9cf0c1e`: multi-turn benchmark improved to `6/20` from `2/20` single-turn
- `b7b89b2`: prompt v2 forbade `open_gripper`, added a few-shot example, and clarified task completion
- `ab933df`: prompt v2 benchmark reached `17/20`
- `56c359e`: tuned temperature to `0.3`, reaching `20/20`

### Phase 2: shape generalization

- `97ced36`: added the shape-generalization environment, `get_object_shape()` API, and shape-conditioned prompt
- `2c7f47d`: fixed the `lift_shape` robosuite import path
- `641a299`: corrected success / reward logic to require lift-from-reset plus grasp contact
- `1da0479`: fixed the prompt to use `sample_grasp_pose(object)` and support the `object` alias
- `4038b0f`: prompt-fix benchmark results reached `25/30`

### Phase 2.75: green-target clutter task

- `eae9fc8`: added the Phase 2.75 clutter simulator with one green target and red distractors
- `benchmark_green_target_clutter_30.log`: completed 30 trials with `24/30` successes
- `scripts/make_success_video_grids.py`: added local tooling to build montage videos for successful clutter trials
- Failure review update: the six clutter failures appear to be clutter-induced grasp / lift failures, not wrong-target selection

### Phase 3: real-world objects

- Planned next milestone: replace randomly generated shape instances with real-world object assets
- Goal: test whether the current prompting and control stack transfers from synthetic shape abstractions to more realistic object geometry and appearance
- Status: initial implementation now exists via a YCB-backed Robosuite lift environment using downloaded ManiSkill YCB assets
- Added low-level env: `franka_robosuite_ycb_lift_low_level`
- Added config: `env_configs/real_objects/franka_qwen_ycb.yaml`
- Smoke test: env registration, reset, object metadata, and prompt-facing APIs all pass locally
- Current workaround: the bridge uses YCB `textured.obj` meshes for collision because this MuJoCo setup would not load the provided `collision.ply` files
- New branch in progress: target-object selection from YCB clutter
- Added low-level env: `franka_robosuite_ycb_target_clutter_low_level`
- Added config: `env_configs/real_objects/franka_qwen_ycb_target_clutter.yaml`
- Current target-clutter behavior: the prompt names a specific YCB target object while `sample_grasp_pose("object")` still resolves to the target object pose
- 30-trial benchmark: `9/30` task-complete with `0.394` average reward
- Current readout: target selection appears to be working, but cluttered grasp reliability is still the dominant bottleneck

### Hand migration work

- `8dcc9aa`: added backward-compatible hand-configuration plumbing while preserving Panda as the default benchmark path
- Active benchmark YAMLs now pin Panda defaults explicitly so the current `20/20`, `25/30`, and `24/30` results remain reproducible
- The low-level Robosuite envs now accept hand metadata such as robot name, IK target link, end-effector body name, and TCP offset
- The low-level Robosuite envs now accept a Robosuite `gripper_types` override so a custom hand can actually be instantiated instead of only relabelled
- Current best candidate for the first five-finger profile: `Panda + InspireRightHand`, because Robosuite already includes Inspire hand assets and a gripper class
- Smoke result: `Panda + InspireRightHand` successfully instantiated and reset in the existing shape-lift low-level env, with Robosuite action dimension increasing to `13`
- Follow-up compatibility result: the shared Robosuite wrapper now steps correctly with the Inspire hand after fixing the old hard-coded action slicing assumption
- Motion result: the existing privileged API can execute `goto_pose(...)` against the Inspire-hand profile when the standard Panda Pyroki server is running
- Added opt-in smoke config: `env_configs/shape_generalization/franka_qwen_shape_inspire_smoke.yaml`
- Caveat: policy behavior is still binary open / close, so the first five-finger benchmark will likely be a compatibility test before any dexterous-hand policy upgrade

## Current local artifacts

- `benchmark_green_target_clutter_30.log`: full 30-trial clutter benchmark output
- `scripts/make_success_video_grids.py`: helper for assembling 4x2 success-video grids
- `docs/hand_configuration.md`: notes on the new configurable hand path and the parameters that need to be swapped for a future five-finger hand

## What is done

- Robosuite privileged baseline is stable at `20/20`
- Phase 2 shape generalization is implemented and benchmarked at `25/30`
- Phase 2.75 green-target clutter is implemented and benchmarked at `24/30`
- The clutter failure review is complete enough to say the misses are grasp-execution failures, not target-selection failures
- Phase 3 real-world objects has started with a first YCB-backed lift environment and prompt/config bridge
- Phase 3 target-object selection from real-object clutter has an initial implementation and smoke-test coverage
- Phase 3 target-object selection from real-object clutter has now been benchmarked once at `9/30`
- The hand path is now configurable without replacing Panda, and the current benchmark configs explicitly preserve Panda settings

## What still needs attention

- Determine the correct mount/frame parameters for the first five-finger profile, likely `Panda + InspireRightHand`, now that reset and simulator stepping both work
- Check whether the current IK / end-effector assumptions still hold across full grasp attempts, not just a single `goto_pose`
- Decide whether the first five-finger evaluation should keep the binary open/close API for comparability or expose richer hand actions immediately
- Decide whether the clutter task needs another prompt pass before expanding it into a broader benchmark tier
- Expand the initial Phase 3 YCB bridge beyond smoke-test level and characterize failure modes
- Run and review the first larger benchmark for the YCB target-clutter variant
- Improve grasp robustness for the YCB target-clutter variant before treating it as a stable benchmark tier
- Turn representative successful trials into montage videos for easier review and sharing
