# Project Status

Last updated: 2026-04-05

## Current branch state

- Branch: `main`
- Remote tracking: `origin/main`
- Latest committed milestone: `8ef3c06` "Log PandaDexRH inspire-hand smoke benchmark result"

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
- The raw `Panda + InspireRightHand` composition was only partially integrated: it instantiated and moved, but the first smoke video showed the hand geometry was not attaching / rendering correctly
- Better path identified: use Robosuite's built-in `PandaDexRH` composition, which already includes the Inspire-hand mount quaternion offset
- `PandaDexRH` instantiates and steps correctly with action dimension `13`, and all Inspire hand bodies appear in the assembled MuJoCo model
- Motion result: the existing privileged API can execute `goto_pose(...)` against the dexterous-hand profile when the standard Panda Pyroki server is running
- Added opt-in smoke config: `env_configs/shape_generalization/franka_qwen_shape_inspire_smoke.yaml`
- Updated smoke result: with `PandaDexRH`, the saved trial video now shows the hand visibly attached in both close-up and overview views
- Corrected 3-trial smoke benchmark result: `1/3` task-complete with average reward `0.333` using the unchanged binary open/close lift policy
- Added the first opt-in dexterous-hand control layer on top of the same low-level env path:
  - `get_hand_capabilities()`
  - `set_hand_preshape("open" | "pregrasp" | "grasp_soft" | "close")`
  - `set_hand_joints([...])` for explicit 6-value Inspire commands
- Backward-compatibility preserved: Panda still uses the original scalar open / close path, and existing Panda benchmark configs were not changed
- Direct control check: the Inspire hand now moves through named preshapes and explicit 6-value commands in simulation without changing the arm stack
- First richer-hand prompt check: a 1-trial smoke run successfully used `get_hand_capabilities()` and `set_hand_preshape(...)` from the prompt, but still failed the lift with reward `0.033`
- Current readout: the model can adopt richer hand actions immediately, but grasp robustness still appears to be the limiting factor rather than API discoverability
- New prompt direction in progress: the Inspire smoke config now emphasizes shape-based preshape choice, two-stage descent, a short test lift, and only one reinforcement grasp before the full lift
- First staged-prompt smoke artifact: the model followed the staged structure closely, but the phrase "if the object seems unstable" led it to invent an unsupported `is_object_stable()` helper and fail in the sandbox
- Prompt tightened again: the reinforcement grasp is now deterministic and the config explicitly forbids inventing helper checks beyond the exposed APIs
- Latest staged-prompt result: the model executed the intended two-stage descent and reinforcement grasp without sandbox errors, but it also redefined provided API names locally; the smoke prompt now explicitly forbids shadowing those APIs
- Prompt wording fix in progress: the Inspire smoke config now explicitly forbids `open_gripper()` / `close_gripper()` so the dexterous-hand path is exclusive rather than optional
- Prompt tightened further: the Inspire smoke config now requires a straight-line script order and explicitly forbids defining helper functions or classes, reducing the chance that the model shadows the real APIs again
- Literal-prompt smoke result: the model now stays on the real Inspire-hand API path, performs real IK-driven approach motion, and no longer falls back to placeholder functions or legacy Panda helpers; however, the grasp still failed with reward `0.0`

## Current local artifacts

- `benchmark_green_target_clutter_30.log`: full 30-trial clutter benchmark output
- `scripts/make_success_video_grids.py`: helper for assembling 4x2 success-video grids
- `docs/hand_configuration.md`: notes on the new configurable hand path and the parameters that need to be swapped for a future five-finger hand
- `benchmark_shape_inspire_smoke_richer_hand_1trial.log`: first prompt-level smoke artifact using the richer Inspire-hand API

## What is done

- Robosuite privileged baseline is stable at `20/20`
- Phase 2 shape generalization is implemented and benchmarked at `25/30`
- Phase 2.75 green-target clutter is implemented and benchmarked at `24/30`
- The clutter failure review is complete enough to say the misses are grasp-execution failures, not target-selection failures
- Phase 3 real-world objects has started with a first YCB-backed lift environment and prompt/config bridge
- Phase 3 target-object selection from real-object clutter has an initial implementation and smoke-test coverage
- Phase 3 target-object selection from real-object clutter has now been benchmarked once at `9/30`
- The hand path is now configurable without replacing Panda, and the current benchmark configs explicitly preserve Panda settings
- The first richer-hand API pass is now implemented for the Inspire smoke path, and the model has already used it in a benchmark trial

## What still needs attention

- Standardize the first five-finger profile around `PandaDexRH` and propagate that profile through the smoke config and future benchmark configs
- Review the richer-hand smoke video and compare it to the earlier binary-hand smoke artifacts to identify whether finger posture or approach motion is the main remaining limiter
- Measure whether the safer staged prompt improves the Inspire smoke result before changing `sample_grasp_pose(...)`
- Confirm that the tightened staged prompt stops sandbox failures from invented helper checks and then compare its grasp outcome against the earlier richer-hand trial
- Confirm that the revised prompt keeps the model on the real Inspire-hand API path and does not fall back to legacy Panda helpers
- Improve the actual grasp target / pose selection next, since prompt tightening alone now produces the intended motion sequence but not a successful lift
- Decide whether the clutter task needs another prompt pass before expanding it into a broader benchmark tier
- Expand the initial Phase 3 YCB bridge beyond smoke-test level and characterize failure modes
- Run and review the first larger benchmark for the YCB target-clutter variant
- Improve grasp robustness for the YCB target-clutter variant before treating it as a stable benchmark tier
- Turn representative successful trials into montage videos for easier review and sharing
