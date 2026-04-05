# Project Plan

Last updated: 2026-04-05

## Immediate next steps

1. Preserve the current Panda benchmark path as the reproducible baseline while hand-configuration work lands.
2. Use the new hand-config plumbing to define the first non-Panda-hand candidate profile without changing existing benchmark YAML behavior.
3. Standardize on `PandaDexRH` as the first five-finger simulation target, since Robosuite already provides the correct Inspire-hand mount composition for Panda.
4. Keep the smoke config and future hand benchmarks aligned to that profile: `robot_name: PandaDexRH`, `robosuite_gripper_type: default`, and the existing Panda IK server settings.
5. Keep the new dexterous-hand API narrow and backward-compatible: preserve `open_gripper()` / `close_gripper()` for Panda while exposing opt-in `get_hand_capabilities()`, `set_hand_preshape(...)`, and `set_hand_joints([...])` for the Inspire path.
6. Use the updated `franka_qwen_shape_inspire_smoke.yaml` config to enforce a safer Inspire-hand grasp routine: shape-based preshape choice, two-stage descent, short test lift, and one deterministic reinforcement grasp using only exposed APIs.
7. Benchmark that prompt-only behavior change before modifying `sample_grasp_pose(...)`, so we can separate prompt gains from low-level grasp-target gains.
8. Keep the clutter-failure review and YCB target-clutter follow-up as the next benchmark tasks once the hand path is stable.

## Short-term goals

1. Keep the already-solved Panda baseline (`20/20`) and shape/clutter results reproducible while making the hand path configurable.
2. Add a first working five-finger hand profile as an opt-in configuration rather than a replacement.
3. Characterize whether the first five-finger benchmark improves more from richer hand actions or from better approach / grasp-pose logic.
4. Raise the green-target clutter task from `24/30` toward the shape-generalization level of `25/30` or better once the hand migration path is understood.
5. Improve the first YCB bridge by replacing the temporary OBJ-for-collision fallback, which currently exists because MuJoCo would not load the provided YCB `collision.ply` files in this setup.
6. Keep the first real-object clutter benchmark focused on graspable YCB objects so the main variable is cluttered target selection rather than impossible grasps.
7. Improve the target-clutter grasp policy or success shaping so the current `9/30` result becomes a more stable real-object clutter baseline.

## Medium-term goals

1. Promote the shape-generalization and green-target clutter tasks into a more formal benchmark suite once failure modes are understood.
2. Reuse the same evaluation harness to compare parallel-jaw Panda behavior against the first five-finger hand configuration on the same lift/clutter tasks.
3. Keep collecting benchmark artifacts that are easy to audit: logs, saved code, overview videos, and montage summaries.
4. Implement Phase 3 by replacing synthetic/generated shapes with real-world objects while keeping the evaluation protocol comparable.
5. Extend Phase 3 from single-object lift into harder real-object variants after the base YCB lift is benchmarked.
6. Extend Phase 3 from single-object lift into specific-target selection from real-object clutter once the base YCB lift and first clutter benchmark are characterized.

## Planned phase sequence

1. Phase 2: shape generalization over generated shapes.
2. Phase 2.75: green-target selection in clutter.
3. Phase 3: real-world objects instead of randomly generated shapes.

## Decision rule for the next iteration

- If the first five-finger hand can be instantiated with Panda and existing Robosuite tasks, prioritize that path over introducing a new arm stack.
- If the five-finger hand only works with a different arm/URDF stack, pause before implementation and compare that migration cost against the value of benchmark continuity.
- If the five-finger hand can run through richer hand actions without disturbing Panda, test that narrow API before widening the control surface further.
- If clutter failures remain dominated by near-success grasps after the hand path is stable, prioritize simulator reward/success shaping and grasp-pose affordances over prompt wording changes.
