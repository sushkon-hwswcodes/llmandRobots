# Benchmark Results

Last updated: 2026-04-06

These are the compact local benchmark readouts we decided to preserve in the
repo without checking in the full generated output trees.

## Local Ollama model

- Model: `ollama/qwen2.5-coder:7b-instruct-q4_K_M`

## Recovered synthetic benchmarks

- Shape generalization
  - Config: `env_configs/shape_generalization/franka_qwen_shape.yaml`
  - Trials: `5`
  - Result: `5/5`
  - Average reward: `1.000`
  - Code generation success: `1.000`
  - Git commit recorded in run output: `0d14778`

- Green-target clutter
  - Config: `env_configs/shape_generalization/franka_qwen_green_target_clutter.yaml`
  - Trials: `30`
  - Result: `24/30`
  - Average reward: `0.807`
  - Code generation success: `1.000`
  - Git commit recorded in run output: `0d14778`

## Real-object path

- YCB single-object lift
  - Config: `env_configs/real_objects/franka_qwen_ycb.yaml`
  - Trials: `20`
  - Result: `13/20`
  - Average reward: `0.750`
  - Code generation success: `1.000`
  - Git commit recorded in run output: `db7f255`

- YCB target clutter
  - Config: `env_configs/real_objects/franka_qwen_ycb_target_clutter.yaml`
  - Trials: `30`
  - Result: `12/30`
  - Average reward: `0.402`
  - Code generation success: `1.000`
  - Git commit recorded in run output: `db7f255`

## Notes

- The synthetic shape benchmark was restored by replacing the dead historical
  Robosuite `lift_shape` dependency with a maintained local shape task wrapper.
- The strongest recent recovery came from restoring center-grasp semantics for
  the maintained Panda / shape / clutter / YCB path.
- Full generated run artifacts remain in `outputs/` on this machine if deeper
  inspection is needed later.
