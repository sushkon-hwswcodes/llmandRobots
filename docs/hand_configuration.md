# Hand Configuration

This repo keeps Panda as the benchmark default while allowing low-level Robosuite
configs to swap the mounted hand / gripper metadata explicitly.

## Active Panda defaults

```yaml
low_level_kwargs:
  robot_name: Panda
  hand_name: panda
  robosuite_gripper_type: default
  ik_robot_name: panda_description
  ik_target_link_name: panda_hand
  eef_body_name: gripper0_right_eef
  tcp_offset: [0.0, 0.0, -0.107]
```

These are the settings that current Panda benchmark YAMLs pin explicitly.

## Minimum fields for a hand swap

To instantiate a different mounted hand in simulation, update:

1. `low_level_kwargs.robot_name`
2. `low_level_kwargs.hand_name`
3. `low_level_kwargs.robosuite_gripper_type`
4. `low_level_kwargs.ik_robot_name`
5. `low_level_kwargs.ik_target_link_name`
6. `low_level_kwargs.eef_body_name`
7. `low_level_kwargs.tcp_offset`

Also keep the matching `api_servers.robot` and `api_servers.target_link` values in
sync with the robot profile you are testing.

## Current smoke-tested non-Panda profile

The simplest currently reachable non-Panda-hand simulation target is:

```yaml
low_level_kwargs:
  robot_name: PandaDexRH
  hand_name: inspire_right
  robosuite_gripper_type: default
```

This profile is kept only as a simulation / instantiation smoke target for now.
The previous prompt-level grasp experiments for that hand path were removed so we
can restart from a smaller surface area.

## Current paused state

- The current Inspire path can instantiate, reset, step, and replay candidate hand command vectors.
- Candidate hand preset videos were exported under `outputs/hand_preset_videos_candidate_v1/`.
- Those candidate presets are not yet validated grasp primitives.
- The current harness keeps the wrist fixed, so it demonstrates articulation, not object pickup quality.
- Any future hand grasp restart should begin with constrained primitives that include:
  - hand shape
  - wrist orientation
  - approach direction / offset
  - optional close profile

## Current external references

- Allegro path: useful for its small named-grasp API and saved-pose workflow.
- RH56DFX path: useful for its public grasp families (`line`, `plane`, `cylinder`) and geometric / force-control planning ideas.
- Neither path is a direct drop-in preset library for this repo's current Robosuite Inspire setup, so the implementation here will still need local validation.

## What is still guaranteed

- Panda benchmark configs remain the reference path.
- Alternate hand profiles can still be instantiated in Robosuite.
- The shared privileged API still supports generic `open_gripper()` /
  `close_gripper()` behavior for dexterous grippers, but the project-local
  named preshape / preset experiment layer has been removed.

## How to validate a hand config

Run the dedicated smoke test:

```bash
uv run pytest tests/test_hand_configs.py -q
```
