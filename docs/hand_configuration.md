# Hand Configuration

This codebase now keeps the current Panda hand as the explicit default while
allowing low-level task configs to override hand-specific parameters.

## Active Panda defaults

The benchmark configs currently pin these values:

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

Those settings preserve the current benchmark behavior and make the Panda runs
reproducible.

## What a future hand swap needs

To test a different hand, update:

1. `low_level_kwargs.robot_name`
2. `low_level_kwargs.robosuite_gripper_type`
3. `low_level_kwargs.ik_robot_name`
4. `low_level_kwargs.ik_target_link_name`
5. `low_level_kwargs.eef_body_name`
6. `low_level_kwargs.tcp_offset`
7. The matching `api_servers.robot` and `api_servers.target_link` values in the YAML

## First five-finger candidate

The current best low-friction target is the built-in Robosuite composite robot:

```yaml
low_level_kwargs:
  robot_name: PandaDexRH
  hand_name: inspire_right
  robosuite_gripper_type: default
```

This path is better than raw `Panda + InspireRightHand` because Robosuite already
defines the necessary gripper mount quaternion offset in `PandaDexRH`.

Current status:
- `PandaDexRH` instantiates, resets, and steps correctly.
- The hand is visibly attached in saved smoke-test video frames.
- The first richer-hand control layer now exists for the Inspire smoke path:
  - `get_hand_capabilities()`
  - `set_hand_preshape("open" | "pregrasp" | "grasp_soft" | "close")`
  - `set_hand_joints([...])` with 6 values
- Panda compatibility is preserved because the older `open_gripper()` and
  `close_gripper()` helpers still exist and still drive the scalar Panda path.

## Current limitation

The richer Inspire-hand API is available, but the grasp strategy is still mostly
the same top-down one-shot lift sequence. This means the next performance gains
are likely to come from better approach / grasp-pose behavior, not just from
adding more hand joints or more named preshapes.
