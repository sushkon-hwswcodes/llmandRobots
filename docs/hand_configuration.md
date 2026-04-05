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

The current best low-friction target is:

```yaml
low_level_kwargs:
  robot_name: Panda
  hand_name: inspire_right
  robosuite_gripper_type: InspireRightHand
```

That profile now instantiates and resets in the existing shape-lift low-level env.
It should be treated as a compatibility smoke target for now, not yet as a
policy-complete dexterous-hand benchmark.

## Current limitation

The policy still uses a binary open / close abstraction. This means a new
five-finger hand can be swapped in at the configuration level, but we should not
expect dexterous-hand benefits until we expose richer grasp primitives and
hand-aware grasp pose generation.
