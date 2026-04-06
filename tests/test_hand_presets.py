import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
from capx.integrations.franka.hand_presets import INSPIRE_COMMAND_LABELS, get_hand_preset_library


def test_inspire_candidate_presets_have_expected_shape() -> None:
    presets = get_hand_preset_library("inspire_right")
    allegro_presets = get_hand_preset_library("inspire_right_allegro_v1")

    assert len(INSPIRE_COMMAND_LABELS) == 6
    for preset in presets.values():
        assert preset.command.shape == (6,)
        assert np.isfinite(preset.command).all()
    for preset in allegro_presets.values():
        assert preset.command.shape == (6,)
        assert np.isfinite(preset.command).all()


def test_inspire_candidate_preset_steps_in_sim() -> None:
    env = FrankaRobosuiteCubeLiftLowLevel(
        privileged=True,
        enable_render=False,
        robot_name="PandaDexRH",
        hand_name="inspire_right",
        robosuite_gripper_type="default",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
    )
    env.reset(seed=0)

    presets = get_hand_preset_library("inspire_right")
    env._set_gripper_command(presets["open"].command)
    for _ in range(10):
        env._step_once()
    open_qpos = env.robosuite_env.sim.data.qpos.copy()

    env._set_gripper_command(presets["pinch"].command)
    for _ in range(10):
        env._step_once()
    pinch_qpos = env.robosuite_env.sim.data.qpos.copy()

    assert env._gripper_action_dim == 6
    assert np.isfinite(pinch_qpos).all()
    assert np.linalg.norm(pinch_qpos - open_qpos) > 1e-3


def test_inspire_allegro_reference_preset_steps_in_sim() -> None:
    env = FrankaRobosuiteCubeLiftLowLevel(
        privileged=True,
        enable_render=False,
        robot_name="PandaDexRH",
        hand_name="inspire_right",
        robosuite_gripper_type="default",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
    )
    env.reset(seed=0)

    presets = get_hand_preset_library("inspire_right_allegro_v1")
    env._set_gripper_command(presets["home"].command)
    for _ in range(10):
        env._step_once()
    home_qpos = env.robosuite_env.sim.data.qpos.copy()

    env._set_gripper_command(presets["pinch_it"].command)
    for _ in range(10):
        env._step_once()
    pinch_qpos = env.robosuite_env.sim.data.qpos.copy()

    assert np.isfinite(home_qpos).all()
    assert np.isfinite(pinch_qpos).all()
    assert np.linalg.norm(pinch_qpos - home_qpos) > 1e-3
