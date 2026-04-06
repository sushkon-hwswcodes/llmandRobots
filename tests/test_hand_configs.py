import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel


def test_pandadexrh_smoke_instantiates_and_steps() -> None:
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

    obs, _ = env.reset(seed=0)

    assert "cube_poses" in obs
    assert env.robosuite_env.action_dim > 7
    assert env._gripper_action_dim == env.robosuite_env.action_dim - 7

    env._set_gripper(1.0)
    for _ in range(3):
        env._step_once()

    env._set_gripper(0.0)
    for _ in range(3):
        env._step_once()

    obs_after = env.get_observation()
    assert "cube_poses" in obs_after
