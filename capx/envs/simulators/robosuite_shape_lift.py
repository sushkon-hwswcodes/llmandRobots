"""Low-level Robosuite Franka environment for shape-generalization experiments.

Identical to FrankaRobosuiteCubeLiftLowLevel but uses LiftShape instead of Lift,
so the grasped object is randomly chosen (box / cylinder / ball) at each reset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from robosuite.environments.manipulation.lift_shape import LiftShape


class FrankaRobosuiteShapeLiftLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Shape Lift: random object shape per episode."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        robot_name: str = "Panda",
        hand_name: str = "panda",
        robosuite_gripper_type: str = "default",
        ik_robot_name: str = "panda_description",
        ik_target_link_name: str = "panda_hand",
        eef_body_name: str = "gripper0_right_eef",
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 0.0, -0.107),
        gripper_open_command: float = -1.0,
        gripper_closed_command: float = 1.0,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
            robot_name=robot_name,
            hand_name=hand_name,
            robosuite_gripper_type=robosuite_gripper_type,
            ik_robot_name=ik_robot_name,
            ik_target_link_name=ik_target_link_name,
            eef_body_name=eef_body_name,
            tcp_offset=tcp_offset,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        )

        lift_kwargs = dict(
            robots=[self.robot_name],
            gripper_types=self.robosuite_gripper_type,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(
                controller=self.controller_cfg
            ),
            horizon=max_steps,
            reward_shaping=True,
        )

        if privileged and not enable_render:
            lift_kwargs.update(
                use_camera_obs=False,
                has_offscreen_renderer=False,
                camera_names=[],
            )

        self.robosuite_env = LiftShape(**lift_kwargs)
        self._initial_cube_height: float | None = None
        self._init_robot_links()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        # Baseline object height after settling; used for robust lift success across varying shape sizes.
        self._initial_cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        info = {"task_prompt": "Pick up the red object and lift it."}
        return obs, info

    def _object_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        obj_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]])
        )
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        obj_robot_base = base_transform @ obj_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [obj_robot_base.translation(), obj_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        if self.task_completed():
            return 1.0

        # Dense shaping: progress is measured by lift above reset baseline (not absolute table height).
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        return float(np.clip((cube_height - baseline) / 0.04, 0.0, 1.0))

    def task_completed(self) -> bool:
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        lifted = cube_height > (baseline + 0.04)
        grasped = self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cube,
        )
        return bool(lifted and grasped)

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._object_pose_dict(robosuite_obs)
        robosuite_obs["cube_poses"] = {
            "primary": np.asarray(pose_dict["primary"], dtype=np.float32),
        }
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs


__all__ = ["FrankaRobosuiteShapeLiftLowLevel"]
