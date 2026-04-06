from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from capx.envs.simulators.robosuite_shape_lift import FrankaRobosuiteShapeLiftLowLevel
from capx.integrations.franka.common import apply_tcp_offset
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.franka.hand_primitives import (
    OPPOSITE_DIRECTION,
    WORLD_DIRECTIONS,
    primitive_quaternion_wxyz,
)


@dataclass(frozen=True)
class ShapePosePreset:
    palm_face: str
    middle_finger_direction: str


DEFAULT_SHAPE_POSE_PRESETS: dict[str, ShapePosePreset] = {
    "box": ShapePosePreset("up", "forward"),
    "cylinder": ShapePosePreset("up", "forward"),
    "ball": ShapePosePreset("back", "down"),
}


def _support_radius(extent_xyz: np.ndarray, direction: np.ndarray) -> float:
    half_extents = 0.5 * np.asarray(extent_xyz, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    return float(np.abs(direction) @ half_extents)


class ExactSystemHandPPOEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reduced PPO wrapper around the exact Robosuite PandaDexRH + Inspire setup.

    The policy controls 6 hand command channels plus a scalar lift fraction while
    wrist orientation and XY position are fixed per shape family candidate.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        fixed_shape: str = "box",
        max_episode_steps: int = 40,
        settle_steps: int = 4,
        lift_height: float = 0.12,
        palm_face: str | None = None,
        middle_finger_direction: str | None = None,
    ) -> None:
        super().__init__()
        self.fixed_shape = fixed_shape
        preset = DEFAULT_SHAPE_POSE_PRESETS[fixed_shape]
        self.palm_face = palm_face or preset.palm_face
        self.middle_finger_direction = middle_finger_direction or preset.middle_finger_direction
        self.max_episode_steps = int(max_episode_steps)
        self.settle_steps = int(settle_steps)
        self.lift_height = float(lift_height)

        self.env = FrankaRobosuiteShapeLiftLowLevel(
            privileged=False,
            enable_render=False,
            robot_name="PandaDexRH",
            hand_name="inspire_right",
            robosuite_gripper_type="default",
            ik_robot_name="panda_description",
            ik_target_link_name="panda_hand",
            eef_body_name="gripper0_right_eef",
            tcp_offset=[0.0, 0.0, -0.107],
            fixed_shape=fixed_shape,
        )
        self.api = FrankaControlPrivilegedApi(self.env)

        self._command_low = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, -0.5], dtype=np.float32)
        self._command_high = np.array([1.5, 1.5, 1.5, 1.5, 3.0, 3.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.concatenate([self._command_low, np.array([0.0], dtype=np.float32)]),
            high=np.concatenate([self._command_high, np.array([1.0], dtype=np.float32)]),
            dtype=np.float32,
        )
        # robot_joint_pos (8) + robot_cartesian_pos (8) + cube pose (7) + rel pos (3) + bbox (3) + current cmd (6) + lift (1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32,
        )

        self._current_command = self._command_low.copy()
        self._current_lift_fraction = 0.0
        self._step_idx = 0
        self._target_position = np.zeros(3, dtype=np.float64)
        self._lift_base_position = np.zeros(3, dtype=np.float64)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._bbox = np.zeros(3, dtype=np.float64)

    def _fast_goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        target_pos = apply_tcp_offset(pos, quat_wxyz, self.api._tcp_offset)
        if self.api.cfg is None:
            self.api.cfg = self.api.ik_solve_fn(target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]))
        else:
            self.api.cfg = self.api.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
                prev_cfg=self.api.cfg,
            )
        joints = np.asarray(self.api.cfg[:-1], dtype=np.float64).reshape(7)
        self.env.move_to_joints_blocking(joints, tolerance=0.05, max_steps=40)

    def _compose_observation(self, obs: dict[str, Any]) -> np.ndarray:
        cube_pose = np.asarray(obs["cube_poses"]["primary"], dtype=np.float32)
        rel_pos = np.asarray(obs["gripper_to_cube_pos"], dtype=np.float32)
        robot_joint = np.asarray(obs["robot_joint_pos"], dtype=np.float32)
        robot_cart = np.asarray(obs["robot_cartesian_pos"], dtype=np.float32)
        return np.concatenate(
            [
                robot_joint,
                robot_cart,
                cube_pose,
                rel_pos,
                self._bbox.astype(np.float32),
                self._current_command.astype(np.float32),
                np.array([self._current_lift_fraction], dtype=np.float32),
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        del info
        self.api.cfg = None
        self._step_idx = 0
        self._current_command = self._command_low.copy()
        self._current_lift_fraction = 0.0

        obj_pos, _, bbox = self.api.get_object_pose("object", return_bbox_extent=True)
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        self._bbox = np.asarray(bbox, dtype=np.float64)
        self._target_quat = primitive_quaternion_wxyz(self.palm_face, self.middle_finger_direction)

        palm_dir = np.asarray(WORLD_DIRECTIONS[self.palm_face], dtype=np.float64)
        approach_dir = np.asarray(WORLD_DIRECTIONS[OPPOSITE_DIRECTION[self.palm_face]], dtype=np.float64)
        radius = _support_radius(self._bbox, palm_dir)

        self._lift_base_position = obj_pos - palm_dir * (radius + 0.025)
        pre_position = self._lift_base_position + approach_dir * 0.08

        self.api.open_gripper()
        self._fast_goto_pose(pre_position, self._target_quat)
        self._fast_goto_pose(self._lift_base_position, self._target_quat)
        obs = self.env.get_observation()
        return self._compose_observation(obs), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(7)
        self._step_idx += 1
        self._current_command = np.clip(action[:6], self._command_low, self._command_high)
        self._current_lift_fraction = float(np.clip(action[6], 0.0, 1.0))

        self.env._set_gripper_command(self._current_command)
        for _ in range(self.settle_steps):
            self.env._step_once()

        commanded_position = self._lift_base_position + np.array(
            [0.0, 0.0, self.lift_height * self._current_lift_fraction], dtype=np.float64
        )
        self._fast_goto_pose(commanded_position, self._target_quat)

        obs = self.env.get_observation()
        reward = float(self.env.compute_reward())
        terminated = bool(self.env.task_completed())
        truncated = self._step_idx >= self.max_episode_steps
        info = {
            "success": terminated,
            "shape": self.fixed_shape,
            "palm_face": self.palm_face,
            "middle_finger_direction": self.middle_finger_direction,
        }
        return self._compose_observation(obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
