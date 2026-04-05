from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.motion.pyroki import init_pyroki


# ------------------------------- Control API ------------------------------
class FrankaControlPrivilegedApi(ApiBase):
    """Robot control helpers for Franka.

    Functions:
      - get_object_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - sample_grasp_pose(object_name: str) -> (position: np.ndarray, quaternion_wxyz: np.ndarray):
      - goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None
      - open_gripper() -> None
      - close_gripper() -> None
    """

    _TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)

    def __init__(self, env: BaseEnv, multi_turn: bool = False) -> None:
        super().__init__(env)
        # Lazy-import to keep startup light
        # from capx.integrations.motion import pyroki_snippets as pks  # type: ignore
        # from capx.integrations.motion.pyroki_context import get_pyroki_context  # type: ignore

        # ctx = get_pyroki_context("panda_description", target_link_name="panda_hand")
        # self._robot = ctx.robot
        # self._target_link_name = ctx.target_link_name
        # self._pks = pks
        self.ik_solve_fn = init_pyroki()
        self.cfg = None
        self.multi_turn = multi_turn

    def functions(self) -> dict[str, Any]:
        base_functions = {
            "get_object_pose": self.get_object_pose,
            "get_object_shape": self.get_object_shape,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            # "home_pose": self.home_pose,
        }
        # if self.multi_turn:
        #     base_functions["breakpoint_code_block"] = self.breakpoint_code_block
        return base_functions

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp pose quaternion OR (0, 0, 1, 0) wxyz as the gripper down orientation if using this for placement position.

        Args:
            object_name: The name of the object to get the pose of.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) object extent in meters of x, y, z axes respectively in the world frame (full side length, not half-length extent). If return_bbox_extent is False, returns None.
        """
        obs = self._env.get_observation()

        # Accept any name that refers to the primary object
        is_primary = (
            ("red" in object_name and "cube" in object_name)
            or object_name in ("object", "red_object", "red object", "cube")
        )
        if is_primary:
            bbox = self._get_primary_bbox()
            return (
                obs["cube_poses"]["primary"][:3],
                obs["cube_poses"]["primary"][3:],
                bbox,
            )
        elif "green" in object_name and "cube" in object_name:
            return (
                obs["cube_poses"]["secondary"][:3],
                obs["cube_poses"]["secondary"][3:],
                np.array([0.05, 0.05, 0.05]),
            )
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def _get_primary_bbox(self) -> np.ndarray:
        """Return full bounding-box extents (x, y, z) in metres for the primary object."""
        rs_env = getattr(self._env, "robosuite_env", None)
        if rs_env is not None and hasattr(rs_env, "_current_size") and hasattr(rs_env, "_current_shape"):
            size = rs_env._current_size   # MuJoCo geom_size: half-extents / radius
            shape = rs_env._current_shape
            if shape == "box":
                return np.array([size[0] * 2, size[1] * 2, size[2] * 2])
            elif shape == "cylinder":
                return np.array([size[0] * 2, size[0] * 2, size[1] * 2])  # [diam, diam, height]
            else:  # ball
                return np.array([size[0] * 2, size[0] * 2, size[0] * 2])
        # Fallback for plain cube environment
        return np.array([0.05, 0.05, 0.05])

    def get_object_shape(self) -> dict:
        """Return the shape type and dimensions of the object to be grasped.

        Returns a dict with keys:
          shape  : str  — "box" | "cylinder" | "ball"
          size   : dict — shape-specific dimensions in metres (full lengths, not half):
            box      → {"x": float, "y": float, "z": float}
            cylinder → {"diameter": float, "height": float}
            ball     → {"diameter": float}
          grasp_hint : str — suggested grasp strategy
        """
        rs_env = getattr(self._env, "robosuite_env", None)
        if rs_env is None or not hasattr(rs_env, "_current_shape"):
            return {"shape": "box", "size": {"x": 0.05, "y": 0.05, "z": 0.05},
                    "grasp_hint": "top-down grasp"}

        shape = rs_env._current_shape
        s = rs_env._current_size  # raw MuJoCo geom_size half-extents

        if shape == "box":
            return {
                "shape": "box",
                "size": {"x": round(float(s[0]) * 2, 4),
                         "y": round(float(s[1]) * 2, 4),
                         "z": round(float(s[2]) * 2, 4)},
                "grasp_hint": (
                    "Top-down grasp. Set grasp_pos[2] = object_pos[2] + size['z']/2 + 0.01. "
                    "Finger gap should exceed max(size['x'], size['y'])."
                ),
            }
        elif shape == "cylinder":
            diameter = round(float(s[0]) * 2, 4)
            height   = round(float(s[1]) * 2, 4)
            return {
                "shape": "cylinder",
                "size": {"diameter": diameter, "height": height},
                "grasp_hint": (
                    "Top-down grasp. Set grasp_pos[2] = object_pos[2] + size['height']/2 + 0.01. "
                    "Finger gap should exceed size['diameter']."
                ),
            }
        else:  # ball
            diameter = round(float(s[0]) * 2, 4)
            return {
                "shape": "ball",
                "size": {"diameter": diameter},
                "grasp_hint": (
                    "Top-down grasp. Set grasp_pos[2] = object_pos[2] + size['diameter']/2 + 0.01. "
                    "Finger gap should exceed size['diameter']."
                ),
            }

    def sample_grasp_pose(self, object_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for.

        Returns:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
        """
        obs = self._env.get_observation()

        if "red" in object_name and "cube" in object_name:
            return obs["cube_poses"]["primary"][:3], np.array([0, 0, 1, 0])
        elif "green" in object_name and "cube" in object_name:
            return obs["cube_poses"]["secondary"][:3], np.array([0, 0, 1, 0])
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> None:
        """Go to pose using Inverse Kinematics.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            position: (3,) XYZ in meters.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            z_approach: (float) Z-axis distance offset for goto_pose insertion approach motion. Will first arrive at position + z_approach meters in Z-axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            None
        """

        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)
        rot = SciRotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        )

        if z_approach != 0.0:
            z_offset_pos = offset_pos + rot.apply(np.array([0, 0, -z_approach]))
            self._solve_and_move(quat_wxyz, z_offset_pos)

        self._solve_and_move(quat_wxyz, offset_pos)

    def _solve_and_move(self, quat_wxyz: np.ndarray, target_pos: np.ndarray) -> None:
        """Solve IK and move to target position (helper to reduce goto_pose duplication)."""
        if self.cfg is None:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
            )
        else:
            self.cfg = self.ik_solve_fn(
                target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
                prev_cfg=self.cfg,
            )
        joints = np.asarray(self.cfg[:-1], dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=40)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=60)

    def home_pose(self) -> None:
        """
        Move the robot to a safe home pose.
        Args:
            None
        Returns:
            None
        """

        # joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
        joints = np.array(
            [
                -2.95353726e-02,
                1.69197371e-01,
                2.39244731e-03,
                -2.64089311e00,
                -2.01237851e-03,
                2.94565778e00,
                8.31390616e-01,
            ]
        )
        self._env.move_to_joints_blocking(joints)
