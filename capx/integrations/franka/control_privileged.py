from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import (
    BaseEnv,
)
from capx.integrations.base_api import ApiBase
from capx.integrations.franka.common import (
    DEFAULT_TCP_OFFSET,
    apply_tcp_offset,
    close_gripper as _close_gripper,
    open_gripper as _open_gripper,
)
from capx.integrations.franka.hand_presets import get_hand_preset_library
from capx.integrations.franka.hand_primitives import (
    WORLD_DIRECTIONS,
    get_hand_primitive_library,
    primitive_quaternion_wxyz,
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

    def __init__(
        self,
        env: BaseEnv,
        multi_turn: bool = False,
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    ) -> None:
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
        if tcp_offset is None:
            tcp_offset = getattr(env, "tcp_offset", DEFAULT_TCP_OFFSET)
        self._tcp_offset = np.asarray(tcp_offset, dtype=np.float64)
        self._hand_name = str(getattr(env, "hand_name", "panda"))
        self._robot_name = str(getattr(env, "robot_name", "Panda"))
        self._gripper_action_dim = int(getattr(env, "_gripper_action_dim", 1))

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
        if self._supports_dexterous_hand():
            base_functions["list_hand_presets"] = self.list_hand_presets
            base_functions["set_hand_preset"] = self.set_hand_preset
            base_functions["list_hand_primitives"] = self.list_hand_primitives
            base_functions["sample_hand_primitive_pose"] = self.sample_hand_primitive_pose
            base_functions["sample_hand_primitive_pregrasp_pose"] = self.sample_hand_primitive_pregrasp_pose
            base_functions["sample_hand_primitive_align_pose"] = self.sample_hand_primitive_align_pose
            base_functions["execute_hand_primitive_soft_close"] = self.execute_hand_primitive_soft_close
            base_functions["execute_hand_primitive_close"] = self.execute_hand_primitive_close
            base_functions["execute_hand_primitive"] = self.execute_hand_primitive
        # if self.multi_turn:
        #     base_functions["breakpoint_code_block"] = self.breakpoint_code_block
        return base_functions

    def _primary_name_aliases(self) -> set[str]:
        aliases = {
            "object",
            "red_object",
            "red object",
            "cube",
            "target",
            "target_object",
            "target object",
            "primary_object",
            "primary object",
            "green object",
            "green cube",
            "mesh",
        }
        rs_env = getattr(self._env, "robosuite_env", None)
        info = getattr(rs_env, "_current_object_info", None)
        if isinstance(info, dict):
            display_name = str(info.get("display_name", "")).strip().lower()
            category = str(info.get("category", "")).strip().lower()
            shape = str(info.get("shape", "")).strip().lower()
            if display_name:
                aliases.add(display_name)
            if category:
                aliases.add(category)
                aliases.add(category.replace("_", " "))
            if shape:
                aliases.add(shape)
        return aliases

    def _is_primary_object_name(self, object_name: str, has_secondary: bool) -> bool:
        name = object_name.lower().strip()
        aliases = self._primary_name_aliases()
        return (
            ("red" in name and "cube" in name)
            or name in aliases
            or (not has_secondary and ("green" in name))
        )

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
        name = object_name.lower().strip()
        has_secondary = "cube_poses" in obs and "secondary" in obs["cube_poses"]

        is_primary = self._is_primary_object_name(name, has_secondary)
        if is_primary:
            bbox = self._get_primary_bbox()
            return (
                obs["cube_poses"]["primary"][:3],
                obs["cube_poses"]["primary"][3:],
                bbox,
            )
        elif "green" in name and "cube" in name:
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
            size = np.asarray(rs_env._current_size, dtype=np.float64)
            shape = rs_env._current_shape
            if shape == "box":
                return np.array([size[0] * 2, size[1] * 2, size[2] * 2])
            elif shape == "cylinder":
                return np.array([size[0] * 2, size[0] * 2, size[1] * 2])
            elif shape == "ball":
                return np.array([size[0] * 2, size[0] * 2, size[0] * 2])
            elif shape == "mesh" and size.shape[0] >= 3:
                return np.array([size[0], size[1], size[2]])
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

        info = getattr(rs_env, "_current_object_info", None)
        if isinstance(info, dict) and info.get("shape") == "mesh":
            size = info.get("size", {})
            return {
                "shape": "mesh",
                "category": info.get("category", "object"),
                "display_name": info.get("display_name", "object"),
                "size": {
                    "x": round(float(size.get("x", 0.05)), 4),
                    "y": round(float(size.get("y", 0.05)), 4),
                    "z": round(float(size.get("z", 0.05)), 4),
                },
                "grasp_hint": info.get("grasp_hint", "Use a top-down grasp near the object center."),
            }

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
        rs_env = getattr(self._env, "robosuite_env", None)
        name = object_name.lower().strip()
        has_secondary = "cube_poses" in obs and "secondary" in obs["cube_poses"]
        is_primary = self._is_primary_object_name(name, has_secondary)

        if getattr(rs_env, "prefer_center_grasp_pose", False):
            info = getattr(rs_env, "_current_object_info", None)
            category = ""
            if isinstance(info, dict):
                category = str(info.get("category", "")).strip().lower()

            # Thin rectangular YCB packages tend to miss with a pure center target.
            # A small upward bias preserves the recovered center-grasp behavior while
            # giving the fingers a cleaner approach on these flatter boxes.
            center_z_bias = 0.0
            if category in {"003_cracker_box", "004_sugar_box", "008_pudding_box"}:
                center_z_bias = 0.005

            if is_primary:
                pos = np.asarray(obs["cube_poses"]["primary"][:3], dtype=np.float64).copy()
                pos[2] += center_z_bias
                return (pos, np.array([0, 0, 1, 0], dtype=np.float64))
            if "green" in name and "cube" in name:
                pos = np.asarray(obs["cube_poses"]["secondary"][:3], dtype=np.float64).copy()
                pos[2] += center_z_bias
                return (pos, np.array([0, 0, 1, 0], dtype=np.float64))

        # Preserve the historical Panda cube-lift behavior for plain cube envs.
        # Those envs do not expose shape metadata, and their earlier benchmarked
        # path sampled a grasp directly at the object center.
        if rs_env is None or not hasattr(rs_env, "_current_shape"):
            if is_primary:
                return (
                    np.asarray(obs["cube_poses"]["primary"][:3], dtype=np.float64).copy(),
                    np.array([0, 0, 1, 0], dtype=np.float64),
                )
            if "green" in name and "cube" in name:
                return (
                    np.asarray(obs["cube_poses"]["secondary"][:3], dtype=np.float64).copy(),
                    np.array([0, 0, 1, 0], dtype=np.float64),
                )

        if is_primary:
            object_pos = np.asarray(obs["cube_poses"]["primary"][:3], dtype=np.float64).copy()
            grasp_pos = self._shape_aware_grasp_position(object_pos)
            grasp_quat = np.array([0, 0, 1, 0], dtype=np.float64)
            return grasp_pos, grasp_quat
        elif "green" in name and "cube" in name:
            object_pos = np.asarray(obs["cube_poses"]["secondary"][:3], dtype=np.float64).copy()
            grasp_pos = self._shape_aware_grasp_position(
                object_pos,
                fallback_shape="box",
                fallback_bbox=np.array([0.05, 0.05, 0.05], dtype=np.float64),
            )
            grasp_quat = np.array([0, 0, 1, 0], dtype=np.float64)
            return grasp_pos, grasp_quat
        else:
            raise ValueError(f"Invalid object name: {object_name}")

    def _shape_aware_grasp_position(
        self,
        object_pos: np.ndarray,
        *,
        fallback_shape: str | None = None,
        fallback_bbox: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a top-down grasp target adjusted for object shape."""
        rs_env = getattr(self._env, "robosuite_env", None)
        shape = fallback_shape
        if shape is None and rs_env is not None and hasattr(rs_env, "_current_shape"):
            shape = str(rs_env._current_shape)
        if shape is None:
            shape = "box"

        bbox = fallback_bbox
        if bbox is None:
            bbox = self._get_primary_bbox()
        bbox = np.asarray(bbox, dtype=np.float64).reshape(3)

        target = np.asarray(object_pos, dtype=np.float64).copy()
        if shape == "box":
            target[2] += bbox[2] * 0.5 + 0.01
        elif shape == "cylinder":
            target[2] += bbox[2] * 0.5 + 0.01
        elif shape == "ball":
            target[2] += bbox[2] * 0.5 + 0.01
        else:
            target[2] += bbox[2] * 0.5 + 0.01

        return target

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
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._tcp_offset)
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

    def _supports_dexterous_hand(self) -> bool:
        return self._gripper_action_dim > 2 or "inspire" in self._hand_name or "Dex" in self._robot_name

    def _inspire_open_pose(self) -> np.ndarray:
        return np.array([-1.5, -1.5, -1.5, -1.5, -3.0, 3.0], dtype=np.float64)

    def _inspire_closed_pose(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5, 1.5, 3.0, 3.0], dtype=np.float64)

    @staticmethod
    def _support_radius(extent_xyz: np.ndarray, direction: np.ndarray) -> float:
        half_extents = 0.5 * np.asarray(extent_xyz, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        return float(np.abs(direction) @ half_extents)

    def list_hand_presets(self) -> list[str]:
        if not self._supports_dexterous_hand():
            return ["open", "close"]
        return sorted(get_hand_preset_library(self._hand_name).keys())

    def set_hand_preset(self, preset_name: str) -> None:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand presets are only available for dexterous hands.")

        self._execute_hand_preset_steps(preset_name)

    def _execute_hand_preset_steps(self, preset_name: str, library_name: str | None = None) -> None:
        library_key = self._hand_name if library_name is None else library_name
        preset = get_hand_preset_library(library_key)[preset_name]
        sequence = preset.sequence if len(preset.sequence) > 0 else (preset.command,)
        stage_steps = preset.stage_steps if len(preset.stage_steps) == len(sequence) else tuple(30 for _ in sequence)
        for command, steps in zip(sequence, stage_steps, strict=True):
            self._env._set_gripper_command(command)
            for _ in range(int(steps)):
                self._env._step_once()

    def list_hand_primitives(self) -> list[str]:
        if not self._supports_dexterous_hand():
            return []
        return sorted(get_hand_primitive_library(self._hand_name).keys())

    def sample_hand_primitive_pose(self, primitive_name: str, object_name: str = "object") -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand primitives are only available for dexterous hands.")

        primitive = get_hand_primitive_library(self._hand_name)[primitive_name]
        quat = primitive_quaternion_wxyz(primitive.palm_face, primitive.middle_finger_direction)
        if primitive.use_sampled_grasp_position:
            grasp_pos, _ = self.sample_grasp_pose(object_name)
            pos = np.asarray(grasp_pos, dtype=np.float64).copy()
            return pos, quat

        obj_pos, _, bbox = self.get_object_pose(object_name, return_bbox_extent=True)
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        bbox = np.asarray(bbox if bbox is not None else np.array([0.05, 0.05, 0.05]), dtype=np.float64)
        palm_dir = np.asarray(WORLD_DIRECTIONS[primitive.palm_face], dtype=np.float64)
        radius = self._support_radius(bbox, palm_dir)
        pos = obj_pos - palm_dir * (radius + 0.025)
        return pos, quat

    def sample_hand_primitive_pregrasp_pose(
        self, primitive_name: str, object_name: str = "object"
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand primitives are only available for dexterous hands.")

        primitive = get_hand_primitive_library(self._hand_name)[primitive_name]
        grasp_pos, quat = self.sample_hand_primitive_pose(primitive_name, object_name)
        approach_dir = np.asarray(WORLD_DIRECTIONS[primitive.approach_direction], dtype=np.float64)
        pregrasp_pos = np.asarray(grasp_pos, dtype=np.float64) + approach_dir * float(primitive.approach_distance)
        return pregrasp_pos, quat

    def sample_hand_primitive_align_pose(
        self, primitive_name: str, object_name: str = "object"
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand primitives are only available for dexterous hands.")

        primitive = get_hand_primitive_library(self._hand_name)[primitive_name]
        grasp_pos, quat = self.sample_hand_primitive_pose(primitive_name, object_name)
        approach_dir = np.asarray(WORLD_DIRECTIONS[primitive.approach_direction], dtype=np.float64)
        align_pos = np.asarray(grasp_pos, dtype=np.float64) + approach_dir * float(primitive.align_distance)
        return align_pos, quat

    def execute_hand_primitive_soft_close(self, primitive_name: str) -> None:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand primitives are only available for dexterous hands.")
        primitive = get_hand_primitive_library(self._hand_name)[primitive_name]
        if primitive.soft_close_preset is None:
            return
        self._execute_hand_preset_steps(primitive.soft_close_preset, primitive.preset_library)
        for _ in range(8):
            self._env._step_once()

    def execute_hand_primitive_close(self, primitive_name: str) -> None:
        self.execute_hand_primitive(primitive_name)

    def execute_hand_primitive(self, primitive_name: str) -> None:
        if not self._supports_dexterous_hand():
            raise ValueError("Hand primitives are only available for dexterous hands.")
        primitive = get_hand_primitive_library(self._hand_name)[primitive_name]
        self._execute_hand_preset_steps(primitive.hand_preset, primitive.preset_library)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        if self._supports_dexterous_hand():
            self._env._set_gripper_command(self._inspire_open_pose())
            for _ in range(40):
                self._env._step_once()
            return
        _open_gripper(self._env, steps=40)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        if self._supports_dexterous_hand():
            self._env._set_gripper_command(self._inspire_closed_pose())
            for _ in range(60):
                self._env._step_once()
            return
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
