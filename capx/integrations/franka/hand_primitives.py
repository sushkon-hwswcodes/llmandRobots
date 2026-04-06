from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


WORLD_DIRECTIONS: dict[str, np.ndarray] = {
    "forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "back": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

# Calibrated from the current PandaDexRH / InspireRightHand wrist frame:
# - local +X aligns best with the palm-facing direction
# - local -Y aligns best with the middle-finger pointing direction
LOCAL_PALM_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
LOCAL_FINGER_AXIS = np.array([0.0, -1.0, 0.0], dtype=np.float64)
LOCAL_BINORMAL_AXIS = np.cross(LOCAL_PALM_AXIS, LOCAL_FINGER_AXIS)


@dataclass(frozen=True)
class HandPrimitive:
    name: str
    preset_library: str
    hand_preset: str
    palm_face: str
    middle_finger_direction: str
    approach_direction: str
    approach_distance: float
    description: str


def _normalized(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return arr / norm


def primitive_quaternion_wxyz(palm_face: str, middle_finger_direction: str) -> np.ndarray:
    palm_world = _normalized(WORLD_DIRECTIONS[palm_face])
    finger_world = _normalized(WORLD_DIRECTIONS[middle_finger_direction])
    if abs(float(np.dot(palm_world, finger_world))) > 1e-6:
        raise ValueError(
            f"palm_face={palm_face!r} and middle_finger_direction={middle_finger_direction!r} must be orthogonal"
        )

    local_basis = np.column_stack([LOCAL_PALM_AXIS, LOCAL_FINGER_AXIS, LOCAL_BINORMAL_AXIS])
    world_binormal = _normalized(np.cross(palm_world, finger_world))
    world_basis = np.column_stack([palm_world, finger_world, world_binormal])
    rotation_matrix = world_basis @ local_basis.T
    quat_xyzw = SciRotation.from_matrix(rotation_matrix).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


INSPIRE_WRIST_PRIMITIVES_V1: dict[str, HandPrimitive] = {
    "top_grasp_4": HandPrimitive(
        name="top_grasp_4",
        preset_library="inspire_right_allegro_v3",
        hand_preset="grasp_4",
        palm_face="down",
        middle_finger_direction="forward",
        approach_direction="up",
        approach_distance=0.08,
        description="Top-down four-finger close with the palm facing down and the middle finger pointing forward.",
    ),
    "top_envelop": HandPrimitive(
        name="top_envelop",
        preset_library="inspire_right_allegro_v3",
        hand_preset="envelop",
        palm_face="down",
        middle_finger_direction="forward",
        approach_direction="up",
        approach_distance=0.08,
        description="Top-down staged envelop primitive: spread first, then broad close.",
    ),
    "side_pinch_it": HandPrimitive(
        name="side_pinch_it",
        preset_library="inspire_right_allegro_v3",
        hand_preset="pinch_it",
        palm_face="left",
        middle_finger_direction="down",
        approach_direction="right",
        approach_distance=0.08,
        description="Side-oriented index-thumb pinch with the palm facing left and the middle finger pointing down.",
    ),
    "side_pinch_mt": HandPrimitive(
        name="side_pinch_mt",
        preset_library="inspire_right_allegro_v3",
        hand_preset="pinch_mt",
        palm_face="right",
        middle_finger_direction="down",
        approach_direction="left",
        approach_distance=0.08,
        description="Side-oriented middle-thumb pinch with the palm facing right and the middle finger pointing down.",
    ),
}
