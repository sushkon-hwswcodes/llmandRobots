#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi


@dataclass(frozen=True)
class AttemptSpec:
    name: str
    primitive: str
    use_align: bool
    pregrasp_steps: int
    final_steps: int
    lift_height: float
    closure_sequence: tuple[str, ...]
    extra_z: float = 0.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    description: str = ""


ATTEMPTS: tuple[AttemptSpec, ...] = (
    AttemptSpec(
        name="power_align_soft",
        primitive="ball_power_down_forward",
        use_align=True,
        pregrasp_steps=5,
        final_steps=7,
        lift_height=0.10,
        closure_sequence=("soft", "close"),
        description="Top-down power-style ball primitive. Kept as a first baseline and usually too blunt in clutter.",
    ),
    AttemptSpec(
        name="wrap_align_poweronly",
        primitive="ball_wrap_back_down",
        use_align=True,
        pregrasp_steps=5,
        final_steps=7,
        lift_height=0.10,
        closure_sequence=("power_grasp",),
        description="Wrap approach but closes too hard too early.",
    ),
    AttemptSpec(
        name="wrap_align_soft_close",
        primitive="ball_wrap_back_down",
        use_align=True,
        pregrasp_steps=5,
        final_steps=7,
        lift_height=0.08,
        closure_sequence=("soft", "close"),
        description="Exact successful apple pickup: side wrap, align, soft close, then full close and short lift.",
    ),
    AttemptSpec(
        name="wrap_align_soft_close_high",
        primitive="ball_wrap_back_down",
        use_align=True,
        pregrasp_steps=5,
        final_steps=7,
        lift_height=0.10,
        closure_sequence=("soft", "close"),
        description="Fallback with a slightly higher post-grasp lift.",
    ),
)


def attempt_names() -> list[str]:
    return [attempt.name for attempt in ATTEMPTS]


def get_attempt(name: str) -> AttemptSpec:
    for attempt in ATTEMPTS:
        if attempt.name == name:
            return attempt
    raise KeyError(f"Unknown attempt {name!r}. Known attempts: {attempt_names()}")


def interpolate_goto(
    api: FrankaControlPrivilegedApi,
    start_position: np.ndarray,
    end_position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    *,
    steps: int,
) -> None:
    for alpha in np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)[1:]:
        position = (1.0 - alpha) * start_position + alpha * end_position
        api.goto_pose(position, quaternion_wxyz)


def run_attempt(env, attempt_name: str) -> dict[str, object]:
    attempt = get_attempt(attempt_name)
    api = FrankaControlPrivilegedApi(env)

    obs = env.get_observation()
    current_position = np.asarray(obs["robot_cartesian_pos"][:3], dtype=np.float64)
    pregrasp_position, grasp_quaternion = api.sample_hand_primitive_pregrasp_pose(attempt.primitive, "object")
    align_position, _ = api.sample_hand_primitive_align_pose(attempt.primitive, "object")
    grasp_position, _ = api.sample_hand_primitive_pose(attempt.primitive, "object")

    pregrasp_position = np.asarray(pregrasp_position, dtype=np.float64).copy()
    align_position = np.asarray(align_position, dtype=np.float64).copy()
    grasp_position = np.asarray(grasp_position, dtype=np.float64).copy()

    for position in (pregrasp_position, align_position, grasp_position):
        position[0] += float(attempt.x_offset)
        position[1] += float(attempt.y_offset)
        position[2] += float(attempt.extra_z)

    api.open_gripper()
    interpolate_goto(api, current_position, pregrasp_position, grasp_quaternion, steps=attempt.pregrasp_steps)

    if attempt.use_align:
        interpolate_goto(
            api,
            pregrasp_position,
            align_position,
            grasp_quaternion,
            steps=max(2, attempt.pregrasp_steps),
        )
        final_start = align_position
    else:
        final_start = pregrasp_position

    interpolate_goto(api, final_start, grasp_position, grasp_quaternion, steps=attempt.final_steps)

    for closure in attempt.closure_sequence:
        if closure == "soft":
            api.execute_hand_primitive_soft_close(attempt.primitive)
        elif closure == "close":
            api.execute_hand_primitive_close(attempt.primitive)
        else:
            api.set_hand_preset(closure)

    lift_position = grasp_position.copy()
    lift_position[2] += float(attempt.lift_height)
    interpolate_goto(api, grasp_position, lift_position, grasp_quaternion, steps=max(3, attempt.pregrasp_steps))

    reward = float(env.compute_reward())
    task_completed = bool(env.task_completed())
    object_position = np.asarray(api.get_object_pose("object")[0], dtype=np.float64)

    return {
        "attempt": attempt.name,
        "description": attempt.description,
        "primitive": attempt.primitive,
        "reward": reward,
        "task_completed": task_completed,
        "object_position": object_position.tolist(),
        "pregrasp_position": pregrasp_position.tolist(),
        "align_position": align_position.tolist(),
        "grasp_position": grasp_position.tolist(),
        "grasp_quaternion_wxyz": np.asarray(grasp_quaternion, dtype=np.float64).tolist(),
        "lift_height": float(attempt.lift_height),
        "closure_sequence": list(attempt.closure_sequence),
    }

