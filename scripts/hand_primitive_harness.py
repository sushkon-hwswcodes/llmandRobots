#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
from capx.integrations.franka.common import apply_tcp_offset
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.franka.hand_presets import INSPIRE_COMMAND_LABELS, get_hand_preset_library
from capx.integrations.franka.hand_primitives import (
    INSPIRE_WRIST_PRIMITIVES_V1,
    WORLD_DIRECTIONS,
    primitive_quaternion_wxyz,
)


def _make_env() -> FrankaRobosuiteCubeLiftLowLevel:
    return FrankaRobosuiteCubeLiftLowLevel(
        privileged=False,
        enable_render=True,
        robot_name="PandaDexRH",
        hand_name="inspire_right",
        robosuite_gripper_type="default",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
    )


def _render_rgb(env: FrankaRobosuiteCubeLiftLowLevel, camera_name: str) -> np.ndarray:
    frame = env.robosuite_env.sim.render(camera_name=camera_name, width=320, height=320, depth=False)
    return np.flipud(frame)


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    if frames:
        imageio.mimwrite(path, frames, fps=fps)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _eef_pose(env: FrankaRobosuiteCubeLiftLowLevel) -> tuple[np.ndarray, np.ndarray]:
    body_id = env.robosuite_env.sim.model.body_name2id("gripper0_right_eef")
    pos = np.asarray(env.robosuite_env.sim.data.xpos[body_id], dtype=np.float64).copy()
    quat = np.asarray(env.robosuite_env.sim.data.xquat[body_id], dtype=np.float64).copy()
    return pos, quat


def _current_joints(env: FrankaRobosuiteCubeLiftLowLevel) -> np.ndarray:
    obs = env.robosuite_env._get_observations()
    return np.asarray(obs["robot0_joint_pos"], dtype=np.float64).copy()


def _move_wrist_with_frames(
    env: FrankaRobosuiteCubeLiftLowLevel,
    api: FrankaControlPrivilegedApi,
    *,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    num_steps: int = 12,
) -> dict[str, Any]:
    quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    target_pos = apply_tcp_offset(pos, quat_wxyz, env.tcp_offset)

    if api.cfg is None:
        api.cfg = api.ik_solve_fn(target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]))
    else:
        api.cfg = api.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
            prev_cfg=api.cfg,
        )
    target_joints = np.asarray(api.cfg[:-1], dtype=np.float64).reshape(7)

    start_joints = _current_joints(env)
    robot_frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []
    sample_indices = {0, max(0, num_steps // 2), max(0, num_steps - 1)}
    for i, alpha in enumerate(np.linspace(0.0, 1.0, num_steps)):
        interp = (1.0 - alpha) * start_joints + alpha * target_joints
        env.move_to_joints_non_blocking(interp)
        if i in sample_indices:
            frame_robot = _render_rgb(env, "robot0_robotview")
            frame_front = _render_rgb(env, "frontview")
            robot_frames.extend([frame_robot] * 4)
            front_frames.extend([frame_front] * 4)

    # Persist the final arm target so subsequent hand-only playback keeps the
    # wrist pose instead of falling back to the previous joint target.
    env._current_joints = target_joints.copy()

    return {
        "target_joints": target_joints.tolist(),
        "robot_frames": robot_frames,
        "front_frames": front_frames,
    }


def _play_preset(env: FrankaRobosuiteCubeLiftLowLevel, preset_name: str, library_name: str) -> dict[str, Any]:
    preset = get_hand_preset_library(library_name)[preset_name]
    robot_frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []
    stages: list[dict[str, Any]] = []

    sequence = tuple(preset.sequence) if preset.sequence else (preset.command,)
    stage_steps = tuple(int(x) for x in preset.stage_steps) if preset.stage_steps else tuple([20] * len(sequence))
    if len(stage_steps) != len(sequence):
        stage_steps = tuple([20] * len(sequence))

    for idx, (command, steps) in enumerate(zip(sequence, stage_steps), start=1):
        env._set_gripper_command(command)
        sample_indices = {0, max(0, steps // 2), max(0, steps - 1)}
        for step_idx in range(steps):
            env._step_once()
            if step_idx in sample_indices:
                frame_robot = _render_rgb(env, "robot0_robotview")
                frame_front = _render_rgb(env, "frontview")
                robot_frames.extend([frame_robot] * 4)
                front_frames.extend([frame_front] * 4)
        stages.append(
            {
                "stage_index": idx,
                "command": {label: float(value) for label, value in zip(INSPIRE_COMMAND_LABELS, command)},
                "steps": steps,
            }
        )
    return {"robot_frames": robot_frames, "front_frames": front_frames, "stages": stages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render wrist-oriented hand primitives.")
    parser.add_argument("--primitive", action="append", default=[], help="Primitive name to render; may be repeated.")
    parser.add_argument("--all-primitives", action="store_true", help="Render all registered wrist primitives.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hand_wrist_primitives_v1",
        help="Directory to save primitive videos and summaries.",
    )
    args = parser.parse_args()

    selected = list(INSPIRE_WRIST_PRIMITIVES_V1.keys()) if args.all_primitives or not args.primitive else args.primitive
    env = _make_env()
    api = FrankaControlPrivilegedApi(env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=0)
    base_position = np.asarray(obs["cube_poses"]["primary"][:3], dtype=np.float64).copy()
    base_position[2] += 0.28

    summary: dict[str, Any] = {
        "base_position": base_position.tolist(),
        "primitives": {},
        "wrist_frame_convention": {
            "palm_face": "local +X",
            "middle_finger_direction": "local -Y",
        },
    }

    for primitive_name in selected:
        primitive = INSPIRE_WRIST_PRIMITIVES_V1[primitive_name]
        env.reset(seed=0)

        before_robot = _render_rgb(env, "robot0_robotview")
        before_front = _render_rgb(env, "frontview")

        quat_wxyz = primitive_quaternion_wxyz(primitive.palm_face, primitive.middle_finger_direction)
        approach_dir = np.asarray(WORLD_DIRECTIONS[primitive.approach_direction], dtype=np.float64)
        pre_position = base_position + primitive.approach_distance * approach_dir

        move_pre = _move_wrist_with_frames(env, api, position=pre_position, quaternion_wxyz=quat_wxyz)
        move_final = _move_wrist_with_frames(env, api, position=base_position, quaternion_wxyz=quat_wxyz)
        after_wrist_robot = _render_rgb(env, "robot0_robotview")
        after_wrist_front = _render_rgb(env, "frontview")

        preset_result = _play_preset(env, primitive.hand_preset, primitive.preset_library)
        final_robot = _render_rgb(env, "robot0_robotview")
        final_front = _render_rgb(env, "frontview")
        eef_pos, eef_quat = _eef_pose(env)

        robot_frames = [before_robot, *move_pre["robot_frames"], *move_final["robot_frames"], after_wrist_robot, *preset_result["robot_frames"], final_robot]
        front_frames = [before_front, *move_pre["front_frames"], *move_final["front_frames"], after_wrist_front, *preset_result["front_frames"], final_front]

        primitive_dir = output_dir / primitive_name
        primitive_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(primitive_dir / "before_robotview.png", before_robot)
        imageio.imwrite(primitive_dir / "before_frontview.png", before_front)
        imageio.imwrite(primitive_dir / "after_robotview.png", final_robot)
        imageio.imwrite(primitive_dir / "after_frontview.png", final_front)
        _write_video(primitive_dir / f"{primitive_name}_robotview.mp4", robot_frames)
        _write_video(primitive_dir / f"{primitive_name}_frontview.mp4", front_frames)

        payload = {
            "description": primitive.description,
            "preset_library": primitive.preset_library,
            "hand_preset": primitive.hand_preset,
            "palm_face": primitive.palm_face,
            "middle_finger_direction": primitive.middle_finger_direction,
            "approach_direction": primitive.approach_direction,
            "approach_distance": primitive.approach_distance,
            "target_position": base_position.tolist(),
            "pre_position": pre_position.tolist(),
            "quaternion_wxyz": quat_wxyz.tolist(),
            "stages": preset_result["stages"],
            "final_eef_position": eef_pos.tolist(),
            "final_eef_quaternion_wxyz": eef_quat.tolist(),
        }
        _write_json(primitive_dir / "summary.json", payload)
        summary["primitives"][primitive_name] = payload

    _write_json(output_dir / "summary.json", summary)
    env.close()


if __name__ == "__main__":
    main()
