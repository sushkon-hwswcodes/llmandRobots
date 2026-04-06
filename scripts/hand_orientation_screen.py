#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

from capx.envs.simulators.robosuite_shape_lift import FrankaRobosuiteShapeLiftLowLevel
from capx.integrations.franka.common import apply_tcp_offset
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.integrations.franka.hand_presets import get_hand_preset_library
from capx.integrations.franka.hand_primitives import (
    OPPOSITE_DIRECTION,
    WORLD_DIRECTIONS,
    iter_all_wrist_orientation_pairs,
    orientation_name,
    primitive_quaternion_wxyz,
)


PRESET_LIBRARY = "inspire_right_allegro_v3"
PRESET_BY_SHAPE = {
    "box": "grasp_4",
    "cylinder": "pinch_mt",
    "ball": "envelop",
}


def _make_env(shape: str) -> FrankaRobosuiteShapeLiftLowLevel:
    return FrankaRobosuiteShapeLiftLowLevel(
        privileged=False,
        enable_render=False,
        robot_name="PandaDexRH",
        hand_name="inspire_right",
        robosuite_gripper_type="default",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
        fixed_shape=shape,
    )


def _play_hand_preset(env: FrankaRobosuiteShapeLiftLowLevel, preset_name: str) -> None:
    preset = get_hand_preset_library(PRESET_LIBRARY)[preset_name]
    sequence = tuple(preset.sequence) if preset.sequence else (preset.command,)
    stage_steps = tuple(int(x) for x in preset.stage_steps) if preset.stage_steps else tuple([20] * len(sequence))
    if len(stage_steps) != len(sequence):
        stage_steps = tuple([20] * len(sequence))

    for command, steps in zip(sequence, stage_steps):
        env._set_gripper_command(command)
        for _ in range(steps):
            env._step_once()


def _fast_goto_pose(
    env: FrankaRobosuiteShapeLiftLowLevel,
    api: FrankaControlPrivilegedApi,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    target_pos = apply_tcp_offset(pos, quat_wxyz, api._tcp_offset)
    if api.cfg is None:
        api.cfg = api.ik_solve_fn(target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]))
    else:
        api.cfg = api.ik_solve_fn(
            target_pose_wxyz_xyz=np.concatenate([quat_wxyz, target_pos]),
            prev_cfg=api.cfg,
        )
    joints = np.asarray(api.cfg[:-1], dtype=np.float64).reshape(7)
    env.move_to_joints_blocking(joints, tolerance=0.05, max_steps=40)


def _support_radius(extent_xyz: np.ndarray, direction: np.ndarray) -> float:
    half_extents = 0.5 * np.asarray(extent_xyz, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    return float(np.abs(direction) @ half_extents)


def _run_trial(shape: str, palm_face: str, finger_dir: str, seed: int) -> dict[str, Any]:
    env = _make_env(shape)
    api = FrankaControlPrivilegedApi(env)
    try:
        return _run_trial_with_env(env, api, shape, palm_face, finger_dir, seed)
    finally:
        env.close()


def _run_trial_with_env(
    env: FrankaRobosuiteShapeLiftLowLevel,
    api: FrankaControlPrivilegedApi,
    shape: str,
    palm_face: str,
    finger_dir: str,
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    del obs
    result: dict[str, Any] = {
        "shape": shape,
        "palm_face": palm_face,
        "middle_finger_direction": finger_dir,
        "orientation_name": orientation_name(palm_face, finger_dir),
        "seed": seed,
        "preset": PRESET_BY_SHAPE[shape],
        "success": False,
        "reward": 0.0,
        "height_gain": 0.0,
        "error": "",
    }

    try:
        obj_pos, _, bbox = api.get_object_pose("object", return_bbox_extent=True)
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        bbox = np.asarray(bbox, dtype=np.float64)
        quat = primitive_quaternion_wxyz(palm_face, finger_dir)

        palm_dir = np.asarray(WORLD_DIRECTIONS[palm_face], dtype=np.float64)
        approach_dir = np.asarray(WORLD_DIRECTIONS[OPPOSITE_DIRECTION[palm_face]], dtype=np.float64)
        radius = _support_radius(bbox, palm_dir)
        # Place the wrist center just outside the object's support radius.
        target_position = obj_pos - palm_dir * (radius + 0.025)
        pre_position = target_position + approach_dir * 0.08
        lift_position = target_position + np.array([0.0, 0.0, 0.12], dtype=np.float64)

        api.open_gripper()
        _fast_goto_pose(env, api, pre_position, quat)
        _fast_goto_pose(env, api, target_position, quat)
        _play_hand_preset(env, PRESET_BY_SHAPE[shape])
        _fast_goto_pose(env, api, lift_position, quat)

        final_pos, _, _ = api.get_object_pose("object", return_bbox_extent=True)
        final_pos = np.asarray(final_pos, dtype=np.float64)
        result["reward"] = float(env.compute_reward())
        result["success"] = bool(env.task_completed())
        result["height_gain"] = float(final_pos[2] - obj_pos[2])
    except Exception as exc:  # broad on purpose for screening
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _write_outputs(output_dir: Path, trial_rows: list[dict[str, Any]], shapes: list[str], orientations: list[tuple[str, str]], trials_per_shape: int) -> None:
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        by_shape[row["shape"]].append(row)

    summary: dict[str, Any] = {
        "trials_per_shape": trials_per_shape,
        "num_orientations": len(orientations),
        "num_shapes": len(shapes),
        "total_trials": len(trial_rows),
        "trial_rows": trial_rows,
        "rankings": {},
    }

    for shape, rows in by_shape.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["orientation_name"]].append(row)

        ranking_rows = []
        for orientation_key, group in grouped.items():
            success_rate = sum(1 for r in group if r["success"]) / len(group)
            avg_reward = sum(float(r["reward"]) for r in group) / len(group)
            avg_height_gain = sum(float(r["height_gain"]) for r in group) / len(group)
            errors = [r["error"] for r in group if r["error"]]
            ranking_rows.append(
                {
                    "orientation_name": orientation_key,
                    "palm_face": group[0]["palm_face"],
                    "middle_finger_direction": group[0]["middle_finger_direction"],
                    "preset": group[0]["preset"],
                    "success_rate": success_rate,
                    "avg_reward": avg_reward,
                    "avg_height_gain": avg_height_gain,
                    "num_errors": len(errors),
                }
            )

        ranking_rows.sort(
            key=lambda r: (r["success_rate"], r["avg_reward"], r["avg_height_gain"]),
            reverse=True,
        )
        summary["rankings"][shape] = ranking_rows

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "trial_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "orientation_name",
                "palm_face",
                "middle_finger_direction",
                "seed",
                "preset",
                "success",
                "reward",
                "height_gain",
                "error",
            ],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)

    with (output_dir / "rankings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "orientation_name",
                "palm_face",
                "middle_finger_direction",
                "preset",
                "success_rate",
                "avg_reward",
                "avg_height_gain",
                "num_errors",
            ],
        )
        writer.writeheader()
        for shape, rows in summary["rankings"].items():
            for row in rows:
                writer.writerow({"shape": shape, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen all 24 wrist orientations against synthetic shape lift objects.")
    parser.add_argument("--trials-per-shape", type=int, default=3, help="Trials per orientation per shape.")
    parser.add_argument(
        "--shape",
        choices=["box", "cylinder", "ball"],
        action="append",
        help="Optional shape filter. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hand_orientation_screen_v1",
        help="Directory for JSON/CSV summaries.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = list(args.shape) if args.shape else ["box", "cylinder", "ball"]
    orientations = iter_all_wrist_orientation_pairs()
    trial_rows: list[dict[str, Any]] = []

    for shape_idx, shape in enumerate(shapes):
        env = _make_env(shape)
        api = FrankaControlPrivilegedApi(env)
        try:
            for orient_idx, (palm_face, finger_dir) in enumerate(orientations):
                for trial_idx in range(args.trials_per_shape):
                    seed = 1000 + shape_idx * 100 + orient_idx * 10 + trial_idx
                    row = _run_trial_with_env(env, api, shape, palm_face, finger_dir, seed)
                    trial_rows.append(row)
                    _write_outputs(output_dir, trial_rows, shapes, orientations, args.trials_per_shape)
                    print(
                        f"[{shape}] {orientation_name(palm_face, finger_dir)} "
                        f"trial {trial_idx + 1}/{args.trials_per_shape} "
                        f"success={int(bool(row['success']))} reward={row['reward']:.3f} "
                        f"height_gain={row['height_gain']:.3f}",
                        flush=True,
                    )
        finally:
            env.close()

    _write_outputs(output_dir, trial_rows, shapes, orientations, args.trials_per_shape)


if __name__ == "__main__":
    main()
