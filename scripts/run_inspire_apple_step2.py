#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from run_inspire_apple_pickup import _capture_scene, _ensure_pyroki_server, _load_snapshot, _make_env, _save_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def _move_linear(api: FrankaControlPrivilegedApi, start: np.ndarray, end: np.ndarray, quat: np.ndarray, steps: int) -> None:
    for alpha in np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)[1:]:
        pos = (1.0 - alpha) * start + alpha * end
        api.goto_pose(pos, quat)


def _all_object_positions(env) -> tuple[list[str], np.ndarray]:
    rs = env.robosuite_env
    names = [rs.cube.root_body] + [obj.root_body for obj in rs._distractors]
    ids = [rs.sim.model.body_name2id(name) for name in names]
    positions = np.asarray([rs.sim.data.body_xpos[idx].copy() for idx in ids], dtype=np.float64)
    return names, positions


def _build_env_with_retries(exact_object_ids, exact_distractor_ids, exact_target_id, seed: int):
    last_error = None
    env = None
    for trial_seed in range(seed, seed + 8):
        try:
            env = _make_env(
                enable_render=True,
                object_ids=exact_object_ids,
                distractor_ids=exact_distractor_ids,
                target_object_id=exact_target_id,
            )
            env.reset(seed=trial_seed, options={"trial": trial_seed})
            return env
        except Exception as exc:
            last_error = exc
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            env = None
    raise RuntimeError(f"Failed to construct scene env: {last_error!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 2 of the Inspire apple pickup search: move to a far safe pregrasp.")
    parser.add_argument(
        "--scene-snapshot",
        default="",
        help="Optional frozen scene snapshot to replay. Leave empty to generate one live fixed scene for this run.",
    )
    parser.add_argument(
        "--scene-metadata",
        default="",
        help="Optional scene metadata json used to rebuild the exact object set before restoring a snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inspire_apple_step2_v1",
        help="Directory for step-2 artifacts.",
    )
    parser.add_argument("--primitive", default="ball_wrap_back_down", help="Primitive used only to pick the side of the apple we stage from.")
    parser.add_argument("--top-down", action="store_true", help="Stage directly above the object center instead of from a side direction.")
    parser.add_argument("--approach-dx", type=float, default=0.0, help="Optional explicit XY approach direction x component.")
    parser.add_argument("--approach-dy", type=float, default=0.0, help="Optional explicit XY approach direction y component.")
    parser.add_argument("--radial-distance", type=float, default=-1.0, help="Optional explicit XY stand-off distance. Negative keeps legacy behavior.")
    parser.add_argument("--extra-approach", type=float, default=0.05, help="Extra horizontal stand-off added on the staged side of the apple.")
    parser.add_argument("--extra-z", type=float, default=0.0, help="Extra world-z clearance added before safe-height clamping.")
    parser.add_argument("--up-clearance", type=float, default=0.10, help="Initial lift from the current pose before translating across the clutter.")
    parser.add_argument("--retreat-dx", type=float, default=0.0, help="Optional XY retreat x applied during the initial lift.")
    parser.add_argument("--retreat-dy", type=float, default=0.0, help="Optional XY retreat y applied during the initial lift.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for a live fixed scene when not using --scene-snapshot.")
    parser.add_argument("--settle-steps", type=int, default=30, help="Idle sim steps before recording the scene and stage-2 motion.")
    parser.add_argument(
        "--safe-vertical-clearance",
        type=float,
        default=0.15,
        help="Final vertical clearance above the apple top for the far safe pregrasp.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyroki_process = _ensure_pyroki_server()
    try:
        exact_object_ids = None
        exact_distractor_ids = None
        exact_target_id = "013_apple"
        if args.scene_metadata:
            scene_metadata = json.loads(Path(args.scene_metadata).read_text(encoding="utf-8"))
            exact_target_id = scene_metadata["target"]["category"]
            exact_distractor_ids = [item["category"] for item in scene_metadata["distractors"]]
            exact_object_ids = [exact_target_id, *exact_distractor_ids]

        env = _build_env_with_retries(
            exact_object_ids=exact_object_ids,
            exact_distractor_ids=exact_distractor_ids,
            exact_target_id=exact_target_id,
            seed=args.seed,
        )
        if args.scene_snapshot:
            snapshot = _load_snapshot(Path(args.scene_snapshot))
            env.restore_state(snapshot)
        for _ in range(int(args.settle_steps)):
            env._step_once()
        scene_metadata = _capture_scene(env, output_dir)

        api = FrankaControlPrivilegedApi(env)
        start_obs = env.get_observation()
        start_position = np.asarray(start_obs["robot_cartesian_pos"][:3], dtype=np.float64)
        start_quaternion = np.asarray(start_obs["robot_cartesian_pos"][3:7], dtype=np.float64)
        object_before = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        object_shape = api.get_object_shape()
        explicit_dir = np.array([float(args.approach_dx), float(args.approach_dy)], dtype=np.float64)
        if args.top_down:
            direction_xy = np.array([0.0, 0.0], dtype=np.float64)
        elif np.linalg.norm(explicit_dir) > 1e-8:
            direction_xy = explicit_dir / np.linalg.norm(explicit_dir)
        else:
            primitive_pregrasp, _primitive_quat = api.sample_hand_primitive_pregrasp_pose(args.primitive, "object")
            primitive_pregrasp = np.asarray(primitive_pregrasp, dtype=np.float64)
            direction_xy = primitive_pregrasp[:2] - object_before[:2]
            if np.linalg.norm(direction_xy) < 1e-8:
                direction_xy = np.array([1.0, 0.0], dtype=np.float64)
            direction_xy = direction_xy / np.linalg.norm(direction_xy)

        # Stage-2 target: safe side-hover with the current hand orientation.
        radial_distance = 0.0 if args.top_down else (
            float(args.radial_distance) if float(args.radial_distance) > 0 else (0.13 + float(args.extra_approach))
        )
        far_pregrasp = np.asarray(
            [
                object_before[0] + direction_xy[0] * radial_distance,
                object_before[1] + direction_xy[1] * radial_distance,
                object_before[2],
            ],
            dtype=np.float64,
        )
        far_pregrasp[2] += float(args.extra_z)

        start_hover = start_position.copy()
        start_hover[0] += float(args.retreat_dx)
        start_hover[1] += float(args.retreat_dy)
        start_hover[2] += float(args.up_clearance)
        transit_hover = far_pregrasp.copy()
        transit_hover[2] = max(float(transit_hover[2]), float(start_hover[2]))
        object_top_z = float(object_before[2]) + 0.5 * float(object_shape["size"]["z"])
        far_pregrasp[2] = max(float(far_pregrasp[2]), object_top_z + float(args.safe_vertical_clearance))

        object_names, positions_before = _all_object_positions(env)
        env.enable_video_capture(True, clear=True, wrist_camera=True)
        api.open_gripper()
        _move_linear(api, start_position, start_hover, start_quaternion, steps=4)
        _move_linear(api, start_hover, transit_hover, start_quaternion, steps=6)
        _move_linear(api, transit_hover, far_pregrasp, start_quaternion, steps=4)

        object_after = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        displacement_m = float(np.linalg.norm(object_after - object_before))
        _object_names_after, positions_after = _all_object_positions(env)
        object_displacements = np.linalg.norm(positions_after - positions_before, axis=1)

        robot_frames = env.get_video_frames(clear=True)
        wrist_frames = env.get_wrist_video_frames(clear=True)
        overview_frames = env.get_overview_video_frames(clear=True)

        _write_video(output_dir / "step2_robotview.mp4", robot_frames)
        _write_video(output_dir / "step2_wristview.mp4", wrist_frames)
        _write_video(output_dir / "step2_frontview.mp4", overview_frames)
        _save_snapshot(output_dir / "step2_snapshot.npz", env.snapshot_state())

        result = {
            "primitive": args.primitive,
            "start_position": start_position.tolist(),
            "start_hover": start_hover.tolist(),
            "transit_hover": transit_hover.tolist(),
            "far_pregrasp": far_pregrasp.tolist(),
            "grasp_quaternion_wxyz": np.asarray(start_quaternion, dtype=np.float64).tolist(),
            "object_before": object_before.tolist(),
            "object_after": object_after.tolist(),
            "target_displacement_m": displacement_m,
            "approach_direction_xy": direction_xy.tolist(),
            "radial_distance": radial_distance,
            "max_object_displacement_m": float(object_displacements.max()),
            "object_displacements_m": {
                name: float(dist) for name, dist in zip(object_names, object_displacements.tolist())
            },
            "scene_snapshot": (str(Path(args.scene_snapshot)) if args.scene_snapshot else None),
            "scene_metadata": scene_metadata,
        }
        _write_json(output_dir / "step2_summary.json", result)
        env.close()
    finally:
        if pyroki_process is not None:
            pyroki_process.terminate()
            try:
                pyroki_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pyroki_process.kill()


if __name__ == "__main__":
    main()
