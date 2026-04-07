#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from run_inspire_apple_pickup import _ensure_pyroki_server, _load_snapshot, _save_snapshot
from run_inspire_apple_step2 import _all_object_positions, _build_env_with_retries


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 4 of the Inspire apple pickup search: high recenter above the apple.")
    parser.add_argument("--step3-snapshot", required=True)
    parser.add_argument("--scene-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raise-distance", type=float, default=0.05)
    parser.add_argument("--move-steps", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyroki_process = _ensure_pyroki_server()
    try:
        scene_metadata = json.loads(Path(args.scene_metadata).read_text(encoding="utf-8"))
        exact_target_id = scene_metadata["target"]["category"]
        exact_distractor_ids = [item["category"] for item in scene_metadata["distractors"]]
        exact_object_ids = [exact_target_id, *exact_distractor_ids]

        env = _build_env_with_retries(exact_object_ids, exact_distractor_ids, exact_target_id, 1)
        env.restore_state(_load_snapshot(Path(args.step3_snapshot)))
        api = FrankaControlPrivilegedApi(env)

        start_obs = env.get_observation()
        start_position = np.asarray(start_obs["robot_cartesian_pos"][:3], dtype=np.float64)
        start_quaternion = np.asarray(start_obs["robot_cartesian_pos"][3:7], dtype=np.float64)
        object_before = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()

        raise_waypoint = start_position.copy()
        raise_waypoint[2] += float(args.raise_distance)
        step4_position = np.array(
            [object_before[0], object_before[1], raise_waypoint[2]],
            dtype=np.float64,
        )

        object_names, positions_before = _all_object_positions(env)
        env.enable_video_capture(True, clear=True, wrist_camera=True)
        _move_linear(api, start_position, raise_waypoint, start_quaternion, steps=int(args.move_steps))
        _move_linear(api, raise_waypoint, step4_position, start_quaternion, steps=int(args.move_steps))

        object_after = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        _, positions_after = _all_object_positions(env)
        object_displacements = np.linalg.norm(positions_after - positions_before, axis=1)

        robot_frames = env.get_video_frames(clear=True)
        wrist_frames = env.get_wrist_video_frames(clear=True)
        overview_frames = env.get_overview_video_frames(clear=True)

        _write_video(output_dir / "step4_robotview.mp4", robot_frames)
        _write_video(output_dir / "step4_wristview.mp4", wrist_frames)
        _write_video(output_dir / "step4_frontview.mp4", overview_frames)
        _save_snapshot(output_dir / "step4_snapshot.npz", env.snapshot_state())

        summary = {
            "step3_snapshot": str(Path(args.step3_snapshot)),
            "start_position": start_position.tolist(),
            "raise_waypoint": raise_waypoint.tolist(),
            "step4_position": step4_position.tolist(),
            "grasp_quaternion_wxyz": start_quaternion.tolist(),
            "object_before": object_before.tolist(),
            "object_after": object_after.tolist(),
            "target_displacement_m": float(np.linalg.norm(object_after - object_before)),
            "max_object_displacement_m": float(object_displacements.max()),
            "object_displacements_m": {
                name: float(dist) for name, dist in zip(object_names, object_displacements.tolist())
            },
        }
        _write_json(output_dir / "step4_summary.json", summary)
        env.close()
    finally:
        if pyroki_process is not None:
            pyroki_process.terminate()
            try:
                pyroki_process.wait(timeout=5)
            except Exception:
                pyroki_process.kill()


if __name__ == "__main__":
    main()
