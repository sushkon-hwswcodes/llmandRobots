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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 3 of the Inspire apple pickup search: move from the locked far pregrasp to a near pregrasp.")
    parser.add_argument("--output-dir", default="outputs/inspire_apple_step3_v1")
    parser.add_argument("--step2-snapshot", required=True)
    parser.add_argument("--scene-metadata", required=True)
    parser.add_argument("--near-horizontal-offset", type=float, default=0.06)
    parser.add_argument("--near-vertical-clearance", type=float, default=0.08)
    parser.add_argument("--step3-steps", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyroki_process = _ensure_pyroki_server()
    try:
        scene_metadata = json.loads(Path(args.scene_metadata).read_text(encoding="utf-8"))
        exact_target_id = scene_metadata["target"]["category"]
        exact_distractor_ids = [item["category"] for item in scene_metadata["distractors"]]
        exact_object_ids = [exact_target_id, *exact_distractor_ids]

        env = None
        last_error = None
        for seed in range(1, 9):
            try:
                env = _make_env(
                    enable_render=True,
                    object_ids=exact_object_ids,
                    distractor_ids=exact_distractor_ids,
                    target_object_id=exact_target_id,
                )
                env.reset(seed=seed, options={"trial": seed})
                break
            except Exception as exc:
                last_error = exc
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                env = None
        if env is None:
            raise RuntimeError(f"Failed to create exact step-3 scene: {last_error!r}")

        env.restore_state(_load_snapshot(Path(args.step2_snapshot)))

        scene_metadata = _capture_scene(env, output_dir)
        api = FrankaControlPrivilegedApi(env)
        start_obs = env.get_observation()
        start_position = np.asarray(start_obs["robot_cartesian_pos"][:3], dtype=np.float64)
        start_quaternion = np.asarray(start_obs["robot_cartesian_pos"][3:7], dtype=np.float64)
        object_before = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        object_shape = api.get_object_shape()
        side_sign = float(np.sign(start_position[0] - object_before[0]))
        if side_sign == 0.0:
            side_sign = 1.0

        object_top_z = float(object_before[2]) + 0.5 * float(object_shape["size"]["z"])
        far_pregrasp = start_position.copy()

        near_pregrasp = np.array(
            [
                object_before[0] + side_sign * float(args.near_horizontal_offset),
                object_before[1],
                object_top_z + float(args.near_vertical_clearance),
            ],
            dtype=np.float64,
        )

        step2_object = object_before.copy()

        env.enable_video_capture(True, clear=True, wrist_camera=True)
        _move_linear(api, far_pregrasp, near_pregrasp, start_quaternion, steps=int(args.step3_steps))
        step3_object = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()

        displacement_m = float(np.linalg.norm(step3_object - step2_object))
        robot_frames = env.get_video_frames(clear=True)
        wrist_frames = env.get_wrist_video_frames(clear=True)
        overview_frames = env.get_overview_video_frames(clear=True)

        _write_video(output_dir / "step3_robotview.mp4", robot_frames)
        _write_video(output_dir / "step3_wristview.mp4", wrist_frames)
        _write_video(output_dir / "step3_frontview.mp4", overview_frames)
        _save_snapshot(output_dir / "step3_snapshot.npz", env.snapshot_state())

        summary = {
            "step2_snapshot": str(Path(args.step2_snapshot)),
            "start_position": start_position.tolist(),
            "far_pregrasp": far_pregrasp.tolist(),
            "near_pregrasp": near_pregrasp.tolist(),
            "grasp_quaternion_wxyz": start_quaternion.tolist(),
            "object_before_scene": object_before.tolist(),
            "object_after_step2": step2_object.tolist(),
            "object_after_step3": step3_object.tolist(),
            "target_displacement_m": displacement_m,
            "scene_metadata": scene_metadata,
        }
        _write_json(output_dir / "step3_summary.json", summary)
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
