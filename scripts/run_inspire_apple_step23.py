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
    parser = argparse.ArgumentParser(description="Replay step 2 and step 3 together on one fixed Inspire apple clutter scene.")
    parser.add_argument("--scene-dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = scene_dir
    step2_dir = scene_dir / "step_02"
    step3_dir = scene_dir / "step_03"
    step2_dir.mkdir(parents=True, exist_ok=True)
    step3_dir.mkdir(parents=True, exist_ok=True)

    pyroki_process = _ensure_pyroki_server()
    try:
        scene_metadata_path = scene_dir / "scene_metadata.json"
        scene_snapshot_path = scene_dir / "scene_snapshot.npz"
        scene_metadata = json.loads(scene_metadata_path.read_text(encoding="utf-8"))
        exact_target_id = scene_metadata["target"]["category"]
        exact_distractor_ids = [item["category"] for item in scene_metadata["distractors"]]
        exact_object_ids = [exact_target_id, *exact_distractor_ids]

        env = _make_env(
            enable_render=True,
            object_ids=exact_object_ids,
            distractor_ids=exact_distractor_ids,
            target_object_id=exact_target_id,
        )
        env.reset(seed=1, options={"trial": 1})
        env.restore_state(_load_snapshot(scene_snapshot_path))

        scene_metadata = _capture_scene(env, output_dir)
        api = FrankaControlPrivilegedApi(env)

        start_obs = env.get_observation()
        start_position = np.asarray(start_obs["robot_cartesian_pos"][:3], dtype=np.float64)
        start_quaternion = np.asarray(start_obs["robot_cartesian_pos"][3:7], dtype=np.float64)
        object_before = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        object_shape = api.get_object_shape()

        primitive_pregrasp, _ = api.sample_hand_primitive_pregrasp_pose("ball_wrap_back_down", "object")
        primitive_pregrasp = np.asarray(primitive_pregrasp, dtype=np.float64)
        side_sign = float(np.sign(primitive_pregrasp[0] - object_before[0]))
        if side_sign == 0.0:
            side_sign = 1.0

        object_top_z = float(object_before[2]) + 0.5 * float(object_shape["size"]["z"])
        start_hover = start_position.copy()
        start_hover[2] += 0.10

        far_pregrasp = np.array(
            [object_before[0] + side_sign * (0.13 + 0.05), object_before[1], object_before[2]],
            dtype=np.float64,
        )
        far_pregrasp[2] = max(float(object_before[2]), object_top_z + 0.15)

        transit_hover = far_pregrasp.copy()
        transit_hover[2] = max(float(transit_hover[2]), float(start_hover[2]))

        near_pregrasp = np.array(
            [object_before[0] + side_sign * 0.06, object_before[1], object_top_z + 0.08],
            dtype=np.float64,
        )

        env.enable_video_capture(True, clear=True, wrist_camera=True)

        api.open_gripper()
        _move_linear(api, start_position, start_hover, start_quaternion, steps=4)
        _move_linear(api, start_hover, transit_hover, start_quaternion, steps=6)
        _move_linear(api, transit_hover, far_pregrasp, start_quaternion, steps=4)

        object_after_step2 = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        step2_snapshot = env.snapshot_state()
        _save_snapshot(step2_dir / "step2_snapshot.npz", step2_snapshot)

        robot_frames = env.get_video_frames(clear=True)
        wrist_frames = env.get_wrist_video_frames(clear=True)
        overview_frames = env.get_overview_video_frames(clear=True)
        _write_video(step2_dir / "step2_robotview.mp4", robot_frames)
        _write_video(step2_dir / "step2_frontview.mp4", overview_frames)
        _write_video(step2_dir / "step2_wristview.mp4", wrist_frames)
        _write_json(
            step2_dir / "step2_summary.json",
            {
                "start_position": start_position.tolist(),
                "start_hover": start_hover.tolist(),
                "transit_hover": transit_hover.tolist(),
                "far_pregrasp": far_pregrasp.tolist(),
                "grasp_quaternion_wxyz": start_quaternion.tolist(),
                "object_before": object_before.tolist(),
                "object_after": object_after_step2.tolist(),
                "target_displacement_m": float(np.linalg.norm(object_after_step2 - object_before)),
                "scene_snapshot": str(scene_snapshot_path),
                "scene_metadata": scene_metadata,
            },
        )

        env.enable_video_capture(True, clear=True, wrist_camera=True)
        _move_linear(api, far_pregrasp, near_pregrasp, start_quaternion, steps=6)

        object_after_step3 = np.asarray(api.get_object_pose("object")[0], dtype=np.float64).copy()
        step3_snapshot = env.snapshot_state()
        _save_snapshot(step3_dir / "step3_snapshot.npz", step3_snapshot)

        robot_frames = env.get_video_frames(clear=True)
        wrist_frames = env.get_wrist_video_frames(clear=True)
        overview_frames = env.get_overview_video_frames(clear=True)

        _write_video(step3_dir / "step3_robotview.mp4", robot_frames)
        _write_video(step3_dir / "step3_frontview.mp4", overview_frames)
        _write_video(step3_dir / "step3_wristview.mp4", wrist_frames)

        _write_json(
            step3_dir / "step3_summary.json",
            {
                "start_position": far_pregrasp.tolist(),
                "far_pregrasp": far_pregrasp.tolist(),
                "near_pregrasp": near_pregrasp.tolist(),
                "grasp_quaternion_wxyz": start_quaternion.tolist(),
                "object_before_scene": object_after_step2.tolist(),
                "object_after_step2": object_after_step2.tolist(),
                "object_after_step3": object_after_step3.tolist(),
                "target_displacement_m": float(np.linalg.norm(object_after_step3 - object_after_step2)),
                "scene_metadata": scene_metadata,
            },
        )
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
