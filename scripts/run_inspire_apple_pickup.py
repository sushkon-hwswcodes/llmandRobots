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
from typing import Any

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.base import get_env

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspire_apple_pickup_program import ATTEMPTS, attempt_names, run_attempt


OBJECT_IDS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "053_mini_soccer_ball",
    "061_foam_brick",
    "062_dice",
    "077_rubiks_cube",
]


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ensure_pyroki_server(host: str = "127.0.0.1", port: int = 8116) -> subprocess.Popen[str] | None:
    if _port_open(host, port):
        return None

    command = [
        sys.executable,
        "-c",
        (
            "from capx.serving.launch_pyroki_server import main; "
            f"main(robot='panda_description', target_link='panda_hand', port={port}, host='{host}')"
        ),
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    deadline = time.time() + 45.0
    while time.time() < deadline:
        if _port_open(host, port):
            return process
        if process.poll() is not None:
            raise RuntimeError("Failed to start PyRoKi server")
        time.sleep(0.5)
    process.terminate()
    raise RuntimeError("Timed out waiting for PyRoKi server")


def _make_env(
    enable_render: bool = True,
    object_ids: list[str] | None = None,
    distractor_ids: list[str] | None = None,
    target_object_id: str = "013_apple",
):
    return get_env(
        "franka_robosuite_ycb_target_clutter_low_level",
        privileged=True,
        enable_render=enable_render,
        robot_name="PandaDexRH",
        hand_name="inspire_right",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
        object_ids=object_ids or OBJECT_IDS,
        distractor_ids=distractor_ids,
        target_object_id=target_object_id,
        num_distractors=len(distractor_ids) if distractor_ids is not None else 10,
        placement_x_range=(-0.28, 0.28),
        placement_y_range=(-0.20, 0.20),
    )


def _render_camera(env, camera_name: str) -> np.ndarray:
    frame = env.robosuite_env.sim.render(
        camera_name=camera_name,
        width=512,
        height=512,
        depth=False,
    )
    return np.flipud(frame)


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def _save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    override = snapshot.get("gripper_command_override")
    np.savez_compressed(
        path,
        sim_state=np.asarray(snapshot["sim_state"], dtype=np.float64),
        robosuite_timestep=np.asarray([int(snapshot.get("robosuite_timestep", 0))], dtype=np.int64),
        step_count=np.asarray([int(snapshot["step_count"])], dtype=np.int64),
        sim_step_count=np.asarray([int(snapshot["sim_step_count"])], dtype=np.int64),
        current_joints=np.asarray(snapshot["current_joints"], dtype=np.float64),
        gripper_fraction=np.asarray([float(snapshot["gripper_fraction"])], dtype=np.float64),
        has_gripper_override=np.asarray([0 if override is None else 1], dtype=np.int64),
        gripper_override=(
            np.zeros(1, dtype=np.float64)
            if override is None
            else np.asarray(override, dtype=np.float64)
        ),
    )


def _load_snapshot(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    has_override = bool(int(data["has_gripper_override"][0]))
    return {
        "sim_state": np.asarray(data["sim_state"], dtype=np.float64),
        "robosuite_timestep": int(data["robosuite_timestep"][0]) if "robosuite_timestep" in data else 0,
        "step_count": int(data["step_count"][0]),
        "sim_step_count": int(data["sim_step_count"][0]),
        "current_joints": np.asarray(data["current_joints"], dtype=np.float64),
        "gripper_fraction": float(data["gripper_fraction"][0]),
        "gripper_command_override": (
            np.asarray(data["gripper_override"], dtype=np.float64) if has_override else None
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_scene(env, output_dir: Path) -> dict[str, Any]:
    robotview = env.render()
    frontview = _render_camera(env, "frontview")
    wristview = env.render_wrist()

    imageio.imwrite(output_dir / "scene_robotview.png", robotview)
    imageio.imwrite(output_dir / "scene_frontview.png", frontview)
    imageio.imwrite(output_dir / "scene_wristview.png", wristview)

    metadata = {
        "task_prompt": f"Pick up the 013 apple and lift it with the Inspire hand.",
        "target": env.robosuite_env._current_object_info,
        "distractors": env.robosuite_env._distractor_infos,
        "attempt_order": attempt_names(),
    }
    _write_json(output_dir / "scene_metadata.json", metadata)
    return metadata


def _run_attempt(env, scene_snapshot: dict[str, Any], attempt_name: str, output_dir: Path) -> dict[str, Any]:
    env.restore_state(scene_snapshot)
    env.enable_video_capture(True, clear=True, wrist_camera=True)
    try:
        result = run_attempt(env, attempt_name)
    except Exception as exc:
        result = {
            "attempt": attempt_name,
            "task_completed": False,
            "reward": float(env.compute_reward()),
            "error": repr(exc),
        }

    robot_frames = env.get_video_frames(clear=True)
    wrist_frames = env.get_wrist_video_frames(clear=True)
    overview_frames = env.get_overview_video_frames(clear=True)

    attempt_dir = output_dir / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=True)
    _write_video(attempt_dir / "robotview.mp4", robot_frames)
    _write_video(attempt_dir / "frontview.mp4", overview_frames)
    _write_video(attempt_dir / "wristview.mp4", wrist_frames)
    _write_json(attempt_dir / "result.json", result)
    return result


def _write_success_program(path: Path, attempt_name: str) -> None:
    path.write_text(
        (
            "import sys\n"
            "from pathlib import Path\n\n"
            "SCRIPT_DIR = Path('/root/llmandRobots/scripts')\n"
            "if str(SCRIPT_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(SCRIPT_DIR))\n\n"
            "import numpy as np\n\n"
            "from inspire_apple_pickup_program import run_attempt\n\n"
            f"RESULT = run_attempt(env, {attempt_name!r})\n"
            "print(RESULT)\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact Inspire-hand apple pickup on a fixed 11-object YCB clutter scene.")
    parser.add_argument(
        "--output-dir",
        default="outputs/inspire_apple_clutter_11_v1",
        help="Directory for the scene snapshot, images, results, and videos.",
    )
    parser.add_argument(
        "--scene-snapshot",
        default="",
        help="Existing scene snapshot .npz to replay. If omitted, a new scene is generated and saved.",
    )
    parser.add_argument(
        "--attempt-name",
        default="",
        help="Run only one named attempt on the scene snapshot. Defaults to the built-in sequence until success.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyroki_process = _ensure_pyroki_server()
    try:
        env = _make_env(enable_render=True)
        if args.scene_snapshot:
            env.reset(options={"trial": 1})
            scene_snapshot = _load_snapshot(Path(args.scene_snapshot))
            env.restore_state(scene_snapshot)
            scene_metadata = _capture_scene(env, output_dir)
            snapshot_path = output_dir / "scene_snapshot.npz"
            _save_snapshot(snapshot_path, scene_snapshot)
        else:
            env.reset(options={"trial": 1})
            scene_snapshot = env.snapshot_state()
            snapshot_path = output_dir / "scene_snapshot.npz"
            _save_snapshot(snapshot_path, scene_snapshot)
            scene_metadata = _capture_scene(env, output_dir)

        ordered_attempts = [args.attempt_name] if args.attempt_name else [attempt.name for attempt in ATTEMPTS]
        results: list[dict[str, Any]] = []
        successful_attempt: str | None = None
        for attempt_name in ordered_attempts:
            result = _run_attempt(env, scene_snapshot, attempt_name, output_dir / "attempts")
            results.append(result)
            if bool(result["task_completed"]):
                successful_attempt = attempt_name
                break

        summary = {
            "scene_snapshot": str(snapshot_path),
            "scene_metadata": scene_metadata,
            "attempt_results": results,
            "successful_attempt": successful_attempt,
            "rerun_command": (
                f"cd /root/llmandRobots && source .venv/bin/activate && "
                f"python scripts/run_inspire_apple_pickup.py --scene-snapshot {snapshot_path} "
                f"--attempt-name {successful_attempt or ordered_attempts[-1]} "
                f"--output-dir {output_dir / 'rerun'}"
            ),
        }
        _write_json(output_dir / "summary.json", summary)

        if successful_attempt is not None:
            _write_success_program(output_dir / "successful_program.py", successful_attempt)
        else:
            raise RuntimeError("No attempt succeeded on this scene")
    finally:
        if pyroki_process is not None:
            pyroki_process.terminate()
            try:
                pyroki_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pyroki_process.kill()


if __name__ == "__main__":
    main()
