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
from capx.integrations.franka.hand_presets import (
    INSPIRE_COMMAND_LABELS,
    get_hand_preset_library,
)


SITE_NAMES = [
    "gripper0_right_site_r_thumb_distal",
    "gripper0_right_site_r_index_distal",
    "gripper0_right_site_r_middle_distal",
    "gripper0_right_site_r_ring_distal",
    "gripper0_right_site_r_pinky_distal",
]


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


def _hand_joint_names(env: FrankaRobosuiteCubeLiftLowLevel) -> list[str]:
    return [name for name in env.robosuite_env.sim.model.joint_names if "gripper0_right_joint" in name]


def _joint_positions(env: FrankaRobosuiteCubeLiftLowLevel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _hand_joint_names(env):
        joint_id = env.robosuite_env.sim.model.joint_name2id(name)
        qpos_idx = env.robosuite_env.sim.model.jnt_qposadr[joint_id]
        out[name] = float(env.robosuite_env.sim.data.qpos[qpos_idx])
    return out


def _site_positions(env: FrankaRobosuiteCubeLiftLowLevel) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name in SITE_NAMES:
        site_id = env.robosuite_env.sim.model.site_name2id(name)
        out[name] = [float(x) for x in env.robosuite_env.sim.data.site_xpos[site_id]]
    return out


def _render_rgb(env: FrankaRobosuiteCubeLiftLowLevel, camera_name: str) -> np.ndarray:
    frame = env.robosuite_env.sim.render(
        camera_name=camera_name,
        width=512,
        height=512,
        depth=False,
    )
    return np.flipud(frame)


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    imageio.mimwrite(path, frames, fps=fps)


def _apply_command(
    env: FrankaRobosuiteCubeLiftLowLevel,
    command: np.ndarray,
    *,
    settle_steps: int,
) -> dict[str, Any]:
    robot_frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []

    env._set_gripper_command(command)
    for _ in range(settle_steps):
        env._step_once()
        robot_frames.append(_render_rgb(env, "robot0_robotview"))
        front_frames.append(_render_rgb(env, "frontview"))
    return {
        "joint_positions": _joint_positions(env),
        "site_positions": _site_positions(env),
        "robot_frames": robot_frames,
        "front_frames": front_frames,
    }


def _apply_preset(
    env: FrankaRobosuiteCubeLiftLowLevel,
    preset: Any,
    *,
    settle_steps: int,
) -> dict[str, Any]:
    sequence = tuple(preset.sequence) if getattr(preset, "sequence", ()) else (preset.command,)
    stage_steps = tuple(int(x) for x in getattr(preset, "stage_steps", ()) if int(x) > 0)
    if len(stage_steps) != len(sequence):
        stage_steps = tuple([settle_steps] * len(sequence))

    robot_frames: list[np.ndarray] = []
    front_frames: list[np.ndarray] = []
    stage_summaries: list[dict[str, Any]] = []

    for idx, (command, steps) in enumerate(zip(sequence, stage_steps), start=1):
        result = _apply_command(env, command, settle_steps=steps)
        robot_frames.extend(result["robot_frames"])
        front_frames.extend(result["front_frames"])
        stage_summaries.append(
            {
                "stage_index": idx,
                "command": {label: float(value) for label, value in zip(INSPIRE_COMMAND_LABELS, command)},
                "after_joint_positions": result["joint_positions"],
                "after_site_positions": result["site_positions"],
            }
        )

    return {
        "joint_positions": stage_summaries[-1]["after_joint_positions"],
        "site_positions": stage_summaries[-1]["after_site_positions"],
        "robot_frames": robot_frames,
        "front_frames": front_frames,
        "stages": stage_summaries,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and log candidate dexterous-hand presets.")
    parser.add_argument("--preset", action="append", default=[], help="Preset name to run; may be repeated.")
    parser.add_argument("--all-presets", action="store_true", help="Run every registered preset.")
    parser.add_argument(
        "--library",
        default="",
        help="Registered hand preset library to use. Defaults to the env hand_name library.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hand_preset_harness",
        help="Directory to save rendered images and JSON summaries.",
    )
    parser.add_argument("--settle-steps", type=int, default=60, help="Simulation steps to hold each preset.")
    parser.add_argument(
        "--sweep-channels",
        action="store_true",
        help="Also sweep one command channel at a time to validate command-to-finger mapping.",
    )
    args = parser.parse_args()

    env = _make_env()
    library_name = args.library or env.hand_name
    preset_library = get_hand_preset_library(library_name)
    selected = list(preset_library.keys()) if args.all_presets or not args.preset else args.preset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "hand_name": env.hand_name,
        "preset_library": library_name,
        "robot_name": env.robot_name,
        "gripper_action_dim": env._gripper_action_dim,
        "command_labels": list(INSPIRE_COMMAND_LABELS),
        "presets": {},
        "channel_sweep": {},
    }

    for preset_name in selected:
        preset = preset_library[preset_name]
        env.reset(seed=0)
        before_robot = _render_rgb(env, "robot0_robotview")
        before_front = _render_rgb(env, "frontview")
        before_joints = _joint_positions(env)
        before_sites = _site_positions(env)

        result = _apply_preset(env, preset, settle_steps=args.settle_steps)

        after_robot = _render_rgb(env, "robot0_robotview")
        after_front = _render_rgb(env, "frontview")

        preset_dir = output_dir / preset_name
        preset_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(preset_dir / "before_robotview.png", before_robot)
        imageio.imwrite(preset_dir / "before_frontview.png", before_front)
        imageio.imwrite(preset_dir / "after_robotview.png", after_robot)
        imageio.imwrite(preset_dir / "after_frontview.png", after_front)
        _write_video(preset_dir / f"{preset_name}_robotview.mp4", [before_robot, *result["robot_frames"], after_robot])
        _write_video(preset_dir / f"{preset_name}_frontview.mp4", [before_front, *result["front_frames"], after_front])

        payload = {
            "description": preset.description,
            "references": list(preset.references),
            "command": {label: float(value) for label, value in zip(INSPIRE_COMMAND_LABELS, preset.command)},
            "stages": result.get("stages", []),
            "before_joint_positions": before_joints,
            "after_joint_positions": result["joint_positions"],
            "before_site_positions": before_sites,
            "after_site_positions": result["site_positions"],
        }
        _write_json(preset_dir / "summary.json", payload)
        summary["presets"][preset_name] = payload

    if args.sweep_channels:
        for idx, label in enumerate(INSPIRE_COMMAND_LABELS):
            env.reset(seed=0)
            command = np.zeros(env._gripper_action_dim, dtype=np.float64)
            command[idx] = 1.5
            result = _apply_command(env, command, settle_steps=args.settle_steps)
            summary["channel_sweep"][label] = {
                "command": command.tolist(),
                "after_joint_positions": result["joint_positions"],
                "after_site_positions": result["site_positions"],
            }

    _write_json(output_dir / "summary.json", summary)
    env.close()


if __name__ == "__main__":
    main()
