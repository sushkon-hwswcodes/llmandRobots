from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from capx.envs.base import get_env
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi
from capx.utils.video_utils import _write_video


PRESETS = [
    "open",
    "pregrasp",
    "wide_enclose",
    "grasp_soft",
    "box_wrap",
    "cylinder_wrap",
    "ball_cup",
    "close",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/inspire_preset_demos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-steps", type=int, default=20)
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for preset_name in PRESETS:
        env = get_env(
            "franka_robosuite_shape_lift_low_level",
            enable_render=False,
            robot_name="PandaDexRH",
            hand_name="inspire_right",
            robosuite_gripper_type="default",
            ik_robot_name="panda_description",
            ik_target_link_name="panda_hand",
            eef_body_name="gripper0_right_eef",
            tcp_offset=[0.0, 0.0, -0.107],
        )
        env.reset(seed=args.seed)
        env.enable_video_capture(True, clear=True)
        api = FrankaControlPrivilegedApi(env)

        grasp_pos, grasp_quat = api.sample_grasp_pose("object")
        hover_pos = np.asarray(grasp_pos, dtype=np.float64) + np.array([0.0, 0.0, 0.06], dtype=np.float64)

        api.set_hand_preshape("open", steps=20)
        api.goto_pose(hover_pos, grasp_quat)
        api.set_hand_preshape(preset_name, steps=25)
        for _ in range(args.hold_steps):
            env._step_once()

        preset_dir = output_root / preset_name
        preset_dir.mkdir(parents=True, exist_ok=True)

        frames = env.get_video_frames(clear=True)
        if frames:
            _write_video(frames, str(preset_dir), suffix="main")

        overview_frames = env.get_overview_video_frames(clear=True)
        if overview_frames:
            _write_video(overview_frames, str(preset_dir), suffix="overview")


if __name__ == "__main__":
    main()
