#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from capx.envs.base import get_env


DEFAULT_OBJECT_IDS = [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-object-id", default="005_tomato_soup_can")
    parser.add_argument("--num-distractors", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = get_env(
        "franka_robosuite_ycb_target_clutter_low_level",
        privileged=True,
        enable_render=True,
        robot_name="Panda",
        hand_name="panda",
        ik_robot_name="panda_description",
        ik_target_link_name="panda_hand",
        eef_body_name="gripper0_right_eef",
        tcp_offset=[0.0, 0.0, -0.107],
        object_ids=DEFAULT_OBJECT_IDS,
        target_object_id=args.target_object_id,
        num_distractors=args.num_distractors,
        placement_x_range=(-0.28, 0.28),
        placement_y_range=(-0.20, 0.20),
    )
    obs, info = env.reset(seed=args.seed, options={"trial": args.seed})

    robotview = env.render()
    frontview = env.robosuite_env.sim.render(
        camera_name="frontview",
        width=512,
        height=512,
        depth=False,
    )[::-1]

    Image.fromarray(robotview).save(output_dir / "scene_robotview.png")
    Image.fromarray(frontview).save(output_dir / "scene_frontview.png")

    metadata = {
        "seed": args.seed,
        "task_prompt": info["task_prompt"],
        "target": env.robosuite_env._current_object_info,
        "distractors": env.robosuite_env._distractor_infos,
        "observation_target_pose": obs["cube_poses"]["primary"].tolist(),
    }
    (output_dir / "scene_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
