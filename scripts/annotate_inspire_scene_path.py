#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_inspire_apple_pickup import _load_snapshot
from run_inspire_apple_step2 import _build_env_with_retries
from capx.integrations.franka.control_privileged import FrankaControlPrivilegedApi


def _font(size: int = 20):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]:
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _draw_path(img_path: Path, out_path: Path, green_pts: list[tuple[int, int]], blue_pts: list[tuple[int, int]], labels: dict[int, tuple[int, int]], title: str) -> None:
    img = Image.open(img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _font(20)
    title_font = _font(24)

    if len(green_pts) >= 2:
        d.line(green_pts, fill=(0, 220, 80, 255), width=7)
    if len(blue_pts) >= 2:
        d.line(blue_pts, fill=(40, 120, 255, 255), width=7)

    for step_idx, xy in labels.items():
        x, y = xy
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
        d.text((x + 10, y - 12), str(step_idx), fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))

    d.rounded_rectangle((12, 12, 388, 72), radius=10, fill=(0, 0, 0, 150))
    d.text((24, 20), title, fill=(255, 255, 255, 255), font=title_font)
    d.text((24, 48), "green: pre-contact  blue: post-grasp", fill=(220, 220, 220, 255), font=_font(16))

    Image.alpha_composite(img, overlay).save(out_path)


def main() -> None:
    scene_dir = Path('outputs/inspire_llm_scene_01')
    meta = json.loads((scene_dir / 'scene_metadata.json').read_text())
    exact_target_id = meta['target']['category']
    exact_distractor_ids = [item['category'] for item in meta['distractors']]
    exact_object_ids = [exact_target_id, *exact_distractor_ids]

    env = _build_env_with_retries(exact_object_ids, exact_distractor_ids, exact_target_id, 1)
    env.restore_state(_load_snapshot(scene_dir / 'scene_snapshot.npz'))
    api = FrankaControlPrivilegedApi(env)
    obs = env.get_observation()
    start = np.asarray(obs['robot_cartesian_pos'][:3], dtype=float)
    apple = np.asarray(api.get_object_pose('object')[0], dtype=float)
    apple_top = float(apple[2]) + 0.5 * float(meta['target']['size']['z'])

    waypoints = {
        '1_start': start.tolist(),
        '2_raise_safe': [float(start[0]), float(start[1]), 0.34],
        '3_transit_high': [float(apple[0] - 0.02), float(apple[1] + 0.01), 0.34],
        '4_center_high': [float(apple[0]), float(apple[1]), 0.31],
        '5_precontact': [float(apple[0]), float(apple[1]), 0.22],
        '6_grasp_close': [float(apple[0]), float(apple[1]), max(apple_top + 0.02, 0.14)],
        '7_lift': [float(apple[0]), float(apple[1]), 0.30],
        '8_retreat': [float(apple[0] - 0.08), float(apple[1] + 0.02), 0.34],
    }

    # Visual overlays are hand-tuned for scene_01; world_waypoints are authoritative.
    robot_labels = {
        1: (257, 44),
        2: (257, 70),
        3: (280, 120),
        4: (270, 215),
        5: (262, 300),
        6: (257, 380),
        7: (257, 250),
        8: (220, 120),
    }
    front_labels = {
        1: (257, 132),
        2: (257, 86),
        3: (273, 90),
        4: (268, 124),
        5: (263, 198),
        6: (259, 287),
        7: (262, 150),
        8: (223, 98),
    }

    _draw_path(
        scene_dir / 'scene_robotview.png',
        scene_dir / 'scene_robotview_planned_path.png',
        [robot_labels[i] for i in [1, 2, 3, 4, 5, 6]],
        [robot_labels[i] for i in [6, 7, 8]],
        robot_labels,
        'Scene 01 Planned Palm Path',
    )
    _draw_path(
        scene_dir / 'scene_frontview.png',
        scene_dir / 'scene_frontview_planned_path.png',
        [front_labels[i] for i in [1, 2, 3, 4, 5, 6]],
        [front_labels[i] for i in [6, 7, 8]],
        front_labels,
        'Scene 01 Planned Palm Path',
    )

    payload = {
        'scene': 'inspire_llm_scene_01',
        'rule': 'Future step runners should follow these world_waypoints in order unless a backtrack is explicitly triggered.',
        'phases': {
            'green_pre_contact': ['1_start', '2_raise_safe', '3_transit_high', '4_center_high', '5_precontact', '6_grasp_close'],
            'blue_post_grasp': ['6_grasp_close', '7_lift', '8_retreat'],
        },
        'world_waypoints_xyz': waypoints,
        'notes': {
            'image_overlays': 'Conceptual drawing for scene_01 inspection.',
            'world_waypoints': 'Authoritative path that later step runners should follow exactly.',
        },
    }
    (scene_dir / 'planned_path.json').write_text(json.dumps(payload, indent=2))
    env.close()


if __name__ == '__main__':
    main()
