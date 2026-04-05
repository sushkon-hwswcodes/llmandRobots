#!/usr/bin/env python3
"""Build 4x2 grid videos from successful trial videos.

Creates two grid outputs from a run directory:
1) video_turn_*.mp4 grid
2) video_turn_*_overview.mp4 grid
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
from pathlib import Path


TRIAL_RE = re.compile(r"trial_(\d+)_sandboxrc_0_reward_1\.000_taskcompleted_1$")


def _trial_index(path: Path) -> int:
    m = TRIAL_RE.search(path.name)
    return int(m.group(1)) if m else 10**9


def find_success_dirs(run_dir: Path) -> list[Path]:
    dirs = []
    for p in run_dir.glob("trial_*_sandboxrc_0_reward_1.000_taskcompleted_1"):
        if p.is_dir() and TRIAL_RE.search(p.name):
            dirs.append(p)
    dirs.sort(key=_trial_index)
    return dirs


def pick_videos(trial_dirs: list[Path], overview: bool) -> list[Path]:
    chosen = []
    pattern = "video_turn_*_overview.mp4" if overview else "video_turn_*.mp4"
    for d in trial_dirs:
        candidates = sorted(d.glob(pattern))
        if overview:
            candidates = [c for c in candidates if "combined" not in c.name]
        else:
            candidates = [c for c in candidates if "overview" not in c.name and "combined" not in c.name]
        if not candidates:
            raise FileNotFoundError(f"No matching videos in {d}")
        chosen.append(candidates[0])
    return chosen


def build_grid(inputs: list[Path], output: Path, cols: int, rows: int, cell_w: int, cell_h: int, fps: int) -> None:
    if len(inputs) != cols * rows:
        raise ValueError(f"Need {cols * rows} inputs, got {len(inputs)}")

    cmd = ["ffmpeg", "-y"]
    for inp in inputs:
        cmd += ["-i", str(inp)]

    pre = []
    for i in range(len(inputs)):
        pre.append(
            f"[{i}:v]fps={fps},scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
    layout = []
    for r in range(rows):
        for c in range(cols):
            layout.append(f"{c * cell_w}_{r * cell_h}")
    inputs_tag = "".join(f"[v{i}]" for i in range(len(inputs)))
    filt = ";".join(pre) + ";" + f"{inputs_tag}xstack=inputs={len(inputs)}:layout=" + "|".join(layout) + ":fill=black[v]"

    cmd += [
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to run folder containing trial_* dirs")
    ap.add_argument("--count", type=int, default=8, help="Number of successful trials to include (default: 8)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: run dir)")
    ap.add_argument("--prefix", default="success8_grid", help="Output filename prefix")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cell-w", type=int, default=640)
    ap.add_argument("--cell-h", type=int, default=360)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success_dirs = find_success_dirs(run_dir)
    if len(success_dirs) < args.count:
        raise RuntimeError(f"Found only {len(success_dirs)} success dirs in {run_dir}, need {args.count}")
    selected = success_dirs[: args.count]

    regular_videos = pick_videos(selected, overview=False)
    overview_videos = pick_videos(selected, overview=True)

    regular_out = out_dir / f"{args.prefix}_video_turn.mp4"
    overview_out = out_dir / f"{args.prefix}_video_turn_overview.mp4"
    meta_out = out_dir / f"{args.prefix}_selected_trials.txt"

    build_grid(regular_videos, regular_out, args.cols, args.rows, args.cell_w, args.cell_h, args.fps)
    build_grid(overview_videos, overview_out, args.cols, args.rows, args.cell_w, args.cell_h, args.fps)

    with meta_out.open("w", encoding="utf-8") as f:
        for d in selected:
            f.write(d.name + "\n")

    print(f"Created: {regular_out}")
    print(f"Created: {overview_out}")
    print(f"Selected list: {meta_out}")


if __name__ == "__main__":
    main()
