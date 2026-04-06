from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INSPIRE_COMMAND_LABELS = (
    "pinky_curl",
    "ring_curl",
    "middle_curl",
    "index_curl",
    "thumb_curl",
    "thumb_swing",
)


@dataclass(frozen=True)
class HandPreset:
    name: str
    command: np.ndarray
    description: str


def _cmd(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


INSPIRE_RIGHT_CANDIDATE_PRESETS: dict[str, HandPreset] = {
    "open": HandPreset(
        name="open",
        command=_cmd(-1.5, -1.5, -1.5, -1.5, -3.0, 3.0),
        description="Fully open the hand using the current Robosuite saturation-style open command.",
    ),
    "soft_close": HandPreset(
        name="soft_close",
        command=_cmd(0.25, 0.25, 0.45, 0.65, 0.55, 0.35),
        description="Light closure intended to settle the fingers around an object without full squeeze.",
    ),
    "power_grasp": HandPreset(
        name="power_grasp",
        command=_cmd(1.2, 1.2, 1.2, 1.2, 1.1, 0.7),
        description="Curl all fingers strongly while keeping the thumb engaged for a wrap-style grasp.",
    ),
    "pinch": HandPreset(
        name="pinch",
        command=_cmd(-0.2, -0.1, 0.15, 1.25, 1.2, 0.9),
        description="Favor index-thumb opposition while leaving the outer fingers mostly out of the way.",
    ),
    "tripod": HandPreset(
        name="tripod",
        command=_cmd(-0.1, 0.2, 0.95, 1.05, 1.2, 0.8),
        description="Bias thumb, index, and middle toward a three-digit grasp configuration.",
    ),
    "lateral": HandPreset(
        name="lateral",
        command=_cmd(-0.35, -0.2, 0.1, 0.95, 0.8, 1.15),
        description="Rotate the thumb across the hand to test side-on contact against the index finger.",
    ),
    "hook": HandPreset(
        name="hook",
        command=_cmd(1.25, 1.25, 1.25, 1.1, -0.4, 0.1),
        description="Curl the four fingers while leaving the thumb mostly disengaged.",
    ),
    "pregrasp_wide": HandPreset(
        name="pregrasp_wide",
        command=_cmd(-0.4, -0.3, -0.1, 0.1, 0.15, 0.35),
        description="A wide pregrasp with gentle thumb swing and minimal finger curl.",
    ),
    "pregrasp_narrow": HandPreset(
        name="pregrasp_narrow",
        command=_cmd(0.1, 0.15, 0.35, 0.55, 0.55, 0.45),
        description="A narrower pregrasp that partially closes the fingers before final closure.",
    ),
}


HAND_PRESET_LIBRARIES: dict[str, dict[str, HandPreset]] = {
    "inspire_right": INSPIRE_RIGHT_CANDIDATE_PRESETS,
    "inspire_left": INSPIRE_RIGHT_CANDIDATE_PRESETS,
}


def get_hand_preset_library(hand_name: str) -> dict[str, HandPreset]:
    key = str(hand_name).strip().lower()
    if key not in HAND_PRESET_LIBRARIES:
        raise KeyError(f"No hand preset library registered for hand_name={hand_name!r}")
    return HAND_PRESET_LIBRARIES[key]
