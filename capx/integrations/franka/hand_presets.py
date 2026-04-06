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
    references: tuple[str, ...] = ()
    sequence: tuple[np.ndarray, ...] = ()
    stage_steps: tuple[int, ...] = ()


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


ALLEGRO_REFERENCE_LINES = (
    "/root/allegro_hand_ros_v5/src/bhand/include/bhand/BHand.h:45",
    "/root/allegro_hand_ros_v5/src/allegro_hand_controllers/src/allegro_node_grasp.cpp:9",
)


INSPIRE_RIGHT_ALLEGRO_V1_PRESETS: dict[str, HandPreset] = {
    "home": HandPreset(
        name="home",
        command=_cmd(-0.65, -0.45, -0.25, -0.05, -0.35, 0.45),
        description="Inspire candidate for Allegro's HOME pose: relaxed fingers with a lightly abducted thumb.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "open": HandPreset(
        name="open",
        command=_cmd(-1.5, -1.5, -1.5, -1.5, -3.0, 3.0),
        description="Fully open hand; kept alongside Allegro-style presets as a visual baseline.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "pinch_it": HandPreset(
        name="pinch_it",
        command=_cmd(-0.95, -0.7, -0.3, 1.35, 1.25, 1.25),
        description="Inspire candidate for Allegro PINCH_IT: favor thumb-index opposition while outer fingers stay mostly clear.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "pinch_mt": HandPreset(
        name="pinch_mt",
        command=_cmd(-0.85, -0.45, 1.2, -0.3, 1.2, 0.8),
        description="Inspire candidate for Allegro PINCH_MT: favor thumb-middle opposition while index stays less engaged.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "grasp_3": HandPreset(
        name="grasp_3",
        command=_cmd(-0.35, 0.45, 1.05, 1.15, 1.2, 0.9),
        description="Inspire candidate for Allegro GRASP_3: emphasize thumb, index, and middle with ring support and reduced pinky curl.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "grasp_4": HandPreset(
        name="grasp_4",
        command=_cmd(1.05, 1.05, 1.05, 1.05, 1.0, 0.75),
        description="Inspire candidate for Allegro GRASP_4: close all four fingers around the palm with the thumb engaged.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "envelop": HandPreset(
        name="envelop",
        command=_cmd(1.3, 1.15, 1.0, 0.9, 0.65, 0.2),
        description="Inspire candidate for Allegro ENVELOP: wrap the fingers with a softer thumb swing for a power-style enclosure.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
}


INSPIRE_RIGHT_ALLEGRO_V2_PRESETS: dict[str, HandPreset] = {
    "home": HandPreset(
        name="home",
        command=_cmd(-0.65, -0.45, -0.25, -0.05, -0.35, 0.45),
        description="Inspire candidate for Allegro's HOME pose: relaxed fingers with a lightly abducted thumb.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "open": HandPreset(
        name="open",
        command=_cmd(-1.5, -1.5, -1.5, -1.5, -3.0, 3.0),
        description="Fully open hand; kept alongside Allegro-style presets as a visual baseline.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "pinch_it": HandPreset(
        name="pinch_it",
        command=_cmd(-1.1, -0.95, -0.55, 1.4, 1.3, 1.35),
        description="Inspire candidate for Allegro PINCH_IT: strong thumb-index opposition while the middle and outer fingers stay more retracted.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "pinch_mt": HandPreset(
        name="pinch_mt",
        command=_cmd(-1.1, -0.9, 1.35, -0.95, 1.3, 0.35),
        description="Inspire candidate for Allegro PINCH_MT: drive middle-thumb opposition while keeping the index clearly out of the pinch.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "grasp_3": HandPreset(
        name="grasp_3",
        command=_cmd(-0.35, 0.45, 1.05, 1.15, 1.2, 0.9),
        description="Inspire candidate for Allegro GRASP_3: emphasize thumb, index, and middle with ring support and reduced pinky curl.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "grasp_4": HandPreset(
        name="grasp_4",
        command=_cmd(1.05, 1.05, 1.05, 1.05, 1.0, 0.75),
        description="Inspire candidate for Allegro GRASP_4: close all four fingers around the palm with the thumb engaged.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
    "envelop": HandPreset(
        name="envelop",
        command=_cmd(0.35, 0.15, -0.05, -0.15, 0.1, 1.35),
        description="Inspire candidate for Allegro ENVELOP: begin from a spread, thumb-abducted shape intended to look and behave broader than grasp_4 before any follow-on closure schedule.",
        references=ALLEGRO_REFERENCE_LINES,
    ),
}


INSPIRE_RIGHT_ALLEGRO_V3_PRESETS: dict[str, HandPreset] = {
    "pinch_it": INSPIRE_RIGHT_ALLEGRO_V2_PRESETS["pinch_it"],
    "pinch_mt": INSPIRE_RIGHT_ALLEGRO_V2_PRESETS["pinch_mt"],
    "grasp_4": INSPIRE_RIGHT_ALLEGRO_V2_PRESETS["grasp_4"],
    # Keep `envelop` as a staged primitive because the intended behavior is
    # "spread first, then close." The current v3 sequence is good enough to
    # preserve as a checkpoint, but it still needs more tuning before we treat
    # it as a settled grasp primitive.
    "envelop": HandPreset(
        name="envelop",
        command=_cmd(0.95, 0.95, 0.9, 0.8, 0.7, 0.45),
        description="Inspire candidate for Allegro ENVELOP as a two-stage primitive: spread the hand first, then close into a broad wrap that should read differently from grasp_4.",
        references=ALLEGRO_REFERENCE_LINES,
        sequence=(
            _cmd(-0.95, -0.75, -0.45, -0.25, -0.2, 1.5),
            _cmd(0.95, 0.95, 0.9, 0.8, 0.7, 0.45),
        ),
        stage_steps=(16, 20),
    ),
}


HAND_PRESET_LIBRARIES: dict[str, dict[str, HandPreset]] = {
    "inspire_right": INSPIRE_RIGHT_CANDIDATE_PRESETS,
    "inspire_left": INSPIRE_RIGHT_CANDIDATE_PRESETS,
    "inspire_right_candidate_v1": INSPIRE_RIGHT_CANDIDATE_PRESETS,
    "inspire_left_candidate_v1": INSPIRE_RIGHT_CANDIDATE_PRESETS,
    "inspire_right_allegro_v1": INSPIRE_RIGHT_ALLEGRO_V1_PRESETS,
    "inspire_left_allegro_v1": INSPIRE_RIGHT_ALLEGRO_V1_PRESETS,
    "inspire_right_allegro_v2": INSPIRE_RIGHT_ALLEGRO_V2_PRESETS,
    "inspire_left_allegro_v2": INSPIRE_RIGHT_ALLEGRO_V2_PRESETS,
    "inspire_right_allegro_v3": INSPIRE_RIGHT_ALLEGRO_V3_PRESETS,
    "inspire_left_allegro_v3": INSPIRE_RIGHT_ALLEGRO_V3_PRESETS,
}


def get_hand_preset_library(hand_name: str) -> dict[str, HandPreset]:
    key = str(hand_name).strip().lower()
    if key not in HAND_PRESET_LIBRARIES:
        raise KeyError(f"No hand preset library registered for hand_name={hand_name!r}")
    return HAND_PRESET_LIBRARIES[key]
