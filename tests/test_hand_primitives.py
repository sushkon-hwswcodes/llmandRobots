import numpy as np

from capx.integrations.franka.hand_primitives import (
    INSPIRE_WRIST_PRIMITIVES_V1,
    WORLD_DIRECTIONS,
    iter_all_wrist_orientation_pairs,
    orientation_name,
    primitive_quaternion_wxyz,
)


def test_primitive_quaternion_is_finite_for_registered_primitives() -> None:
    for primitive in INSPIRE_WRIST_PRIMITIVES_V1.values():
        quat = primitive_quaternion_wxyz(primitive.palm_face, primitive.middle_finger_direction)
        assert quat.shape == (4,)
        assert np.isfinite(quat).all()
        assert np.linalg.norm(quat) > 0.99


def test_registered_primitive_axes_are_orthogonal() -> None:
    for primitive in INSPIRE_WRIST_PRIMITIVES_V1.values():
        palm = WORLD_DIRECTIONS[primitive.palm_face]
        finger = WORLD_DIRECTIONS[primitive.middle_finger_direction]
        assert abs(float(np.dot(palm, finger))) < 1e-6


def test_all_wrist_orientation_pairs_cover_24_unique_orientations() -> None:
    pairs = iter_all_wrist_orientation_pairs()
    assert len(pairs) == 24
    assert len({orientation_name(p, f) for p, f in pairs}) == 24
