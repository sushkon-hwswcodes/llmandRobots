from capx.envs.simulators.robosuite_ycb_lift import _YCB_CANDIDATES
from capx.integrations.franka.ycb_shape_families import (
    YCB_OBJECT_TO_FAMILY,
    get_ycb_shape_family,
    list_local_ycb_object_ids,
)


def test_current_ycb_benchmark_candidates_have_shape_families() -> None:
    for object_id in _YCB_CANDIDATES:
        family = get_ycb_shape_family(object_id)
        assert family


def test_local_ycb_assets_are_all_covered_by_shape_families() -> None:
    local_ids = list_local_ycb_object_ids()
    assert local_ids, "Expected local ManiSkill YCB assets to be installed for this test"
    missing = sorted(set(local_ids) - set(YCB_OBJECT_TO_FAMILY))
    assert not missing, f"Unmapped local YCB object ids: {missing}"


def test_each_ycb_object_maps_to_exactly_one_family() -> None:
    assert len(YCB_OBJECT_TO_FAMILY) == len(set(YCB_OBJECT_TO_FAMILY))
