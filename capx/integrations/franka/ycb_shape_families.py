from __future__ import annotations

from pathlib import Path


# Coarse grasp-relevant families for YCB objects. These are intended to be the
# training units for offline primitive / PPO discovery, not perfect semantic
# categories.
YCB_SHAPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "can_cylinder": (
        "002_master_chef_can",
        "005_tomato_soup_can",
        "007_tuna_fish_can",
        "010_potted_meat_can",
    ),
    "flat_box": (
        "003_cracker_box",
        "004_sugar_box",
        "008_pudding_box",
        "009_gelatin_box",
        "077_rubiks_cube",
    ),
    "tall_bottle": (
        "006_mustard_bottle",
        "019_pitcher_base",
        "021_bleach_cleanser",
        "022_windex_bottle",
    ),
    "fruit_roundish": (
        "011_banana",
        "012_strawberry",
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum",
    ),
    "cup_container": (
        "024_bowl",
        "025_mug",
        "029_plate",
        "065-a_cups",
        "065-b_cups",
        "065-c_cups",
        "065-d_cups",
        "065-e_cups",
        "065-f_cups",
        "065-g_cups",
        "065-h_cups",
        "065-i_cups",
        "065-j_cups",
    ),
    "ball_sphere": (
        "053_mini_soccer_ball",
        "054_softball",
        "055_baseball",
        "056_tennis_ball",
        "057_racquetball",
        "058_golf_ball",
        "062_dice",
        "063-a_marbles",
        "063-b_marbles",
    ),
    "flat_utensil": (
        "028_skillet_lid",
        "030_fork",
        "031_spoon",
        "032_knife",
        "033_spatula",
    ),
    "handled_tool": (
        "035_power_drill",
        "037_scissors",
        "040_large_marker",
        "042_adjustable_wrench",
        "043_phillips_screwdriver",
        "044_flat_screwdriver",
        "048_hammer",
    ),
    "clamp_like": (
        "038_padlock",
        "050_medium_clamp",
        "051_large_clamp",
        "052_extra_large_clamp",
        "071_nine_hole_peg_test",
    ),
    "block_brick": (
        "036_wood_block",
        "061_foam_brick",
        "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks",
        "073-a_lego_duplo",
        "073-b_lego_duplo",
        "073-c_lego_duplo",
        "073-d_lego_duplo",
        "073-e_lego_duplo",
        "073-f_lego_duplo",
        "073-g_lego_duplo",
    ),
    "articulated_or_irregular": (
        "026_sponge",
        "059_chain",
        "072-a_toy_airplane",
        "072-b_toy_airplane",
        "072-c_toy_airplane",
        "072-d_toy_airplane",
        "072-e_toy_airplane",
    ),
}


YCB_OBJECT_TO_FAMILY: dict[str, str] = {
    object_id: family_name
    for family_name, object_ids in YCB_SHAPE_FAMILIES.items()
    for object_id in object_ids
}


def get_ycb_shape_family(object_id: str) -> str:
    key = str(object_id).strip()
    if key not in YCB_OBJECT_TO_FAMILY:
        raise KeyError(f"No YCB shape family registered for object_id={object_id!r}")
    return YCB_OBJECT_TO_FAMILY[key]


def list_registered_ycb_object_ids() -> list[str]:
    return sorted(YCB_OBJECT_TO_FAMILY)


def list_local_ycb_object_ids(asset_root: str | Path = "/root/.maniskill/data/assets/mani_skill2_ycb/models") -> list[str]:
    root = Path(asset_root)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())
