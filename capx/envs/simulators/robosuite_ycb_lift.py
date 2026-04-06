"""Low-level Robosuite Franka environments for Phase 3 real-world YCB objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import MujocoXMLObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


_YCB_CANDIDATES = [
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "008_pudding_box",
]


class YCBMeshObject(MujocoXMLObject):
    """Runtime-generated XML object wrapper around ManiSkill YCB meshes."""

    def __init__(self, model_id: str, metadata: dict[str, Any], xml_path: Path, name: str | None = None):
        self.model_id = model_id
        self.metadata = metadata
        super().__init__(
            str(xml_path),
            name=name or model_id,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


def _display_name(model_id: str) -> str:
    return model_id.replace("_", " ")


def _resolve_maniskill_asset_dir() -> Path:
    env = os.environ.get("MANISKILL_ASSET_DIR")
    if env:
        candidate = Path(env).expanduser()
        if candidate.exists():
            return candidate

    try:
        import mani_skill  # type: ignore

        candidate = Path(mani_skill.ASSET_DIR)
        if candidate.exists():
            return candidate
    except Exception:
        pass

    candidate = Path.home() / ".maniskill" / "data"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not find ManiSkill asset dir. Set MANISKILL_ASSET_DIR or download assets with "
        "`python -m mani_skill.utils.download_asset -y ycb`."
    )


def _load_ycb_metadata(asset_root: Path) -> dict[str, Any]:
    info_path = asset_root / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _full_extents(metadata: dict[str, Any]) -> np.ndarray:
    bbox = metadata["bbox"]
    mins = np.asarray(bbox["min"], dtype=np.float64)
    maxs = np.asarray(bbox["max"], dtype=np.float64)
    scale = float(metadata.get("scales", [1.0])[0])
    return (maxs - mins) * scale


def _write_ycb_xml(asset_root: Path, model_id: str, metadata: dict[str, Any]) -> Path:
    model_dir = asset_root / "assets" / "mani_skill2_ycb" / "models" / model_id
    visual_path = (model_dir / "textured.obj").resolve()
    collision_path = (model_dir / "textured.obj").resolve()
    texture_path = (model_dir / "texture_map.png").resolve()

    extents = _full_extents(metadata)
    half = extents / 2.0
    radius = float(np.linalg.norm(extents[:2]) / 2.0)
    density = float(metadata.get("density", 1000))

    out_dir = Path("/tmp/capx_ycb_xml")
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{model_id}.xml"

    xml = dedent(
        f"""
        <mujoco model=\"{model_id}\">
          <asset>
            <mesh file=\"{collision_path}\" name=\"{model_id}_collision_mesh\" scale=\"1 1 1\"/>
            <mesh file=\"{visual_path}\" name=\"{model_id}_visual_mesh\" scale=\"1 1 1\"/>
            <texture file=\"{texture_path}\" type=\"2d\" name=\"tex-{model_id}\" />
            <material name=\"mat-{model_id}\" texture=\"tex-{model_id}\" texuniform=\"true\"/>
          </asset>
          <worldbody>
            <body>
              <body name=\"object\">
                <geom name=\"collision\" pos=\"0 0 0\" mesh=\"{model_id}_collision_mesh\" type=\"mesh\"
                      solimp=\"0.998 0.998 0.001\" solref=\"0.001 1\" density=\"{density}\"
                      friction=\"0.95 0.3 0.1\" group=\"0\" condim=\"4\"/>
                <geom name=\"visual\" pos=\"0 0 0\" mesh=\"{model_id}_visual_mesh\" type=\"mesh\"
                      material=\"mat-{model_id}\" contype=\"0\" conaffinity=\"0\" group=\"1\"/>
              </body>
              <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 {-half[2]:.6f}\" name=\"bottom_site\"/>
              <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"0 0 {half[2]:.6f}\" name=\"top_site\"/>
              <site rgba=\"0 0 0 0\" size=\"0.005\" pos=\"{radius:.6f} 0 0\" name=\"horizontal_radius_site\"/>
            </body>
          </worldbody>
        </mujoco>
        """
    ).strip() + "\n"
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


class LiftYCBObject(Lift):
    """Lift task with one random YCB object per episode."""

    def __init__(self, *args, object_ids: list[str] | None = None, **kwargs):
        self.asset_root = _resolve_maniskill_asset_dir()
        self._ycb_metadata = _load_ycb_metadata(self.asset_root)
        self.object_ids = object_ids or list(_YCB_CANDIDATES)
        self._current_shape = "mesh"
        self._current_size = np.zeros(3)
        self._current_object_info: dict[str, Any] = {}
        self.prefer_center_grasp_pose = True
        super().__init__(*args, **kwargs)

    def _make_ycb_object(self) -> YCBMeshObject:
        idx = int(self.rng.integers(0, len(self.object_ids)))
        model_id = self.object_ids[idx]
        metadata = self._ycb_metadata[model_id]
        xml_path = _write_ycb_xml(self.asset_root, model_id, metadata)
        extents = _full_extents(metadata)
        self._current_size = extents.copy()
        self._current_object_info = {
            "shape": "mesh",
            "category": model_id,
            "display_name": _display_name(model_id),
            "size": {
                "x": round(float(extents[0]), 4),
                "y": round(float(extents[1]), 4),
                "z": round(float(extents[2]), 4),
            },
            "grasp_hint": (
                "Use a top-down grasp near the object's center. Prefer sample_grasp_pose('object') "
                "over deriving the grasp pose from size alone."
            ),
        }
        return YCBMeshObject(model_id=model_id, metadata=metadata, xml_path=xml_path)

    def _load_model(self):
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self.cube = self._make_ycb_object()

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="YCBObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.04, 0.04],
                y_range=[-0.04, 0.04],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

    def _setup_references(self):
        super()._setup_references()
        if self._current_object_info:
            size = self._current_object_info["size"]
            self._current_size = np.array([size["x"], size["y"], size["z"]], dtype=np.float64)


class FrankaRobosuiteYCBLiftLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka YCB Lift: one random real-world object per episode."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        robot_name: str = "Panda",
        hand_name: str = "panda",
        robosuite_gripper_type: str = "default",
        ik_robot_name: str = "panda_description",
        ik_target_link_name: str = "panda_hand",
        eef_body_name: str = "gripper0_right_eef",
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 0.0, -0.107),
        gripper_open_command: float = -1.0,
        gripper_closed_command: float = 1.0,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
            robot_name=robot_name,
            hand_name=hand_name,
            robosuite_gripper_type=robosuite_gripper_type,
            ik_robot_name=ik_robot_name,
            ik_target_link_name=ik_target_link_name,
            eef_body_name=eef_body_name,
            tcp_offset=tcp_offset,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        )

        lift_kwargs = dict(
            robots=[self.robot_name],
            gripper_types=self.robosuite_gripper_type,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
            object_ids=list(_YCB_CANDIDATES),
        )

        if privileged and not enable_render:
            lift_kwargs.update(
                use_camera_obs=False,
                has_offscreen_renderer=False,
                camera_names=[],
            )

        self.robosuite_env = LiftYCBObject(**lift_kwargs)
        self._initial_cube_height: float | None = None
        self._init_robot_links()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        self._initial_cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        display_name = self.robosuite_env._current_object_info.get("display_name", "object")
        info = {"task_prompt": f"Pick up the object on the table and lift it. The current object is {display_name}."}
        return obs, info

    def _object_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        obj_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]])
        )
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        obj_robot_base = base_transform @ obj_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate([obj_robot_base.translation(), obj_robot_base.rotation().wxyz])
            ],
        }

    def compute_reward(self) -> float:
        if self.task_completed():
            return 1.0
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        return float(np.clip((cube_height - baseline) / 0.04, 0.0, 1.0))

    def task_completed(self) -> bool:
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        lifted = cube_height > (baseline + 0.04)
        grasped = self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cube,
        )
        return bool(lifted and grasped)

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._object_pose_dict(robosuite_obs)
        robosuite_obs["cube_poses"] = {
            "primary": np.asarray(pose_dict["primary"], dtype=np.float32),
        }
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs


class LiftYCBTargetClutter(Lift):
    """Lift task with one specified YCB target among YCB distractors."""

    def __init__(
        self,
        *args,
        object_ids: list[str] | None = None,
        num_distractors: int = 4,
        **kwargs,
    ):
        self.asset_root = _resolve_maniskill_asset_dir()
        self._ycb_metadata = _load_ycb_metadata(self.asset_root)
        self.object_ids = object_ids or list(_YCB_CANDIDATES)
        self.num_distractors = num_distractors
        self._current_shape = "mesh"
        self._current_size = np.zeros(3)
        self._current_object_info: dict[str, Any] = {}
        self._distractor_infos: list[dict[str, Any]] = []
        self.prefer_center_grasp_pose = True
        super().__init__(*args, **kwargs)

    def _make_named_ycb_object(self, model_id: str, name: str) -> YCBMeshObject:
        metadata = self._ycb_metadata[model_id]
        xml_path = _write_ycb_xml(self.asset_root, model_id, metadata)
        return YCBMeshObject(model_id=model_id, metadata=metadata, xml_path=xml_path, name=name)

    def _info_for_model(self, model_id: str) -> dict[str, Any]:
        extents = _full_extents(self._ycb_metadata[model_id])
        return {
            "shape": "mesh",
            "category": model_id,
            "display_name": _display_name(model_id),
            "size": {
                "x": round(float(extents[0]), 4),
                "y": round(float(extents[1]), 4),
                "z": round(float(extents[2]), 4),
            },
            "grasp_hint": (
                "Use a top-down grasp near the object's center. Prefer sample_grasp_pose('object') "
                "over deriving the grasp pose from size alone."
            ),
        }

    def _load_model(self):
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        target_idx = int(self.rng.integers(0, len(self.object_ids)))
        target_id = self.object_ids[target_idx]
        distractor_pool = [obj_id for obj_id in self.object_ids if obj_id != target_id]
        if len(distractor_pool) < self.num_distractors:
            distractor_ids = list(self.rng.choice(distractor_pool, size=self.num_distractors, replace=True))
        else:
            distractor_ids = list(self.rng.choice(distractor_pool, size=self.num_distractors, replace=False))

        self.cube = self._make_named_ycb_object(target_id, "target")
        self._current_object_info = self._info_for_model(target_id)
        self._current_size = np.array(
            [
                self._current_object_info["size"]["x"],
                self._current_object_info["size"]["y"],
                self._current_object_info["size"]["z"],
            ],
            dtype=np.float64,
        )

        self._distractors = []
        self._distractor_infos = []
        for idx, distractor_id in enumerate(distractor_ids):
            self._distractors.append(self._make_named_ycb_object(distractor_id, f"distractor_{idx}"))
            self._distractor_infos.append(self._info_for_model(distractor_id))

        all_objects = [self.cube] + self._distractors

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            for obj in all_objects:
                self.placement_initializer.add_objects(obj)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="YCBClutterSampler",
                mujoco_objects=all_objects,
                x_range=[-0.18, 0.18],
                y_range=[-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        if self._current_object_info:
            size = self._current_object_info["size"]
            self._current_size = np.array([size["x"], size["y"], size["z"]], dtype=np.float64)


class FrankaRobosuiteYCBTargetClutterLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka task: pick the specified YCB target from YCB clutter."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
        robot_name: str = "Panda",
        hand_name: str = "panda",
        robosuite_gripper_type: str = "default",
        ik_robot_name: str = "panda_description",
        ik_target_link_name: str = "panda_hand",
        eef_body_name: str = "gripper0_right_eef",
        tcp_offset: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 0.0, -0.107),
        gripper_open_command: float = -1.0,
        gripper_closed_command: float = 1.0,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
            robot_name=robot_name,
            hand_name=hand_name,
            robosuite_gripper_type=robosuite_gripper_type,
            ik_robot_name=ik_robot_name,
            ik_target_link_name=ik_target_link_name,
            eef_body_name=eef_body_name,
            tcp_offset=tcp_offset,
            gripper_open_command=gripper_open_command,
            gripper_closed_command=gripper_closed_command,
        )

        clutter_candidates = [
            "004_sugar_box",
            "005_tomato_soup_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
        ]
        lift_kwargs = dict(
            robots=[self.robot_name],
            gripper_types=self.robosuite_gripper_type,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
            object_ids=clutter_candidates,
            num_distractors=4,
        )

        if privileged and not enable_render:
            lift_kwargs.update(
                use_camera_obs=False,
                has_offscreen_renderer=False,
                camera_names=[],
            )

        self.robosuite_env = LiftYCBTargetClutter(**lift_kwargs)
        self._initial_cube_height: float | None = None
        self._init_robot_links()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi

        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        self._initial_cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )

        robosuite_obs = self.robosuite_env._get_observations()
        self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
        self._current_joints[6] -= np.pi

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

        target_name = self.robosuite_env._current_object_info.get("display_name", "target object")
        clutter_names = ", ".join(info["display_name"] for info in self.robosuite_env._distractor_infos)
        info = {
            "task_prompt": (
                f"Pick up the {target_name} and lift it. Ignore the other objects in the clutter. "
                f"Other objects on the table may include: {clutter_names}."
            )
        }
        return obs, info

    def _object_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        obj_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]])
        )
        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        obj_robot_base = base_transform @ obj_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate([obj_robot_base.translation(), obj_robot_base.rotation().wxyz])
            ],
        }

    def compute_reward(self) -> float:
        if self.task_completed():
            return 1.0
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        return float(np.clip((cube_height - baseline) / 0.04, 0.0, 1.0))

    def task_completed(self) -> bool:
        cube_height = float(
            self.robosuite_env.sim.data.body_xpos[self.robosuite_env.cube_body_id][2]
        )
        baseline = (
            self._initial_cube_height
            if self._initial_cube_height is not None
            else float(self.robosuite_env.model.mujoco_arena.table_offset[2])
        )
        lifted = cube_height > (baseline + 0.04)
        grasped = self.robosuite_env._check_grasp(
            gripper=self.robosuite_env.robots[0].gripper,
            object_geoms=self.robosuite_env.cube,
        )
        return bool(lifted and grasped)

    def get_observation(self) -> dict[str, Any]:
        robosuite_obs = self.robosuite_env._get_observations()
        pose_dict = self._object_pose_dict(robosuite_obs)
        robosuite_obs["cube_poses"] = {
            "primary": np.asarray(pose_dict["primary"], dtype=np.float32),
        }
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs


__all__ = [
    "FrankaRobosuiteYCBLiftLowLevel",
    "FrankaRobosuiteYCBTargetClutterLowLevel",
]
