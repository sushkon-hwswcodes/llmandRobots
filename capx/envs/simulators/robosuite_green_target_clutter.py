"""Robosuite Franka cluttered lift task: one green target + multiple red distractors."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv


class LiftGreenTargetClutter(Lift):
    """Lift task with clutter: one random-shape green target and random red distractors."""

    SHAPES = ["box", "cylinder", "ball"]

    def __init__(self, *args, num_red_distractors: int = 6, **kwargs):
        self.num_red_distractors = num_red_distractors
        self._current_shape = "box"
        self._current_size = np.zeros(3)
        super().__init__(*args, **kwargs)

    def _make_random_object(self, name: str, rgba: list[float], material: CustomMaterial):
        shape_idx = int(self.rng.integers(0, len(self.SHAPES)))
        shape = self.SHAPES[shape_idx]
        if shape == "box":
            obj = BoxObject(
                name=name,
                size_min=[0.018, 0.018, 0.018],
                size_max=[0.045, 0.045, 0.045],
                rgba=rgba,
                material=material,
                rng=self.rng,
            )
        elif shape == "cylinder":
            obj = CylinderObject(
                name=name,
                size_min=[0.015, 0.020],  # [radius, half-height]
                size_max=[0.035, 0.055],
                rgba=rgba,
                material=material,
                rng=self.rng,
            )
        else:
            obj = BallObject(
                name=name,
                size_min=[0.015],  # [radius]
                size_max=[0.035],
                rgba=rgba,
                material=material,
            )
        return obj, shape

    def _load_model(self):
        # Run ManipulationEnv setup but replace Lift single-object construction with clutter setup.
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        # Primary target object is green.
        self.cube, self._current_shape = self._make_random_object(
            name="target",
            rgba=[0.0, 1.0, 0.0, 1.0],
            material=greenwood,
        )

        self._distractors = []
        for i in range(self.num_red_distractors):
            obj, _ = self._make_random_object(
                name=f"distractor_{i}",
                rgba=[1.0, 0.0, 0.0, 1.0],
                material=redwood,
            )
            self._distractors.append(obj)

        all_objects = [self.cube] + self._distractors

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            for obj in all_objects:
                self.placement_initializer.add_objects(obj)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ClutterSampler",
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
        try:
            geom_name = self.cube.contact_geoms[0]
            geom_id = self.sim.model.geom_name2id(geom_name)
            self._current_size = self.sim.model.geom_size[geom_id].copy()
        except Exception:
            if hasattr(self.cube, "size") and self.cube.size:
                arr = np.zeros(3)
                for i, v in enumerate(self.cube.size[:3]):
                    arr[i] = v
                self._current_size = arr


class FrankaRobosuiteGreenTargetClutterLowLevel(RobosuiteBaseEnv):
    """Low-level env wrapper for cluttered green-target lifting."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )

        lift_kwargs = dict(
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(
                controller=self.controller_cfg
            ),
            horizon=max_steps,
            reward_shaping=True,
            num_red_distractors=6,
        )

        if privileged and not enable_render:
            lift_kwargs.update(
                use_camera_obs=False,
                has_offscreen_renderer=False,
                camera_names=[],
            )

        self.robosuite_env = LiftGreenTargetClutter(**lift_kwargs)
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

        info = {"task_prompt": "Pick up the green target object and lift it. Ignore red objects."}
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
                for x in np.concatenate(
                    [obj_robot_base.translation(), obj_robot_base.rotation().wxyz]
                )
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


__all__ = ["FrankaRobosuiteGreenTargetClutterLowLevel"]

