import os
import time

import pytest
import tyro

os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.base import list_envs
from capx.envs.tasks import CodeExecutionEnvBase, get_config, get_exec_env, list_configs, list_exec_envs
from capx.integrations.base_api import list_apis


def test_environments(env_name: str = "franka_robosuite_pick_place_code_env") -> bool:
    print("Available environments: ", list_envs())
    print("Available execution environments: ", list_exec_envs())
    print("Available APIs: ", list_apis())
    print("Available configurations: ", list_configs())
    cfg = get_config(env_name)
    env: CodeExecutionEnvBase = get_exec_env(env_name)(cfg)
    env.enable_video_capture(True)
    start = time.time()
    obs, info = env.reset()
    end = time.time()
    print("Time taken to reset: ", end - start)
    print("Observation keys: ", list(obs.keys()))
    print("Prompt: ", obs["full_prompt"][1]["content"])
    start = time.time()
    obs_next, reward, terminated, truncated, info_step = env.step(env.oracle_code)
    end = time.time()
    print("Time taken: ", end - start)
    video_frames = env.get_video_frames()
    if video_frames:
        import imageio

        imageio.mimsave("test_video.mp4", video_frames, fps=30)
        print("Video saved to test_video.mp4")
    if reward != 1.0:
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info_step)
        return False
    print("Success")
    return True


ACTIVE_ENV_TESTS = [
    "franka_robosuite_pick_place_code_env",
    "franka_lift_code_env",
    "franka_nut_assembly_code_env",
]

OPTIONAL_ENV_TESTS = [
    "franka_nut_assembly_code_env_visual",
    "franka_restack_code_env",
    "franka_robosuite_spill_wipe_code_env",
    "two_arm_handover_code_env",
    "franka_libero_code_env",
    "r1pro_radio_code_env",
    "r1pro_trash_code_env",
]


@pytest.mark.parametrize("env_name", ACTIVE_ENV_TESTS)
def test_active_environments(env_name: str) -> None:
    assert test_environments(env_name)


@pytest.mark.parametrize("env_name", OPTIONAL_ENV_TESTS)
def test_optional_environments(env_name: str) -> None:
    cfg = get_config(env_name)
    if cfg.low_level not in list_envs():
        pytest.skip(f"low-level env '{cfg.low_level}' is not registered in this install")
    assert test_environments(env_name)


if __name__ == "__main__":
    tyro.cli(test_environments)
