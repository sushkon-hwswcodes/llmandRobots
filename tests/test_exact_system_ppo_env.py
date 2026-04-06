import numpy as np

from capx.integrations.franka.exact_system_ppo_env import ExactSystemHandPPOEnv


def test_exact_system_ppo_env_reset_and_step_smoke() -> None:
    env = ExactSystemHandPPOEnv(fixed_shape="box", max_episode_steps=2)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert info == {}

        action = np.concatenate([np.zeros(6, dtype=np.float32), np.array([0.0], dtype=np.float32)])
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["shape"] == "box"
    finally:
        env.close()
