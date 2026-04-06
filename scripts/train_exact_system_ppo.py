#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from capx.integrations.franka.exact_system_ppo_env import ExactSystemHandPPOEnv

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[assignment]


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


@dataclass
class Config:
    shape: str = "box"
    total_timesteps: int = 4096
    num_envs: int = 1
    num_steps: int = 16
    learning_rate: float = 3e-4
    seed: int = 1
    cuda: bool = True
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    log_dir: str = "runs/exact_system_ppo"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train PPO directly on the exact Robosuite PandaDexRH + Inspire setup.")
    parser.add_argument("--shape", choices=["box", "cylinder", "ball"], default="box")
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.set_defaults(cuda=True)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.1)
    parser.add_argument("--log-dir", default="runs/exact_system_ppo")
    ns = parser.parse_args()
    return Config(**vars(ns))


def make_env(shape: str, seed: int):
    def thunk():
        env = ExactSystemHandPPOEnv(fixed_shape=shape)
        env.reset(seed=seed)
        return env

    return thunk


def main() -> None:
    cfg = parse_args()
    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_iterations = max(1, cfg.total_timesteps // batch_size)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(cfg.shape, cfg.seed + i) for i in range(cfg.num_envs)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    run_name = f"{cfg.shape}__ppo__{cfg.seed}__{int(time.time())}"
    log_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    if SummaryWriter is None:
        class _NoOpWriter:
            def add_scalar(self, *args, **kwargs):  # noqa: ANN002, ANN003
                return None

            def close(self) -> None:
                return None

        writer = _NoOpWriter()
    else:
        writer = SummaryWriter(log_dir)

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs, act_dim), device=device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            clipped_action = torch.clamp(action, action_low, action_high)
            actions[step] = clipped_action
            logprobs[step] = logprob

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(clipped_action.cpu().numpy())
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            reward_t = torch.as_tensor(reward_np, dtype=torch.float32, device=device).view(-1)
            done_t = torch.as_tensor(np.logical_or(term_np, trunc_np), dtype=torch.float32, device=device).view(-1)
            rewards[step] = reward_t
            next_done = done_t

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    next_values = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + cfg.gamma * next_values * next_not_done - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1, act_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)
        writer.add_scalar("losses/value_loss", float(v_loss.item()), global_step)
        writer.add_scalar("losses/policy_loss", float(pg_loss.item()), global_step)
        writer.add_scalar("losses/entropy", float(entropy_loss.item()), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl.item()), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs) if clipfracs else 0.0), global_step)
        writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
        writer.add_scalar("charts/mean_reward", float(rewards.mean().item()), global_step)
        writer.add_scalar("charts/success_rate", float((rewards >= 1.0).float().mean().item()), global_step)
        print(
            f"iter={iteration}/{num_iterations} "
            f"shape={cfg.shape} mean_reward={rewards.mean().item():.3f} "
            f"approx_kl={approx_kl.item():.4f}"
        )

    ckpt_path = os.path.join(log_dir, "final_ckpt.pt")
    torch.save(agent.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    writer.close()
    envs.close()


if __name__ == "__main__":
    main()
