from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY
from hrl_poc.training.evaluate_skill import load_policy


class Agent(Protocol):
    def act(self, obs: dict[str, object], deterministic: bool = True) -> tuple[int, float, float]:
        ...


@dataclass
class RandomAgent:
    action_space_n: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: dict[str, object], deterministic: bool = True) -> tuple[int, float, float]:
        return int(self.rng.integers(0, self.action_space_n)), 0.0, 0.0


def evaluate_agent(agent: Agent, skill: str, episodes: int, seed: int) -> dict[str, object]:
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)
    successes = []
    lengths = []
    rewards = []
    for episode_idx in range(episodes):
        obs, _info = env.reset(seed=seed + episode_idx, options={"skill_name": skill})
        done = False
        while not done:
            action, _log_prob, _value = agent.act(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(bool(info["success"]))
        lengths.append(int(info["steps"]))
        rewards.append(float(info["reward_total"]))
    rate = float(np.mean(successes)) if successes else 0.0
    se = math.sqrt(rate * (1.0 - rate) / episodes) if 0.0 < rate < 1.0 else 0.0
    return {
        "episodes": episodes,
        "successes": int(np.sum(successes)),
        "success_rate": rate,
        "ci95": [max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)],
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    args = parse_args()
    policy, checkpoint = load_policy(args.checkpoint)
    skill = args.skill or str(checkpoint.get("skill", "go"))
    policy_config = dict(checkpoint.get("policy_config", {}))
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)

    torch.manual_seed(args.seed)
    untrained = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), **policy_config)
    untrained.eval()
    random_agent = RandomAgent(action_space_n=env.action_space.n, seed=args.seed)

    rows = {
        "trained": evaluate_agent(policy, skill, args.episodes, args.seed),
        "random": evaluate_agent(random_agent, skill, args.episodes, args.seed),
        "untrained": evaluate_agent(untrained, skill, args.episodes, args.seed),
    }
    if args.nonconditioned_checkpoint:
        nonconditioned, _meta = load_policy(args.nonconditioned_checkpoint)
        rows["non_conditioned"] = evaluate_agent(nonconditioned, skill, args.episodes, args.seed)

    trained_rate = float(rows["trained"]["success_rate"])
    for name, row in rows.items():
        row["trained_delta_over_baseline"] = trained_rate - float(row["success_rate"]) if name != "trained" else 0.0
    print(json.dumps({"skill": skill, "checkpoint": str(Path(args.checkpoint)), "results": rows}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a skill checkpoint against random, untrained, and optional non-conditioned baselines.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skill", type=str, default="")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--nonconditioned-checkpoint", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
