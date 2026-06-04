from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY
from hrl_poc.training.logging import write_execution_record


@dataclass(frozen=True)
class EvalResult:
    checkpoint: str
    skill: str
    episodes: int
    successes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    avg_length: float
    avg_reward: float
    paired_delta_vs_baseline: float | None = None
    paired_successes_vs_baseline: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "skill": self.skill,
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "success_ci95_low": self.success_ci95_low,
            "success_ci95_high": self.success_ci95_high,
            "avg_length": self.avg_length,
            "avg_reward": self.avg_reward,
            "paired_delta_vs_baseline": self.paired_delta_vs_baseline,
            "paired_successes_vs_baseline": self.paired_successes_vs_baseline,
        }


def load_policy(checkpoint_path: str | Path, device: str = "cpu") -> tuple[SkillPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    skill = str(checkpoint["skill"])
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)
    policy_config = dict(checkpoint.get("policy_config", {}))
    if "condition_on_skill" not in policy_config:
        policy_config["condition_on_skill"] = not dict(checkpoint.get("args", {})).get("disable_skill_conditioning", False)
    policy = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), **policy_config)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    episodes: int,
    seed: int,
    paired_baseline: SkillPolicy | None = None,
    log_dir: str | Path | None = None,
    start_index: int = 0,
) -> EvalResult:
    policy, checkpoint = load_policy(checkpoint_path)
    skill = str(checkpoint["skill"])
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)
    successes = []
    lengths = []
    rewards = []
    paired_delta = 0
    paired_successes = 0

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        obs, _info = env.reset(seed=episode_seed, options={"skill_name": skill})
        done = False
        while not done:
            action, _log_prob, _value = policy.act(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(bool(info["success"]))
        lengths.append(int(info["steps"]))
        rewards.append(float(info["reward_total"]))
        if log_dir is not None:
            record = env.make_execution_record()
            write_execution_record(record, env.trajectory, log_dir, episode_idx=start_index + episode_idx)

        if paired_baseline is not None:
            baseline_env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)
            baseline_obs, _baseline_info = baseline_env.reset(seed=episode_seed, options={"skill_name": skill})
            baseline_done = False
            while not baseline_done:
                baseline_action, _baseline_log_prob, _baseline_value = paired_baseline.act(baseline_obs, deterministic=True)
                baseline_obs, _baseline_reward, baseline_terminated, baseline_truncated, baseline_info = baseline_env.step(baseline_action)
                baseline_done = baseline_terminated or baseline_truncated
            paired_delta += int(info["success"]) - int(baseline_info["success"])
            paired_successes += int(info["success"]) - int(baseline_info["success"])

    success_rate = float(np.mean(successes)) if successes else 0.0
    se = math.sqrt(success_rate * (1.0 - success_rate) / episodes) if 0.0 < success_rate < 1.0 else 0.0
    return EvalResult(
        checkpoint=str(checkpoint_path),
        skill=skill,
        episodes=episodes,
        successes=int(np.sum(successes)),
        success_rate=success_rate,
        success_ci95_low=max(0.0, success_rate - 1.96 * se),
        success_ci95_high=min(1.0, success_rate + 1.96 * se),
        avg_length=float(np.mean(lengths)) if lengths else 0.0,
        avg_reward=float(np.mean(rewards)) if rewards else 0.0,
        paired_delta_vs_baseline=float(paired_delta / episodes) if paired_baseline is not None else None,
        paired_successes_vs_baseline=int(paired_successes) if paired_baseline is not None else None,
    )


def compare_checkpoints(
    baseline_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    baseline_policy, baseline_meta = load_policy(baseline_checkpoint)
    candidate_result = evaluate_checkpoint(
        candidate_checkpoint,
        episodes=episodes,
        seed=seed,
        paired_baseline=baseline_policy,
    )
    baseline_result = evaluate_checkpoint(baseline_checkpoint, episodes=episodes, seed=seed)
    candidate_result_dict = candidate_result.to_dict()
    candidate_result_dict["baseline_checkpoint"] = str(baseline_checkpoint)
    candidate_result_dict["baseline_success_rate"] = baseline_result.success_rate
    candidate_result_dict["baseline_success_ci95_low"] = baseline_result.success_ci95_low
    candidate_result_dict["baseline_success_ci95_high"] = baseline_result.success_ci95_high
    candidate_result_dict["delta_success_rate"] = candidate_result.success_rate - baseline_result.success_rate
    candidate_result_dict["delta_successes"] = candidate_result.successes - baseline_result.successes
    candidate_result_dict["baseline_skill"] = baseline_meta["skill"]
    return candidate_result_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate skill checkpoints on matched seeds.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--baseline-checkpoint", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline_checkpoint:
        result = compare_checkpoints(args.baseline_checkpoint, args.checkpoint, args.episodes, args.seed)
    else:
        result = evaluate_checkpoint(
            args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            log_dir=args.log_dir or None,
            start_index=args.start_index,
        ).to_dict()
    if args.json:
        print(json.dumps(result))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
