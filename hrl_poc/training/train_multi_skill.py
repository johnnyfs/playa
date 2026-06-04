from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY
from hrl_poc.training.train_skill import collect_rollout, evaluate, update_ppo


def main() -> None:
    args = parse_args()
    skills = [skill.strip() for skill in args.skills.split(",") if skill.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir or f"runs/multi_{int(time.time())}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skills[0])
    policy = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), condition_on_skill=not args.disable_skill_conditioning)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    episodes_done = 0
    update_idx = 0
    while episodes_done < args.episodes:
        rollout = collect_mixed_rollout(policy, skills, args.batch_episodes, args.seed + episodes_done)
        update_ppo(policy, optimizer, rollout, args)
        episodes_done += rollout["episode_count"]
        update_idx += 1
        print(
            json.dumps(
                {
                    "phase": "train",
                    "episodes": episodes_done,
                    "success_rate": float(np.mean(rollout["successes"])),
                    "episode_length": float(np.mean(rollout["lengths"])),
                    "reward": float(np.mean(rollout["episode_rewards"])),
                }
            )
        )

        if episodes_done % args.eval_every < args.batch_episodes or episodes_done >= args.episodes:
            for skill in skills:
                stats = evaluate(policy, skill, args.eval_episodes, args.seed + 30_000 + episodes_done, log_dir / skill, start_index=episodes_done)
                print(json.dumps({"phase": "eval", "episodes": episodes_done, "skill": skill, **stats}))

        if update_idx % args.checkpoint_every == 0 or episodes_done >= args.episodes:
            checkpoint_path = checkpoint_dir / f"skill_policy_{episodes_done:06d}.pt"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "skill": skills[0],
                    "trained_skills": skills,
                    "episodes": episodes_done,
                    "args": vars(args),
                    "policy_config": {"condition_on_skill": not args.disable_skill_conditioning},
                },
                checkpoint_path,
            )

    print(json.dumps({"phase": "done", "run_dir": str(run_dir)}))


def collect_mixed_rollout(policy: SkillPolicy, skills: list[str], batch_episodes: int, seed: int) -> dict[str, object]:
    merged = {
        "observations": [],
        "actions": [],
        "old_log_probs": [],
        "values": [],
        "returns": [],
        "episode_count": 0,
        "successes": [],
        "lengths": [],
        "episode_rewards": [],
    }
    for episode_idx in range(batch_episodes):
        skill = skills[episode_idx % len(skills)]
        rollout = collect_rollout(policy, skill, batch_episodes=1, seed=seed + episode_idx)
        merged["observations"].extend(rollout["observations"])
        merged["actions"].append(rollout["actions"])
        merged["old_log_probs"].append(rollout["old_log_probs"])
        merged["values"].append(rollout["values"])
        merged["returns"].append(rollout["returns"])
        merged["episode_count"] += 1
        merged["successes"].extend(rollout["successes"])
        merged["lengths"].extend(rollout["lengths"])
        merged["episode_rewards"].extend(rollout["episode_rewards"])
    merged["actions"] = torch.cat(merged["actions"])
    merged["old_log_probs"] = torch.cat(merged["old_log_probs"])
    merged["values"] = torch.cat(merged["values"])
    merged["returns"] = torch.cat(merged["returns"])
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one shared low-level policy over a fixed skill mix.")
    parser.add_argument("--skills", type=str, default="go,collect,attack")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--go-aux-coef", "--navigate-aux-coef", dest="navigate_aux_coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--disable-skill-conditioning", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
