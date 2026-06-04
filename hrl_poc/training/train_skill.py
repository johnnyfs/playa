from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY, get_skill_id
from hrl_poc.training.logging import write_execution_record


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir or f"runs/{args.skill}_{int(time.time())}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = SkillConditionedEnv(GridWorldEnv(), skill_name=args.skill)
    policy = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), condition_on_skill=not args.disable_skill_conditioning)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    eval_stats = evaluate(policy, args.skill, args.eval_episodes, args.seed + 10_000, log_dir, start_index=0)
    print(json.dumps({"phase": "eval", "episodes": 0, **eval_stats}))

    episodes_done = 0
    update_idx = 0
    while episodes_done < args.episodes:
        rollout = collect_rollout(policy, args.skill, args.batch_episodes, args.seed + episodes_done)
        update_ppo(policy, optimizer, rollout, args)
        episodes_done += rollout["episode_count"]
        update_idx += 1

        train_stats = {
            "phase": "train",
            "episodes": episodes_done,
            "success_rate": float(np.mean(rollout["successes"])),
            "episode_length": float(np.mean(rollout["lengths"])),
            "reward": float(np.mean(rollout["episode_rewards"])),
        }
        print(json.dumps(train_stats))

        if episodes_done % args.eval_every < args.batch_episodes or episodes_done >= args.episodes:
            eval_stats = evaluate(
                policy,
                args.skill,
                args.eval_episodes,
                args.seed + 20_000 + episodes_done,
                log_dir,
                start_index=episodes_done,
            )
            print(json.dumps({"phase": "eval", "episodes": episodes_done, **eval_stats}))

        if update_idx % args.checkpoint_every == 0 or episodes_done >= args.episodes:
            checkpoint_path = checkpoint_dir / f"skill_policy_{episodes_done:06d}.pt"
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "skill": args.skill,
                    "episodes": episodes_done,
                    "args": vars(args),
                    "policy_config": {"condition_on_skill": not args.disable_skill_conditioning},
                },
                checkpoint_path,
            )

    print(json.dumps({"phase": "done", "run_dir": str(run_dir)}))


def collect_rollout(policy: SkillPolicy, skill_name: str, batch_episodes: int, seed: int) -> dict[str, object]:
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill_name)
    observations = []
    actions = []
    old_log_probs = []
    values = []
    rewards_by_episode: list[list[float]] = []
    successes = []
    lengths = []
    episode_rewards = []

    for episode_idx in range(batch_episodes):
        obs, _info = env.reset(seed=seed + episode_idx, options={"skill_name": skill_name})
        done = False
        episode_rewards_current = []
        while not done:
            action, log_prob, value = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            old_log_probs.append(log_prob)
            values.append(value)
            episode_rewards_current.append(float(reward))
            obs = next_obs
            done = terminated or truncated
        rewards_by_episode.append(episode_rewards_current)
        successes.append(bool(info["success"]))
        lengths.append(int(info["steps"]))
        episode_rewards.append(float(info["reward_total"]))

    returns = []
    for rewards in rewards_by_episode:
        running = 0.0
        episode_returns = []
        for reward in reversed(rewards):
            running = reward + 0.99 * running
            episode_returns.append(running)
        returns.extend(reversed(episode_returns))

    return {
        "observations": observations,
        "actions": torch.as_tensor(actions, dtype=torch.long),
        "old_log_probs": torch.as_tensor(old_log_probs, dtype=torch.float32),
        "values": torch.as_tensor(values, dtype=torch.float32),
        "returns": torch.as_tensor(returns, dtype=torch.float32),
        "episode_count": batch_episodes,
        "successes": successes,
        "lengths": lengths,
        "episode_rewards": episode_rewards,
    }


def update_ppo(policy: SkillPolicy, optimizer: torch.optim.Optimizer, rollout: dict[str, object], args: argparse.Namespace) -> None:
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    returns = rollout["returns"]
    old_values = rollout["values"]
    assert isinstance(observations, list)
    assert isinstance(actions, torch.Tensor)
    assert isinstance(old_log_probs, torch.Tensor)
    assert isinstance(returns, torch.Tensor)
    assert isinstance(old_values, torch.Tensor)

    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = np.arange(len(actions))

    for _ in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.minibatch_size):
            batch_idx = indices[start : start + args.minibatch_size]
            batch_obs = [observations[int(i)] for i in batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            logits, values = policy(batch_obs)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy()
            ratio = torch.exp(log_probs - batch_old_log_probs)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * batch_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss
            if args.navigate_aux_coef > 0:
                target_actions, mask = go_oracle_actions(batch_obs)
                if mask.any():
                    aux_loss = nn.functional.cross_entropy(logits[mask], target_actions[mask])
                    loss = loss + args.navigate_aux_coef * aux_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()


def go_oracle_actions(obs_batch: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    mask = []
    go_ids = {get_skill_id("go"), get_skill_id("go_evading")}
    for obs in obs_batch:
        skill_id = int(np.asarray(obs["skill_id"]).reshape(-1)[0])
        if skill_id not in go_ids:
            targets.append(5)
            mask.append(False)
            continue
        state = obs["state"]
        player = np.asarray(state["player"], dtype=np.int64)
        params = np.asarray(obs["skill_params"], dtype=np.float32)
        goal = np.rint(params[:2] * (state["terrain"].shape[0] - 1)).astype(np.int64)
        action = shortest_path_action(np.asarray(state["terrain"]), tuple(player), tuple(goal))
        targets.append(action)
        mask.append(True)
    return torch.as_tensor(targets, dtype=torch.long), torch.as_tensor(mask, dtype=torch.bool)


def shortest_path_action(terrain: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> int:
    if player == goal:
        return 5
    size = terrain.shape[0]
    moves = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]
    queue = [player]
    came_from: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    seen = {player}
    for pos in queue:
        if pos == goal:
            break
        x, y = pos
        for dx, dy, action in moves:
            nxt = (x + dx, y + dy)
            nx, ny = nxt
            if 0 <= nx < size and 0 <= ny < size and terrain[ny, nx] == 0 and nxt not in seen:
                seen.add(nxt)
                came_from[nxt] = (pos, action)
                queue.append(nxt)
    if goal not in came_from:
        dx = int(goal[0] - player[0])
        dy = int(goal[1] - player[1])
        if abs(dx) >= abs(dy) and dx != 0:
            return 3 if dx > 0 else 2
        if dy != 0:
            return 1 if dy > 0 else 0
        return 5

    pos = goal
    action = 5
    while came_from[pos][0] != player:
        pos = came_from[pos][0]
    action = came_from[pos][1]
    return action


@torch.no_grad()
def evaluate(
    policy: SkillPolicy,
    skill_name: str,
    episodes: int,
    seed: int,
    log_dir: Path,
    start_index: int,
) -> dict[str, float]:
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill_name)
    successes = []
    lengths = []
    rewards = []
    for idx in range(episodes):
        obs, _info = env.reset(seed=seed + idx, options={"skill_name": skill_name})
        done = False
        while not done:
            action, _log_prob, _value = policy.act(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes.append(bool(info["success"]))
        lengths.append(int(info["steps"]))
        rewards.append(float(info["reward_total"]))
        record = env.make_execution_record()
        write_execution_record(record, env.trajectory, log_dir, episode_idx=start_index + idx)
    return {
        "success_rate": float(np.mean(successes)),
        "episode_length": float(np.mean(lengths)),
        "reward": float(np.mean(rewards)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared skill-conditioned PPO policy.")
    parser.add_argument("--skill", choices=sorted(SKILL_REGISTRY), default="go")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=80)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--go-aux-coef", "--navigate-aux-coef", dest="navigate_aux_coef", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--disable-skill-conditioning", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
