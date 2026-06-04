from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hrl_poc.envs.gridworld import Action
from hrl_poc.ladder.config import LadderLevel
from hrl_poc.ladder.episode import EpisodePolicy, EpisodeResult, SkillEpisode
from hrl_poc.ladder.world import generate_go_task
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.semantic.state import Vec2
from hrl_poc.skills.registry import SKILL_REGISTRY, get_skill_id
from hrl_poc.skills.schema import SkillCall


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    trainer: str
    level_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "trainer": self.trainer,
            "level_id": self.level_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class LevelEvalResult:
    level_id: str
    checkpoint: Checkpoint
    episodes: int
    successes: int
    pass_rate: float
    required_pass_rate: float
    passed: bool
    episode_results: list[EpisodeResult]

    @property
    def failures(self) -> list[EpisodeResult]:
        return [result for result in self.episode_results if not result.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "checkpoint": self.checkpoint.to_dict(),
            "episodes": self.episodes,
            "successes": self.successes,
            "pass_rate": self.pass_rate,
            "required_pass_rate": self.required_pass_rate,
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
        }


class SkillTrainer(Protocol):
    name: str

    def train_level(self, level: LadderLevel, checkpoint_dir: Path) -> Checkpoint:
        ...

    def evaluate_level(
        self,
        checkpoint: Checkpoint,
        level: LadderLevel,
        seed_base: int,
        trajectory_dir: Path | None = None,
    ) -> LevelEvalResult:
        ...


class BaseGoTrainer:
    name = "base"

    def train_level(self, level: LadderLevel, checkpoint_dir: Path) -> Checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{level.id}_{self.name}.json"
        payload = {
            "trainer": self.name,
            "level_id": level.id,
            "skill": "go",
            "train": level.train.to_dict(),
            "note": "Placeholder checkpoint. Replace this trainer with PPO behind the SkillTrainer interface.",
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return Checkpoint(path=path, trainer=self.name, level_id=level.id, metadata=payload)

    def evaluate_level(
        self,
        checkpoint: Checkpoint,
        level: LadderLevel,
        seed_base: int,
        trajectory_dir: Path | None = None,
    ) -> LevelEvalResult:
        episode_results = []
        target_radius = int(level.task.get("target_radius_px", 4))
        for idx in range(level.eval.heldout_seeds):
            seed = int(seed_base + idx)
            world, target = generate_go_task(level.world, level.task, seed)
            skill_call = SkillCall("go", {"target": target})
            path = None
            if trajectory_dir is not None:
                path = trajectory_dir / f"{level.id}_seed_{seed}.json"
            episode = SkillEpisode(
                level_id=level.id,
                world=world,
                skill_call=skill_call,
                max_steps=int(level.task.get("max_steps", 64)),
                seed=seed,
                target_radius_px=target_radius,
            )
            episode_results.append(episode.run(self.policy(seed), trajectory_path=path))

        successes = sum(1 for result in episode_results if result.success)
        episodes = len(episode_results)
        pass_rate = successes / episodes if episodes else 0.0
        return LevelEvalResult(
            level_id=level.id,
            checkpoint=checkpoint,
            episodes=episodes,
            successes=successes,
            pass_rate=pass_rate,
            required_pass_rate=level.eval.pass_rate,
            passed=pass_rate >= level.eval.pass_rate,
            episode_results=episode_results,
        )

    def policy(self, seed: int) -> EpisodePolicy:
        raise NotImplementedError


class RandomPolicyTrainer(BaseGoTrainer):
    name = "random"

    def policy(self, seed: int) -> EpisodePolicy:
        return RandomPolicy(seed)


class HeuristicGoTrainer(BaseGoTrainer):
    name = "heuristic_go"

    def policy(self, seed: int) -> EpisodePolicy:
        return HeuristicGoPolicy()


class PpoGoTrainer(BaseGoTrainer):
    name = "ppo_go"

    def __init__(
        self,
        batch_episodes: int = 8,
        lr: float = 3e-4,
        ppo_epochs: int = 4,
        minibatch_size: int = 128,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
    ):
        self.batch_episodes = batch_episodes
        self.lr = lr
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def train_level(self, level: LadderLevel, checkpoint_dir: Path) -> Checkpoint:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        env_size = level.world.env_size
        action_space = generate_go_task(level.world, level.task, seed=0)[0].env().action_space
        policy = SkillPolicy(action_space, num_skills=len(SKILL_REGISTRY), grid_size=env_size)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        steps_budget = int(level.train.steps)
        train_seed_count = max(1, int(level.train.seeds))
        steps_done = 0
        episode_idx = 0

        while steps_done < steps_budget:
            rollout = self._collect_rollout(policy, level, seed_base=episode_idx % train_seed_count)
            if rollout["actions"].numel() == 0:
                break
            self._update_ppo(policy, optimizer, rollout)
            steps_done += int(rollout["step_count"])
            episode_idx += int(rollout["episode_count"])

        path = checkpoint_dir / f"{level.id}_{self.name}.pt"
        payload = {
            "policy_state_dict": policy.state_dict(),
            "skill": "go",
            "trainer": self.name,
            "level_id": level.id,
            "steps_done": steps_done,
            "episodes_done": episode_idx,
            "train": level.train.to_dict(),
            "policy_config": {"grid_size": env_size, "condition_on_skill": True},
            "trainer_config": {
                "batch_episodes": self.batch_episodes,
                "lr": self.lr,
                "ppo_epochs": self.ppo_epochs,
                "minibatch_size": self.minibatch_size,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "max_grad_norm": self.max_grad_norm,
            },
        }
        torch.save(payload, path)
        return Checkpoint(
            path=path,
            trainer=self.name,
            level_id=level.id,
            metadata={
                "skill": "go",
                "steps_done": steps_done,
                "episodes_done": episode_idx,
                "policy_config": payload["policy_config"],
            },
        )

    def evaluate_level(
        self,
        checkpoint: Checkpoint,
        level: LadderLevel,
        seed_base: int,
        trajectory_dir: Path | None = None,
    ) -> LevelEvalResult:
        policy = self._load_policy(checkpoint.path, level)
        episode_results = []
        target_radius = int(level.task.get("target_radius_px", 4))
        for idx in range(level.eval.heldout_seeds):
            seed = int(seed_base + idx)
            world, target = generate_go_task(level.world, level.task, seed)
            skill_call = SkillCall("go", {"target": target})
            path = None
            if trajectory_dir is not None:
                path = trajectory_dir / f"{level.id}_seed_{seed}.json"
            episode = SkillEpisode(
                level_id=level.id,
                world=world,
                skill_call=skill_call,
                max_steps=int(level.task.get("max_steps", 64)),
                seed=seed,
                target_radius_px=target_radius,
            )
            episode_results.append(episode.run(PpoEpisodePolicy(policy), trajectory_path=path))

        successes = sum(1 for result in episode_results if result.success)
        episodes = len(episode_results)
        pass_rate = successes / episodes if episodes else 0.0
        return LevelEvalResult(
            level_id=level.id,
            checkpoint=checkpoint,
            episodes=episodes,
            successes=successes,
            pass_rate=pass_rate,
            required_pass_rate=level.eval.pass_rate,
            passed=pass_rate >= level.eval.pass_rate,
            episode_results=episode_results,
        )

    def policy(self, seed: int) -> EpisodePolicy:
        raise RuntimeError("PpoGoTrainer loads policies from checkpoints for evaluation")

    def _collect_rollout(self, policy: SkillPolicy, level: LadderLevel, seed_base: int) -> dict[str, Any]:
        observations = []
        actions = []
        old_log_probs = []
        values = []
        rewards_by_episode: list[list[float]] = []
        successes = []
        lengths = []
        episode_rewards = []
        max_steps = int(level.task.get("max_steps", 64))
        target_radius = int(level.task.get("target_radius_px", 4))

        for idx in range(self.batch_episodes):
            seed = seed_base + idx
            world, target = generate_go_task(level.world, level.task, seed)
            env = world.env()
            raw_obs = env._observation()
            skill_call = SkillCall("go", {"target": target})
            prev_distance = _distance_px(tuple(int(v) for v in raw_obs["player"]), target, level.world)
            episode_rewards_current = []
            success = False
            failure_reason = None

            for step_idx in range(1, max_steps + 1):
                obs = _augment_obs(raw_obs, skill_call)
                action, log_prob, value = policy.act(obs)
                next_raw_obs, _base_reward, terminated, _truncated, info = env.step(action)
                player = (int(info["player"]["x"]), int(info["player"]["y"]))
                distance = _distance_px(player, target, level.world)
                reward = _bounded((prev_distance - distance) / max(1, level.world.tile_size_px) * 0.1 - 0.01)
                prev_distance = distance
                if distance <= target_radius:
                    success = True
                    reward = 1.0
                elif terminated or int(info.get("health", 0)) <= 0:
                    failure_reason = "player_dead"
                    reward = -1.0
                elif step_idx >= max_steps:
                    failure_reason = "timeout"
                    reward = -1.0

                observations.append(obs)
                actions.append(action)
                old_log_probs.append(log_prob)
                values.append(value)
                episode_rewards_current.append(float(reward))
                raw_obs = next_raw_obs
                if success or failure_reason is not None:
                    lengths.append(step_idx)
                    break

            rewards_by_episode.append(episode_rewards_current)
            successes.append(success)
            episode_rewards.append(float(sum(episode_rewards_current)))

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
            "episode_count": self.batch_episodes,
            "step_count": len(actions),
            "successes": successes,
            "lengths": lengths,
            "episode_rewards": episode_rewards,
        }

    def _update_ppo(self, policy: SkillPolicy, optimizer: torch.optim.Optimizer, rollout: dict[str, Any]) -> None:
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
        if actions.numel() == 0:
            return

        advantages = returns - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        indices = np.arange(len(actions))
        minibatch_size = max(1, min(self.minibatch_size, len(indices)))

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), minibatch_size):
                batch_idx = indices[start : start + minibatch_size]
                batch_obs = [observations[int(i)] for i in batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                logits, predicted_values = policy(batch_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(predicted_values, batch_returns)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                optimizer.step()

    def _load_policy(self, checkpoint_path: Path, level: LadderLevel) -> SkillPolicy:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        env_size = int(checkpoint.get("policy_config", {}).get("grid_size", level.world.env_size))
        action_space = generate_go_task(level.world, level.task, seed=0)[0].env().action_space
        policy = SkillPolicy(
            action_space,
            num_skills=len(SKILL_REGISTRY),
            grid_size=env_size,
            condition_on_skill=bool(checkpoint.get("policy_config", {}).get("condition_on_skill", True)),
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        return policy


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any], info: dict[str, Any], skill_call: SkillCall) -> int:
        return int(self.rng.integers(0, len(Action)))


class HeuristicGoPolicy:
    def act(self, observation: dict[str, Any], info: dict[str, Any], skill_call: SkillCall) -> int:
        target = _target(skill_call)
        player = tuple(int(v) for v in observation["player"])
        return shortest_path_action(np.asarray(observation["terrain"]), player, (target.x, target.y))


class PpoEpisodePolicy:
    def __init__(self, policy: SkillPolicy):
        self.policy = policy

    def act(self, observation: dict[str, Any], info: dict[str, Any], skill_call: SkillCall) -> int:
        obs = _augment_obs(observation, skill_call)
        action, _log_prob, _value = self.policy.act(obs, deterministic=True)
        return int(action)


def shortest_path_action(terrain: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> int:
    if player == goal:
        return int(Action.WAIT)
    height, width = terrain.shape
    moves = [
        (0, -1, int(Action.UP)),
        (0, 1, int(Action.DOWN)),
        (-1, 0, int(Action.LEFT)),
        (1, 0, int(Action.RIGHT)),
    ]
    queue: deque[tuple[int, int]] = deque([player])
    came_from: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    seen = {player}
    while queue:
        pos = queue.popleft()
        if pos == goal:
            break
        x, y = pos
        for dx, dy, action in moves:
            nxt = (x + dx, y + dy)
            nx, ny = nxt
            if 0 <= nx < width and 0 <= ny < height and terrain[ny, nx] == 0 and nxt not in seen:
                seen.add(nxt)
                came_from[nxt] = (pos, action)
                queue.append(nxt)

    if goal not in seen:
        return _greedy_action(terrain, player, goal)

    pos = goal
    while came_from[pos][0] != player:
        pos = came_from[pos][0]
    return came_from[pos][1]


def make_trainer(name: str) -> SkillTrainer:
    if name == "heuristic" or name == "heuristic_go":
        return HeuristicGoTrainer()
    if name == "ppo" or name == "ppo_go":
        return PpoGoTrainer()
    if name == "random":
        return RandomPolicyTrainer()
    raise ValueError(f"Unknown ladder trainer {name!r}")


def _target(skill_call: SkillCall) -> Vec2:
    target = skill_call.params["target"]
    if isinstance(target, Vec2):
        return target
    if isinstance(target, dict):
        return Vec2.from_dict(target)
    return Vec2(int(target[0]), int(target[1]))  # type: ignore[index]


def _greedy_action(terrain: np.ndarray, player: tuple[int, int], goal: tuple[int, int]) -> int:
    candidates = []
    for dx, dy, action in (
        (0, -1, int(Action.UP)),
        (0, 1, int(Action.DOWN)),
        (-1, 0, int(Action.LEFT)),
        (1, 0, int(Action.RIGHT)),
    ):
        nxt = (player[0] + dx, player[1] + dy)
        x, y = nxt
        if 0 <= y < terrain.shape[0] and 0 <= x < terrain.shape[1] and terrain[y, x] == 0:
            distance = abs(goal[0] - x) + abs(goal[1] - y)
            candidates.append((distance, action))
    if not candidates:
        return int(Action.WAIT)
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _augment_obs(raw_obs: dict[str, Any], skill_call: SkillCall) -> dict[str, Any]:
    terrain = np.asarray(raw_obs["terrain"])
    target = _target(skill_call)
    scale = max(1, terrain.shape[0] - 1)
    skill_params = np.zeros(5, dtype=np.float32)
    skill_params[0] = target.x / scale
    skill_params[1] = target.y / scale
    return {
        "state": raw_obs,
        "skill_id": np.array([get_skill_id("go")], dtype=np.int64),
        "skill_params": skill_params,
    }


def _distance_px(player: tuple[int, int], target: Vec2, world: Any) -> int:
    return (abs(player[0] - target.x) + abs(player[1] - target.y)) * int(world.tile_size_px)


def _bounded(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))
