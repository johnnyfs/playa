from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hrl_poc.envs.gridworld import Action
from hrl_poc.ladder.config import LadderLevel
from hrl_poc.ladder.episode import EpisodePolicy, EpisodeResult, SkillEpisode
from hrl_poc.ladder.world import generate_go_task
from hrl_poc.semantic.state import Vec2
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
