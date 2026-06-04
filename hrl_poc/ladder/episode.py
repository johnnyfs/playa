from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hrl_poc.envs.gridworld import Action
from hrl_poc.ladder.world import GeneratedWorld, WorldSpec
from hrl_poc.semantic.state import Vec2
from hrl_poc.skills.schema import SkillCall


class EpisodePolicy(Protocol):
    def act(self, observation: dict[str, Any], info: dict[str, Any], skill_call: SkillCall) -> int:
        ...


@dataclass
class EpisodeResult:
    skill_call: SkillCall
    level_id: str
    seed: int
    success: bool
    failure: bool
    failure_reason: str | None
    steps: int
    reward_total: float
    initial_state_summary: dict[str, Any]
    final_state_summary: dict[str, Any]
    trajectory_path: Path | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_call": self.skill_call.to_dict(),
            "level_id": self.level_id,
            "seed": self.seed,
            "success": bool(self.success),
            "failure": bool(self.failure),
            "failure_reason": self.failure_reason,
            "steps": int(self.steps),
            "reward_total": float(self.reward_total),
            "initial_state_summary": self.initial_state_summary,
            "final_state_summary": self.final_state_summary,
            "trajectory_path": str(self.trajectory_path) if self.trajectory_path else None,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


class SkillEpisode:
    def __init__(
        self,
        level_id: str,
        world: GeneratedWorld,
        skill_call: SkillCall,
        max_steps: int,
        seed: int,
        target_radius_px: int = 4,
    ):
        self.level_id = level_id
        self.world = world
        self.skill_call = skill_call
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.target_radius_px = int(target_radius_px)

    def run(self, policy: EpisodePolicy, trajectory_path: str | Path | None = None) -> EpisodeResult:
        env = self.world.env()
        observation = env._observation()
        info = env.get_summary()
        info["world_state"] = env.world_state()
        target = _target(self.skill_call)
        initial_world_state = env.world_state().to_dict()
        initial_player = _player_tuple(info)
        prev_distance_px = _distance_px(initial_player, target, self.world.spec)
        reward_total = 0.0
        trajectory: list[dict[str, Any]] = []
        blocked_steps = 0
        positions = [initial_player]

        if self._is_success(initial_player, target):
            reward_total = 1.0
            result = self._result(
                env=env,
                success=True,
                failure=False,
                failure_reason=None,
                steps=0,
                reward_total=reward_total,
                initial_state_summary=initial_world_state,
                trajectory=trajectory,
                trajectory_path=trajectory_path,
                blocked_steps=blocked_steps,
                positions=positions,
                target=target,
            )
            return result

        if self.max_steps <= 0:
            return self._result(
                env=env,
                success=False,
                failure=True,
                failure_reason="timeout",
                steps=0,
                reward_total=-1.0,
                initial_state_summary=initial_world_state,
                trajectory=trajectory,
                trajectory_path=trajectory_path,
                blocked_steps=blocked_steps,
                positions=positions,
                target=target,
            )

        success = False
        failure = False
        failure_reason = None
        steps = 0
        for step_idx in range(1, self.max_steps + 1):
            action = int(policy.act(observation, info, self.skill_call))
            observation, _base_reward, terminated, _truncated, info = env.step(action)
            steps = step_idx
            player = _player_tuple(info)
            positions.append(player)
            distance_px = _distance_px(player, target, self.world.spec)
            reward = _bounded((prev_distance_px - distance_px) / max(1, self.world.spec.tile_size_px) * 0.1 - 0.01)
            prev_distance_px = distance_px

            if info.get("movement_blocked", False):
                blocked_steps += 1

            if self._is_success(player, target):
                success = True
                reward = 1.0
            elif terminated or int(info.get("health", 0)) <= 0:
                failure = True
                failure_reason = "player_dead"
                reward = -1.0
            elif step_idx >= self.max_steps:
                failure = True
                failure_reason = "timeout"
                reward = -1.0

            reward_total += reward
            trajectory.append(
                {
                    "step": step_idx,
                    "action": Action(action).name,
                    "player": info["player"],
                    "target": target.to_dict(),
                    "distance_px": distance_px,
                    "reward": float(reward),
                    "movement_blocked": bool(info.get("movement_blocked", False)),
                    "health": int(info.get("health", 0)),
                    "success": success,
                    "failure": failure,
                    "failure_reason": failure_reason,
                }
            )
            if success or failure:
                break

        return self._result(
            env=env,
            success=success,
            failure=failure,
            failure_reason=failure_reason,
            steps=steps,
            reward_total=reward_total,
            initial_state_summary=initial_world_state,
            trajectory=trajectory,
            trajectory_path=trajectory_path,
            blocked_steps=blocked_steps,
            positions=positions,
            target=target,
        )

    def _is_success(self, player: tuple[int, int], target: Vec2) -> bool:
        return _distance_px(player, target, self.world.spec) <= self.target_radius_px

    def _result(
        self,
        env: Any,
        success: bool,
        failure: bool,
        failure_reason: str | None,
        steps: int,
        reward_total: float,
        initial_state_summary: dict[str, Any],
        trajectory: list[dict[str, Any]],
        trajectory_path: str | Path | None,
        blocked_steps: int,
        positions: list[tuple[int, int]],
        target: Vec2,
    ) -> EpisodeResult:
        final_world_state = env.world_state().to_dict()
        final_player = positions[-1]
        final_distance = _distance_px(final_player, target, self.world.spec)
        progress_values = [
            _distance_px(pos, target, self.world.spec)
            for pos in positions
        ]
        average_progress = float(progress_values[0] - progress_values[-1]) if progress_values else 0.0
        stuck = len(positions[-8:]) >= 8 and len(set(positions[-8:])) <= 2
        path: Path | None = None
        if trajectory_path is not None:
            path = Path(trajectory_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
        diagnostics = {
            "failed_seed": self.seed if failure else None,
            "failure_reason": failure_reason,
            "initial_position": {"x": positions[0][0], "y": positions[0][1]},
            "target_position": target.to_dict(),
            "final_position": {"x": final_player[0], "y": final_player[1]},
            "steps": int(steps),
            "final_distance_to_target_px": int(final_distance),
            "screen_position": final_world_state.get("screen_position"),
            "stuck": bool(stuck),
            "hit_obstacle": blocked_steps > 0,
            "blocked_steps": int(blocked_steps),
            "average_progress_before_timeout_px": average_progress if failure_reason == "timeout" else None,
        }
        return EpisodeResult(
            skill_call=self.skill_call,
            level_id=self.level_id,
            seed=self.seed,
            success=success,
            failure=failure,
            failure_reason=failure_reason,
            steps=steps,
            reward_total=float(reward_total),
            initial_state_summary=initial_state_summary,
            final_state_summary=final_world_state,
            trajectory_path=path,
            diagnostics=diagnostics,
        )


def _target(skill_call: SkillCall) -> Vec2:
    target = skill_call.params["target"]
    if isinstance(target, Vec2):
        return target
    if isinstance(target, dict):
        return Vec2.from_dict(target)
    return Vec2(int(target[0]), int(target[1]))  # type: ignore[index]


def _player_tuple(info: dict[str, Any]) -> tuple[int, int]:
    return int(info["player"]["x"]), int(info["player"]["y"])


def _distance_px(player: tuple[int, int], target: Vec2, spec: WorldSpec) -> int:
    return (abs(player[0] - target.x) + abs(player[1] - target.y)) * spec.tile_size_px


def _bounded(value: float) -> float:
    return float(max(-1.0, min(1.0, value)))
