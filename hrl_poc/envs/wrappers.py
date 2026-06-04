from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.semantic.state import Vec2
from hrl_poc.skills.evaluator import SkillEvaluator
from hrl_poc.skills.registry import SKILL_IDS, canonical_skill_name, get_skill_id, get_skill_spec, make_skill_call
from hrl_poc.skills.schema import SkillCall


PARAM_VECTOR_SIZE = 5


class SkillConditionedEnv(gym.Wrapper):
    """Turns a base GridWorld into one episode for a concrete SkillCall."""

    def __init__(self, env: GridWorldEnv, skill_name: str | None = None, skill_call: SkillCall | None = None):
        super().__init__(env)
        self.skill_name = canonical_skill_name(skill_name or "go")
        self.skill_call = normalize_skill_call(skill_call) if skill_call else None
        self.fixed_skill_call = skill_call is not None
        self.skill_spec = get_skill_spec(self.skill_call.skill_name if self.skill_call else self.skill_name)
        self.evaluator = SkillEvaluator()
        self.observation_space = spaces.Dict(
            {
                "state": env.observation_space,
                "skill_id": spaces.Box(0, len(SKILL_IDS) - 1, shape=(1,), dtype=np.int64),
                "skill_params": spaces.Box(-1.0, 1.0, shape=(PARAM_VECTOR_SIZE,), dtype=np.float32),
            }
        )
        self.action_space = env.action_space
        self.initial_summary: dict[str, Any] | None = None
        self.elapsed_steps = 0
        self.reward_total = 0.0
        self.success = False
        self.failure = False
        self.failure_reason: str | None = None
        self.trajectory: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        env_options = options.get("env_options")
        obs, info = self.env.reset(seed=seed, options=env_options)

        call_data = options.get("skill_call")
        if isinstance(call_data, SkillCall):
            self.skill_call = normalize_skill_call(call_data)
        elif isinstance(call_data, dict):
            self.skill_call = normalize_skill_call(SkillCall.from_dict(call_data))
        elif self.skill_call is None or not self.fixed_skill_call:
            requested = canonical_skill_name(str(options.get("skill_name") or self.skill_name))
            self.skill_call = sample_skill_call(self.env, requested)

        self.skill_spec = get_skill_spec(self.skill_call.skill_name, self.skill_call.params)
        self.initial_summary = self.env.get_summary()
        self.elapsed_steps = 0
        self.reward_total = 0.0
        self.success = False
        self.failure = False
        self.failure_reason = None
        self.trajectory = []
        info["world_state"] = self.env.world_state()
        return self._augment_observation(obs), info

    def step(self, action: int):
        if self.skill_call is None:
            raise RuntimeError("SkillConditionedEnv.step called before reset")

        prev_state = self.env.world_state()
        prev_state.attrs["skill_steps"] = self.elapsed_steps
        obs, _base_reward, base_terminated, _base_truncated, _base_info = self.env.step(action)
        self.elapsed_steps += 1
        next_state = self.env.world_state()
        next_state.attrs["skill_steps"] = self.elapsed_steps

        result = self.evaluator.evaluate(prev_state, next_state, self.skill_call, self.skill_spec)
        if base_terminated and not result.done:
            result = self.evaluator.evaluate(prev_state, next_state, self.skill_call, self.skill_spec)

        self.success = result.success
        self.failure = result.failure
        self.failure_reason = result.failure_reason
        self.reward_total += float(result.reward)
        summary = self.env.get_summary()
        self.trajectory.append(
            {
                "step": self.elapsed_steps,
                "action": int(action),
                "reward": float(result.reward),
                "player": summary["player"],
                "health": summary["health"],
                "success": bool(result.success),
                "failure": bool(result.failure),
                "failure_reason": result.failure_reason,
            }
        )

        info = {
            **summary,
            "world_state": next_state,
            "success": bool(result.success),
            "failure": bool(result.failure),
            "failure_reason": result.failure_reason,
            "reward_total": float(self.reward_total),
            "steps": self.elapsed_steps,
            "skill_call": self.skill_call.to_dict(),
        }
        return self._augment_observation(obs), float(result.reward), result.done, False, info

    def make_execution_record(self, trajectory_path: str = "") -> dict[str, Any]:
        if self.skill_call is None:
            raise RuntimeError("No skill call has been run")
        summary = self.env.get_summary()
        return {
            "skill_call": self.skill_call.to_dict(),
            "success": bool(self.success),
            "failure": bool(self.failure),
            "failure_reason": self.failure_reason,
            "reward_total": float(self.reward_total),
            "steps": int(self.elapsed_steps),
            "initial_position": (self.initial_summary or {}).get("player", {}),
            "final_position": summary["player"],
            "trajectory_path": trajectory_path,
        }

    def _augment_observation(self, state_obs: dict[str, np.ndarray]) -> dict[str, Any]:
        if self.skill_call is None:
            raise RuntimeError("No active SkillCall")
        return {
            "state": state_obs,
            "skill_id": np.array([get_skill_id(self.skill_call.skill_name, self.skill_call.params)], dtype=np.int64),
            "skill_params": encode_skill_params(self.skill_call, self.env),
        }


def normalize_skill_call(skill_call: SkillCall) -> SkillCall:
    return make_skill_call(skill_call.skill_name, **dict(skill_call.params))


def encode_skill_params(skill_call: SkillCall, env: GridWorldEnv) -> np.ndarray:
    params = np.zeros(PARAM_VECTOR_SIZE, dtype=np.float32)
    scale = max(1, env.size - 1)
    name = canonical_skill_name(skill_call.skill_name, skill_call.params)
    if name in {"go", "go_evading"}:
        target = _vec_param(skill_call.params["target"])
        params[0] = target.x / scale
        params[1] = target.y / scale
        params[2] = 1.0 if name == "go_evading" else 0.0
        params[3] = (target.x - int(env.player[0])) / scale
        params[4] = (target.y - int(env.player[1])) / scale
    elif name in {"attack", "collect"}:
        entity_id = str(skill_call.params["enemy_id"] if name == "attack" else skill_call.params["item_id"])
        entity = env.world_state().entities.get(entity_id)
        if entity is not None:
            params[0] = entity.pos.x / scale
            params[1] = entity.pos.y / scale
            params[2] = 1.0 if name == "attack" else 0.5
    return params


def sample_skill_call(env: GridWorldEnv, skill_name: str) -> SkillCall:
    name = canonical_skill_name(skill_name)
    if name in {"go", "go_evading"}:
        target = _sample_walkable_cell(env)
        params: dict[str, object] = {"target": Vec2(*target)}
        if name == "go_evading":
            params["mode"] = "evading"
        return SkillCall(name, params)
    if name == "attack":
        enemies = [enemy for enemy in env.enemies if enemy["alive"]]
        enemy = enemies[int(env.np_random.integers(0, len(enemies)))]
        return SkillCall("attack", {"enemy_id": str(enemy["id"])})
    if name == "collect":
        items = [item for item in env.items if item["present"]]
        item = items[int(env.np_random.integers(0, len(items)))]
        return SkillCall("collect", {"item_id": str(item["id"])})
    raise ValueError(f"Unknown skill {skill_name!r}")


def _sample_walkable_cell(env: GridWorldEnv) -> tuple[int, int]:
    for _ in range(500):
        x = int(env.np_random.integers(0, env.size))
        y = int(env.np_random.integers(0, env.size))
        if env.terrain[y, x] == 0 and (x, y) != tuple(int(v) for v in env.player):
            return x, y
    raise RuntimeError("Could not sample a walkable skill target")


def _vec_param(value: object) -> Vec2:
    if isinstance(value, Vec2):
        return value
    if isinstance(value, dict):
        return Vec2.from_dict(value)
    return Vec2(int(value[0]), int(value[1]))  # type: ignore[index]
