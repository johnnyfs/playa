from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.skills.predicates import PredicateContext, evaluate_failure, evaluate_shaping, evaluate_success
from hrl_poc.skills.registry import SKILL_REGISTRY, get_skill_spec
from hrl_poc.skills.schema import EnemyType, ItemType, SkillCall, Vec2


PARAM_VECTOR_SIZE = 5


class SkillConditionedEnv(gym.Wrapper):
    """Turns a base GridWorld into one episode for a concrete SkillCall."""

    def __init__(self, env: GridWorldEnv, skill_name: str | None = None, skill_call: SkillCall | None = None):
        super().__init__(env)
        self.skill_name = skill_name
        self.skill_call = skill_call
        self.fixed_skill_call = skill_call is not None
        self.skill_spec = get_skill_spec(skill_call.name if skill_call else skill_name or "navigate")
        self.observation_space = spaces.Dict(
            {
                "state": env.observation_space,
                "skill_id": spaces.Box(0, len(SKILL_REGISTRY) - 1, shape=(1,), dtype=np.int64),
                "skill_params": spaces.Box(-1.0, 1.0, shape=(PARAM_VECTOR_SIZE,), dtype=np.float32),
            }
        )
        self.action_space = env.action_space
        self.initial_state: dict[str, Any] | None = None
        self.initial_summary: dict[str, Any] | None = None
        self.previous_state: dict[str, Any] | None = None
        self.elapsed_steps = 0
        self.reward_total = 0.0
        self.success = False
        self.failure_reason: str | None = None
        self.trajectory: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        env_options = options.get("env_options")
        obs, info = self.env.reset(seed=seed, options=env_options)

        call_data = options.get("skill_call")
        if isinstance(call_data, SkillCall):
            self.skill_call = call_data
        elif isinstance(call_data, dict):
            self.skill_call = SkillCall.from_dict(call_data)
        elif self.skill_call is None or not self.fixed_skill_call:
            requested = options.get("skill_name") or self.skill_name or "navigate"
            self.skill_call = sample_skill_call(self.env, requested)

        self.skill_spec = get_skill_spec(self.skill_call.name)
        self.initial_state = self.env.get_state()
        self.initial_summary = self.env.get_summary()
        self.previous_state = self.initial_state
        self.elapsed_steps = 0
        self.reward_total = 0.0
        self.success = False
        self.failure_reason = None
        self.trajectory = []
        return self._augment_observation(obs), info

    def step(self, action: int):
        if self.skill_call is None or self.initial_state is None or self.previous_state is None:
            raise RuntimeError("SkillConditionedEnv.step called before reset")

        previous_state = self.env.get_state()
        obs, _base_reward, base_terminated, _base_truncated, _base_info = self.env.step(action)
        self.elapsed_steps += 1
        state = self.env.get_state()
        ctx = PredicateContext(
            skill_call=self.skill_call,
            initial_state=self.initial_state,
            previous_state=previous_state,
            state=state,
            elapsed_steps=self.elapsed_steps,
            max_steps=self.skill_spec.max_steps,
        )

        success, success_reason = evaluate_success(ctx, self.skill_spec.success_predicates)
        failure, failure_reason = evaluate_failure(ctx, self.skill_spec.failure_predicates)
        if base_terminated and not failure:
            failure, failure_reason = True, "player_dead"

        shaping = evaluate_shaping(ctx, self.skill_spec.shaping_metrics)
        reward = -0.01 + 0.08 * shaping
        if success:
            reward += 1.0
        elif failure:
            reward -= 1.0

        self.success = success
        self.failure_reason = None if success else failure_reason
        self.reward_total += float(reward)
        self.previous_state = state
        self.trajectory.append(
            {
                "step": self.elapsed_steps,
                "action": int(action),
                "reward": float(reward),
                "player": {"x": int(state["player"][0]), "y": int(state["player"][1])},
                "health": int(state["health"]),
                "success": bool(success),
                "failure_reason": self.failure_reason,
            }
        )

        terminated = bool(success or failure)
        info = {
            "success": bool(success),
            "success_reason": success_reason,
            "failure_reason": self.failure_reason,
            "reward_total": float(self.reward_total),
            "steps": self.elapsed_steps,
            "skill_call": self.skill_call.to_dict(),
        }
        return self._augment_observation(obs), float(reward), terminated, False, info

    def make_execution_record(self, trajectory_path: str = "") -> dict[str, Any]:
        if self.skill_call is None:
            raise RuntimeError("No skill call has been run")
        return {
            "skill_call": self.skill_call.to_dict(),
            "initial_summary": self.initial_summary or {},
            "final_summary": self.env.get_summary(),
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "reward_total": float(self.reward_total),
            "steps": int(self.elapsed_steps),
            "trajectory_path": trajectory_path,
        }

    def _augment_observation(self, state_obs: dict[str, np.ndarray]) -> dict[str, Any]:
        if self.skill_call is None:
            raise RuntimeError("No active SkillCall")
        return {
            "state": state_obs,
            "skill_id": np.array([self.skill_spec.skill_id], dtype=np.int64),
            "skill_params": encode_skill_params(self.skill_call, self.env.size),
        }


def encode_skill_params(skill_call: SkillCall, grid_size: int) -> np.ndarray:
    scale = max(1, grid_size - 1)
    params = np.zeros(PARAM_VECTOR_SIZE, dtype=np.float32)
    if skill_call.name in {"navigate", "avoid"}:
        goal = _vec_param(skill_call, "goal")
        params[0] = goal.x / scale
        params[1] = goal.y / scale
    elif skill_call.name == "fight":
        target = _vec_param(skill_call, "target")
        params[0] = target.x / scale
        params[1] = target.y / scale
        params[2] = float(EnemyType(skill_call.params["enemy_type"]))
    elif skill_call.name == "pickup":
        target = _vec_param(skill_call, "target")
        params[0] = target.x / scale
        params[1] = target.y / scale
        params[2] = float(ItemType(skill_call.params["item_type"]))
    return params


def sample_skill_call(env: GridWorldEnv, skill_name: str) -> SkillCall:
    if skill_name in {"navigate", "avoid"}:
        goal = _sample_walkable_cell(env)
        return SkillCall(skill_name, {"goal": Vec2(*goal)})
    if skill_name == "fight":
        enemies = [enemy for enemy in env.enemies if enemy["alive"]]
        enemy = enemies[int(env.np_random.integers(0, len(enemies)))]
        x, y = (int(v) for v in enemy["pos"])
        return SkillCall("fight", {"enemy_type": EnemyType(enemy["type"]), "target": Vec2(x, y)})
    if skill_name == "pickup":
        items = [item for item in env.items if item["present"]]
        item = items[int(env.np_random.integers(0, len(items)))]
        x, y = (int(v) for v in item["pos"])
        return SkillCall("pickup", {"item_type": ItemType(item["type"]), "target": Vec2(x, y)})
    raise ValueError(f"Unknown skill {skill_name!r}")


def _sample_walkable_cell(env: GridWorldEnv) -> tuple[int, int]:
    for _ in range(500):
        x = int(env.np_random.integers(0, env.size))
        y = int(env.np_random.integers(0, env.size))
        if env.terrain[y, x] == 0 and (x, y) != tuple(int(v) for v in env.player):
            return x, y
    raise RuntimeError("Could not sample a walkable skill target")


def _vec_param(skill_call: SkillCall, name: str) -> Vec2:
    value = skill_call.params[name]
    if isinstance(value, Vec2):
        return value
    if isinstance(value, dict):
        return Vec2.from_dict(value)
    return Vec2(int(value[0]), int(value[1]))
