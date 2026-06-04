from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from hrl_poc.envs.gridworld import GRID_SIZE, MAX_ENEMIES, MAX_ITEMS, START_HEALTH
from hrl_poc.envs.wrappers import PARAM_VECTOR_SIZE


class SkillPolicy(nn.Module):
    """Shared actor-critic policy pi(action | state, skill_id, skill_params)."""

    def __init__(
        self,
        action_space: gym.Space,
        num_skills: int,
        grid_size: int = GRID_SIZE,
        max_enemies: int = MAX_ENEMIES,
        max_items: int = MAX_ITEMS,
        hidden_size: int = 128,
        skill_embedding_dim: int = 8,
        condition_on_skill: bool = True,
    ):
        super().__init__()
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("SkillPolicy currently expects a discrete action space")
        self.grid_size = grid_size
        self.max_enemies = max_enemies
        self.max_items = max_items
        self.condition_on_skill = condition_on_skill
        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim) if condition_on_skill else None
        state_dim = grid_size * grid_size + 2 + 1 + max_enemies * 5 + max_items * 4
        input_dim = state_dim + skill_embedding_dim + PARAM_VECTOR_SIZE if condition_on_skill else state_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_space.n)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs_batch: list[dict[str, Any]]):
        state_features, skill_ids, skill_params = encode_observation_batch(
            obs_batch,
            grid_size=self.grid_size,
            max_enemies=self.max_enemies,
            max_items=self.max_items,
            device=next(self.parameters()).device,
        )
        if self.condition_on_skill:
            if self.skill_embedding is None:
                raise RuntimeError("Skill embedding missing for skill-conditioned policy")
            skill_features = self.skill_embedding(skill_ids)
            features = torch.cat([state_features, skill_features, skill_params], dim=-1)
        else:
            features = state_features
        hidden = self.backbone(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def distribution_and_value(self, obs_batch: list[dict[str, Any]]):
        logits, values = self.forward(obs_batch)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(self, obs: dict[str, Any], deterministic: bool = False) -> tuple[int, float, float]:
        dist, value = self.distribution_and_value([obs])
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(self, obs_batch: list[dict[str, Any]], actions: torch.Tensor):
        dist, values = self.distribution_and_value(obs_batch)
        return dist.log_prob(actions), dist.entropy(), values


def encode_observation_batch(
    obs_batch: list[dict[str, Any]],
    grid_size: int,
    max_enemies: int,
    max_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_rows = []
    skill_ids = []
    skill_params = []
    for obs in obs_batch:
        state = obs["state"]
        terrain = np.asarray(state["terrain"], dtype=np.float32).reshape(-1)
        player = np.asarray(state["player"], dtype=np.float32) / max(1, grid_size - 1)
        health = np.asarray(state["health"], dtype=np.float32) / START_HEALTH
        enemies = np.asarray(state["enemies"], dtype=np.float32).reshape(max_enemies, 5)
        enemies[:, 0:2] = np.where(enemies[:, 0:2] >= 0, enemies[:, 0:2] / max(1, grid_size - 1), -1.0)
        enemies[:, 3] = np.where(enemies[:, 3] >= 0, enemies[:, 3] / 2.0, -1.0)
        items = np.asarray(state["items"], dtype=np.float32).reshape(max_items, 4)
        items[:, 0:2] = np.where(items[:, 0:2] >= 0, items[:, 0:2] / max(1, grid_size - 1), -1.0)
        state_rows.append(np.concatenate([terrain, player, health, enemies.reshape(-1), items.reshape(-1)]))
        skill_ids.append(int(np.asarray(obs["skill_id"]).reshape(-1)[0]))
        skill_params.append(np.asarray(obs["skill_params"], dtype=np.float32))

    state_tensor = torch.as_tensor(np.stack(state_rows), dtype=torch.float32, device=device)
    skill_id_tensor = torch.as_tensor(skill_ids, dtype=torch.long, device=device)
    skill_param_tensor = torch.as_tensor(np.stack(skill_params), dtype=torch.float32, device=device)
    return state_tensor, skill_id_tensor, skill_param_tensor
