from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hrl_poc.skills.schema import EnemyType, ItemType


GRID_SIZE = 12
MAX_ENEMIES = 3
MAX_ITEMS = 3
START_HEALTH = 5


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ATTACK = 4
    WAIT = 5


class GridWorldEnv(gym.Env):
    """Small structured GridWorld for skill-conditioned RL episodes."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = GRID_SIZE, max_enemies: int = MAX_ENEMIES, max_items: int = MAX_ITEMS):
        super().__init__()
        self.size = size
        self.max_enemies = max_enemies
        self.max_items = max_items
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Dict(
            {
                "terrain": spaces.Box(0, 1, shape=(size, size), dtype=np.int8),
                "player": spaces.Box(0, size - 1, shape=(2,), dtype=np.int16),
                "health": spaces.Box(0, START_HEALTH, shape=(1,), dtype=np.int16),
                "enemies": spaces.Box(-1, max(size, 2), shape=(max_enemies, 5), dtype=np.int16),
                "items": spaces.Box(-1, max(size, 2), shape=(max_items, 4), dtype=np.int16),
            }
        )
        self.terrain = np.zeros((size, size), dtype=np.int8)
        self.player = np.array([0, 0], dtype=np.int16)
        self.health = START_HEALTH
        self.enemies: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.steps = 0
        self.terrain = np.zeros((self.size, self.size), dtype=np.int8)
        self.enemies = []
        self.items = []
        self.health = int(options.get("health", START_HEALTH))

        if options:
            self._reset_from_options(options)
        else:
            self._random_reset()

        return self._observation(), self.get_summary()

    def step(self, action: int):
        self.steps += 1
        action = int(action)
        reward = -0.01

        if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            self._move(action)
            reward += self._collect_item()
        elif action == Action.ATTACK:
            reward += self._attack()

        self._enemy_contact_damage()
        terminated = self.health <= 0
        truncated = False
        info = self.get_summary()
        return self._observation(), reward, terminated, truncated, info

    def get_state(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "terrain": self.terrain.copy(),
            "player": tuple(int(v) for v in self.player),
            "health": int(self.health),
            "enemies": [dict(enemy) for enemy in self.enemies],
            "items": [dict(item) for item in self.items],
            "steps": int(self.steps),
        }

    def get_summary(self) -> dict[str, Any]:
        return {
            "player": {"x": int(self.player[0]), "y": int(self.player[1])},
            "health": int(self.health),
            "alive_enemies": [
                {
                    "type": int(enemy["type"]),
                    "x": int(enemy["pos"][0]),
                    "y": int(enemy["pos"][1]),
                    "hp": int(enemy["hp"]),
                }
                for enemy in self.enemies
                if enemy["alive"]
            ],
            "present_items": [
                {
                    "type": int(item["type"]),
                    "x": int(item["pos"][0]),
                    "y": int(item["pos"][1]),
                }
                for item in self.items
                if item["present"]
            ],
            "steps": int(self.steps),
        }

    def render(self):
        canvas = np.full((self.size, self.size), ".", dtype=object)
        canvas[self.terrain == 1] = "#"
        for item in self.items:
            if item["present"]:
                x, y = item["pos"]
                canvas[y, x] = "i"
        for enemy in self.enemies:
            if enemy["alive"]:
                x, y = enemy["pos"]
                canvas[y, x] = "e"
        x, y = self.player
        canvas[y, x] = "@"
        return "\n".join("".join(row) for row in canvas)

    def _reset_from_options(self, options: dict[str, Any]) -> None:
        for x, y in options.get("walls", []):
            self.terrain[int(y), int(x)] = 1
        self.player = np.array(options.get("player", (0, 0)), dtype=np.int16)
        self.enemies = [
            {
                "type": EnemyType(enemy.get("type", EnemyType.GRUNT)),
                "pos": np.array(enemy["pos"], dtype=np.int16),
                "hp": int(enemy.get("hp", 1)),
                "alive": bool(enemy.get("alive", True)),
            }
            for enemy in options.get("enemies", [])
        ][: self.max_enemies]
        self.items = [
            {
                "type": ItemType(item.get("type", ItemType.COIN)),
                "pos": np.array(item["pos"], dtype=np.int16),
                "present": bool(item.get("present", True)),
            }
            for item in options.get("items", [])
        ][: self.max_items]

    def _random_reset(self) -> None:
        wall_count = 14
        for _ in range(wall_count):
            x, y = self._empty_cell()
            self.terrain[y, x] = 1
        self.player = np.array(self._empty_cell(), dtype=np.int16)
        for _ in range(self.np_random.integers(1, self.max_enemies + 1)):
            enemy_type = EnemyType(int(self.np_random.integers(0, len(EnemyType))))
            hp = 2 if enemy_type == EnemyType.BRUTE else 1
            self.enemies.append({"type": enemy_type, "pos": np.array(self._empty_cell(), dtype=np.int16), "hp": hp, "alive": True})
        for _ in range(self.np_random.integers(1, self.max_items + 1)):
            item_type = ItemType(int(self.np_random.integers(0, len(ItemType))))
            self.items.append({"type": item_type, "pos": np.array(self._empty_cell(), dtype=np.int16), "present": True})

    def _observation(self) -> dict[str, np.ndarray]:
        enemies = np.full((self.max_enemies, 5), -1, dtype=np.int16)
        for idx, enemy in enumerate(self.enemies[: self.max_enemies]):
            x, y = enemy["pos"]
            enemies[idx] = [x, y, int(enemy["type"]), int(enemy["hp"]), int(enemy["alive"])]
        items = np.full((self.max_items, 4), -1, dtype=np.int16)
        for idx, item in enumerate(self.items[: self.max_items]):
            x, y = item["pos"]
            items[idx] = [x, y, int(item["type"]), int(item["present"])]
        return {
            "terrain": self.terrain.copy(),
            "player": self.player.copy(),
            "health": np.array([self.health], dtype=np.int16),
            "enemies": enemies,
            "items": items,
        }

    def _move(self, action: int) -> None:
        dx, dy = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
        }[Action(action)]
        target = self.player + np.array([dx, dy], dtype=np.int16)
        if self._is_walkable(tuple(int(v) for v in target)):
            self.player = target

    def _attack(self) -> float:
        for enemy in self.enemies:
            if enemy["alive"] and _manhattan(self.player, enemy["pos"]) <= 1:
                enemy["hp"] -= 1
                if enemy["hp"] <= 0:
                    enemy["alive"] = False
                    return 0.5
                return 0.1
        return -0.02

    def _collect_item(self) -> float:
        reward = 0.0
        for item in self.items:
            if item["present"] and np.array_equal(item["pos"], self.player):
                item["present"] = False
                reward += 0.25
                if item["type"] == ItemType.POTION:
                    self.health = min(START_HEALTH, self.health + 2)
        return reward

    def _enemy_contact_damage(self) -> None:
        for enemy in self.enemies:
            if enemy["alive"] and _manhattan(self.player, enemy["pos"]) <= 1:
                self.health -= 1
                return

    def _empty_cell(self) -> tuple[int, int]:
        occupied = {tuple(int(v) for v in self.player)}
        occupied.update(tuple(int(v) for v in enemy["pos"]) for enemy in self.enemies if enemy["alive"])
        occupied.update(tuple(int(v) for v in item["pos"]) for item in self.items if item["present"])
        for _ in range(500):
            x = int(self.np_random.integers(0, self.size))
            y = int(self.np_random.integers(0, self.size))
            if self.terrain[y, x] == 0 and (x, y) not in occupied:
                return x, y
        raise RuntimeError("Could not sample an empty GridWorld cell")

    def _is_walkable(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and self.terrain[y, x] == 0


def _manhattan(a: np.ndarray, b: np.ndarray) -> int:
    return int(abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])))
