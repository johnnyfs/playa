from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil
from typing import Any

import numpy as np

from hrl_poc.envs.gridworld import GRID_SIZE, START_HEALTH, GridWorldEnv
from hrl_poc.semantic.state import Vec2


DEFAULT_TILE_SIZE_PX = 16


class TerrainSpec(StrEnum):
    EMPTY = "Empty"
    BORDER_WALLS = "BorderWalls"
    SINGLE_WALL = "SingleWall"
    SIMPLE_MAZE = "SimpleMaze"


@dataclass(frozen=True)
class SpawnSpec:
    x: int | None = None
    y: int | None = None

    @classmethod
    def from_config(cls, data: Any | None) -> "SpawnSpec":
        if data is None or data == "random":
            return cls()
        if isinstance(data, dict):
            return cls(
                x=None if data.get("x") is None else int(data["x"]),
                y=None if data.get("y") is None else int(data["y"]),
            )
        if isinstance(data, list | tuple) and len(data) == 2:
            return cls(x=int(data[0]), y=int(data[1]))
        raise ValueError(f"Invalid spawn spec: {data!r}")

    @property
    def fixed(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass(frozen=True)
class EnemySpec:
    id: str
    type: int = 0
    pos: tuple[int, int] = (0, 0)
    hp: int = 1


@dataclass(frozen=True)
class PickupSpec:
    id: str
    type: int = 0
    pos: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class WorldSpec:
    world_width_screens: int = 1
    world_height_screens: int = 1
    screen_width_px: int = GRID_SIZE * DEFAULT_TILE_SIZE_PX
    screen_height_px: int = GRID_SIZE * DEFAULT_TILE_SIZE_PX
    tile_size_px: int = DEFAULT_TILE_SIZE_PX
    player_start: SpawnSpec = field(default_factory=SpawnSpec)
    terrain: TerrainSpec = TerrainSpec.EMPTY
    enemies: tuple[EnemySpec, ...] = ()
    pickups: tuple[PickupSpec, ...] = ()
    seed: int = 0

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "WorldSpec":
        data = dict(data or {})
        terrain = data.get("terrain", TerrainSpec.EMPTY.value)
        enemies = tuple(
            EnemySpec(
                id=str(enemy.get("id", f"enemy_{idx}")),
                type=int(enemy.get("type", 0)),
                pos=_pair(enemy.get("pos", (0, 0))),
                hp=int(enemy.get("hp", 1)),
            )
            for idx, enemy in enumerate(data.get("enemies", []))
        )
        pickups = tuple(
            PickupSpec(
                id=str(item.get("id", f"item_{idx}")),
                type=int(item.get("type", 0)),
                pos=_pair(item.get("pos", (0, 0))),
            )
            for idx, item in enumerate(data.get("pickups", data.get("items", [])))
        )
        return cls(
            world_width_screens=int(data.get("world_width_screens", 1)),
            world_height_screens=int(data.get("world_height_screens", 1)),
            screen_width_px=int(data.get("screen_width_px", GRID_SIZE * DEFAULT_TILE_SIZE_PX)),
            screen_height_px=int(data.get("screen_height_px", GRID_SIZE * DEFAULT_TILE_SIZE_PX)),
            tile_size_px=int(data.get("tile_size_px", DEFAULT_TILE_SIZE_PX)),
            player_start=SpawnSpec.from_config(data.get("player_start")),
            terrain=TerrainSpec(str(terrain)),
            enemies=enemies,
            pickups=pickups,
            seed=int(data.get("seed", 0)),
        )

    @property
    def screen_width_tiles(self) -> int:
        return max(1, self.screen_width_px // self.tile_size_px)

    @property
    def screen_height_tiles(self) -> int:
        return max(1, self.screen_height_px // self.tile_size_px)

    @property
    def world_width_tiles(self) -> int:
        return self.world_width_screens * self.screen_width_tiles

    @property
    def world_height_tiles(self) -> int:
        return self.world_height_screens * self.screen_height_tiles

    @property
    def env_size(self) -> int:
        return max(self.world_width_tiles, self.world_height_tiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_width_screens": self.world_width_screens,
            "world_height_screens": self.world_height_screens,
            "screen_width_px": self.screen_width_px,
            "screen_height_px": self.screen_height_px,
            "tile_size_px": self.tile_size_px,
            "player_start": {"x": self.player_start.x, "y": self.player_start.y},
            "terrain": self.terrain.value,
            "enemies": [
                {"id": enemy.id, "type": enemy.type, "pos": list(enemy.pos), "hp": enemy.hp}
                for enemy in self.enemies
            ],
            "pickups": [
                {"id": item.id, "type": item.type, "pos": list(item.pos)}
                for item in self.pickups
            ],
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GeneratedWorld:
    spec: WorldSpec
    walls: tuple[tuple[int, int], ...]
    player: tuple[int, int]
    env_options: dict[str, Any]

    def env(self) -> GridWorldEnv:
        env = GridWorldEnv(size=self.spec.env_size)
        env.reset(seed=self.spec.seed, options=self.env_options)
        return env


def generate_world(spec: WorldSpec, seed: int, player: tuple[int, int] | None = None) -> GeneratedWorld:
    rng = np.random.default_rng(spec.seed + seed)
    walls = tuple(sorted(generate_walls(spec, rng)))
    chosen_player = player or choose_player_start(spec, walls, rng)
    env_options = {
        "player": chosen_player,
        "walls": list(walls),
        "enemies": [
            {"id": enemy.id, "type": enemy.type, "pos": enemy.pos, "hp": enemy.hp}
            for enemy in spec.enemies
        ],
        "items": [
            {"id": item.id, "type": item.type, "pos": item.pos}
            for item in spec.pickups
        ],
        "health": START_HEALTH,
        "screen_width_tiles": spec.screen_width_tiles,
        "screen_height_tiles": spec.screen_height_tiles,
    }
    return GeneratedWorld(spec=spec, walls=walls, player=chosen_player, env_options=env_options)


def generate_walls(spec: WorldSpec, rng: np.random.Generator) -> set[tuple[int, int]]:
    width = spec.world_width_tiles
    height = spec.world_height_tiles
    size = spec.env_size
    walls = {
        (x, y)
        for y in range(size)
        for x in range(size)
        if x >= width or y >= height
    }
    if spec.terrain == TerrainSpec.EMPTY:
        return walls

    if spec.terrain in {TerrainSpec.BORDER_WALLS, TerrainSpec.SINGLE_WALL, TerrainSpec.SIMPLE_MAZE}:
        walls.update((x, 0) for x in range(width))
        walls.update((x, height - 1) for x in range(width))
        walls.update((0, y) for y in range(height))
        walls.update((width - 1, y) for y in range(height))

    if spec.terrain == TerrainSpec.SINGLE_WALL and width >= 5 and height >= 5:
        wall_x = width // 2
        gap_y = int(rng.integers(1, height - 1))
        for y in range(1, height - 1):
            if y != gap_y:
                walls.add((wall_x, y))

    if spec.terrain == TerrainSpec.SIMPLE_MAZE and width >= 7 and height >= 7:
        for wall_x in range(3, width - 2, 4):
            gap_y = int(rng.integers(1, height - 1))
            for y in range(1, height - 1):
                if y != gap_y:
                    walls.add((wall_x, y))

    return walls


def choose_player_start(spec: WorldSpec, walls: tuple[tuple[int, int], ...] | set[tuple[int, int]], rng: np.random.Generator) -> tuple[int, int]:
    if spec.player_start.fixed:
        pos = (int(spec.player_start.x), int(spec.player_start.y))  # type: ignore[arg-type]
        if not is_walkable(spec, walls, pos):
            raise ValueError(f"Configured player_start is not walkable: {pos}")
        return pos
    cells = walkable_cells(spec, walls)
    if not cells:
        raise ValueError("World has no walkable cells")
    return cells[int(rng.integers(0, len(cells)))]


def generate_go_task(
    spec: WorldSpec,
    task: dict[str, Any],
    seed: int,
) -> tuple[GeneratedWorld, Vec2]:
    terrain_rng = np.random.default_rng(spec.seed + seed)
    task_rng = np.random.default_rng(spec.seed + seed + 17)
    walls = tuple(sorted(generate_walls(spec, terrain_rng)))
    target_distance_tiles = max(1, ceil(int(task.get("target_distance_px", spec.tile_size_px)) / spec.tile_size_px))
    screen_offset = task.get("target_screen_offset")

    for _ in range(1000):
        player = _sample_player_for_task(spec, walls, task_rng, screen_offset)
        candidates = _target_candidates(spec, walls, player, target_distance_tiles, screen_offset)
        if not candidates:
            continue
        target = candidates[int(task_rng.integers(0, len(candidates)))]
        if path_exists(spec, walls, player, target):
            world = generate_world(spec, seed=seed, player=player)
            return world, Vec2(*target)

    raise ValueError(f"Could not generate valid go task for terrain={spec.terrain} seed={seed}")


def walkable_cells(spec: WorldSpec, walls: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    wall_set = set(walls)
    return [
        (x, y)
        for y in range(spec.world_height_tiles)
        for x in range(spec.world_width_tiles)
        if (x, y) not in wall_set
    ]


def is_walkable(spec: WorldSpec, walls: tuple[tuple[int, int], ...] | set[tuple[int, int]], pos: tuple[int, int]) -> bool:
    x, y = pos
    return 0 <= x < spec.world_width_tiles and 0 <= y < spec.world_height_tiles and pos not in set(walls)


def path_exists(
    spec: WorldSpec,
    walls: tuple[tuple[int, int], ...] | set[tuple[int, int]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    wall_set = set(walls)
    if not is_walkable(spec, wall_set, start) or not is_walkable(spec, wall_set, target):
        return False
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        pos = queue.popleft()
        if pos == target:
            return True
        x, y = pos
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (x + dx, y + dy)
            if nxt not in seen and is_walkable(spec, wall_set, nxt):
                seen.add(nxt)
                queue.append(nxt)
    return False


def _sample_player_for_task(
    spec: WorldSpec,
    walls: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    screen_offset: Any,
) -> tuple[int, int]:
    cells = walkable_cells(spec, walls)
    if spec.player_start.fixed:
        return choose_player_start(spec, walls, rng)
    if screen_offset is None:
        return cells[int(rng.integers(0, len(cells)))]

    offset_x, offset_y = _pair(screen_offset)
    viable = []
    for cell in cells:
        screen_x = cell[0] // spec.screen_width_tiles
        screen_y = cell[1] // spec.screen_height_tiles
        target_screen = (screen_x + offset_x, screen_y + offset_y)
        if 0 <= target_screen[0] < spec.world_width_screens and 0 <= target_screen[1] < spec.world_height_screens:
            viable.append(cell)
    if not viable:
        raise ValueError(f"No viable start screen for offset {screen_offset!r}")
    return viable[int(rng.integers(0, len(viable)))]


def _target_candidates(
    spec: WorldSpec,
    walls: tuple[tuple[int, int], ...],
    player: tuple[int, int],
    target_distance_tiles: int,
    screen_offset: Any,
) -> list[tuple[int, int]]:
    cells = walkable_cells(spec, walls)
    if screen_offset is not None:
        offset_x, offset_y = _pair(screen_offset)
        player_screen = (player[0] // spec.screen_width_tiles, player[1] // spec.screen_height_tiles)
        wanted_screen = (player_screen[0] + offset_x, player_screen[1] + offset_y)
        cells = [
            cell
            for cell in cells
            if cell[0] // spec.screen_width_tiles == wanted_screen[0]
            and cell[1] // spec.screen_height_tiles == wanted_screen[1]
        ]
        return [cell for cell in cells if _manhattan(player, cell) >= target_distance_tiles]

    return [
        cell
        for cell in cells
        if cell != player and _manhattan(player, cell) == target_distance_tiles
    ]


def _pair(data: Any) -> tuple[int, int]:
    if isinstance(data, dict):
        return int(data["x"]), int(data["y"])
    if isinstance(data, list | tuple) and len(data) == 2:
        return int(data[0]), int(data[1])
    raise ValueError(f"Expected coordinate pair, got {data!r}")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
