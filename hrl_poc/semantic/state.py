from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Vec2:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vec2":
        return cls(x=int(data["x"]), y=int(data["y"]))


@dataclass
class EntityState:
    id: str
    type: str
    kind: str
    pos: Vec2
    alive: bool = True
    hp: int | None = None
    hostile: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TileState:
    pos: Vec2
    walkable: bool
    hazardous: bool = False
    type: str = "floor"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    tick: int
    global_position: Vec2
    local_position: Vec2
    region_id: str
    player_id: str
    entities: dict[str, EntityState]
    tiles: dict[tuple[int, int], TileState]
    inventory: dict[str, int]
    attrs: dict[str, Any] = field(default_factory=dict)
