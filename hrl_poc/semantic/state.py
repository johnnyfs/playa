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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "kind": self.kind,
            "pos": self.pos.to_dict(),
            "alive": bool(self.alive),
            "hp": self.hp,
            "hostile": bool(self.hostile),
            "attrs": dict(self.attrs),
        }


@dataclass
class TileState:
    pos: Vec2
    walkable: bool
    hazardous: bool = False
    type: str = "floor"
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pos": self.pos.to_dict(),
            "walkable": bool(self.walkable),
            "hazardous": bool(self.hazardous),
            "type": self.type,
            "attrs": dict(self.attrs),
        }


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

    def to_dict(self) -> dict[str, Any]:
        player = self.entities.get(self.player_id)
        solid_tiles = [tile.pos.to_dict() for tile in self.tiles.values() if not tile.walkable]
        terrain_grid = self.attrs.get("terrain_grid")
        return {
            "tick": int(self.tick),
            "world_position": self.global_position.to_dict(),
            "screen_position": self.attrs.get("screen_position", self.local_position.to_dict()),
            "global_position": self.global_position.to_dict(),
            "local_position": self.local_position.to_dict(),
            "region_id": self.region_id,
            "player_id": self.player_id,
            "player_position": player.pos.to_dict() if player else self.global_position.to_dict(),
            "player_hitbox": self.attrs.get("player_hitbox"),
            "player_facing": self.attrs.get("player_facing"),
            "terrain_grid": terrain_grid,
            "solid_tiles": solid_tiles,
            "enemies": [
                entity.to_dict()
                for entity in self.entities.values()
                if entity.kind == "enemy"
            ],
            "pickups": [
                entity.to_dict()
                for entity in self.entities.values()
                if entity.kind == "item"
            ],
            "inventory": dict(self.inventory),
            "active_sword_hitbox": self.attrs.get("active_sword_hitbox"),
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "tiles": {f"{x},{y}": tile.to_dict() for (x, y), tile in self.tiles.items()},
            "attrs": dict(self.attrs),
        }
