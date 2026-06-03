from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class EnemyType(IntEnum):
    GRUNT = 0
    BRUTE = 1


class ItemType(IntEnum):
    COIN = 0
    POTION = 1


@dataclass(frozen=True)
class Vec2:
    x: int
    y: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vec2":
        return cls(x=int(data["x"]), y=int(data["y"]))


@dataclass(frozen=True)
class SkillSpec:
    name: str
    skill_id: int
    parameter_schema: dict[str, str]
    success_predicates: list[str]
    failure_predicates: list[str]
    shaping_metrics: list[str]
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "skill_id": self.skill_id,
            "parameter_schema": dict(self.parameter_schema),
            "success_predicates": list(self.success_predicates),
            "failure_predicates": list(self.failure_predicates),
            "shaping_metrics": list(self.shaping_metrics),
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpec":
        return cls(
            name=str(data["name"]),
            skill_id=int(data["skill_id"]),
            parameter_schema=dict(data["parameter_schema"]),
            success_predicates=list(data["success_predicates"]),
            failure_predicates=list(data["failure_predicates"]),
            shaping_metrics=list(data["shaping_metrics"]),
            max_steps=int(data["max_steps"]),
        )


@dataclass(frozen=True)
class SkillCall:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": {key: _serialize_value(value) for key, value in self.params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCall":
        params = dict(data.get("params", {}))
        for key, value in list(params.items()):
            if isinstance(value, dict) and set(value.keys()) == {"x", "y"}:
                params[key] = Vec2.from_dict(value)
            elif key == "enemy_type":
                params[key] = EnemyType(value)
            elif key == "item_type":
                params[key] = ItemType(value)
        return cls(name=str(data["name"]), params=params)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Vec2):
        return value.to_dict()
    if isinstance(value, IntEnum):
        return int(value)
    return value
