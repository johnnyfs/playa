from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from hrl_poc.semantic.state import Vec2


class EnemyType(IntEnum):
    GRUNT = 0
    BRUTE = 1


class ItemType(IntEnum):
    COIN = 0
    POTION = 1


@dataclass
class SkillSpec:
    name: str
    params_schema: dict[str, Any]
    success_predicates: list[Any]
    failure_predicates: list[Any]
    progress_rules: list[Any]
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params_schema": self.params_schema,
            "success_predicates": [repr(predicate) for predicate in self.success_predicates],
            "failure_predicates": [repr(predicate) for predicate in self.failure_predicates],
            "progress_rules": [repr(rule) for rule in self.progress_rules],
            "max_steps": self.max_steps,
        }


@dataclass
class SkillCall:
    skill_name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.skill_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "params": {key: _serialize_value(value) for key, value in self.params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillCall":
        name = str(data.get("skill_name", data.get("name")))
        params = dict(data.get("params", {}))
        for key, value in list(params.items()):
            if isinstance(value, dict) and set(value.keys()) == {"x", "y"}:
                params[key] = Vec2.from_dict(value)
        return cls(skill_name=name, params=params)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Vec2):
        return value.to_dict()
    if isinstance(value, IntEnum):
        return int(value)
    return value
