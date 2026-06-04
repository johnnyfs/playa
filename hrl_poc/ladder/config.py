from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hrl_poc.ladder.world import WorldSpec


@dataclass(frozen=True)
class TrainSpec:
    seeds: int = 0
    steps: int = 0

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "TrainSpec":
        data = dict(data or {})
        return cls(seeds=int(data.get("seeds", 0)), steps=int(data.get("steps", 0)))

    def to_dict(self) -> dict[str, int]:
        return {"seeds": self.seeds, "steps": self.steps}


@dataclass(frozen=True)
class EvalSpec:
    heldout_seeds: int = 1
    pass_rate: float = 1.0

    @classmethod
    def from_config(cls, data: dict[str, Any] | None) -> "EvalSpec":
        data = dict(data or {})
        return cls(
            heldout_seeds=int(data.get("heldout_seeds", 1)),
            pass_rate=float(data.get("pass_rate", 1.0)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {"heldout_seeds": self.heldout_seeds, "pass_rate": self.pass_rate}


@dataclass(frozen=True)
class LadderLevel:
    id: str
    description: str
    world: WorldSpec
    task: dict[str, Any]
    train: TrainSpec = field(default_factory=TrainSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "LadderLevel":
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            world=WorldSpec.from_config(data.get("world")),
            task=dict(data.get("task", {})),
            train=TrainSpec.from_config(data.get("train")),
            eval=EvalSpec.from_config(data.get("eval")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "world": self.world.to_dict(),
            "task": dict(self.task),
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict(),
        }


@dataclass(frozen=True)
class LadderConfig:
    skill: str
    levels: tuple[LadderLevel, ...]

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "LadderConfig":
        return cls(
            skill=str(data["skill"]),
            levels=tuple(LadderLevel.from_config(level) for level in data.get("levels", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "levels": [level.to_dict() for level in self.levels]}


def load_ladder_config(path: str | Path) -> LadderConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Ladder config must be a mapping: {config_path}")
    return LadderConfig.from_config(data)
