from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrl_poc.semantic.state import EntityState, Vec2, WorldState
from hrl_poc.skills.schema import SkillCall


@dataclass(frozen=True)
class EvalContext:
    prev_state: WorldState
    next_state: WorldState
    skill_call: SkillCall
    max_steps: int

    @property
    def elapsed_steps(self) -> int:
        return int(self.next_state.attrs.get("skill_steps", 0))


class Ref:
    def resolve(self, state: WorldState, ctx: EvalContext) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class PlayerRef(Ref):
    def resolve(self, state: WorldState, ctx: EvalContext) -> EntityState:
        return state.entities[state.player_id]


@dataclass(frozen=True)
class ParamRef(Ref):
    name: str

    def resolve(self, state: WorldState, ctx: EvalContext) -> Any:
        return ctx.skill_call.params[self.name]


@dataclass(frozen=True)
class EntityByIdRef(Ref):
    ref_or_string: Ref | str

    def resolve(self, state: WorldState, ctx: EvalContext) -> EntityState | None:
        entity_id = self.ref_or_string.resolve(state, ctx) if isinstance(self.ref_or_string, Ref) else self.ref_or_string
        return state.entities.get(str(entity_id))


@dataclass(frozen=True)
class EntityPosRef(Ref):
    entity_ref: Ref

    def resolve(self, state: WorldState, ctx: EvalContext) -> Vec2 | None:
        entity = self.entity_ref.resolve(state, ctx)
        return None if entity is None else entity.pos


@dataclass(frozen=True)
class ConstRef(Ref):
    value: Any

    def resolve(self, state: WorldState, ctx: EvalContext) -> Any:
        return self.value


def resolve_value(value: Any, state: WorldState, ctx: EvalContext) -> Any:
    if isinstance(value, Ref):
        return value.resolve(state, ctx)
    return value
