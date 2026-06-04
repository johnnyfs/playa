from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrl_poc.semantic.state import EntityState, WorldState
from hrl_poc.skills.predicates import _as_pos, manhattan
from hrl_poc.skills.refs import EvalContext, resolve_value


class Metric:
    def value(self, state: WorldState, ctx: EvalContext) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Distance(Metric):
    a: Any
    b: Any

    def value(self, state: WorldState, ctx: EvalContext) -> float:
        a = _as_pos(resolve_value(self.a, state, ctx))
        b = _as_pos(resolve_value(self.b, state, ctx))
        if a is None or b is None:
            return 99.0
        return float(manhattan(a, b))


@dataclass(frozen=True)
class Health(Metric):
    entity: Any

    def value(self, state: WorldState, ctx: EvalContext) -> float:
        entity = resolve_value(self.entity, state, ctx)
        if not isinstance(entity, EntityState) or entity.hp is None:
            return 0.0
        return float(entity.hp)


@dataclass(frozen=True)
class EnemyProximity(Metric):
    entity: Any

    def value(self, state: WorldState, ctx: EvalContext) -> float:
        entity = resolve_value(self.entity, state, ctx)
        if not isinstance(entity, EntityState):
            return 0.0
        distances = [
            manhattan(entity.pos, other.pos)
            for other in state.entities.values()
            if other.hostile and other.alive
        ]
        if not distances:
            return 0.0
        nearest = min(distances)
        return float(max(0, 6 - nearest))


@dataclass(frozen=True)
class InventoryCount(Metric):
    item_id: Any

    def value(self, state: WorldState, ctx: EvalContext) -> float:
        item_id = str(resolve_value(self.item_id, state, ctx))
        return float(state.inventory.get(item_id, 0))


@dataclass(frozen=True)
class ContactFlag(Metric):
    def value(self, state: WorldState, ctx: EvalContext) -> float:
        return 1.0 if state.attrs.get("enemy_contact", False) else 0.0
