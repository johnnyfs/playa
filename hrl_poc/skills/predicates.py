from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hrl_poc.semantic.state import EntityState, Vec2, WorldState
from hrl_poc.skills.refs import EvalContext, Ref, resolve_value


class Predicate:
    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Near(Predicate):
    a: Any
    b: Any
    radius: int = 0

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        a = _as_pos(resolve_value(self.a, state, ctx))
        b = _as_pos(resolve_value(self.b, state, ctx))
        return a is not None and b is not None and manhattan(a, b) <= self.radius


@dataclass(frozen=True)
class Alive(Predicate):
    entity: Any

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        entity = resolve_value(self.entity, state, ctx)
        return isinstance(entity, EntityState) and entity.alive


@dataclass(frozen=True)
class Dead(Predicate):
    entity: Any

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        entity = resolve_value(self.entity, state, ctx)
        return isinstance(entity, EntityState) and not entity.alive


@dataclass(frozen=True)
class Exists(Predicate):
    entity: Any

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        return resolve_value(self.entity, state, ctx) is not None


@dataclass(frozen=True)
class HasInventoryItem(Predicate):
    item_type: Any

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        item_key = str(resolve_value(self.item_type, state, ctx))
        return state.inventory.get(item_key, 0) > ctx.prev_state.inventory.get(item_key, 0)


@dataclass(frozen=True)
class Timeout(Predicate):
    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        return ctx.elapsed_steps >= ctx.max_steps


@dataclass(frozen=True)
class EnemyContact(Predicate):
    radius: int = 1

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        player = state.entities[state.player_id]
        return any(
            entity.hostile and entity.alive and manhattan(player.pos, entity.pos) <= self.radius
            for entity in state.entities.values()
        )


@dataclass(frozen=True)
class And(Predicate):
    predicates: tuple[Predicate, ...]

    def __init__(self, *predicates: Predicate):
        object.__setattr__(self, "predicates", tuple(predicates))

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        return all(predicate.evaluate(state, ctx) for predicate in self.predicates)


@dataclass(frozen=True)
class Or(Predicate):
    predicates: tuple[Predicate, ...]

    def __init__(self, *predicates: Predicate):
        object.__setattr__(self, "predicates", tuple(predicates))

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        return any(predicate.evaluate(state, ctx) for predicate in self.predicates)


@dataclass(frozen=True)
class Not(Predicate):
    predicate: Predicate

    def evaluate(self, state: WorldState, ctx: EvalContext) -> bool:
        return not self.predicate.evaluate(state, ctx)


def _as_pos(value: Any) -> Vec2 | None:
    if isinstance(value, Vec2):
        return value
    if isinstance(value, EntityState):
        return value.pos
    if isinstance(value, dict) and {"x", "y"} <= set(value):
        return Vec2.from_dict(value)
    if isinstance(value, tuple | list) and len(value) == 2:
        return Vec2(int(value[0]), int(value[1]))
    return None


def manhattan(a: Vec2, b: Vec2) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)
