from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from hrl_poc.skills.schema import EnemyType, ItemType, SkillCall, Vec2


@dataclass(frozen=True)
class PredicateContext:
    skill_call: SkillCall
    initial_state: dict[str, Any]
    previous_state: dict[str, Any]
    state: dict[str, Any]
    elapsed_steps: int
    max_steps: int


SuccessPredicate = Callable[[PredicateContext], bool]
FailurePredicate = Callable[[PredicateContext], bool]
ShapingMetric = Callable[[PredicateContext], float]


def at_goal(ctx: PredicateContext) -> bool:
    goal = _vec_param(ctx.skill_call, "goal")
    return tuple(ctx.state["player"]) == (goal.x, goal.y) and ctx.state["health"] > 0


def at_goal_and_safe(ctx: PredicateContext) -> bool:
    return at_goal(ctx) and _nearest_enemy_distance(ctx.state) > 1


def target_enemy_dead(ctx: PredicateContext) -> bool:
    target = _vec_param(ctx.skill_call, "target")
    enemy_type = EnemyType(ctx.skill_call.params["enemy_type"])
    if not _enemy_exists(ctx.initial_state, target, enemy_type):
        return False
    return not _enemy_exists(ctx.state, target, enemy_type, alive_only=True) and ctx.state["health"] > 0


def target_item_collected(ctx: PredicateContext) -> bool:
    target = _vec_param(ctx.skill_call, "target")
    item_type = ItemType(ctx.skill_call.params["item_type"])
    if not _item_exists(ctx.initial_state, target, item_type):
        return False
    return not _item_exists(ctx.state, target, item_type, present_only=True) and ctx.state["health"] > 0


def player_dead(ctx: PredicateContext) -> bool:
    return ctx.state["health"] <= 0


def timeout(ctx: PredicateContext) -> bool:
    return ctx.elapsed_steps >= ctx.max_steps


def target_enemy_missing(ctx: PredicateContext) -> bool:
    target = _vec_param(ctx.skill_call, "target")
    enemy_type = EnemyType(ctx.skill_call.params["enemy_type"])
    return not _enemy_exists(ctx.initial_state, target, enemy_type)


def target_item_missing(ctx: PredicateContext) -> bool:
    target = _vec_param(ctx.skill_call, "target")
    item_type = ItemType(ctx.skill_call.params["item_type"])
    return not _item_exists(ctx.initial_state, target, item_type)


def progress_to_goal(ctx: PredicateContext) -> float:
    goal = _vec_param(ctx.skill_call, "goal")
    return float(_distance(ctx.previous_state["player"], goal) - _distance(ctx.state["player"], goal))


def progress_to_target(ctx: PredicateContext) -> float:
    target = _vec_param(ctx.skill_call, "target")
    return float(_distance(ctx.previous_state["player"], target) - _distance(ctx.state["player"], target))


def distance_from_enemies(ctx: PredicateContext) -> float:
    prev = _nearest_enemy_distance(ctx.previous_state)
    current = _nearest_enemy_distance(ctx.state)
    return 0.1 * float(current - prev)


def target_enemy_damage(ctx: PredicateContext) -> float:
    target = _vec_param(ctx.skill_call, "target")
    enemy_type = EnemyType(ctx.skill_call.params["enemy_type"])
    prev_hp = _enemy_hp(ctx.previous_state, target, enemy_type)
    current_hp = _enemy_hp(ctx.state, target, enemy_type)
    return float(max(0, prev_hp - current_hp))


SUCCESS_PREDICATES: dict[str, SuccessPredicate] = {
    "at_goal": at_goal,
    "at_goal_and_safe": at_goal_and_safe,
    "target_enemy_dead": target_enemy_dead,
    "target_item_collected": target_item_collected,
}

FAILURE_PREDICATES: dict[str, FailurePredicate] = {
    "player_dead": player_dead,
    "timeout": timeout,
    "target_enemy_missing": target_enemy_missing,
    "target_item_missing": target_item_missing,
}

SHAPING_METRICS: dict[str, ShapingMetric] = {
    "progress_to_goal": progress_to_goal,
    "progress_to_target": progress_to_target,
    "distance_from_enemies": distance_from_enemies,
    "target_enemy_damage": target_enemy_damage,
}


def evaluate_success(ctx: PredicateContext, names: list[str]) -> tuple[bool, str | None]:
    for name in names:
        if SUCCESS_PREDICATES[name](ctx):
            return True, name
    return False, None


def evaluate_failure(ctx: PredicateContext, names: list[str]) -> tuple[bool, str | None]:
    for name in names:
        if FAILURE_PREDICATES[name](ctx):
            return True, name
    return False, None


def evaluate_shaping(ctx: PredicateContext, names: list[str]) -> float:
    return sum(SHAPING_METRICS[name](ctx) for name in names)


def _vec_param(skill_call: SkillCall, name: str) -> Vec2:
    value = skill_call.params[name]
    if isinstance(value, Vec2):
        return value
    if isinstance(value, dict):
        return Vec2.from_dict(value)
    return Vec2(int(value[0]), int(value[1]))


def _distance(pos: tuple[int, int], target: Vec2) -> int:
    return abs(int(pos[0]) - target.x) + abs(int(pos[1]) - target.y)


def _nearest_enemy_distance(state: dict[str, Any]) -> int:
    alive = [enemy for enemy in state["enemies"] if enemy["alive"]]
    if not alive:
        return 99
    px, py = state["player"]
    return min(abs(px - int(enemy["pos"][0])) + abs(py - int(enemy["pos"][1])) for enemy in alive)


def _enemy_exists(state: dict[str, Any], target: Vec2, enemy_type: EnemyType, alive_only: bool = False) -> bool:
    return any(
        enemy["type"] == enemy_type
        and tuple(int(v) for v in enemy["pos"]) == (target.x, target.y)
        and (not alive_only or enemy["alive"])
        for enemy in state["enemies"]
    )


def _enemy_hp(state: dict[str, Any], target: Vec2, enemy_type: EnemyType) -> int:
    for enemy in state["enemies"]:
        if enemy["type"] == enemy_type and tuple(int(v) for v in enemy["pos"]) == (target.x, target.y):
            return int(enemy["hp"]) if enemy["alive"] else 0
    return 0


def _item_exists(state: dict[str, Any], target: Vec2, item_type: ItemType, present_only: bool = False) -> bool:
    return any(
        item["type"] == item_type
        and tuple(int(v) for v in item["pos"]) == (target.x, target.y)
        and (not present_only or item["present"])
        for item in state["items"]
    )
