from __future__ import annotations

from hrl_poc.skills.schema import EnemyType, ItemType, SkillCall, SkillSpec, Vec2


SKILL_REGISTRY: dict[str, SkillSpec] = {
    "navigate": SkillSpec(
        name="navigate",
        skill_id=0,
        parameter_schema={"goal": "Vec2"},
        success_predicates=["at_goal"],
        failure_predicates=["player_dead", "timeout"],
        shaping_metrics=["progress_to_goal"],
        max_steps=48,
    ),
    "avoid": SkillSpec(
        name="avoid",
        skill_id=1,
        parameter_schema={"goal": "Vec2"},
        success_predicates=["at_goal_and_safe"],
        failure_predicates=["player_dead", "timeout"],
        shaping_metrics=["progress_to_goal", "distance_from_enemies"],
        max_steps=56,
    ),
    "fight": SkillSpec(
        name="fight",
        skill_id=2,
        parameter_schema={"enemy_type": "EnemyType", "target": "Vec2"},
        success_predicates=["target_enemy_dead"],
        failure_predicates=["player_dead", "timeout", "target_enemy_missing"],
        shaping_metrics=["progress_to_target", "target_enemy_damage"],
        max_steps=36,
    ),
    "pickup": SkillSpec(
        name="pickup",
        skill_id=3,
        parameter_schema={"item_type": "ItemType", "target": "Vec2"},
        success_predicates=["target_item_collected"],
        failure_predicates=["player_dead", "timeout", "target_item_missing"],
        shaping_metrics=["progress_to_target"],
        max_steps=48,
    ),
}


def get_skill_spec(name: str) -> SkillSpec:
    try:
        return SKILL_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(SKILL_REGISTRY))
        raise ValueError(f"Unknown skill {name!r}. Known skills: {known}") from exc


def make_skill_call(name: str, **params: object) -> SkillCall:
    if name in {"navigate", "avoid"}:
        goal = params.get("goal")
        if isinstance(goal, tuple):
            params["goal"] = Vec2(int(goal[0]), int(goal[1]))
    elif name == "fight":
        enemy_type = params.get("enemy_type", EnemyType.GRUNT)
        params["enemy_type"] = EnemyType(enemy_type)
        target = params.get("target")
        if isinstance(target, tuple):
            params["target"] = Vec2(int(target[0]), int(target[1]))
    elif name == "pickup":
        item_type = params.get("item_type", ItemType.COIN)
        params["item_type"] = ItemType(item_type)
        target = params.get("target")
        if isinstance(target, tuple):
            params["target"] = Vec2(int(target[0]), int(target[1]))
    return SkillCall(name=name, params=dict(params))
