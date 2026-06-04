from __future__ import annotations

from typing import Any

from hrl_poc.semantic.state import Vec2
from hrl_poc.skills.evaluator import ProgressRule
from hrl_poc.skills.metrics import ContactFlag, Distance, EnemyProximity, Health, InventoryCount
from hrl_poc.skills.predicates import Alive, And, Dead, EnemyContact, HasInventoryItem, Near, Timeout
from hrl_poc.skills.refs import EntityByIdRef, EntityPosRef, ParamRef, PlayerRef
from hrl_poc.skills.schema import SkillCall, SkillSpec


PLAYER = PlayerRef()


SKILL_REGISTRY: dict[str, SkillSpec] = {
    "go": SkillSpec(
        name="go",
        params_schema={"target": Vec2},
        success_predicates=[Near(PLAYER, ParamRef("target"), radius=0)],
        failure_predicates=[Timeout(), Dead(PLAYER)],
        progress_rules=[ProgressRule(Distance(PLAYER, ParamRef("target")), direction="decrease", weight=0.18)],
        max_steps=48,
    ),
    "go_evading": SkillSpec(
        name="go_evading",
        params_schema={"target": Vec2, "mode": "evading"},
        success_predicates=[Near(PLAYER, ParamRef("target"), radius=0)],
        failure_predicates=[EnemyContact(radius=1), Timeout(), Dead(PLAYER)],
        progress_rules=[
            ProgressRule(Distance(PLAYER, ParamRef("target")), direction="decrease", weight=0.16),
            ProgressRule(EnemyProximity(PLAYER), direction="decrease", weight=0.08),
            ProgressRule(ContactFlag(), direction="decrease", weight=0.2),
        ],
        max_steps=56,
    ),
    "attack": SkillSpec(
        name="attack",
        params_schema={"enemy_id": str},
        success_predicates=[And(Dead(EntityByIdRef(ParamRef("enemy_id"))), Alive(PLAYER))],
        failure_predicates=[Timeout(), Dead(PLAYER)],
        progress_rules=[
            ProgressRule(Health(EntityByIdRef(ParamRef("enemy_id"))), direction="decrease", weight=0.35),
            ProgressRule(Health(PLAYER), direction="increase", weight=0.15),
            ProgressRule(Distance(PLAYER, EntityPosRef(EntityByIdRef(ParamRef("enemy_id")))), direction="decrease", weight=0.12),
        ],
        max_steps=40,
    ),
    "collect": SkillSpec(
        name="collect",
        params_schema={"item_id": str},
        success_predicates=[HasInventoryItem(ParamRef("item_id"))],
        failure_predicates=[Timeout(), Dead(PLAYER)],
        progress_rules=[
            ProgressRule(Distance(PLAYER, EntityPosRef(EntityByIdRef(ParamRef("item_id")))), direction="decrease", weight=0.16),
            ProgressRule(InventoryCount(ParamRef("item_id")), direction="increase", weight=0.25),
        ],
        max_steps=48,
    ),
}


SKILL_IDS: dict[str, int] = {name: idx for idx, name in enumerate(SKILL_REGISTRY)}

ALIASES = {
    "navigate": "go",
    "avoid": "go_evading",
    "fight": "attack",
    "pickup": "collect",
}


def canonical_skill_name(name: str, params: dict[str, Any] | None = None) -> str:
    if name == "go" and params and params.get("mode") == "evading":
        return "go_evading"
    return ALIASES.get(name, name)


def get_skill_spec(name: str, params: dict[str, Any] | None = None) -> SkillSpec:
    canonical = canonical_skill_name(name, params)
    try:
        return SKILL_REGISTRY[canonical]
    except KeyError as exc:
        known = ", ".join(sorted(SKILL_REGISTRY))
        raise ValueError(f"Unknown skill {name!r}. Known skills: {known}") from exc


def get_skill_id(name: str, params: dict[str, Any] | None = None) -> int:
    return SKILL_IDS[get_skill_spec(name, params).name]


def make_skill_call(name: str, **params: object) -> SkillCall:
    canonical = canonical_skill_name(name, params)
    if "goal" in params and "target" not in params:
        params["target"] = params.pop("goal")
    target = params.get("target")
    if isinstance(target, tuple):
        params["target"] = Vec2(int(target[0]), int(target[1]))
    if canonical == "go_evading":
        params.setdefault("mode", "evading")
    return SkillCall(skill_name=canonical, params=dict(params))
