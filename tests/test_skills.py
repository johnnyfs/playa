from hrl_poc.envs.gridworld import Action, GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.semantic.state import EntityState, Vec2, WorldState
from hrl_poc.skills.evaluator import SkillEvaluator
from hrl_poc.skills.predicates import Alive, Dead, EnemyContact, Near, Timeout
from hrl_poc.skills.refs import EntityByIdRef, ParamRef, PlayerRef
from hrl_poc.skills.registry import get_skill_spec
from hrl_poc.skills.schema import SkillCall


def test_semantic_predicates_resolve_refs():
    state = WorldState(
        tick=0,
        global_position=Vec2(1, 1),
        local_position=Vec2(1, 1),
        region_id="gridworld",
        player_id="player",
        entities={
            "player": EntityState("player", "player", "player", Vec2(1, 1), hp=5),
            "enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(2, 1), alive=False, hp=0, hostile=True),
        },
        tiles={},
        inventory={},
        attrs={"skill_steps": 1},
    )
    call = SkillCall("attack", {"enemy_id": "enemy_0"})
    spec = get_skill_spec("attack")
    ctx = __import__("hrl_poc.skills.refs", fromlist=["EvalContext"]).EvalContext(state, state, call, spec.max_steps)

    assert Near(PlayerRef(), Vec2(1, 1)).evaluate(state, ctx)
    assert Alive(PlayerRef()).evaluate(state, ctx)
    assert Dead(EntityByIdRef(ParamRef("enemy_id"))).evaluate(state, ctx)
    assert not Timeout().evaluate(state, ctx)


def test_gridworld_exposes_world_state_in_info():
    env = GridWorldEnv()
    _obs, info = env.reset(
        seed=1,
        options={
            "player": (0, 0),
            "walls": [],
            "enemies": [{"id": "enemy_0", "type": 0, "pos": (2, 1), "hp": 1}],
            "items": [{"id": "item_0", "type": 0, "pos": (1, 0)}],
            "health": 5,
        },
    )

    world = info["world_state"]
    assert world.player_id == "player"
    assert world.entities["enemy_0"].hostile
    assert world.entities["item_0"].kind == "item"
    assert world.tiles[(0, 0)].walkable


def test_skill_evaluator_go_success():
    env = SkillConditionedEnv(GridWorldEnv())
    call = SkillCall("go", {"target": Vec2(1, 0)})
    _obs, _info = env.reset(
        seed=1,
        options={
            "skill_call": call,
            "env_options": {
                "player": (0, 0),
                "walls": [],
                "enemies": [],
                "items": [],
                "health": 5,
            },
        },
    )

    _obs, reward, terminated, _truncated, info = env.step(Action.RIGHT)

    assert terminated
    assert reward == 1.0
    assert info["success"]
    assert not info["failure"]


def test_skill_evaluator_attack_success():
    env = SkillConditionedEnv(GridWorldEnv())
    call = SkillCall("attack", {"enemy_id": "enemy_0"})
    env.reset(
        seed=1,
        options={
            "skill_call": call,
            "env_options": {
                "player": (1, 1),
                "walls": [],
                "enemies": [{"id": "enemy_0", "type": 0, "pos": (2, 1), "hp": 1}],
                "items": [],
                "health": 5,
            },
        },
    )

    _obs, reward, terminated, _truncated, info = env.step(Action.ATTACK)
    record = env.make_execution_record("trajectory.json")

    assert terminated
    assert reward == 1.0
    assert info["success"]
    assert record["success"]
    assert record["skill_call"] == call.to_dict()


def test_skill_evaluator_evading_fails_on_contact():
    prev = WorldState(
        tick=0,
        global_position=Vec2(0, 0),
        local_position=Vec2(0, 0),
        region_id="gridworld",
        player_id="player",
        entities={
            "player": EntityState("player", "player", "player", Vec2(0, 0), hp=5),
            "enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(4, 4), hp=1, hostile=True),
        },
        tiles={},
        inventory={},
    )
    nxt = WorldState(
        tick=1,
        global_position=Vec2(0, 0),
        local_position=Vec2(0, 0),
        region_id="gridworld",
        player_id="player",
        entities={
            "player": EntityState("player", "player", "player", Vec2(0, 0), hp=5),
            "enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(1, 0), hp=1, hostile=True),
        },
        tiles={},
        inventory={},
        attrs={"skill_steps": 1},
    )
    call = SkillCall("go_evading", {"target": Vec2(3, 3), "mode": "evading"})
    result = SkillEvaluator().evaluate(prev, nxt, call, get_skill_spec("go_evading"))

    assert EnemyContact().evaluate(nxt, __import__("hrl_poc.skills.refs", fromlist=["EvalContext"]).EvalContext(prev, nxt, call, 10))
    assert result.failure
    assert result.done
    assert result.reward == -1.0


def test_go_shaping_rewards_progress_and_penalizes_regress():
    spec = get_skill_spec("go")
    call = SkillCall("go", {"target": Vec2(4, 0)})
    prev = _world(player=Vec2(0, 0), entities={})
    closer = _world(player=Vec2(1, 0), entities={}, skill_steps=1)
    farther = _world(player=Vec2(0, 1), entities={}, skill_steps=1)

    progress = SkillEvaluator().evaluate(prev, closer, call, spec)
    regress = SkillEvaluator().evaluate(prev, farther, call, spec)

    assert progress.reward > 0.1
    assert regress.reward < -0.1


def test_attack_shaping_rewards_damage_and_penalizes_player_damage():
    spec = get_skill_spec("attack")
    call = SkillCall("attack", {"enemy_id": "enemy_0"})
    prev = _world(
        player=Vec2(1, 1),
        player_hp=5,
        entities={"enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(2, 1), hp=2, hostile=True)},
    )
    damaged_enemy = _world(
        player=Vec2(1, 1),
        player_hp=5,
        entities={"enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(2, 1), hp=1, hostile=True)},
        skill_steps=1,
    )
    damaged_player = _world(
        player=Vec2(1, 1),
        player_hp=4,
        entities={"enemy_0": EntityState("enemy_0", "grunt", "enemy", Vec2(2, 1), hp=2, hostile=True)},
        skill_steps=1,
    )

    hit = SkillEvaluator().evaluate(prev, damaged_enemy, call, spec)
    hurt = SkillEvaluator().evaluate(prev, damaged_player, call, spec)

    assert hit.reward > 0.2
    assert hurt.reward < -0.1


def _world(
    player: Vec2,
    entities: dict[str, EntityState],
    player_hp: int = 5,
    inventory: dict[str, int] | None = None,
    skill_steps: int = 0,
) -> WorldState:
    all_entities = {"player": EntityState("player", "player", "player", player, hp=player_hp, alive=player_hp > 0)}
    all_entities.update(entities)
    return WorldState(
        tick=skill_steps,
        global_position=player,
        local_position=player,
        region_id="gridworld",
        player_id="player",
        entities=all_entities,
        tiles={},
        inventory=inventory or {},
        attrs={"skill_steps": skill_steps},
    )
