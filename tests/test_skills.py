from hrl_poc.envs.gridworld import Action, GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.skills.predicates import PredicateContext, target_enemy_dead
from hrl_poc.skills.registry import get_skill_spec
from hrl_poc.skills.schema import EnemyType, SkillCall, SkillSpec, Vec2


def test_skill_spec_and_call_serialization_round_trip():
    spec = get_skill_spec("fight")
    spec_round_trip = SkillSpec.from_dict(spec.to_dict())
    assert spec_round_trip == spec

    call = SkillCall("fight", {"enemy_type": EnemyType.GRUNT, "target": Vec2(2, 1)})
    call_round_trip = SkillCall.from_dict(call.to_dict())
    assert call_round_trip == call


def test_fight_predicate_succeeds_when_target_enemy_dead():
    call = SkillCall("fight", {"enemy_type": EnemyType.GRUNT, "target": Vec2(2, 1)})
    initial_state = {
        "player": (1, 1),
        "health": 5,
        "enemies": [{"type": EnemyType.GRUNT, "pos": (2, 1), "hp": 1, "alive": True}],
        "items": [],
    }
    final_state = {
        "player": (1, 1),
        "health": 5,
        "enemies": [{"type": EnemyType.GRUNT, "pos": (2, 1), "hp": 0, "alive": False}],
        "items": [],
    }
    ctx = PredicateContext(call, initial_state, initial_state, final_state, elapsed_steps=1, max_steps=10)

    assert target_enemy_dead(ctx)


def test_skill_wrapper_terminates_on_navigate_success_and_rewards_it():
    env = SkillConditionedEnv(GridWorldEnv())
    call = SkillCall("navigate", {"goal": Vec2(1, 0)})
    obs, _info = env.reset(
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

    assert obs["skill_id"].tolist() == [0]
    assert obs["skill_params"][0] > 0

    _obs, reward, terminated, truncated, info = env.step(Action.RIGHT)

    assert terminated
    assert not truncated
    assert reward > 0.9
    assert info["success"]
    assert info["failure_reason"] is None


def test_skill_wrapper_assigns_negative_reward_when_moving_away():
    env = SkillConditionedEnv(GridWorldEnv())
    call = SkillCall("navigate", {"goal": Vec2(0, 0)})
    env.reset(
        seed=1,
        options={
            "skill_call": call,
            "env_options": {
                "player": (1, 0),
                "walls": [],
                "enemies": [],
                "items": [],
                "health": 5,
            },
        },
    )

    _obs, reward, terminated, _truncated, info = env.step(Action.RIGHT)

    assert not terminated
    assert reward < -0.05
    assert not info["success"]


def test_skill_wrapper_fight_success_record():
    env = SkillConditionedEnv(GridWorldEnv())
    call = SkillCall("fight", {"enemy_type": EnemyType.GRUNT, "target": Vec2(2, 1)})
    env.reset(
        seed=1,
        options={
            "skill_call": call,
            "env_options": {
                "player": (1, 1),
                "walls": [],
                "enemies": [{"type": EnemyType.GRUNT, "pos": (2, 1), "hp": 1}],
                "items": [],
                "health": 5,
            },
        },
    )

    _obs, _reward, terminated, _truncated, info = env.step(Action.ATTACK)
    record = env.make_execution_record("trajectory.json")

    assert terminated
    assert info["success"]
    assert record["success"]
    assert record["skill_call"] == call.to_dict()
    assert record["trajectory_path"] == "trajectory.json"
