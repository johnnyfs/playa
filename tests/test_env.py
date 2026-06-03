from hrl_poc.envs.gridworld import Action, GridWorldEnv


def test_gridworld_reset_and_step():
    env = GridWorldEnv()
    obs, info = env.reset(
        seed=1,
        options={
            "player": (0, 0),
            "walls": [],
            "enemies": [],
            "items": [],
            "health": 5,
        },
    )

    assert obs["terrain"].shape == (12, 12)
    assert obs["player"].tolist() == [0, 0]
    assert info["health"] == 5

    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)

    assert obs["player"].tolist() == [1, 0]
    assert reward < 0
    assert not terminated
    assert not truncated
    assert info["player"] == {"x": 1, "y": 0}


def test_gridworld_attack_kills_adjacent_enemy():
    env = GridWorldEnv()
    env.reset(
        seed=1,
        options={
            "player": (1, 1),
            "walls": [],
            "enemies": [{"type": 0, "pos": (2, 1), "hp": 1}],
            "items": [],
            "health": 5,
        },
    )

    _obs, reward, terminated, _truncated, info = env.step(Action.ATTACK)

    assert reward > 0
    assert not terminated
    assert info["alive_enemies"] == []
