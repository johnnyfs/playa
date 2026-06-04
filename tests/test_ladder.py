from __future__ import annotations

import json
from pathlib import Path

import yaml

from hrl_poc.ladder.config import load_ladder_config
from hrl_poc.ladder.episode import SkillEpisode
from hrl_poc.ladder.runner import run_ladder
from hrl_poc.ladder.trainers import HeuristicGoPolicy
from hrl_poc.ladder.world import (
    TerrainSpec,
    WorldSpec,
    generate_go_task,
    generate_walls,
    generate_world,
    path_exists,
)
from hrl_poc.semantic.state import Vec2
from hrl_poc.skills.schema import SkillCall


def test_ladder_config_parses():
    config = load_ladder_config("ladders/go.yaml")

    assert config.skill == "go"
    assert len(config.levels) >= 5
    assert config.levels[0].world.terrain == TerrainSpec.EMPTY


def test_world_spec_produces_deterministic_worlds_by_seed():
    spec = WorldSpec(terrain=TerrainSpec.SINGLE_WALL)
    world_a, target_a = generate_go_task(spec, {"target_distance_px": 64}, seed=123)
    world_b, target_b = generate_go_task(spec, {"target_distance_px": 64}, seed=123)

    assert world_a.walls == world_b.walls
    assert world_a.player == world_b.player
    assert target_a == target_b
    assert world_a.env().world_state().to_dict() == world_b.env().world_state().to_dict()


def test_generated_go_target_respects_distance_constraints():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    world, target = generate_go_task(spec, {"target_distance_px": 64}, seed=7)

    distance_px = (abs(world.player[0] - target.x) + abs(world.player[1] - target.y)) * spec.tile_size_px
    assert distance_px == 64


def test_generated_target_is_walkable():
    spec = WorldSpec(terrain=TerrainSpec.BORDER_WALLS)
    world, target = generate_go_task(spec, {"target_distance_px": 64}, seed=8)

    assert (target.x, target.y) not in set(world.walls)


def test_bfs_path_validation_works_on_empty_room():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    walls = generate_walls(spec, __import__("numpy").random.default_rng(1))

    assert path_exists(spec, walls, (0, 0), (3, 3))


def test_bfs_path_validation_fails_through_solid_wall():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    walls = {(2, y) for y in range(spec.world_height_tiles)}

    assert not path_exists(spec, walls, (0, 1), (4, 1))


def test_go_success_triggers_at_target():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    world = generate_world(spec, seed=1, player=(1, 1))
    episode = SkillEpisode(
        level_id="tiny",
        world=world,
        skill_call=SkillCall("go", {"target": Vec2(1, 1)}),
        max_steps=10,
        seed=1,
    )

    result = episode.run(HeuristicGoPolicy())

    assert result.success
    assert not result.failure
    assert result.steps == 0


def test_go_timeout_fails_episode():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    world = generate_world(spec, seed=1, player=(0, 0))
    episode = SkillEpisode(
        level_id="timeout",
        world=world,
        skill_call=SkillCall("go", {"target": Vec2(5, 5)}),
        max_steps=0,
        seed=1,
    )

    result = episode.run(HeuristicGoPolicy())

    assert result.failure
    assert result.failure_reason == "timeout"


def test_episode_result_serializes_to_json():
    spec = WorldSpec(terrain=TerrainSpec.EMPTY)
    world = generate_world(spec, seed=1, player=(1, 1))
    episode = SkillEpisode("json", world, SkillCall("go", {"target": Vec2(1, 1)}), max_steps=1, seed=1)

    payload = json.loads(episode.run(HeuristicGoPolicy()).to_json())

    assert payload["success"] is True
    assert payload["skill_call"]["params"]["target"] == {"x": 1, "y": 1}


def test_ladder_runner_runs_tiny_level_with_heuristic_policy(tmp_path: Path):
    config_path = _write_tiny_ladder(tmp_path, max_steps=5, pass_rate=1.0)

    run_dir = run_ladder(
        config_path,
        reports_dir=tmp_path / "reports",
        competence_dir=tmp_path / "competence",
        max_eval_seeds=3,
    )

    results = json.loads((run_dir / "level_results.json").read_text())
    assert results[0]["passed"] is True
    assert (run_dir / "summary.md").exists()


def test_competence_frontier_file_is_written(tmp_path: Path):
    config_path = _write_tiny_ladder(tmp_path, max_steps=5, pass_rate=1.0)

    run_ladder(config_path, reports_dir=tmp_path / "reports", competence_dir=tmp_path / "competence")

    frontier = json.loads((tmp_path / "competence" / "go.json").read_text())
    assert frontier["skill"] == "go"
    assert frontier["levels"]["tiny_go"]["status"] == "passed"


def test_failure_report_is_written_when_pass_rate_below_threshold(tmp_path: Path):
    config_path = _write_tiny_ladder(tmp_path, max_steps=0, pass_rate=1.0)

    run_dir = run_ladder(config_path, reports_dir=tmp_path / "reports", competence_dir=tmp_path / "competence")

    failure_report = run_dir / "failures" / "tiny_go_failures.json"
    payload = json.loads(failure_report.read_text())
    assert payload["aggregate"]["level_id"] == "tiny_go"
    assert payload["aggregate"]["failures"] > 0


def test_ladder_runner_can_use_ppo_trainer_for_tiny_level(tmp_path: Path):
    config_path = _write_tiny_ladder(tmp_path, max_steps=5, pass_rate=0.0, train_steps=4)

    run_dir = run_ladder(
        config_path,
        reports_dir=tmp_path / "reports",
        competence_dir=tmp_path / "competence",
        trainer_name="ppo",
        max_eval_seeds=1,
    )

    checkpoints = list((run_dir / "checkpoints").glob("*.pt"))
    results = json.loads((run_dir / "level_results.json").read_text())
    assert checkpoints
    assert results[0]["checkpoint"]["trainer"] == "ppo_go"


def test_ppo_trainer_initializes_next_level_from_previous_checkpoint(tmp_path: Path):
    config_path = _write_two_level_tiny_ladder(tmp_path)

    run_dir = run_ladder(
        config_path,
        reports_dir=tmp_path / "reports",
        competence_dir=tmp_path / "competence",
        trainer_name="ppo",
        max_eval_seeds=1,
    )

    results = json.loads((run_dir / "level_results.json").read_text())
    second_init = results[1]["checkpoint"]["metadata"]["initialization"]
    assert second_init["status"] == "loaded"
    assert second_init["source_level_id"] == "tiny_go_l0"


def test_ladder_runner_only_level_runs_selected_level(tmp_path: Path):
    config_path = _write_two_level_tiny_ladder(tmp_path)

    run_dir = run_ladder(
        config_path,
        reports_dir=tmp_path / "reports",
        competence_dir=tmp_path / "competence",
        only_level="tiny_go_l1",
    )

    results = json.loads((run_dir / "level_results.json").read_text())
    assert [result["level_id"] for result in results] == ["tiny_go_l1"]


def _write_tiny_ladder(tmp_path: Path, max_steps: int, pass_rate: float, train_steps: int = 1) -> Path:
    config_path = tmp_path / "tiny_go.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "skill": "go",
                "levels": [
                    {
                        "id": "tiny_go",
                        "description": "tiny go",
                        "world": {"terrain": "Empty"},
                        "task": {"target_distance_px": 16, "max_steps": max_steps},
                        "train": {"seeds": 1, "steps": train_steps},
                        "eval": {"heldout_seeds": 2, "pass_rate": pass_rate},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_two_level_tiny_ladder(tmp_path: Path) -> Path:
    config_path = tmp_path / "two_level_tiny_go.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "skill": "go",
                "levels": [
                    {
                        "id": "tiny_go_l0",
                        "description": "tiny go l0",
                        "world": {"terrain": "Empty"},
                        "task": {"target_distance_px": 16, "max_steps": 5},
                        "train": {"seeds": 1, "steps": 4},
                        "eval": {"heldout_seeds": 1, "pass_rate": 0.0},
                    },
                    {
                        "id": "tiny_go_l1",
                        "description": "tiny go l1",
                        "world": {"terrain": "Empty"},
                        "task": {"target_distance_px": 16, "max_steps": 5},
                        "train": {"seeds": 1, "steps": 4},
                        "eval": {"heldout_seeds": 1, "pass_rate": 0.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return config_path
