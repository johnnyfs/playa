from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from hrl_poc.ladder.config import LadderConfig, LadderLevel, load_ladder_config
from hrl_poc.ladder.trainers import LevelEvalResult, make_trainer


def run_ladder(
    config_path: str | Path,
    reports_dir: str | Path = "reports",
    competence_dir: str | Path = "competence",
    trainer_name: str = "heuristic",
    continue_on_failure: bool = False,
    seed_base: int = 100_000,
    max_eval_seeds: int | None = None,
    only_level: str | None = None,
) -> Path:
    config = load_ladder_config(config_path)
    if config.skill != "go":
        raise ValueError("The current ladder harness only fully wires the go(target) skill")
    levels = tuple(level for level in config.levels if only_level is None or level.id == only_level)
    if only_level is not None and not levels:
        raise ValueError(f"Unknown ladder level {only_level!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(reports_dir) / config.skill / f"run_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    failure_dir = run_dir / "failures"
    trajectory_dir = run_dir / "trajectories"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yaml")

    trainer = make_trainer(trainer_name)
    level_results: list[LevelEvalResult] = []
    frontier = _initial_frontier(config)
    latest_passed_checkpoint = None

    for level_idx, level in enumerate(levels):
        active_level = _with_max_eval_seeds(level, max_eval_seeds)
        frontier["levels"][active_level.id]["status"] = "training"
        checkpoint = trainer.train_level(active_level, checkpoint_dir, initial_checkpoint=latest_passed_checkpoint)
        result = trainer.evaluate_level(
            checkpoint=checkpoint,
            level=active_level,
            seed_base=seed_base + level_idx * 10_000,
            trajectory_dir=trajectory_dir,
        )
        level_results.append(result)
        frontier["levels"][active_level.id] = {
            "status": "passed" if result.passed else "failed",
            "pass_rate": result.pass_rate,
            "required_pass_rate": result.required_pass_rate,
            "trainer": checkpoint.trainer,
            "checkpoint": str(checkpoint.path),
            "initialization": checkpoint.metadata.get("initialization"),
            "train_metrics": checkpoint.metadata.get("train_metrics", {}).get("final"),
        }
        if result.passed:
            latest_passed_checkpoint = checkpoint
            frontier["latest_passed_level"] = active_level.id
        else:
            _write_failure_report(failure_dir, result)
            if not continue_on_failure:
                break

    _write_level_results(run_dir, level_results)
    _write_summary(run_dir, config, level_results)
    _write_frontier(competence_dir, config.skill, frontier)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a competence ladder for a fixed low-level skill.")
    parser.add_argument("--config", required=True, help="Path to ladder YAML config.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--competence-dir", default="competence")
    parser.add_argument("--trainer", choices=["heuristic", "heuristic_go", "ppo", "ppo_go", "random"], default="heuristic")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--seed-base", type=int, default=100_000)
    parser.add_argument("--max-eval-seeds", type=int, default=0, help="Optional cap for smoke runs.")
    parser.add_argument("--only-level", default="", help="Run only one level id for focused tuning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_ladder(
        config_path=args.config,
        reports_dir=args.reports_dir,
        competence_dir=args.competence_dir,
        trainer_name=args.trainer,
        continue_on_failure=args.continue_on_failure,
        seed_base=args.seed_base,
        max_eval_seeds=args.max_eval_seeds or None,
        only_level=args.only_level or None,
    )
    print(json.dumps({"run_dir": str(run_dir)}))


def _initial_frontier(config: LadderConfig) -> dict[str, Any]:
    return {
        "skill": config.skill,
        "latest_passed_level": None,
        "levels": {
            level.id: {
                "status": "untried",
                "pass_rate": None,
                "required_pass_rate": level.eval.pass_rate,
            }
            for level in config.levels
        },
    }


def _with_max_eval_seeds(level: LadderLevel, max_eval_seeds: int | None) -> LadderLevel:
    if max_eval_seeds is None:
        return level
    return replace(level, eval=replace(level.eval, heldout_seeds=min(level.eval.heldout_seeds, max_eval_seeds)))


def _write_level_results(run_dir: Path, results: list[LevelEvalResult]) -> None:
    payload = [result.to_dict() for result in results]
    (run_dir / "level_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_failure_report(failure_dir: Path, result: LevelEvalResult) -> None:
    timeout_count = sum(1 for failure in result.failures if failure.failure_reason == "timeout")
    aggregate = {
        "level_id": result.level_id,
        "pass_rate": result.pass_rate,
        "required_pass_rate": result.required_pass_rate,
        "failures": len(result.failures),
        "episodes": result.episodes,
        "timeout_rate": timeout_count / result.episodes if result.episodes else 0.0,
        "average_progress_before_timeout_px": _mean(
            failure.diagnostics.get("average_progress_before_timeout_px")
            for failure in result.failures
            if failure.diagnostics.get("average_progress_before_timeout_px") is not None
        ),
        "hit_obstacle_count": sum(1 for failure in result.failures if failure.diagnostics.get("hit_obstacle")),
        "stuck_count": sum(1 for failure in result.failures if failure.diagnostics.get("stuck")),
    }
    payload = {
        "aggregate": aggregate,
        "failures": [failure.diagnostics for failure in result.failures],
    }
    (failure_dir / f"{result.level_id}_failures.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_summary(run_dir: Path, config: LadderConfig, results: list[LevelEvalResult]) -> None:
    lines = [
        f"# Competence Ladder Run: {config.skill}",
        "",
        "| Level | Trainer | Pass Rate | Required | Status | Train Steps | Init |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for result in results:
        status = "passed" if result.passed else "failed"
        metadata = result.checkpoint.metadata
        initialization = metadata.get("initialization") or {}
        init_status = str(initialization.get("status", "n/a"))
        steps_done = int(metadata.get("steps_done", 0))
        lines.append(
            f"| {result.level_id} | {result.checkpoint.trainer} | {result.pass_rate:.3f} | "
            f"{result.required_pass_rate:.3f} | {status} | {steps_done} | {init_status} |"
        )
    if not results:
        lines.append("| none | 0.000 | 0.000 | untried |")
    lines.extend(
        [
            "",
            "Failures are written under `failures/`; per-episode trajectories are under `trajectories/`.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_frontier(competence_dir: str | Path, skill: str, frontier: dict[str, Any]) -> None:
    path = Path(competence_dir) / f"{skill}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")


def _mean(values: Any) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


if __name__ == "__main__":
    main()
