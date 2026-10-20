from __future__ import annotations

import argparse
import html
import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from hrl_poc.ladder.config import LadderConfig, LadderLevel, load_ladder_config
from hrl_poc.ladder.trainers import LevelEvalResult, make_trainer
from hrl_poc.ladder.world import validate_go_task


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
    released_artifact = None

    for level_idx, level in enumerate(levels):
        active_level = _with_max_eval_seeds(level, max_eval_seeds)
        validation = _validate_level(active_level, seed_base + level_idx * 10_000)
        frontier["levels"][active_level.id]["status"] = "training"
        frontier["levels"][active_level.id]["validation"] = validation["summary"]
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
            "validation": validation["summary"],
        }
        if result.passed:
            latest_passed_checkpoint = checkpoint
            frontier["latest_passed_level"] = active_level.id
            released_artifact = _write_released_checkpoint(
                competence_dir,
                config.skill,
                checkpoint,
                level_id=active_level.id,
                pass_rate=result.pass_rate,
                required_pass_rate=result.required_pass_rate,
            )
        else:
            _write_failure_report(failure_dir, result)
            if not continue_on_failure:
                break

    if released_artifact is not None:
        frontier["released_checkpoint"] = released_artifact

    _write_level_results(run_dir, level_results)
    _write_summary(run_dir, config, level_results)
    _write_html_report(run_dir, config, level_results, frontier)
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
        "released_checkpoint": None,
        "levels": {
            level.id: {
                "status": "untried",
                "pass_rate": None,
                "required_pass_rate": level.eval.pass_rate,
                "validation": None,
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
            "Released checkpoints are copied into `competence/` when a level passes.",
            "Open `summary.html` for a visual view of training progress and frontier state.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html_report(run_dir: Path, config: LadderConfig, results: list[LevelEvalResult], frontier: dict[str, Any]) -> None:
    rows = []
    for result in results:
        metadata = result.checkpoint.metadata
        init = metadata.get("initialization") or {}
        train_metrics = metadata.get("train_metrics") or {}
        history = train_metrics.get("history", [])
        validation = frontier["levels"][result.level_id].get("validation") or {}
        rows.append(
            {
                "level_id": result.level_id,
                "trainer": result.checkpoint.trainer,
                "status": "passed" if result.passed else "failed",
                "pass_rate": result.pass_rate,
                "required_pass_rate": result.required_pass_rate,
                "steps_done": int(metadata.get("steps_done", 0)),
                "init_status": str(init.get("status", "n/a")),
                "validation_possible": bool(validation.get("possible", False)),
                "checkpoint": str(result.checkpoint.path),
                "released": frontier.get("released_checkpoint"),
                "history": history,
                "final_metrics": train_metrics.get("final", {}),
            }
        )

    released = frontier.get("released_checkpoint") or {}
    latest_passed = frontier.get("latest_passed_level")
    sections = []
    for row in rows:
        history = row["history"]
        success_values = [float(item.get("success_rate", 0.0)) for item in history]
        reward_values = [float(item.get("avg_reward", 0.0)) for item in history]
        sections.append(
            f"""
            <section class="level">
              <div class="level-head">
                <div>
                  <h2>{html.escape(row["level_id"])}</h2>
                  <p>{html.escape(row["trainer"])} · {html.escape(row["status"])} · validation: {"ok" if row["validation_possible"] else "failed"}</p>
                </div>
                <div class="badge">{row["pass_rate"]:.3f} / {row["required_pass_rate"]:.3f}</div>
              </div>
              <div class="meta-grid">
                <div><span>checkpoint</span><code>{html.escape(row["checkpoint"])}</code></div>
                <div><span>steps</span><code>{row["steps_done"]}</code></div>
                <div><span>init</span><code>{html.escape(row["init_status"])}</code></div>
                <div><span>train success</span><code>{float(row["final_metrics"].get("success_rate", 0.0)):.3f}</code></div>
              </div>
              <div class="charts">
                <div class="chart">
                  <div class="chart-label">train success rate</div>
                  {_sparkline_svg(success_values, "#2563eb")}
                </div>
                <div class="chart">
                  <div class="chart-label">train reward</div>
                  {_sparkline_svg(reward_values, "#16a34a")}
                </div>
              </div>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Competence Ladder - {html.escape(config.skill)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f5;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --accent: #2563eb;
      --good: #16a34a;
      --bad: #dc2626;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #fafaf9 0%, #f3f4f6 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      padding: 20px 22px;
      margin-bottom: 20px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    .hero h1 {{ font-size: 28px; margin-bottom: 8px; }}
    .hero p {{ color: var(--muted); line-height: 1.5; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px 14px;
      background: #fcfcfb;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 6px;
    }}
    .stat strong {{
      font-size: 16px;
      font-weight: 650;
    }}
    .level {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px;
      margin-top: 16px;
      box-shadow: var(--shadow);
    }}
    .level-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .level-head h2 {{ font-size: 20px; margin-bottom: 4px; }}
    .level-head p {{ color: var(--muted); font-size: 14px; }}
    .badge {{
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 8px 12px;
      font-variant-numeric: tabular-nums;
      background: #f9fafb;
      white-space: nowrap;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .meta-grid div {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fcfcfb;
      overflow: hidden;
    }}
    .meta-grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .chart {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px 12px;
      background: #ffffff;
    }}
    .chart-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }}
    .chart svg {{
      width: 100%;
      height: 72px;
      display: block;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .frontier {{
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Competence Ladder: {html.escape(config.skill)}</h1>
      <p>Latest passed level: {html.escape(str(latest_passed or "none"))}. Released checkpoint: {html.escape(str(released.get("released_checkpoint", "none")) if released else "none")}.</p>
      <div class="summary-grid">
        <div class="stat"><span>levels run</span><strong>{len(results)}</strong></div>
        <div class="stat"><span>latest passed</span><strong>{html.escape(str(latest_passed or "none"))}</strong></div>
        <div class="stat"><span>released checkpoint</span><strong>{html.escape(str(released.get("released_checkpoint", "none")) if released else "none")}</strong></div>
        <div class="stat"><span>skill</span><strong>{html.escape(config.skill)}</strong></div>
      </div>
    </section>
    <section class="frontier">
      <h2>Frontier</h2>
      <p class="empty">The JSON frontier is also written to <code>competence/{html.escape(config.skill)}.json</code>.</p>
    </section>
    {''.join(sections) if sections else '<p class="empty">No levels were run.</p>'}
  </main>
</body>
</html>
"""
    (run_dir / "summary.html").write_text(html_doc, encoding="utf-8")


def _sparkline_svg(values: list[float], color: str) -> str:
    width = 220
    height = 64
    padding = 6
    if not values:
        return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="empty chart"><text x="10" y="34" fill="#9ca3af" font-size="12">no data</text></svg>'
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span == 0:
        span = 1.0
    points = []
    for idx, value in enumerate(values):
        x = padding + (width - 2 * padding) * (idx / max(1, len(values) - 1))
        y = height - padding - ((value - minimum) / span) * (height - 2 * padding)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="training curve">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#e5e7eb" stroke-width="1" />'
        f"</svg>"
    )


def _write_frontier(competence_dir: str | Path, skill: str, frontier: dict[str, Any]) -> None:
    path = Path(competence_dir) / f"{skill}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")


def _write_released_checkpoint(
    competence_dir: str | Path,
    skill: str,
    checkpoint: Any,
    level_id: str,
    pass_rate: float,
    required_pass_rate: float,
) -> dict[str, Any]:
    competence_path = Path(competence_dir)
    competence_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(checkpoint.path)
    suffix = source_path.suffix or ".ckpt"
    released_path = competence_path / f"{skill}_released{suffix}"
    shutil.copy2(source_path, released_path)
    manifest = {
        "skill": skill,
        "level_id": level_id,
        "source_checkpoint": str(source_path),
        "released_checkpoint": str(released_path),
        "trainer": checkpoint.trainer,
        "pass_rate": pass_rate,
        "required_pass_rate": required_pass_rate,
        "metadata": dict(checkpoint.metadata),
    }
    (competence_path / f"{skill}_released.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _mean(values: Any) -> float | None:
    nums = [float(value) for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _validate_level(level: LadderLevel, seed_base: int, samples: int = 4) -> dict[str, Any]:
    validations = [validate_go_task(level.world, level.task, seed_base + idx) for idx in range(samples)]
    summary = {
        "possible": all(validation.possible for validation in validations),
        "samples": [validation.to_dict() for validation in validations],
    }
    if not summary["possible"]:
        details = "; ".join(
            f"seed={validation.seed} error={validation.error or 'none'} possible={validation.possible}"
            for validation in validations
        )
        raise ValueError(f"Level {level.id} is not solvable as configured: {details}")
    return {"summary": summary}


if __name__ == "__main__":
    main()
