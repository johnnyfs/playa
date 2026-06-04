from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class SweepRow:
    run_dir: str
    skill: str
    seed: int
    checkpoint: str
    checkpoint_episodes: int
    eval_episodes: int
    successes: int
    success_rate: float
    ci_low: float
    ci_high: float
    delta_vs_previous: float | None
    avg_reward: float
    avg_length: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "skill": self.skill,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "checkpoint_episodes": self.checkpoint_episodes,
            "eval_episodes": self.eval_episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "delta_vs_previous": self.delta_vs_previous,
            "avg_reward": self.avg_reward,
            "avg_length": self.avg_length,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sweep evaluation rows into a comparison table.")
    parser.add_argument("--run-dir", action="append", default=[], help="Sweep run directory. Can be repeated.")
    parser.add_argument("--glob", dest="glob_pattern", default="runs/*/sweep_summary.json", help="Glob for auto-discovery.")
    parser.add_argument("--markdown", action="store_true", help="Print a markdown table.")
    parser.add_argument("--csv", dest="csv_path", default="", help="Optional CSV output path.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of a table.")
    return parser.parse_args()


def load_rows(run_dirs: list[str], glob_pattern: str) -> list[SweepRow]:
    summary_files: list[Path] = []
    for run_dir in run_dirs:
        summary_files.append(Path(run_dir) / "sweep_summary.json")
    if not summary_files:
        summary_files = sorted(Path().glob(glob_pattern))

    rows: list[SweepRow] = []
    for summary_path in summary_files:
        if not summary_path.exists():
            continue
        run_dir = summary_path.parent
        entries = json.loads(summary_path.read_text())
        previous_rate_by_run: float | None = None
        for entry in sorted(entries, key=_checkpoint_episodes_from_path):
            checkpoint_path = Path(entry["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            seed = int(checkpoint["args"]["seed"])
            checkpoint_episodes = int(checkpoint["episodes"])
            eval_episodes = int(entry["episodes"])
            successes = int(entry["successes"])
            success_rate = float(entry["success_rate"])
            se = math.sqrt(success_rate * (1.0 - success_rate) / eval_episodes) if 0.0 < success_rate < 1.0 else 0.0
            delta_vs_previous = None if previous_rate_by_run is None else success_rate - previous_rate_by_run
            previous_rate_by_run = success_rate
            rows.append(
                SweepRow(
                    run_dir=str(run_dir),
                    skill=str(entry["skill"]),
                    seed=seed,
                    checkpoint=str(checkpoint_path),
                    checkpoint_episodes=checkpoint_episodes,
                    eval_episodes=eval_episodes,
                    successes=successes,
                    success_rate=success_rate,
                    ci_low=max(0.0, success_rate - 1.96 * se),
                    ci_high=min(1.0, success_rate + 1.96 * se),
                    delta_vs_previous=delta_vs_previous,
                    avg_reward=float(entry["avg_reward"]),
                    avg_length=float(entry["avg_length"]),
                )
            )
    return rows


def _checkpoint_episodes_from_path(row: dict[str, Any]) -> int:
    checkpoint = Path(row["checkpoint"]).stem
    try:
        return int(checkpoint.split("_")[-1])
    except ValueError:
        return 0


def render_markdown(rows: list[SweepRow]) -> str:
    header = [
        "run_dir",
        "seed",
        "checkpoint_episodes",
        "eval_episodes",
        "successes",
        "success_rate",
        "ci95",
        "delta_vs_previous",
        "avg_reward",
        "avg_length",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        delta = "" if row.delta_vs_previous is None else f"{row.delta_vs_previous:+.4f}"
        ci = f"[{row.ci_low:.4f}, {row.ci_high:.4f}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    row.run_dir,
                    str(row.seed),
                    str(row.checkpoint_episodes),
                    str(row.eval_episodes),
                    f"{row.successes}/{row.eval_episodes}",
                    f"{row.success_rate:.4f}",
                    ci,
                    delta,
                    f"{row.avg_reward:.4f}",
                    f"{row.avg_length:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(rows: list[SweepRow], path: str) -> None:
    fieldnames = list(rows[0].to_dict().keys()) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_dir, args.glob_pattern)
    if args.csv_path:
        write_csv(rows, args.csv_path)
    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2))
    elif args.markdown or not args.csv_path:
        print(render_markdown(rows))


if __name__ == "__main__":
    main()
