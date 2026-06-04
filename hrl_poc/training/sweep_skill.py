from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a longer training budget and evaluate checkpoints.")
    parser.add_argument("--skill", choices=["go", "go_evading", "attack", "collect"], default="go")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--go-aux-coef", "--navigate-aux-coef", dest="navigate_aux_coef", type=float, default=2.0)
    parser.add_argument("--compare-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir or f"runs/sweep_{args.skill}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "-m",
        "hrl_poc.training.train_skill",
        "--skill",
        args.skill,
        "--episodes",
        str(args.episodes),
        "--batch-episodes",
        str(args.batch_episodes),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-every",
        str(args.eval_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--seed",
        str(args.seed),
        "--run-dir",
        str(run_dir),
        "--go-aux-coef",
        str(args.navigate_aux_coef),
    ]
    subprocess.run(train_cmd, check=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("skill_policy_*.pt"))
    eval_summary = []
    baseline = checkpoints[0] if args.compare_baseline and checkpoints else None
    for checkpoint in checkpoints:
        eval_cmd = [
            sys.executable,
            "-m",
            "hrl_poc.training.evaluate_skill",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            str(max(args.eval_episodes, 64)),
            "--seed",
            str(args.seed + 50_000),
            "--json",
        ]
        if baseline is not None and checkpoint != baseline:
            eval_cmd.extend(["--baseline-checkpoint", str(baseline)])
        completed = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        eval_summary.append(payload)
        print(json.dumps(payload))

    (run_dir / "sweep_summary.json").write_text(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
