from pathlib import Path

import json
import torch

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY
from hrl_poc.training.summarize_sweeps import load_rows, render_markdown


def _checkpoint(tmp_path: Path, episodes: int, seed: int) -> Path:
    env = SkillConditionedEnv(GridWorldEnv(), skill_name="go")
    policy = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY))
    path = tmp_path / "checkpoints" / f"skill_policy_{episodes:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "skill": "go",
            "episodes": episodes,
            "args": {"seed": seed},
        },
        path,
    )
    return path


def test_load_rows_and_render_markdown(tmp_path: Path):
    ckpt1 = _checkpoint(tmp_path, episodes=16, seed=11)
    ckpt2 = _checkpoint(tmp_path, episodes=32, seed=11)
    summary_path = tmp_path / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            [
                {
                    "checkpoint": str(ckpt1),
                    "skill": "go",
                    "episodes": 4,
                    "successes": 1,
                    "success_rate": 0.25,
                    "success_ci95_low": 0.0,
                    "success_ci95_high": 0.5,
                    "avg_length": 10.0,
                    "avg_reward": 1.0,
                    "paired_delta_vs_baseline": None,
                    "paired_successes_vs_baseline": None,
                },
                {
                    "checkpoint": str(ckpt2),
                    "skill": "go",
                    "episodes": 4,
                    "successes": 2,
                    "success_rate": 0.5,
                    "success_ci95_low": 0.0,
                    "success_ci95_high": 0.75,
                    "avg_length": 9.0,
                    "avg_reward": 1.5,
                    "paired_delta_vs_baseline": 0.25,
                    "paired_successes_vs_baseline": 1,
                    "baseline_checkpoint": str(ckpt1),
                    "baseline_success_rate": 0.25,
                    "baseline_success_ci95_low": 0.0,
                    "baseline_success_ci95_high": 0.5,
                    "delta_success_rate": 0.25,
                    "delta_successes": 1,
                    "baseline_skill": "go",
                },
            ]
        )
    )

    rows = load_rows([str(tmp_path)], "")
    assert len(rows) == 2
    assert rows[0].seed == 11
    assert rows[0].delta_vs_previous is None
    assert rows[1].delta_vs_previous == 0.25

    markdown = render_markdown(rows)
    assert "checkpoint_episodes" in markdown
    assert "delta_vs_previous" in markdown
