from pathlib import Path

import torch

from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY
from hrl_poc.training.evaluate_skill import compare_checkpoints, evaluate_checkpoint


def _save_checkpoint(tmp_path: Path, skill: str = "go") -> Path:
    env = SkillConditionedEnv(GridWorldEnv(), skill_name=skill)
    policy = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY))
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "skill": skill,
            "episodes": 0,
            "args": {"seed": 7},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_evaluate_checkpoint_returns_serializable_metrics(tmp_path: Path):
    checkpoint = _save_checkpoint(tmp_path)
    result = evaluate_checkpoint(checkpoint, episodes=4, seed=123)

    payload = result.to_dict()
    assert payload["checkpoint"] == str(checkpoint)
    assert payload["episodes"] == 4
    assert 0.0 <= payload["success_rate"] <= 1.0
    assert payload["success_ci95_low"] <= payload["success_rate"] <= payload["success_ci95_high"]


def test_compare_checkpoints_uses_matched_seeds(tmp_path: Path):
    baseline = _save_checkpoint(tmp_path, skill="go")
    candidate = _save_checkpoint(tmp_path, skill="go")

    payload = compare_checkpoints(baseline, candidate, episodes=8, seed=456)

    assert payload["baseline_checkpoint"] == str(baseline)
    assert payload["delta_success_rate"] == 0.0
    assert payload["delta_successes"] == 0
