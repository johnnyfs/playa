# HRL Skill Learning POC

Phase 1 is a small Python prototype for parameterized skill learning in a simple hierarchical RL harness. It does not include an LLM yet. The point is to prove that skills can be defined as typed calls, executed as testable episodes, trained with a shared skill-conditioned low-level policy, and logged for later offline analysis.

## What Phase 1 Proves

- A Gymnasium-compatible `GridWorldEnv` exposes structured state, not pixels.
- Skills are represented as serializable `SkillSpec` and `SkillCall` objects.
- Skill success, failure, timeout, and shaping are executable predicates.
- `SkillConditionedEnv` turns a `SkillCall` into a single RL episode.
- A shared PyTorch actor-critic policy learns `pi(action | state, skill_name, skill_params)`.
- `train_skill.py` runs an online PPO-style training loop and saves checkpoints.
- Evaluation episodes write compact execution records and minimal trajectories.

## What Is Not Included Yet

- No learned high-level skill chooser.
- No LLM planner, evaluator, or skill authoring loop.
- No offline RL.
- No replay buffer or dataset curation.
- No PufferLib dependency. The environment and wrapper use Gymnasium spaces and a flat policy input path, so adding a PufferLib adapter later should be straightforward.

## Setup

This repo is managed with `uv`.

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```

## Train A Skill

Train `Navigate(goal)` online:

```bash
uv run python -m hrl_poc.training.train_skill --skill navigate
```

Short smoke run:

```bash
uv run python -m hrl_poc.training.train_skill --skill navigate --episodes 16 --batch-episodes 4 --eval-episodes 2
```

Other predefined skills:

```bash
uv run python -m hrl_poc.training.train_skill --skill fight
uv run python -m hrl_poc.training.train_skill --skill pickup
uv run python -m hrl_poc.training.train_skill --skill avoid
```

Training prints JSON metrics with success rate, average episode length, and average reward. Checkpoints are written under `runs/<skill>_<timestamp>/checkpoints`.

`Navigate(goal)` also uses a small online auxiliary shortest-path action loss by default. It is there to make the Phase 1 learning demo reliable in a tiny repo; set `--navigate-aux-coef 0` to run only the PPO losses.

## Execution Logs

Every evaluated episode writes a JSONL record:

```json
{
  "skill_call": {"name": "navigate", "params": {"goal": {"x": 5, "y": 3}}},
  "initial_summary": {},
  "final_summary": {},
  "success": true,
  "failure_reason": null,
  "reward_total": 1.2,
  "steps": 8,
  "trajectory_path": "runs/navigate_.../eval/trajectories/episode_000000.json"
}
```

These records are intentionally compact. They are meant to become the future bridge to offline RL, skill evaluation datasets, or LLM-generated analysis.

## Future LLM Extension Points

- Planner: propose a `SkillCall`, such as `Pickup(item_type=POTION, target=Vec2(...))`.
- Test authoring: add or modify predicate names in a `SkillSpec`.
- Evaluator: read execution records and trajectories, then explain failures.
- Curriculum: generate distributions over skill calls based on recent success rates.

The current code keeps those boundaries explicit: schema in `hrl_poc/skills/schema.py`, executable predicates in `hrl_poc/skills/predicates.py`, skill-to-episode conversion in `hrl_poc/envs/wrappers.py`, and execution records in `hrl_poc/training/logging.py`.
