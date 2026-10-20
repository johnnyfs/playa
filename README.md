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

Train `go(target)` online:

```bash
uv run python -m hrl_poc.training.train_skill --skill go
```

Short smoke run:

```bash
uv run python -m hrl_poc.training.train_skill --skill go --episodes 16 --batch-episodes 4 --eval-episodes 2
```

Other predefined skills:

```bash
uv run python -m hrl_poc.training.train_skill --skill attack
uv run python -m hrl_poc.training.train_skill --skill collect
uv run python -m hrl_poc.training.train_skill --skill go_evading
```

Train one shared policy over a fixed skill mix:

```bash
uv run python -m hrl_poc.training.train_multi_skill --skills go,collect,attack
```

Longer training sweep with matched-seed checkpoint evaluation:

```bash
uv run python -m hrl_poc.training.sweep_skill --skill go --episodes 1024 --compare-baseline
```

Direct checkpoint comparison:

```bash
uv run python -m hrl_poc.training.evaluate_skill --checkpoint runs/.../checkpoints/skill_policy_001024.pt --baseline-checkpoint runs/.../checkpoints/skill_policy_000000.pt --episodes 200 --json
```

Compare a trained checkpoint against random, untrained, and optional non-conditioned baselines:

```bash
uv run python -m hrl_poc.training.benchmark_skill --checkpoint runs/.../checkpoints/skill_policy_001024.pt --skill go --episodes 200 --nonconditioned-checkpoint runs/.../checkpoints/skill_policy_001024.pt
```

Sweep summary table:

```bash
uv run python -m hrl_poc.training.summarize_sweeps --run-dir runs/sweep_go --markdown
```

The summary table shows `seed`, `checkpoint_episodes`, `success_rate`, and `delta_vs_previous_checkpoint` within the same run. That delta is the right default comparison for a checkpoint curve; cross-seed comparison is a separate analysis.

Training prints JSON metrics with success rate, average episode length, and average reward. Checkpoints are written under `runs/<skill>_<timestamp>/checkpoints`.

`go(target)` also uses a small online auxiliary shortest-path action loss by default. It is there to make the Phase 1 learning demo reliable in a tiny repo; set `--go-aux-coef 0` to run only the PPO losses.

## Competence Ladder Harness

The competence ladder is a semi-automated validation harness for fixed low-level skills. It runs curriculum levels in order:

```text
Skill -> Ladder -> Level -> Train -> Evaluate -> Promote/Fail -> Report
```

The ladder harness validates skills before they are exposed to a high-level planner.
A skill is not considered available at full complexity until it passes held-out randomized evaluations.
Failures are logged in a form that a future agent can inspect to adjust curriculum, model architecture, or skill definitions.

The first fully wired ladder is `go(target)`. The harness can generate deterministic curriculum worlds from `WorldSpec`, including empty rooms, border walls, and a single-wall detour terrain. Each held-out episode records the semantic state, target, trajectory, reward total, failure reason, blocked movement diagnostics, final distance, stuck detection, and timeout progress.

For this ladder, `go` is treated as a local navigation instruction: the simulator still executes against an absolute target position, but the policy conditioning is driven by target delta, distance, and cross-screen flag rather than raw world coordinates. That keeps the skill definition local while still allowing adjacent-screen tasks.

Run the `go` ladder:

```bash
uv run ladder --config ladders/go.yaml
```

Equivalent module form:

```bash
uv run python -m hrl_poc.ladder.runner --config ladders/go.yaml
```

For a quick smoke run:

```bash
uv run ladder --config ladders/go.yaml --max-eval-seeds 2
```

Promotion is config-driven. Each level declares `eval.heldout_seeds` and `eval.pass_rate`; the runner evaluates held-out seeds and promotes only when the observed pass rate meets or exceeds the configured threshold. By default, the run stops at the first failed promotion. Use `--continue-on-failure` to keep evaluating later levels.

Reports are written under:

```text
reports/
  go/
    run_YYYYMMDD_HHMMSS/
      config.yaml
      level_results.json
      summary.md
      checkpoints/
      failures/
      trajectories/
```

The current competence frontier is written to:

```text
competence/go.json
```

When a level passes, the latest released checkpoint is copied into `competence/go_released.*` and a manifest is written to `competence/go_released.json`.

Each run also writes `reports/go/run_YYYYMMDD_HHMMSS/summary.html`, a self-contained HTML view with per-level progress curves and validation state.

The runner currently provides `HeuristicGoTrainer` and `RandomPolicyTrainer` implementations behind a small trainer interface. The heuristic trainer is a sanity check that proves the ladder plumbing works; PPO can replace it later without changing ladder configs, report formats, or competence frontier semantics.

An opt-in PPO trainer is also available:

```bash
uv run ladder --config ladders/go.yaml --trainer ppo
```

This trains a real `SkillPolicy` checkpoint for each level using that level's generated curriculum episodes and `train.steps`, then evaluates the saved checkpoint on held-out seeds. `go(target)` conditioning includes both absolute target coordinates and relative target offset. PPO also uses a small shortest-path auxiliary action loss for `go`, which makes the first ladder level learn reliably in this compact environment. When a level passes, the next compatible PPO level initializes from the latest passed checkpoint; incompatible model shapes are recorded as skipped initialization in the checkpoint metadata. The default remains the heuristic trainer because it is the fastest way to smoke-test curriculum generation and reporting.

For focused PPO iteration on one level:

```bash
uv run ladder --config ladders/go.yaml --trainer ppo --only-level go_l0_one_step_empty --max-eval-seeds 4
```

PPO checkpoints include compact training metrics (`steps_done`, `episodes_done`, update count, train success rate, average reward, and average episode length). These are copied into `level_results.json`, `summary.md`, and `competence/go.json`, and the latest passed checkpoint is published to `competence/go_released.*`.

## Execution Logs

Every evaluated episode writes a JSONL record:

```json
{
  "skill_call": {"skill_name": "go", "params": {"target": {"x": 5, "y": 3}}},
  "initial_summary": {},
  "final_summary": {},
  "success": true,
  "failure_reason": null,
  "reward_total": 1.2,
  "steps": 8,
  "trajectory_path": "runs/go_.../eval/trajectories/episode_000000.json"
}
```

These records are intentionally compact. They are meant to become the future bridge to offline RL, skill evaluation datasets, or LLM-generated analysis.

## Future LLM Extension Points

- Planner: propose a `SkillCall`, such as `Pickup(item_type=POTION, target=Vec2(...))`.
- Test authoring: add or modify predicate names in a `SkillSpec`.
- Evaluator: read execution records and trajectories, then explain failures.
- Curriculum: generate distributions over skill calls based on recent success rates.
- Competence gating: read `competence/<skill>.json` and expose only skills/levels that passed held-out ladder evaluation.

The current code keeps those boundaries explicit: schema in `hrl_poc/skills/schema.py`, executable predicates in `hrl_poc/skills/predicates.py`, skill-to-episode conversion in `hrl_poc/envs/wrappers.py`, and execution records in `hrl_poc/training/logging.py`.
