from __future__ import annotations

from dataclasses import dataclass

from hrl_poc.semantic.state import WorldState
from hrl_poc.skills.refs import EvalContext
from hrl_poc.skills.schema import SkillCall, SkillSpec


@dataclass(frozen=True)
class ProgressRule:
    metric: object
    direction: str = "decrease"
    weight: float = 1.0


@dataclass(frozen=True)
class SkillEvalResult:
    reward: float
    success: bool
    failure: bool
    done: bool
    failure_reason: str | None


class SkillEvaluator:
    def evaluate(
        self,
        prev_state: WorldState,
        next_state: WorldState,
        skill_call: SkillCall,
        skill_spec: SkillSpec,
    ) -> SkillEvalResult:
        ctx = EvalContext(prev_state=prev_state, next_state=next_state, skill_call=skill_call, max_steps=skill_spec.max_steps)
        success = any(predicate.evaluate(next_state, ctx) for predicate in skill_spec.success_predicates)
        failure_reason = None
        failure = False
        if not success:
            for predicate in skill_spec.failure_predicates:
                if predicate.evaluate(next_state, ctx):
                    failure = True
                    failure_reason = type(predicate).__name__
                    break

        reward = 0.0
        if success:
            reward += 1.0
        elif failure:
            reward -= 1.0
        else:
            reward += self._progress_reward(prev_state, next_state, ctx, skill_spec.progress_rules)
        return SkillEvalResult(
            reward=float(max(-1.0, min(1.0, reward))),
            success=success,
            failure=failure,
            done=success or failure,
            failure_reason=failure_reason,
        )

    def _progress_reward(self, prev_state: WorldState, next_state: WorldState, ctx: EvalContext, rules: list[ProgressRule]) -> float:
        reward = -0.02
        for rule in rules:
            prev_value = rule.metric.value(prev_state, ctx)
            next_value = rule.metric.value(next_state, ctx)
            delta = next_value - prev_value
            signed = -delta if rule.direction == "decrease" else delta
            reward += rule.weight * max(-1.0, min(1.0, signed))
        return float(max(-0.25, min(0.25, reward)))
