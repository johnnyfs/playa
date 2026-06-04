from hrl_poc.envs.gridworld import GridWorldEnv
from hrl_poc.envs.wrappers import SkillConditionedEnv
from hrl_poc.policies.skill_policy import SkillPolicy
from hrl_poc.skills.registry import SKILL_REGISTRY


def test_skill_conditioned_and_nonconditioned_policy_forward():
    env = SkillConditionedEnv(GridWorldEnv(), skill_name="go")
    obs, _info = env.reset(seed=1, options={"skill_name": "go"})

    conditioned = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), condition_on_skill=True)
    nonconditioned = SkillPolicy(env.action_space, num_skills=len(SKILL_REGISTRY), condition_on_skill=False)

    conditioned_logits, conditioned_values = conditioned([obs])
    nonconditioned_logits, nonconditioned_values = nonconditioned([obs])

    assert conditioned_logits.shape == nonconditioned_logits.shape == (1, env.action_space.n)
    assert conditioned_values.shape == nonconditioned_values.shape == (1,)
