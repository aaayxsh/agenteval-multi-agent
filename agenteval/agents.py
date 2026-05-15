from __future__ import annotations

from dataclasses import dataclass

from .metrics import (
    score_clarity,
    score_completeness,
    score_goal_alignment,
    score_groundedness,
    score_safety,
)
from .schemas import AgentScore, EvalCase


@dataclass
class JudgeAgent:
    agent_name: str
    dimension: str

    def evaluate(self, case: EvalCase) -> AgentScore:
        raise NotImplementedError


class GoalAlignmentAgent(JudgeAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="Goal Alignment Agent", dimension="goal_alignment")

    def evaluate(self, case: EvalCase) -> AgentScore:
        score, evidence = score_goal_alignment(case.system_output, case.user_prompt, case.reference_points)
        return AgentScore(
            agent_name=self.agent_name,
            dimension=self.dimension,
            score=score,
            rationale="Measures whether the response addresses the user request and core reference concepts.",
            evidence=evidence,
        )


class CompletenessAgent(JudgeAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="Completeness Agent", dimension="completeness")

    def evaluate(self, case: EvalCase) -> AgentScore:
        score, evidence = score_completeness(case.system_output, case.expected_criteria)
        return AgentScore(
            agent_name=self.agent_name,
            dimension=self.dimension,
            score=score,
            rationale="Checks whether expected requirements are covered with enough detail.",
            evidence=evidence,
        )


class GroundednessAgent(JudgeAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="Groundedness Agent", dimension="groundedness")

    def evaluate(self, case: EvalCase) -> AgentScore:
        score, evidence = score_groundedness(case.system_output, case.reference_points)
        return AgentScore(
            agent_name=self.agent_name,
            dimension=self.dimension,
            score=score,
            rationale="Rewards overlap with trusted reference points and penalizes exaggerated claims.",
            evidence=evidence,
        )


class ClarityAgent(JudgeAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="Clarity Agent", dimension="clarity")

    def evaluate(self, case: EvalCase) -> AgentScore:
        score, evidence = score_clarity(case.system_output)
        return AgentScore(
            agent_name=self.agent_name,
            dimension=self.dimension,
            score=score,
            rationale="Estimates readability, structure, and actionability.",
            evidence=evidence,
        )


class SafetyAgent(JudgeAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="Safety Agent", dimension="safety")

    def evaluate(self, case: EvalCase) -> AgentScore:
        score, evidence = score_safety(case.system_output, case.safety_flags)
        return AgentScore(
            agent_name=self.agent_name,
            dimension=self.dimension,
            score=score,
            rationale="Checks for risky patterns based on the case safety flags.",
            evidence=evidence,
        )

