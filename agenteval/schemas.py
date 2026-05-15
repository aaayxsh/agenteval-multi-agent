from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str
    application_name: str
    use_case: str
    user_prompt: str
    system_output: str
    reference_points: List[str]
    expected_criteria: List[str]
    safety_flags: List[str] = Field(default_factory=list)


class AgentScore(BaseModel):
    agent_name: str
    dimension: str
    score: float = Field(ge=0, le=100)
    rationale: str
    evidence: List[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    case_id: str
    application_name: str
    use_case: str
    overall_score: float = Field(ge=0, le=100)
    utility_band: str
    recommendation: str
    strengths: List[str]
    weaknesses: List[str]
    agent_scores: List[AgentScore]
    llm_commentary: Optional[str] = None

