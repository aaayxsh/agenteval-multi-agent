from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional

from .agents import ClarityAgent, CompletenessAgent, GoalAlignmentAgent, GroundednessAgent, SafetyAgent
from .schemas import EvalCase, EvaluationResult


def _utility_band(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Needs Improvement"
    return "High Risk"


def _recommendation(score: float) -> str:
    if score >= 85:
        return "Ship with confidence and monitor for regression."
    if score >= 70:
        return "Good candidate for release after prompt and coverage polish."
    if score >= 55:
        return "Revise prompt design, content coverage, and safety review before release."
    return "Do not ship without substantial correction and retesting."


def evaluate_case(
    case: EvalCase,
    llm_client: Optional[object] = None,
    system_prompt_path: Optional[str | Path] = None,
) -> EvaluationResult:
    agents = [
        GoalAlignmentAgent(),
        CompletenessAgent(),
        GroundednessAgent(),
        ClarityAgent(),
        SafetyAgent(),
    ]
    agent_scores = [agent.evaluate(case) for agent in agents]

    weighted_scores = {
        "goal_alignment": 0.26,
        "completeness": 0.23,
        "groundedness": 0.21,
        "clarity": 0.15,
        "safety": 0.15,
    }
    overall_score = round(
        sum(item.score * weighted_scores[item.dimension] for item in agent_scores),
        2,
    )
    utility_band = _utility_band(overall_score)
    recommendation = _recommendation(overall_score)

    strong_dimensions = [item.dimension for item in agent_scores if item.score >= 75]
    weak_dimensions = [item.dimension for item in agent_scores if item.score < 65]

    strengths = [f"Strong {dimension.replace('_', ' ')} performance." for dimension in strong_dimensions] or [
        "No standout strengths detected from the current rubric."
    ]
    weaknesses = [f"Improve {dimension.replace('_', ' ')} coverage." for dimension in weak_dimensions] or [
        "No major weaknesses surfaced from the current rubric."
    ]

    llm_commentary = None
    if llm_client and getattr(llm_client, "is_available", lambda: False)():
        system_prompt = ""
        if system_prompt_path and Path(system_prompt_path).exists():
            system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
        user_prompt = (
            f"Case ID: {case.case_id}\n"
            f"Application: {case.application_name}\n"
            f"Use case: {case.use_case}\n"
            f"User prompt: {case.user_prompt}\n"
            f"System output: {case.system_output}\n"
            f"Agent scores: {[item.model_dump() for item in agent_scores]}\n"
            "Write a short evaluation memo with a verdict, the biggest risk, and the single best improvement."
        )
        llm_commentary = llm_client.generate_markdown(system_prompt, user_prompt)

    return EvaluationResult(
        case_id=case.case_id,
        application_name=case.application_name,
        use_case=case.use_case,
        overall_score=overall_score,
        utility_band=utility_band,
        recommendation=recommendation,
        strengths=strengths,
        weaknesses=weaknesses,
        agent_scores=agent_scores,
        llm_commentary=llm_commentary,
    )


def evaluate_dataset(
    cases: Iterable[EvalCase],
    llm_client: Optional[object] = None,
    system_prompt_path: Optional[str | Path] = None,
) -> dict[str, object]:
    results = [
        evaluate_case(case, llm_client=llm_client, system_prompt_path=system_prompt_path).model_dump()
        for case in cases
    ]

    scores = [item["overall_score"] for item in results]
    bands = [item["utility_band"] for item in results]
    summary = {
        "case_count": len(results),
        "average_overall_score": round(mean(scores), 2) if scores else 0.0,
        "pass_rate": (sum(1 for score in scores if score >= 70) / len(scores)) if scores else 0.0,
        "most_common_band": Counter(bands).most_common(1)[0][0] if bands else "N/A",
    }
    return {"summary": summary, "results": results}

