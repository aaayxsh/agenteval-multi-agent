from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    def load_dotenv() -> bool:
        return False

from agenteval.dataset import load_eval_cases
from agenteval.evaluator import evaluate_dataset
from agenteval.llm_client import OptionalOpenAIClient


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    st.set_page_config(page_title="AgentEval Dashboard", page_icon="📊", layout="wide")
    st.title("📊 AgentEval: Multi-Agent Utility Evaluation")
    st.caption("Evaluate the usefulness of LLM-powered application outputs with multi-agent scoring.")

    with st.sidebar:
        input_path = st.text_input("Dataset Path", value=str(BASE_DIR / "data" / "sample_eval_dataset.jsonl"))
        use_llm = st.toggle("Use OpenAI commentary if configured", value=True)

    cases = load_eval_cases(input_path)
    llm_client = OptionalOpenAIClient(enabled=use_llm)
    report = evaluate_dataset(cases, llm_client=llm_client, system_prompt_path=BASE_DIR / "prompts" / "judge_system_prompt.md")

    summary = report["summary"]
    st.subheader("Evaluation Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Cases", summary["case_count"])
    summary_cols[1].metric("Average Score", f"{summary['average_overall_score']:.1f}")
    summary_cols[2].metric("Pass Rate", f"{summary['pass_rate']:.0%}")
    summary_cols[3].metric("Top Utility Band", summary["most_common_band"])

    st.subheader("Case Results")
    results_df = pd.DataFrame(report["results"])
    st.dataframe(
        results_df[["case_id", "application_name", "use_case", "overall_score", "utility_band", "recommendation"]],
        use_container_width=True,
    )
    st.bar_chart(results_df.set_index("case_id")["overall_score"])

    selected_case = st.selectbox("Inspect Case", options=results_df["case_id"].tolist())
    selected = next(item for item in report["results"] if item["case_id"] == selected_case)

    st.subheader(f"Detailed Review: {selected_case}")
    st.markdown(f"**Application:** {selected['application_name']}")
    st.markdown(f"**Use Case:** {selected['use_case']}")
    st.markdown(f"**Utility Band:** {selected['utility_band']}")
    st.markdown(f"**Recommendation:** {selected['recommendation']}")

    st.markdown("**Strengths**")
    for item in selected["strengths"]:
        st.write(f"- {item}")

    st.markdown("**Weaknesses**")
    for item in selected["weaknesses"]:
        st.write(f"- {item}")

    st.markdown("**Agent Scores**")
    agent_scores_df = pd.DataFrame(selected["agent_scores"])
    st.dataframe(agent_scores_df, use_container_width=True)

    if selected.get("llm_commentary"):
        st.markdown("**OpenAI Commentary**")
        st.markdown(selected["llm_commentary"])


if __name__ == "__main__":
    main()
