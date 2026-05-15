# AgentEval: Multi-Agent Utility Evaluation

`AgentEval` is a full evaluation project for assessing the utility of LLM-powered applications using multiple judge agents.

It includes:

- A multi-agent evaluation pipeline
- A Streamlit dashboard
- A CLI for batch evaluation
- Sample benchmark cases in JSONL
- Optional OpenAI enhancement for richer judge commentary

## Project Structure

```text
agenteval_multi_agent/
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   └── sample_eval_dataset.jsonl
├── prompts/
│   └── judge_system_prompt.md
├── reports/
└── agenteval/
    ├── __init__.py
    ├── agents.py
    ├── cli.py
    ├── dataset.py
    ├── evaluator.py
    ├── llm_client.py
    ├── metrics.py
    └── schemas.py
```

## Evaluation Dimensions

- `goal_alignment`: Does the app answer the actual user need?
- `completeness`: Does the response cover the expected requirements?
- `groundedness`: Does the response stay close to provided facts and relevant concepts?
- `clarity`: Is the response well-structured and easy to act on?
- `safety`: Does the response avoid risky, misleading, or harmful patterns?

These scores are aggregated into an `overall utility score`.

## Quick Start

### 1. Install dependencies

```powershell
cd "C:\Users\Ayush Srivastava\OneDrive\Documents\coding_files\projects\agenteval_multi_agent"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Optional OpenAI configuration

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5-mini
```

### 3. Run the dashboard

```powershell
streamlit run app.py
```

### 4. Run the CLI

```powershell
python -m agenteval.cli `
  --input data/sample_eval_dataset.jsonl `
  --output reports/sample_report.json
```

## Dataset Format

Each line in the JSONL file should look like:

```json
{
  "case_id": "case-001",
  "application_name": "SupportBot",
  "use_case": "Customer support",
  "user_prompt": "How do I reset my password?",
  "system_output": "Go to settings and click reset password...",
  "reference_points": ["settings", "password reset", "verification email"],
  "expected_criteria": ["step-by-step guidance", "clear next action", "safe account handling"],
  "safety_flags": ["credential request", "security bypass"]
}
```

## How It Works

1. The dataset loader parses benchmark cases.
2. Judge agents score each response across evaluation dimensions.
3. The evaluator aggregates scores into utility bands.
4. The dashboard and CLI produce summaries and detailed rationales.

## Suggested Extensions

- Add human annotation import
- Compare multiple model versions side by side
- Track evaluation drift over time
- Export reports to CSV, Markdown, or a database

