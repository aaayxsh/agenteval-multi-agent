from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience
    def load_dotenv() -> bool:
        return False

from .dataset import load_eval_cases
from .evaluator import evaluate_dataset
from .llm_client import OptionalOpenAIClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentEval on a JSONL dataset.")
    parser.add_argument("--input", required=True, help="Path to JSONL evaluation dataset.")
    parser.add_argument("--output", default="reports/eval_report.json", help="Path to write JSON report.")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI commentary if configured.")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    cases = load_eval_cases(args.input)
    report = evaluate_dataset(
        cases,
        llm_client=OptionalOpenAIClient(enabled=args.use_llm),
        system_prompt_path=Path("prompts") / "judge_system_prompt.md",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluated {report['summary']['case_count']} cases")
    print(f"Average utility score: {report['summary']['average_overall_score']}")
    print(f"Pass rate: {report['summary']['pass_rate']:.0%}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
