from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .schemas import EvalCase


def load_eval_cases(path: str | Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase(**json.loads(line)))
    return cases

