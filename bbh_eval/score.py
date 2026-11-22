"""Exact-match scoring for single, direct BBH answers (no model dependencies)."""

import re


def score_answer(task: str, answer: str, gold: str, question: str) -> dict:
    """Ignore outer whitespace. Only a single allowed label counts as valid."""
    if task == "boolean_expressions":
        allowed = {"True", "False"}
    elif task == "date_understanding":
        # Use the choices actually present: some questions have five, others six.
        allowed = set(re.findall(r"^\s*(\([A-F]\))\s+\S", question, re.MULTILINE))
    else:
        raise ValueError(f"Unsupported task: {task}")

    if gold not in allowed:
        raise ValueError(f"Invalid gold label {gold!r} for {task}; inspect the input file.")
    prediction = answer.strip()
    return {
        "prediction": prediction,
        "valid_format": prediction in allowed,
        "exact_match": prediction == gold,
    }
