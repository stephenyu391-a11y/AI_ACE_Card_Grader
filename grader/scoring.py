"""The validator's card-grading score, copied from the subnet rules.

Source of truth: scorevision/validator/central/private_track/scoring.py in the
turbovision repository (TCG_GRADE_TOLERANCE and _TCG_FIELD_WEIGHTS).
"""
import numpy as np

FIELDS = ["surface", "edges", "corners", "centering", "grade"]
WEIGHTS = {"surface": 0.25, "edges": 0.25, "corners": 0.25, "centering": 0.10, "grade": 0.15}
GRADES = np.arange(1, 11)

# REWARD[true - 1, answer - 1]: 1.0 exact, 0.5 one grade off, 0 otherwise.
REWARD = np.maximum(0.0, 1.0 - np.abs(GRADES[:, None] - GRADES[None, :]) / 2.0)


def field_score(answer: int, truth: int) -> float:
    return max(0.0, 1.0 - abs(int(answer) - int(truth)) / 2.0)


def card_score(answer: dict, truth: dict) -> float:
    """Score for one card; both dicts map field name -> whole grade 1..10."""
    return sum(WEIGHTS[f] * field_score(answer[f], truth[f]) for f in FIELDS)


def best_answer(probabilities: np.ndarray) -> int:
    """Grade with the highest expected score, given P(true grade = 1..10).

    One grade off still earns half, so this is not always the most likely grade.
    """
    return int(GRADES[np.argmax(probabilities @ REWARD)])
