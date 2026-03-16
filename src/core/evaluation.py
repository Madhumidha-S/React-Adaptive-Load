from collections import defaultdict
from typing import Dict, List, Tuple


def confusion_matrix_from_trace(
    trace_events: List[Dict[str, object]]
) -> Dict[Tuple[str, str], int]:
    """
    Build a simple confusion matrix from per-step trace events.

    Each trace event is expected to have:
      - predicted_top1: the component ID predicted with highest probability
      - actual_next: the ground truth next component ID

    The result is a dictionary keyed by (actual, predicted) pairs with counts.
    This corresponds to the detailed error analysis described in the paper.
    """
    matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    for ev in trace_events:
        actual = ev.get("actual_next")
        predicted = ev.get("predicted_top1")
        if actual is None:
            continue
        key = (actual, predicted or "<NONE>")
        matrix[key] += 1
    return dict(matrix)


def top1_accuracy_from_trace(trace_events: List[Dict[str, object]]) -> float:
    """
    Compute top-1 accuracy directly from trace events.

    This is similar to the Accuracy metric defined in Equation (3)
    of the paper but scoped to a single run over the provided trace.
    """
    correct = 0
    total = 0
    for ev in trace_events:
        actual = ev.get("actual_next")
        predicted = ev.get("predicted_top1")
        if actual is None:
            continue
        total += 1
        if predicted == actual:
            correct += 1
    return (correct / total) if total else 0.0

