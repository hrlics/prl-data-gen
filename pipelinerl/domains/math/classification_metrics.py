"""pipelinerl.domains.math.classification_metrics

Classification metrics for boxed answers.

This module extracts the final answer from a LaTeX ``\\boxed{...}`` and computes:
- 5-class classification metrics for labels {1,2,3,4,5}
- per-class metrics and macro/weighted aggregates
- MAE/MSE between predicted label and ground-truth label
- Within-1 Accuracy
"""

from typing import Dict, List, Sequence
import logging
import re

logger = logging.getLogger(__name__)


try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
except ImportError as e:
    raise RuntimeError(
        "scikit-learn is required for classification metrics. "
        "Please install it with `pip install scikit-learn`."
    ) from e


LABELS: Sequence[int] = (1, 2, 3, 4, 5)


def extract_boxed_content(text: str) -> str | None:
    """
    Extract content from \\boxed{...} format.
    
    Args:
        text: The text potentially containing \\boxed{...}
    
    Returns:
        The content inside \\boxed{...}, or None if no boxed format found.
    """
    if not text:
        return None
    
    # Prefer a small parser over regex so we handle nested braces like:
    #   \boxed{\text{4}}
    # and also take the *last* boxed occurrence when multiple exist.
    needle = "\\boxed{"  # this matches both "\boxed{" and "\\boxed{" inputs

    search_upto = len(text)
    while True:
        start = text.rfind(needle, 0, search_upto)
        if start == -1:
            return None

        i = start + len(needle)
        depth = 1
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len(needle) : i].strip()
            i += 1

        # Unbalanced braces for this occurrence; try an earlier \boxed{...
        search_upto = start


def answer_to_label(answer: str) -> int:
    """Parse a 1-5 class label from a ``\\boxed{...}`` answer.

    Rules:
    - Only accepts answers in ``\\boxed{...}`` format.
    - Extracts a single digit 1..5 (not part of a larger number).

    Returns:
        The integer label (1..5) or -1 if invalid/unparsable.
    """

    extracted = extract_boxed_content(answer)
    if extracted is None:
        return -1

    # Robustly find a standalone digit 1..5, ignoring LaTeX wrappers like \text{...}
    # Match a digit 1..5 that is not part of a larger number.
    matches = re.findall(r"(?<!\d)([1-5])(?!\d)", extracted)
    if not matches:
        return -1
    try:
        value = int(matches[-1])
    except ValueError:
        return -1

    return value if value in LABELS else -1


def answer_to_binary(answer: str) -> int:
    """Convert legacy yes/no boxed answer to binary 0/1.

    Deprecated: kept for backward compatibility with older scripts/tests.
    New 1-5 classification should use :func:`answer_to_label`.

    Returns:
        1 for "yes", 0 for "no", -1 for invalid/unparsable (including non-boxed format)
    """

    extracted = extract_boxed_content(answer)
    if extracted is None:
        return -1

    answer_lower = extracted.lower().strip()
    if "yes" in answer_lower:
        return 1
    if "no" in answer_lower:
        return 0
    return -1


def calculate_classification_metrics(
    predictions: List[str], # its the raw model outputs, not yet converted to labels
    ground_truths: List[str],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate classification metrics for 5-class labels {1,2,3,4,5}.

    Metrics:
    - Per-class: ACC, Precision, Recall, F1, MAE, MSE, support
    - Macro + weighted: ACC, Precision, Recall, F1
    - Overall: ACC/Precision/Recall/F1 (weighted averages for P/R/F1), MAE, MSE
    
    Args:
        predictions: List of model outputs containing ``\\boxed{...}`` with a label 1..5
        ground_truths: List of ground truth strings containing ``\\boxed{...}`` with a label 1..5
        prefix: Optional prefix for metric names (e.g., "" for train, "test/" for test)
    
    Returns:
        Dictionary of classification metrics
    """
    if len(predictions) != len(ground_truths):
        logger.error(f"Predictions and ground truths have different lengths: {len(predictions)} vs {len(ground_truths)}")
        return {}

    # Convert answers to integer labels
    pred_labels = [answer_to_label(pred) for pred in predictions]
    gt_labels = [answer_to_label(gt) for gt in ground_truths]

    # Filter out invalid predictions and ground truths
    valid_pairs = [(p, g) for p, g in zip(pred_labels, gt_labels) if p != -1 and g != -1]
    
    if len(valid_pairs) == 0:
        logger.warning("No valid prediction/ground truth pairs found")
        metrics: Dict[str, float] = {
            f"{prefix}classification/total_samples": float(len(predictions)),
            f"{prefix}classification/valid_samples": 0.0,
            f"{prefix}classification/ACC": 0.0,
            f"{prefix}classification/Precision": 0.0,
            f"{prefix}classification/Recall": 0.0,
            f"{prefix}classification/F1": 0.0,
            f"{prefix}classification/macro/ACC": 0.0,
            f"{prefix}classification/macro/Precision": 0.0,
            f"{prefix}classification/macro/Recall": 0.0,
            f"{prefix}classification/macro/F1": 0.0,
            f"{prefix}classification/weighted/ACC": 0.0,
            f"{prefix}classification/weighted/Precision": 0.0,
            f"{prefix}classification/weighted/Recall": 0.0,
            f"{prefix}classification/weighted/F1": 0.0,
            f"{prefix}classification/overall/MAE": 0.0,
            f"{prefix}classification/overall/MSE": 0.0,
        }
        for label in LABELS:
            metrics[f"{prefix}classification/class_{label}/support"] = 0.0
            metrics[f"{prefix}classification/class_{label}/ACC"] = 0.0
            metrics[f"{prefix}classification/class_{label}/Precision"] = 0.0
            metrics[f"{prefix}classification/class_{label}/Recall"] = 0.0
            metrics[f"{prefix}classification/class_{label}/F1"] = 0.0
            metrics[f"{prefix}classification/class_{label}/MAE"] = 0.0
            metrics[f"{prefix}classification/class_{label}/MSE"] = 0.0
        return metrics
    
    preds = [p for p, _g in valid_pairs]
    gts = [g for _p, g in valid_pairs]

    # Regression-style distance metrics over labels
    diffs = [float(p - g) for p, g in valid_pairs]
    mae_overall = sum(abs(d) for d in diffs) / len(diffs)
    mse_overall = sum((d * d) for d in diffs) / len(diffs)

    metrics: Dict[str, float] = {
        f"{prefix}classification/total_samples": float(len(predictions)),
        f"{prefix}classification/valid_samples": float(len(valid_pairs)),
        f"{prefix}classification/overall/MAE": float(mae_overall),
        f"{prefix}classification/overall/MSE": float(mse_overall),
    }


    # Per-class MAE/MSE and within-1 accuracy grouped by ground-truth class
    for label in LABELS:
        class_pairs = [(p, g) for p, g in valid_pairs if g == label]
        metrics[f"{prefix}classification/class_{label}/support"] = float(len(class_pairs))
        if not class_pairs:
            metrics[f"{prefix}classification/class_{label}/MAE"] = 0.0
            metrics[f"{prefix}classification/class_{label}/MSE"] = 0.0
            metrics[f"{prefix}classification/class_{label}/within1_ACC"] = 0.0
        else:
            class_diffs = [float(p - g) for p, g in class_pairs]
            metrics[f"{prefix}classification/class_{label}/MAE"] = float(
                sum(abs(d) for d in class_diffs) / len(class_diffs)
            )
            metrics[f"{prefix}classification/class_{label}/MSE"] = float(
                sum((d * d) for d in class_diffs) / len(class_diffs)
            )
            # within-1 accuracy: proportion of predictions within 1 of ground truth
            within1 = sum(1 for d in class_diffs if abs(d) <= 1)
            metrics[f"{prefix}classification/class_{label}/within1_ACC"] = float(within1) / len(class_diffs)

    # Overall within-1 accuracy (all valid pairs)
    if valid_pairs:
        overall_within1 = sum(1 for p, g in valid_pairs if abs(p - g) <= 1) / len(valid_pairs)
    else:
        overall_within1 = 0.0
    metrics[f"{prefix}classification/overall/within1_ACC"] = float(overall_within1)

    try:
        # Overall accuracy
        overall_acc = accuracy_score(gts, preds)

        # Per-class precision/recall/f1/support
        per_class_p, per_class_r, per_class_f1, supports = precision_recall_fscore_support(
            gts,
            preds,
            labels=list(LABELS),
            average=None,
            zero_division=0,
        )

        # Per-class ACC: correct within each ground-truth class / support
        per_class_acc: list[float] = []
        for label, support in zip(LABELS, supports):
            if support == 0:
                per_class_acc.append(0.0)
                continue
            correct = sum(1 for p, g in valid_pairs if g == label and p == g)
            per_class_acc.append(float(correct / support))

        for idx, label in enumerate(LABELS):
            metrics[f"{prefix}classification/class_{label}/ACC"] = float(per_class_acc[idx])
            metrics[f"{prefix}classification/class_{label}/Precision"] = float(per_class_p[idx])
            metrics[f"{prefix}classification/class_{label}/Recall"] = float(per_class_r[idx])
            metrics[f"{prefix}classification/class_{label}/F1"] = float(per_class_f1[idx])

        # Macro/weighted aggregates
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gts, preds, labels=list(LABELS), average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            gts, preds, labels=list(LABELS), average="weighted", zero_division=0
        )

        # Macro ACC = mean per-class ACC over classes that exist in ground truth.
        present_accs = [acc for acc, support in zip(per_class_acc, supports) if support > 0]
        macro_acc = float(sum(present_accs) / len(present_accs)) if present_accs else 0.0

        weighted_acc = float(overall_acc)  # weighted by support equals overall accuracy

        metrics |= {
            # Backward-compatible overall keys
            f"{prefix}classification/ACC": float(overall_acc),
            f"{prefix}classification/Precision": float(weighted_p),
            f"{prefix}classification/Recall": float(weighted_r),
            f"{prefix}classification/F1": float(weighted_f1),

            # Explicit macro/weighted
            f"{prefix}classification/macro/ACC": float(macro_acc),
            f"{prefix}classification/macro/Precision": float(macro_p),
            f"{prefix}classification/macro/Recall": float(macro_r),
            f"{prefix}classification/macro/F1": float(macro_f1),

            f"{prefix}classification/weighted/ACC": float(weighted_acc),
            f"{prefix}classification/weighted/Precision": float(weighted_p),
            f"{prefix}classification/weighted/Recall": float(weighted_r),
            f"{prefix}classification/weighted/F1": float(weighted_f1),
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {e}")
        # Return what we have (counts + MAE/MSE), plus zeroed classification metrics.
        metrics |= {
            f"{prefix}classification/ACC": 0.0,
            f"{prefix}classification/Precision": 0.0,
            f"{prefix}classification/Recall": 0.0,
            f"{prefix}classification/F1": 0.0,
            f"{prefix}classification/macro/ACC": 0.0,
            f"{prefix}classification/macro/Precision": 0.0,
            f"{prefix}classification/macro/Recall": 0.0,
            f"{prefix}classification/macro/F1": 0.0,
            f"{prefix}classification/weighted/ACC": 0.0,
            f"{prefix}classification/weighted/Precision": 0.0,
            f"{prefix}classification/weighted/Recall": 0.0,
            f"{prefix}classification/weighted/F1": 0.0,
        }
        for label in LABELS:
            metrics.setdefault(f"{prefix}classification/class_{label}/ACC", 0.0)
            metrics.setdefault(f"{prefix}classification/class_{label}/Precision", 0.0)
            metrics.setdefault(f"{prefix}classification/class_{label}/Recall", 0.0)
            metrics.setdefault(f"{prefix}classification/class_{label}/F1", 0.0)
        return metrics
