from typing import Dict, List, Set, Tuple

def compute_entity_f05(predicted_ids: Set[str], true_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f0.5) for a single Source 1 entity according to the challenge rules:
    - If true_ids is empty (singleton):
        - if predicted_ids is empty: scores 1.0 (precision=1.0, recall=1.0, f0.5=1.0)
        - if predicted_ids is non-empty: scores 0.0
    - If true_ids is non-empty:
        - if predicted_ids is empty: scores 0.0
        - otherwise:
            precision = |pred & true| / |pred|
            recall = |pred & true| / |true|
            f0.5 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
    """
    # Singleton case
    if not true_ids:
        if not predicted_ids:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0

    # Non-singleton but empty prediction
    if not predicted_ids:
        return 0.0, 0.0, 0.0

    tp = len(predicted_ids & true_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    denom = (0.25 * precision) + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return precision, recall, f05

def evaluate_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Computes Macro-averaged Precision, Recall, and F0.5 across all Source 1 entities in ground truth.
    Includes dedicated singleton tracking.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0, "singleton_acc": 0.0}

    sum_p = 0.0
    sum_r = 0.0
    sum_f = 0.0
    
    singleton_total = 0
    singleton_correct = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f = compute_entity_f05(pred_set, true_set)
        sum_p += p
        sum_r += r
        sum_f += f

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1

    return {
        "macro_precision": round(sum_p / total_entities, 5),
        "macro_recall": round(sum_r / total_entities, 5),
        "macro_f05": round(sum_f / total_entities, 5),
        "singleton_accuracy": round(singleton_correct / singleton_total, 5) if singleton_total else 1.0,
        "total_evaluated": total_entities,
        "total_singletons": singleton_total
    }
