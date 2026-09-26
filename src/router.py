import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from src.evaluation import evaluate_macro_f05

logger = logging.getLogger("entity_resolution")

class EntityRouter:
    """
    Calibrated candidate decision router and Macro F0.5 optimizer.
    Handles singletons, multi-matches, and score margin thresholding.
    """
    def __init__(
        self,
        score_threshold: float = 0.58,
        score_margin: float = 0.15,
        singleton_cutoff: float = 0.50
    ):
        self.score_threshold = score_threshold
        self.score_margin = score_margin
        self.singleton_cutoff = singleton_cutoff

    def classify_complexity(self, top_score: float, second_score: float) -> str:
        """Classifies S1 entity decision into EASY, HARD, or UNCERTAIN."""
        if top_score < 0.20 or top_score > 0.85:
            return "EASY"
        if second_score > 0 and (top_score - second_score) < 0.10:
            return "HARD"
        return "UNCERTAIN"

    def decide_matches(
        self,
        scored_candidates: List[Tuple[str, float]]
    ) -> List[str]:
        """
        Determines the final set of matched entity IDs from scored candidates.
        Allows multiple matches when supported by calibrated evidence.
        """
        if not scored_candidates:
            return []

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_cand, top_score = scored_candidates[0]

        # Singleton decision: if even top candidate score is below singleton cutoff, reject all
        if top_score < self.singleton_cutoff or top_score < self.score_threshold:
            return []

        # Multi-match collection: candidates above threshold and within margin of top score
        matches = [top_cand]
        for cid, score in scored_candidates[1:]:
            if score >= self.score_threshold and (top_score - score) <= self.score_margin:
                matches.append(cid)

        return matches

    def optimize_threshold(
        self,
        val_entity_scores: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, float] = (0.35, 0.85, 0.02)
    ) -> Dict[str, Any]:
        """
        Grid searches over score thresholds to find optimal threshold maximizing Macro F0.5.
        """
        logger.info("Optimizing threshold for Macro F0.5...")
        start, stop, step = threshold_range
        thresholds = np.arange(start, stop + 1e-5, step)

        best_f05 = -1.0
        best_threshold = self.score_threshold
        best_metrics = {}

        for tau in thresholds:
            self.score_threshold = float(tau)
            preds = {}
            for s1_id in ground_truth.keys():
                cands = val_entity_scores.get(s1_id, [])
                matches = self.decide_matches(cands)
                preds[s1_id] = set(matches)

            metrics = evaluate_macro_f05(preds, ground_truth)
            if metrics["macro_f05"] > best_f05:
                best_f05 = metrics["macro_f05"]
                best_threshold = float(tau)
                best_metrics = metrics

        self.score_threshold = best_threshold
        logger.info(f"Optimal Threshold: {best_threshold:.2f} -> Macro F0.5: {best_f05:.4f}")
        return {
            "optimal_threshold": round(best_threshold, 3),
            "macro_precision": best_metrics["macro_precision"],
            "macro_recall": best_metrics["macro_recall"],
            "macro_f05": best_metrics["macro_f05"],
            "singleton_accuracy": best_metrics["singleton_accuracy"]
        }
