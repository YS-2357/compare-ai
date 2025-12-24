"""평가 결과 집계."""

from __future__ import annotations

from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


def aggregate_scores(results: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """여러 평가자의 점수를 모델별로 집계."""

    logger.debug("aggregate_scores:시작 results=%d evaluations=%d", len(results), len(evaluations))
    try:
        targets = [r["model"] for r in results]
        aggregated: list[dict[str, Any]] = []
        for target in targets:
            collected_scores = []
            rationales = []
            for ev in evaluations:
                evaluator = ev.get("evaluator")
                for sc in ev.get("scores", []):
                    if sc.get("model") == target:
                        collected_scores.append(sc.get("score"))
                        if sc.get("rationale"):
                            rationales.append(f"[{evaluator}] {sc['rationale']}")
            valid_scores = [s for s in collected_scores if isinstance(s, (int, float)) and s >= 0]
            avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
            logger.debug("집계 점수 target=%s scores=%s avg=%s", target, valid_scores, avg)
            aggregated.append(
                {
                    "model": target,
                    "score": avg if avg is not None else -1,
                    "raw_scores": collected_scores,
                    "rationale": "\n".join(rationales) if rationales else None,
                }
            )

        sorted_items = sorted(
            aggregated,
            key=lambda x: x["score"] if isinstance(x.get("score"), (int, float)) else -1,
            reverse=True,
        )
        last_score = None
        current_rank = 0
        for item in sorted_items:
            score_val = item.get("score")
            if score_val != last_score:
                current_rank += 1
                last_score = score_val
            item["rank"] = current_rank
        logger.debug("aggregate_scores:종료 aggregated=%d", len(sorted_items))
        return sorted_items
    except Exception as exc:
        logger.warning("집계 실패: %s", exc)
        return [
            {
                "model": r["model"],
                "score": -1,
                "raw_scores": [],
                "rationale": "aggregation failed",
                "rank": idx + 1,
            }
            for idx, r in enumerate(results)
        ]
