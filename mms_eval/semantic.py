"""The paper's entropy, finite-sample rank calibration and candidate proportion."""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING
from typing import Sequence

import numpy as np

SCORE_NAMES = ("entropy", "second_probability", "one_minus_max_probability",
               "negative_top1_top2_margin", "gini")
MMS_ALGORITHM_VERSION = "entropy-rank-strict-v1"


def posterior_scores(logits: np.ndarray, temperature: float = 1.0) -> dict:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 2 or z.shape[0] == 0 or not np.isfinite(z).all():
        raise ValueError("logits must be finite nonempty [N,C], with C>=2")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # Negative overflow denotes negligible mass. Some platforms map longdouble
    # to float64, so clamp it explicitly before computing p * log(p).
    wide = z.astype(np.longdouble)
    with np.errstate(over='ignore'):
        z = np.maximum((wide-wide.max(axis=1, keepdims=True))/np.longdouble(temperature),
                       np.longdouble(-1e300)).astype(np.float64)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    entropy = -(p * logp).sum(1)
    # Stable sorting breaks equal probabilities by ascending original class ID.
    order = np.argsort(-p, axis=1, kind="stable")
    top1 = p[np.arange(len(p)), order[:, 0]]
    top2 = p[np.arange(len(p)), order[:, 1]]
    return {"probabilities": p, "log_probabilities": logp, "class_order": order,
            "entropy": entropy, "second_probability": top2,
            "one_minus_max_probability": 1 - top1,
            "negative_top1_top2_margin": -(top1 - top2),
            "gini": 1 - np.square(p).sum(1)}


def rank_calibrate(values: Sequence[float], alpha: float = .05) -> dict:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    n = len(scores)
    # Decimal prevents 19.000000000000004 being rounded to the wrong order statistic.
    k = int((Decimal(n + 1) * (Decimal(1) - Decimal(str(alpha)))).to_integral_value(
        rounding=ROUND_CEILING))
    return {"n": n, "k": k, "alpha": float(alpha),
            "threshold": float(np.partition(scores, k - 1)[k - 1]) if k <= n else "infinity",
            "inequality": ">", "algorithm": MMS_ALGORITHM_VERSION}


def apply_threshold(values: Sequence[float], calibration: dict) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    if calibration.get("inequality") != ">":
        raise ValueError("MMS calibration requires the strict > boundary")
    threshold = calibration["threshold"]
    if threshold == "infinity":
        return np.zeros(len(scores), dtype=bool)
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError("Invalid calibrated threshold")
    return scores > float(threshold)


def wilson_interval(positive: int, total: int, z: float = 1.959963984540054) -> list | None:
    if total < 0 or positive < 0 or positive > total:
        raise ValueError("Require 0 <= numerator <= denominator")
    if total == 0:
        return None
    p = positive / total
    factor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / factor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / factor
    return [max(0., center - radius), min(1., center + radius)]


def aggregate_mms(scores: dict, calibrations: dict, classes: list[str]) -> dict:
    p = np.asarray(scores["probabilities"])
    if p.ndim != 2 or p.shape[1] != len(classes) or len(p) == 0:
        raise ValueError("Posterior shape and category list must agree and contain images")
    flags = {name: apply_threshold(scores[name], calibrations[name]) for name in SCORE_NAMES}
    n = len(p)
    candidates = flags["entropy"]
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    order = scores["class_order"]
    for i in np.flatnonzero(candidates):
        a, b = sorted([int(order[i, 0]), int(order[i, 1])])
        matrix[a, b] += 1
    count = int(candidates.sum())
    assert int(matrix.sum()) == count
    return {
        "status": "single_image_only" if n == 1 else "ok",
        "n": n, "candidate_count": count,
        "value": None if n == 1 else count / n,
        "single_image_candidate": bool(candidates[0]) if n == 1 else None,
        "reason": "One image receives a candidate flag, not a model prevalence claim" if n == 1 else None,
        "ci95": None if n == 1 else wilson_interval(count, n),
        "ci_assumptions": "fixed evaluator and reference; independent generated draws; not semantic truth",
        "mean_entropy": float(scores["entropy"].mean()),
        "classes": classes, "pair_counts_upper_triangle": matrix.tolist(),
        "pair_rates_upper_triangle": (matrix / n).tolist(),
        "predicted_class_counts": np.bincount(order[:, 0], minlength=len(classes)).tolist(),
        "baselines": {name: {"candidate_count": int(f.sum()), "n": n,
                              "candidate_fraction": float(f.mean())} for name, f in flags.items()},
    }


def make_score_rows(records: list[dict], logits: np.ndarray, calibrations: dict,
                    classes: list[str], *, temperature: float = 1,
                    evaluator_id: str = "A", protocol_id: str = "") -> list[dict]:
    scores = posterior_scores(logits, temperature)
    if len(records) != len(logits):
        raise ValueError("Image identities and logits length differ")
    flags = {name: apply_threshold(scores[name], calibrations[name]) for name in SCORE_NAMES}
    rows = []
    for i, record in enumerate(records):
        order = scores["class_order"][i]
        row = {k: record[k] for k in ("image_id", "setting_id", "source_id", "domain", "cohort_role", "pair_id", "parent_id", "seed") if k in record}
        row.update({"protocol_id": protocol_id, "evaluator_id": evaluator_id,
                    "temperature": float(temperature), "top1_class": classes[order[0]],
                    "top2_class": classes[order[1]], "probabilities": scores["probabilities"][i].tolist()})
        row.update({name: float(scores[name][i]) for name in SCORE_NAMES})
        row.update({f"flag_{name}": bool(flags[name][i]) for name in SCORE_NAMES})
        row["mmr_original"] = {str(t): bool(scores["second_probability"][i] >= t)
                              for t in (.1, .11, .13, .16, .2)}
        rows.append(row)
    return rows
