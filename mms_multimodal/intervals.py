"""Wilson and Hoeffding confidence intervals for MMS candidate fractions.

Both pipelines already report `mms.ci95`, a Wilson score interval for the
candidate fraction under independent draws. The paper's Eq. (10) instead gives
the distribution-free Hoeffding bound, which this module inverts into an
interval of the same shape so the two can be reported side by side.

This module intentionally lives outside `mms_eval/` and outside
`mms_multimodal/core.py`: every frozen reference records a hash of those
files, so extending reporting must not modify them. Intervals are computed
from already-saved reports and written to a separate `intervals.json`,
leaving the original report bytes untouched.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

from mms_eval.semantic import wilson_interval
from mms_eval.utils import read_json, sha256_file, write_json


def z_score(confidence: float) -> float:
    """Two-sided normal quantile; 0.95 gives the 1.959963984540054 default."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def hoeffding_interval(positive: int, total: int, confidence: float = .95) -> list | None:
    """Distribution-free interval inverted from the paper's Eq. (10).

    Pr(|MMS_hat - MMS| >= eps) <= 2 exp(-2 N eps^2); solving at level delta
    gives eps = sqrt(ln(2/delta) / (2N)). The bound holds for independent
    draws from any distribution and is conservative next to Wilson.
    """
    if total < 0 or positive < 0 or positive > total:
        raise ValueError("Require 0 <= numerator <= denominator")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if total == 0:
        return None
    epsilon = math.sqrt(math.log(2 / (1 - confidence)) / (2 * total))
    p = positive / total
    return [max(0., p - epsilon), min(1., p + epsilon)]


def intervals_for_report(report: dict, *, confidence: float = .95) -> dict:
    """Both intervals for one saved report.json (either pipeline flavour)."""
    mms = report.get("mms")
    if not isinstance(mms, dict):
        raise ValueError("Report has no 'mms' block; not an MMS report")
    n, count = mms.get("n"), mms.get("candidate_count")
    if not isinstance(n, int) or n < 0 or not isinstance(count, int) or count < 0:
        raise ValueError("Report mms block needs integer n and candidate_count")
    result = {
        "n": n, "candidate_count": count,
        "value": None if n == 0 else count / n,
        "confidence": confidence,
        "wilson_interval": None if n <= 1 else wilson_interval(count, n, z=z_score(confidence)),
        "hoeffding_interval": None if n <= 1 else hoeffding_interval(count, n, confidence),
        "hoeffding_half_width": None if n <= 1 else math.sqrt(
            math.log(2 / (1 - confidence)) / (2 * n)),
    }
    if n <= 1:
        result["reason"] = ("One sample receives a candidate flag; a proportion interval "
                            "is not a model prevalence claim")
    result["notes"] = {
        "wilson_interval": ("Score interval for a binomial proportion under independent "
                            "draws; matches mms.ci95 in pipeline reports at 95%"),
        "hoeffding_interval": ("Distribution-free bound inverted from Eq. (10); valid for "
                               "independent draws from any distribution, wider than Wilson"),
    }
    return result


def write_intervals_for_report(report_path: str | Path, *, confidence: float = .95) -> dict:
    """Read one report.json, write intervals.json beside it, return the payload."""
    path = Path(report_path).resolve()
    result = intervals_for_report(read_json(path), confidence=confidence)
    result["report_path"] = str(path)
    result["report_sha256"] = sha256_file(path)
    write_json(path.parent / "intervals.json", result)
    return result
