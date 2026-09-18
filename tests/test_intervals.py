"""Wilson + Hoeffding interval reporting for saved MMS reports."""
import math

import pytest

from mms_eval import semantic
from mms_eval.utils import read_json, write_json
from mms_multimodal.intervals import (hoeffding_interval, intervals_for_report,
                                      write_intervals_for_report, z_score)


def test_hoeffding_matches_paper_equation_ten():
    count, n = 845, 5000
    epsilon = math.sqrt(math.log(2 / .05) / (2 * n))
    p = count / n
    assert hoeffding_interval(count, n) == pytest.approx([p - epsilon, p + epsilon])
    assert hoeffding_interval(count, n)[1] - hoeffding_interval(count, n)[0] \
        == pytest.approx(2 * epsilon)


def test_hoeffding_clips_to_unit_range_and_rejects_bad_input():
    assert hoeffding_interval(0, 8)[0] == 0
    assert hoeffding_interval(8, 8)[1] == 1
    assert hoeffding_interval(0, 0) is None
    with pytest.raises(ValueError):
        hoeffding_interval(2, 1)
    with pytest.raises(ValueError):
        hoeffding_interval(1, 1, confidence=1.0)


def test_z_score_generalises_wilson_default():
    assert z_score(.95) == pytest.approx(1.959963984540054, abs=1e-12)


def test_wilson_matches_pipeline_implementation():
    for count, n in [(845, 5000), (1, 40), (0, 7), (400, 400)]:
        row = intervals_for_report({"mms": {"n": n, "candidate_count": count}})
        assert row["wilson_interval"] == pytest.approx(semantic.wilson_interval(count, n))


def test_hoeffding_is_more_conservative_than_wilson():
    for count, n in [(845, 5000), (341, 5000), (236, 5000), (7, 25)]:
        row = intervals_for_report({"mms": {"n": n, "candidate_count": count}})
        wilson_width = row["wilson_interval"][1] - row["wilson_interval"][0]
        hoeffding_width = row["hoeffding_interval"][1] - row["hoeffding_interval"][0]
        assert hoeffding_width > wilson_width


def synthetic_report(n=5000, count=845, status="ok", **extra):
    return {"n": n, "mms": {"n": n, "candidate_count": count, "value": count / n,
                            "status": status}, **extra}


def test_intervals_for_report_variants(tmp_path):
    row = intervals_for_report(synthetic_report())
    assert row["value"] == pytest.approx(845 / 5000)
    assert row["hoeffding_half_width"] == pytest.approx(math.sqrt(math.log(40) / 10000))
    single = intervals_for_report(synthetic_report(n=1, count=0, status="single_sample_only"))
    assert single["wilson_interval"] is None and single["hoeffding_interval"] is None
    assert "reason" in single
    with pytest.raises(ValueError):
        intervals_for_report({"n": 10})  # quality-only report without an mms block


def test_write_intervals_leaves_report_bytes_untouched(tmp_path):
    report_path = tmp_path / "report.json"
    write_json(report_path, synthetic_report())
    original = report_path.read_bytes()
    row = write_intervals_for_report(report_path)
    saved = read_json(tmp_path / "intervals.json")
    assert report_path.read_bytes() == original
    assert saved["report_sha256"] == row["report_sha256"]
    assert saved["wilson_interval"] == row["wilson_interval"]
    assert saved["hoeffding_interval"] == row["hoeffding_interval"]


def test_cli_over_directory_skips_non_mms_reports(tmp_path):
    import subprocess
    import sys as _sys
    from pathlib import Path
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    write_json(good_dir / "report.json", synthetic_report())
    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()
    write_json(quality_dir / "report.json", {"n": 10, "distribution": {"fid": {"value": 1.0}}})
    script = Path(__file__).resolve().parent.parent / "scripts" / "report_intervals.py"
    import os
    env = {**os.environ, "PYTHONPATH": str(script.parent.parent)}
    result = subprocess.run(
        [_sys.executable, str(script), str(tmp_path), "--no-write"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert "0.1690" in result.stdout
    assert "skipped" in result.stdout
