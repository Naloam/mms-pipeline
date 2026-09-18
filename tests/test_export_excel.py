"""Excel export of per-sample scores and summary from saved evaluations."""
import json

import pytest

from mms_eval.utils import write_json, write_jsonl

pytest.importorskip("openpyxl")
from scripts.export_excel import build_workbook  # noqa: E402


def synthetic_evaluation(tmp_path):
    classes = ["cat", "dog", "wild"]
    scores = [
        {"image_id": "g:000", "setting_id": "smoke", "path": "/img/000.png",
         "top1_class": "cat", "top2_class": "dog", "probabilities": [.5, .3, .2],
         "entropy": 1.03, "second_probability": .3, "one_minus_max_probability": .5,
         "negative_top1_top2_margin": -.2, "gini": .62,
         "flag_entropy": True, "flag_second_probability": False,
         "flag_one_minus_max_probability": False, "flag_negative_top1_top2_margin": False,
         "flag_gini": True, "mmr_original": {"0.1": True, "0.2": False}},
        {"image_id": "g:001", "setting_id": "smoke", "path": "/img/001.png",
         "top1_class": "wild", "top2_class": "wild", "probabilities": [.01, .01, .98],
         "entropy": .09, "second_probability": .01, "one_minus_max_probability": .02,
         "negative_top1_top2_margin": -.97, "gini": .04,
         "flag_entropy": False, "flag_second_probability": False,
         "flag_one_minus_max_probability": False, "flag_negative_top1_top2_margin": False,
         "flag_gini": False, "mmr_original": {"0.1": False, "0.2": False}},
    ]
    report = {
        "n": 2, "protocol_id": "smoke_v1", "reference_signature": "abc",
        "calibrations": {"entropy": {"threshold": .9, "k": 950, "n": 999, "alpha": .05,
                                     "inequality": ">"}},
        "mms": {"n": 2, "candidate_count": 1, "value": .5, "mean_entropy": .56,
                "classes": classes, "predicted_class_counts": [1, 0, 1],
                "pair_counts_upper_triangle": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                "pair_rates_upper_triangle": [[0, .5, 0], [0, 0, 0], [0, 0, 0]],
                "baselines": {"gini": {"candidate_count": 1, "n": 2, "candidate_fraction": .5}}},
    }
    write_json(tmp_path / "report.json", report)
    write_jsonl(tmp_path / "scores.jsonl", scores)
    return tmp_path


def test_export_builds_all_sheets_with_expected_cells(tmp_path):
    from openpyxl import load_workbook
    evaluation = synthetic_evaluation(tmp_path)
    result = build_workbook(evaluation, tmp_path / "out.xlsx")
    assert result["n"] == 2
    book = load_workbook(tmp_path / "out.xlsx")
    assert book.sheetnames == ["per_image", "summary", "pair_matrix"]

    per = book["per_image"]
    header = [cell.value for cell in per[1]]
    for expected in ["image_id", "p_cat", "p_dog", "p_wild", "entropy", "flag_entropy",
                     "mmr_p2>=0.1"]:
        assert expected in header
    first = {name: per.cell(row=2, column=header.index(name) + 1).value for name in header}
    assert first["entropy"] == 1.03 and first["flag_entropy"] is True
    assert first["p_cat"] == .5 and first["mmr_p2>=0.1"] is True
    assert per.cell(row=3, column=header.index("flag_entropy") + 1).value is False

    summary = {row[0].value: row[1].value for row in summary_rows(book["summary"]) if row[0].value}
    assert summary["MMS_candidate_fraction"] == .5
    assert summary["wilson95_lower"] < .5 < summary["wilson95_upper"]
    assert summary["hoeffding95_lower"] < summary["wilson95_lower"]
    assert summary["hoeffding95_upper"] > summary["wilson95_upper"]
    flat = [cell.value for row in summary_rows(book["summary"]) for cell in row]
    assert .9 in flat  # frozen entropy threshold appears in the threshold table

    pair = book["pair_matrix"]
    values = [cell.value for row in pair.iter_rows() for cell in row]
    assert 1 in values and .5 in values


def summary_rows(sheet):
    return [[cell for cell in row] for row in sheet.iter_rows()]


def test_export_fails_cleanly_without_report(tmp_path):
    from scripts.export_excel import build_workbook
    with pytest.raises(FileNotFoundError):
        build_workbook(tmp_path, tmp_path / "out.xlsx")
