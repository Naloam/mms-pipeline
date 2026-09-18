"""Export one evaluation's per-sample scores and summary into a single Excel file.

Reads report.json + scores.jsonl (either pipeline flavour) and writes
scores.xlsx with three sheets: per_image (one row per sample, posterior
entropy first among scores), summary (MMS, thresholds, both confidence
intervals, baselines) and pair_matrix (Eq. 12 category-pair decomposition).
This is a post-processing viewer next to the frozen outputs; it never
modifies report.json or scores.jsonl.
"""

import argparse
import json
from pathlib import Path

from mms_multimodal.intervals import intervals_for_report

SCORE_ORDER = ("entropy", "second_probability", "one_minus_max_probability",
               "negative_top1_top2_margin", "gini")


def _load(directory: Path) -> tuple[dict, list[dict]]:
    report = json.loads((directory / "report.json").read_text())
    rows = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines() if line]
    return report, rows


def _per_image_columns(rows: list[dict], classes: list[str]) -> list[str]:
    preferred = ["image_id", "sample_id", "setting_id", "source_id", "domain",
                 "cohort_role", "path", "protocol_id", "evaluator_id", "temperature",
                 "top1_class", "top2_class"]
    preferred += [f"p_{name}" for name in classes]
    preferred += list(SCORE_ORDER)
    preferred += [f"flag_{name}" for name in SCORE_ORDER]
    flattened = [_row_values(row, classes) for row in rows]
    seen, columns = set(), []
    for name in preferred + sorted({key for row in flattened for key in row} - set(preferred)):
        if name not in seen:
            seen.add(name)
            columns.append(name)
    return columns


def _row_values(row: dict, classes: list[str]) -> dict:
    values = dict(row)
    probabilities = values.pop("probabilities", None)
    if probabilities is not None:
        for name, value in zip(classes, probabilities):
            values[f"p_{name}"] = value
    mmr = values.pop("mmr_original", None)
    if isinstance(mmr, dict):
        for threshold, flagged in sorted(mmr.items(), key=lambda item: float(item[0])):
            values[f"mmr_p2>={threshold}"] = flagged
    return values


def _threshold_table(report: dict) -> list[tuple]:
    calibrations = report.get("calibrations")
    if not calibrations:
        return []
    return [(name, cal["threshold"], cal["k"], cal["n"], cal["alpha"], cal["inequality"])
            for name, cal in sorted(calibrations.items())]


def build_workbook(directory: Path, output: Path):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    report, rows = _load(directory)
    mms = report["mms"]
    classes = mms["classes"]
    intervals = intervals_for_report(report)
    book = Workbook()
    bold = Font(bold=True)

    sheet = book.active
    sheet.title = "per_image"
    columns = _per_image_columns(rows, classes)
    sheet.append(columns)
    for cell in sheet[1]:
        cell.font = bold
    for row in rows:
        values = _row_values(row, classes)
        sheet.append([values.get(name) for name in columns])
    sheet.freeze_panes = "A2"

    summary = book.create_sheet("summary")
    entries = [
        ("n", mms["n"]), ("candidate_count", mms["candidate_count"]),
        ("MMS_candidate_fraction", mms["value"]),
        ("mean_entropy", mms["mean_entropy"]),
    ]
    for name, interval in (("wilson95", intervals["wilson_interval"]),
                           ("hoeffding95", intervals["hoeffding_interval"])):
        if interval is not None:
            entries += [(f"{name}_lower", interval[0]), (f"{name}_upper", interval[1])]
    entries += [
        ("modality", report.get("modality")), ("protocol_id", report.get("protocol_id")),
        ("evaluator_id", report.get("evaluator_id")),
        ("reference_signature", report.get("reference_signature")),
        ("checkpoint_sha256", report.get("checkpoint_sha256") or report.get("evaluator_checkpoint_sha256")),
        ("classes", ", ".join(classes)),
        ("predicted_class_counts", ", ".join(str(v) for v in mms["predicted_class_counts"])),
        ("interpretation", mms.get("ci_assumptions")),
    ]
    if intervals.get("reason"):
        entries.append(("single_sample_reason", intervals["reason"]))
    for key, value in entries:
        if value is not None:
            summary.append([key, value])
    thresholds = _threshold_table(report)
    if thresholds:
        summary.append([])
        summary.append(["score", "threshold_tau", "k", "n_cal", "alpha", "inequality"])
        for cell in summary[summary.max_row]:
            cell.font = bold
        for row in thresholds:
            summary.append(list(row))
    summary.append([])
    summary.append(["baseline", "candidate_count", "n", "candidate_fraction"])
    for cell in summary[summary.max_row]:
        cell.font = bold
    for name, stats in sorted(mms.get("baselines", {}).items()):
        summary.append([name, stats["candidate_count"], stats["n"], stats["candidate_fraction"]])

    pairs = book.create_sheet("pair_matrix")
    pairs.append(["Counts: rows=top1 class, cols=top2 class (upper triangle)"])
    pairs.append([""] + classes)
    counts = mms["pair_counts_upper_triangle"]
    for name, row in zip(classes, counts):
        pairs.append([name] + row)
    pairs.append([])
    pairs.append(["Rates: each cell divided by n; all cells sum to the global MMS"])
    pairs.append([""] + classes)
    rates = mms["pair_rates_upper_triangle"]
    for name, row in zip(classes, rates):
        pairs.append([name] + row)

    book.save(output)
    return {"output": str(output), "n": mms["n"], "sheets": book.sheetnames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", help="directory holding report.json and scores.jsonl")
    parser.add_argument("--output", help="default: <evaluation_dir>/scores.xlsx")
    args = parser.parse_args()
    directory = Path(args.evaluation_dir).resolve()
    output = Path(args.output).resolve() if args.output else directory / "scores.xlsx"
    print(json.dumps(build_workbook(directory, output), ensure_ascii=False))


if __name__ == "__main__":
    main()
