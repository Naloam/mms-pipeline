"""Print and save Wilson + Hoeffding intervals for saved MMS report.json files."""

import argparse
import sys
from pathlib import Path

from mms_eval.utils import read_json
from mms_multimodal.intervals import intervals_for_report, write_intervals_for_report


def collect(path: str) -> list[Path]:
    target = Path(path)
    if target.is_dir():
        return sorted(target.rglob("report.json"))
    return [target]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="report.json files or directories to search")
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--no-write", action="store_true",
                        help="only print; do not write intervals.json beside each report")
    args = parser.parse_args()

    reports = [p for path in args.paths for p in collect(path)]
    if not reports:
        sys.exit("No report.json found")
    width = max(len(str(p)) for p in reports)
    print(f"{'report':<{width}}  {'n':>6}  {'cand':>6}  {'MMS':>8}  "
          f"{'wilson95':>21}  {'hoeffding95':>21}")
    for path in reports:
        try:
            if args.no_write:
                row = intervals_for_report(read_json(path), confidence=args.confidence)
            else:
                row = write_intervals_for_report(path, confidence=args.confidence)
        except (ValueError, KeyError) as error:
            print(f"{str(path):<{width}}  skipped ({error})")
            continue

        def fmt(interval):
            return "[" + ", ".join(f"{v:.4f}" for v in interval) + "]" \
                if interval else "n/a (single sample)"

        print(f"{str(path):<{width}}  {row['n']:>6}  {row['candidate_count']:>6}  "
              f"{row['value']:>8.4f}  {fmt(row['wilson_interval']):>21}  "
              f"{fmt(row['hoeffding_interval']):>21}")
    if args.no_write:
        print("(not written; remove --no-write to save intervals.json beside each report)")


if __name__ == "__main__":
    main()
