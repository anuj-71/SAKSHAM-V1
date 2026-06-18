import argparse
import json
from collections import Counter
from typing import Dict

from tools.normalize_labels import (
    TARGET_LABELS,
    detect_label_column,
    load_csv_rows,
    normalize_rows,
)


def build_target_count_report(input_path: str, label_column: str = None) -> Dict:
    fieldnames, rows = load_csv_rows(input_path)
    resolved_label_column = detect_label_column(fieldnames, preferred=label_column)
    normalized_rows = normalize_rows(rows, label_column=resolved_label_column)

    target_counts = Counter()
    unmapped_counts = Counter()
    for row in normalized_rows:
        saksham_label = row.get("saksham_label", "")
        if saksham_label:
            target_counts[saksham_label] += 1
        else:
            unmapped_counts[row.get("normalized_label", "")] += 1

    return {
        "input_path": input_path,
        "label_column": resolved_label_column,
        "total_rows": len(normalized_rows),
        "matched_rows": sum(target_counts.values()),
        "target_counts": {label: target_counts.get(label, 0) for label in TARGET_LABELS},
        "top_unmapped_labels": [
            {"label": label, "count": count}
            for label, count in unmapped_counts.most_common(25)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Count CISLR rows that map into the SAKSHAM target vocabulary.")
    parser.add_argument("--input", required=True, help="Path to the source CSV metadata file.")
    parser.add_argument("--label-column", help="Optional explicit label/gloss column name.")
    parser.add_argument("--json-output", help="Optional path to save the report as JSON.")
    args = parser.parse_args()

    report = build_target_count_report(args.input, label_column=args.label_column)

    print(f"Input file: {report['input_path']}")
    print(f"Detected label column: {report['label_column']}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Matched rows: {report['matched_rows']}")
    print("Target label counts:")
    for label in TARGET_LABELS:
        print(f"- {label}: {report['target_counts'][label]}")

    if report["top_unmapped_labels"]:
        print("Top unmapped normalized labels:")
        for entry in report["top_unmapped_labels"]:
            display_label = entry["label"] or "<empty>"
            print(f"- {display_label}: {entry['count']}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as json_file:
            json.dump(report, json_file, indent=2)
        print(f"JSON report: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
