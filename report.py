#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — Print a summary of face tagging results

Usage:
    ./pt.py report --output results.json
"""

import os
import sys
import json
import argparse
from collections import Counter


def report(output_json: str):
    """Print a quick summary of the JSON results."""
    if not os.path.exists(output_json):
        print(f"No results file found: {output_json}")
        sys.exit(1)

    with open(output_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    print(f"\n{'─'*50}")
    print(f"  Results Summary: {output_json}")
    print(f"{'─'*50}")
    print(f"  Total entries   : {len(rows)}")
    print(f"  Unique photos   : {len(set(r['file_name'] for r in rows))}")
    print()

    counts = Counter(r["matched_name"] for r in rows if r.get("matched_name"))
    for name, count in counts.most_common():
        print(f"  {name:<25} {count} face(s)")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Print a summary of face tagging results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="results.json", help="JSON file to summarise (default: results.json)")

    args = parser.parse_args()
    report(args.output)


if __name__ == "__main__":
    main()
