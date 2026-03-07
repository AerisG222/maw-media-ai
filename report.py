#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — Print a summary of scan results

Usage:
    ./pt.py report --output results.json
    ./pt.py report --output results.json --type faces
    ./pt.py report --output results.json --type objects
    ./pt.py report --output results.json --type scenes
"""

import os
import sys
import json
import argparse
from collections import Counter


def report_faces(rows: list):
    face_rows = [r for r in rows if r.get("scan_type") == "faces"]
    if not face_rows:
        print("  No face scan results found.")
        return

    matched = [r for r in face_rows if r.get("matched_name") not in (
        "No face detected", "ERROR", "Unknown", None
    )]
    unknown  = [r for r in face_rows if r.get("matched_name") == "Unknown"]
    no_face  = [r for r in face_rows if r.get("matched_name") == "No face detected"]
    errors   = [r for r in face_rows if r.get("matched_name") == "ERROR"]

    print(f"\n  {'─'*46}")
    print(f"  FACES")
    print(f"  {'─'*46}")
    print(f"  Total face entries  : {len(face_rows)}")
    print(f"  Matched             : {len(matched)}")
    print(f"  Unknown             : {len(unknown)}")
    print(f"  No face detected    : {len(no_face)}")
    print(f"  Errors              : {len(errors)}")
    print()

    counts = Counter(r["matched_name"] for r in matched)
    for name, count in counts.most_common():
        print(f"    {name:<25} {count} face(s)")


def report_objects(rows: list):
    obj_rows = [r for r in rows if r.get("scan_type") == "objects"]
    if not obj_rows:
        print("  No object scan results found.")
        return

    all_labels = []
    for r in obj_rows:
        all_labels.extend(r.get("labels", []))

    total_objects = sum(r.get("object_count", 0) for r in obj_rows)
    photos_with_objects = sum(1 for r in obj_rows if r.get("object_count", 0) > 0)

    print(f"\n  {'─'*46}")
    print(f"  OBJECTS")
    print(f"  {'─'*46}")
    print(f"  Photos scanned      : {len(obj_rows)}")
    print(f"  Photos with objects : {photos_with_objects}")
    print(f"  Total detections    : {total_objects}")
    print()

    counts = Counter(all_labels)
    for label, count in counts.most_common(15):
        print(f"    {label:<25} {count} detection(s)")

    if len(counts) > 15:
        print(f"    ... and {len(counts) - 15} more labels")


def report_scenes(rows: list):
    scene_rows = [r for r in rows if r.get("scan_type") == "scenes"]
    if not scene_rows:
        print("  No scene scan results found.")
        return

    top_scenes = [r.get("top_scene") for r in scene_rows if r.get("top_scene")]
    all_scene_labels = []
    for r in scene_rows:
        all_scene_labels.extend(s["label"] for s in r.get("scenes", []))

    print(f"\n  {'─'*46}")
    print(f"  SCENES")
    print(f"  {'─'*46}")
    print(f"  Photos scanned      : {len(scene_rows)}")
    print(f"  Photos classified   : {len(top_scenes)}")
    print()
    print(f"  Top scene per photo:")
    counts = Counter(top_scenes)
    for label, count in counts.most_common(15):
        print(f"    {label:<25} {count} photo(s)")

    if len(counts) > 15:
        print(f"    ... and {len(counts) - 15} more scenes")


def report(output_json: str, report_type: str):
    """Print a summary of scan results, optionally filtered by scan type."""
    if not os.path.exists(output_json):
        print(f"No results file found: {output_json}")
        sys.exit(1)

    with open(output_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    unique_photos = len(set(r["file_name"] for r in rows))

    print(f"\n  {'═'*46}")
    print(f"  Results Summary: {output_json}")
    print(f"  {'═'*46}")
    print(f"  Total entries       : {len(rows)}")
    print(f"  Unique photos       : {unique_photos}")

    if report_type in ("all", "faces"):
        report_faces(rows)
    if report_type in ("all", "objects"):
        report_objects(rows)
    if report_type in ("all", "scenes"):
        report_scenes(rows)

    print(f"\n  {'═'*46}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Print a summary of scan results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", default="results.json",
                        help="JSON results file to summarise (default: results.json)")
    parser.add_argument("--type",   default="all",
                        choices=["all", "faces", "objects", "scenes"],
                        help="Which scan type to report on (default: all)")

    args = parser.parse_args()
    report(args.output, args.type)


if __name__ == "__main__":
    main()
