#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan.py — Scan a photo library and tag faces

Usage:
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --skip-processed
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    DeepFace,
    MODEL_NAME, DETECTOR, THRESHOLD,
    file_hash, list_images, load_db, save_db, match_face,
)


def scan(photos_folder: str, db_path: str, output_json: str, skip_processed: bool):
    """Scan all photos, detect faces, match against known people, and write results to JSON."""
    db = load_db(db_path)

    if not db["people"]:
        print("ERROR: No people enrolled yet. Run the 'enroll' command first.")
        sys.exit(1)

    images = list_images(photos_folder)
    if not images:
        print(f"No images found in: {photos_folder}")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Scanning {len(images)} photos in: {photos_folder}")
    print(f"  Known people: {', '.join(sorted(db['people'].keys()))}")
    print(f"  Threshold: {THRESHOLD}  |  Output: {output_json}")
    print(f"{'─'*50}\n")

    # Load existing results so we can append
    existing_rows = []
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing_rows = json.load(f)

    results = []
    skipped = 0
    errors  = 0

    for img_path in tqdm(images, desc="Scanning photos"):
        img_hash = file_hash(str(img_path))

        if skip_processed and img_hash in db["processed"]:
            skipped += 1
            continue

        try:
            faces = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False,
            )
        except Exception as e:
            results.append({
                "file_path":    str(img_path),
                "file_name":    img_path.name,
                "face_index":   0,
                "matched_name": "ERROR",
                "confidence":   None,
                "distance":     None,
                "face_region":  {},
                "scanned_at":   datetime.now().isoformat(timespec="seconds"),
                "notes":        str(e),
            })
            errors += 1
            continue

        if not faces:
            results.append({
                "file_path":    str(img_path),
                "file_name":    img_path.name,
                "face_index":   0,
                "matched_name": "No face detected",
                "confidence":   None,
                "distance":     None,
                "face_region":  {},
                "scanned_at":   datetime.now().isoformat(timespec="seconds"),
                "notes":        "",
            })
        else:
            for i, face_data in enumerate(faces):
                embedding = np.array(face_data["embedding"])
                name, dist = match_face(embedding, db["people"])
                confidence = round((1 - dist) * 100, 1)
                region = face_data.get("facial_area", {})

                results.append({
                    "file_path":    str(img_path),
                    "file_name":    img_path.name,
                    "face_index":   i,
                    "matched_name": name,
                    "confidence":   confidence,
                    "distance":     round(dist, 4),
                    "face_region":  {
                        "x": region.get("x"),
                        "y": region.get("y"),
                        "w": region.get("w"),
                        "h": region.get("h"),
                    },
                    "scanned_at":   datetime.now().isoformat(timespec="seconds"),
                    "notes":        "",
                })

        db["processed"].add(img_hash)

    # Merge with existing results and write JSON
    all_rows = existing_rows + results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    save_db(db, db_path)

    print(f"\n{'─'*50}")
    print(f"  ✓ Done!")
    print(f"  Photos scanned  : {len(images) - skipped}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors          : {errors}")
    print(f"  Faces found     : {sum(1 for r in results if r['matched_name'] not in ('No face detected', 'ERROR'))}")
    print(f"  Results saved → {output_json}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a photo library and tag faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--photos",         required=True,          help="Folder of photos to scan")
    parser.add_argument("--db",             default="faces.db",     help="Path to face database file (default: faces.db)")
    parser.add_argument("--output",         default="results.json", help="Output JSON file (default: results.json)")
    parser.add_argument("--skip-processed", action="store_true",    help="Skip photos already in the database")

    args = parser.parse_args()
    scan(args.photos, args.db, args.output, args.skip_processed)


if __name__ == "__main__":
    main()
