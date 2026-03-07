#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan.py — Scan a photo library for faces, objects, and/or scenes

Usage:
    # Scan for everything (default)
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json

    # Scan for specific types only
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types objects
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types scenes
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces objects

    # Skip already-processed photos
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --skip-processed
"""

import os
import sys
import json
import argparse

from tqdm import tqdm

from common import file_hash, find_images, save_db

ALL_SCAN_TYPES = ["faces", "objects", "scenes"]


def scan(
    photos_folder: str,
    db_path: str,
    output_json: str,
    skip_processed: bool,
    scan_types: list[str],
):
    """Orchestrate scanning of photos across one or more scan types."""

    # ── Validate scan types ───────────────────────────────────────────────
    for st in scan_types:
        if st not in ALL_SCAN_TYPES:
            print(f"ERROR: Unknown scan type '{st}'. Choose from: {', '.join(ALL_SCAN_TYPES)}")
            sys.exit(1)

    do_faces   = "faces"   in scan_types
    do_objects = "objects" in scan_types
    do_scenes  = "scenes"  in scan_types

    # ── Load scanners ─────────────────────────────────────────────────────
    face_db      = None
    object_model = None
    scene_model  = None

    if do_faces:
        from faces import load_face_model, detect_faces
        face_db = load_face_model(db_path)

    if do_objects:
        from objects import load_object_model, detect_objects
        print("  Loading object detection model...")
        object_model = load_object_model()

    if do_scenes:
        from scenes import load_scene_model, classify_scene
        print("  Loading scene classification model...")
        scene_model = load_scene_model()

    # ── Find images ───────────────────────────────────────────────────────
    images = find_images(photos_folder)
    if not images:
        print(f"ERROR: No images found in: {photos_folder}")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Scanning {len(images)} photos in: {photos_folder}")
    print(f"  Scan types : {', '.join(scan_types)}")
    if do_faces:
        from common import THRESHOLD
        print(f"  Known people   : {', '.join(sorted(face_db['people'].keys()))}")
        print(f"  Face threshold : {THRESHOLD}")
    print(f"  Output     : {output_json}")
    print(f"{'─'*50}\n")

    # ── Load existing results ─────────────────────────────────────────────
    existing_rows = []
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing_rows = json.load(f)

    results      = []
    skipped      = 0
    errors       = 0
    face_count   = 0
    object_count = 0
    scene_count  = 0

    for img_path in tqdm(images, desc="Scanning photos"):
        img_hash = file_hash(str(img_path))

        if skip_processed and face_db and img_hash in face_db["processed"]:
            skipped += 1
            continue

        # ── Faces ─────────────────────────────────────────────────────────
        if do_faces:
            face_results = detect_faces(img_path, face_db)
            results.extend(face_results)
            face_count += sum(
                1 for r in face_results
                if r["matched_name"] not in ("No face detected", "ERROR")
            )
            errors += sum(1 for r in face_results if r["matched_name"] == "ERROR")

        # ── Objects ───────────────────────────────────────────────────────
        if do_objects:
            obj_result = detect_objects(img_path, object_model)
            results.append(obj_result)
            object_count += obj_result["object_count"]

        # ── Scenes ────────────────────────────────────────────────────────
        if do_scenes:
            scene_result = classify_scene(img_path, scene_model)
            results.append(scene_result)
            if scene_result.get("top_scene"):
                scene_count += 1

        if face_db:
            face_db["processed"].add(img_hash)

    # ── Write results ─────────────────────────────────────────────────────
    all_rows = existing_rows + results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    if face_db:
        save_db(face_db, db_path)

    print(f"\n{'─'*50}")
    print(f"  ✓ Done!")
    print(f"  Photos scanned   : {len(images) - skipped}")
    print(f"  Skipped (cached) : {skipped}")
    print(f"  Errors           : {errors}")
    if do_faces:
        print(f"  Faces matched    : {face_count}")
    if do_objects:
        print(f"  Objects detected : {object_count}")
    if do_scenes:
        print(f"  Scenes classified: {scene_count}")
    print(f"  Results saved  → {output_json}")
    print(f"{'─'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a photo library for faces, objects, and/or scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--photos",         required=True,
                        help="Folder of photos to scan (searched recursively)")
    parser.add_argument("--db",             default="faces.db",
                        help="Path to face database file (default: faces.db)")
    parser.add_argument("--output",         default="results.json",
                        help="Output JSON file (default: results.json)")
    parser.add_argument("--skip-processed", action="store_true",
                        help="Skip photos already in the database cache")
    parser.add_argument("--scan-types",     nargs="+", default=ALL_SCAN_TYPES,
                        help="Scan types to run: faces objects scenes (default: all)")

    args = parser.parse_args()
    scan(args.photos, args.db, args.output, args.skip_processed, args.scan_types)


if __name__ == "__main__":
    main()
