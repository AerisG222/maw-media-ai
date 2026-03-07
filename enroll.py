#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enroll.py — Enroll known people into the face database

Usage:
    ./pt.py enroll --known ./known_people --db faces.db

Folder structure:
    known_people/
        Alice/
            alice1.jpg
            alice2.jpg
        Bob/
            bob1.jpg
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    DeepFace,
    MODEL_NAME, DETECTOR,
    list_images, load_db, save_db,
)


def enroll(known_folder: str, db_path: str):
    """Build face embeddings for all known people."""
    known_folder = Path(known_folder)
    if not known_folder.exists():
        print(f"ERROR: Known people folder not found: {known_folder}")
        sys.exit(1)

    db = load_db(db_path)

    people_dirs = [d for d in known_folder.iterdir() if d.is_dir()]
    if not people_dirs:
        print("ERROR: No sub-folders found. Create one folder per person inside your known_people directory.")
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Enrolling {len(people_dirs)} people from: {known_folder}")
    print(f"  Model: {MODEL_NAME}  |  Detector: {DETECTOR}")
    print(f"{'─'*50}\n")

    for person_dir in sorted(people_dirs):
        name = person_dir.name
        images = list_images(str(person_dir))

        if not images:
            print(f"  ⚠  {name}: no images found, skipping.")
            continue

        print(f"  Enrolling: {name} ({len(images)} photos)")
        embeddings = []

        for img_path in tqdm(images, desc=f"    {name}", leave=False):
            try:
                result = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=True,
                )
                for face in result:
                    embeddings.append(np.array(face["embedding"]))
            except Exception as e:
                print(f"    ⚠  Skipped {img_path.name}: {e}")

        if embeddings:
            db["people"][name] = embeddings
            print(f"    ✓  {name}: {len(embeddings)} face(s) enrolled\n")
        else:
            print(f"    ✗  {name}: no valid faces detected — check your reference photos\n")

    save_db(db, db_path)
    print(f"\nEnrollment complete. {len(db['people'])} people in database.")


def main():
    parser = argparse.ArgumentParser(
        description="Enroll known people from reference photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--known", required=True, help="Folder containing one sub-folder per person")
    parser.add_argument("--db",    default="faces.db", help="Path to face database file (default: faces.db)")

    args = parser.parse_args()
    enroll(args.known, args.db)


if __name__ == "__main__":
    main()
