#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Run prep.sh first to set up the conda environment, then:
#   conda activate photo-tagger
#   ./photo-tagger.py <command>
"""
photo-tagger.py — Local Facial Recognition Photo Tagger
--------------------------------------------------------
Scans a folder of photos, detects faces, matches them against known people,
and saves results to a JSON file.

Setup:
    ./prep.sh                       # create conda environment and install dependencies
    conda activate photo-tagger     # enter the environment

Usage:
    # Step 1: Enroll known people (run once, or when you add new people)
    ./photo-tagger.py enroll --known ./known_people --db faces.db

    # Step 2: Scan your photo library
    ./photo-tagger.py scan --photos ./my_photos --db faces.db --output results.json

    # Step 3: Re-scan only new/unprocessed photos
    ./photo-tagger.py scan --photos ./my_photos --db faces.db --output results.json --skip-processed

    # Step 4: View a summary of results
    ./photo-tagger.py report --output results.json
"""

import os
import sys
import json
import argparse
import hashlib
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Optional GPU acceleration ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow noise

try:
    from deepface import DeepFace
except Exception as e:
    print("ERROR: deepface failed to import.")
    print(f"  Active Python : {sys.executable}")
    print(f"  Active env    : {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f"  Error         : {e}")
    print()
    print("  Possible fixes:")
    print("  1. Make sure you are in the conda environment:  conda activate photo-tagger")
    print("  2. Or install manually:                         pip install deepface opencv-python tqdm tf-keras")
    sys.exit(1)

def configure_gpu():
    """
    Detect and configure GPU acceleration.
    - Enables memory growth to prevent TensorFlow grabbing all VRAM at once.
    - Prints a clear summary of what hardware will be used.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            names = []
            for i, gpu in enumerate(gpus):
                details = tf.config.experimental.get_device_details(gpu)
                name = details.get("device_name", f"GPU:{i}")
                names.append(name)
            print(f"  🚀 GPU acceleration enabled: {', '.join(names)}")
        else:
            # Check if this is Apple Silicon and suggest tensorflow-metal
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                print("  ⚠  No GPU found. For Apple Silicon speedup, run:")
                print("     pip install tensorflow-macos tensorflow-metal")
            else:
                print("  ℹ  No GPU found — running on CPU.")
                print("     For NVIDIA GPU support, install CUDA + cuDNN and run:")
                print("     pip install tensorflow[and-cuda]")
    except Exception as e:
        print(f"  ⚠  GPU check failed ({e}) — continuing on CPU.")

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_NAME      = "ArcFace"          # Best accuracy for personal collections
DETECTOR        = "retinaface"       # Best detector; fall back to "opencv" if slow
DISTANCE_METRIC = "cosine"
THRESHOLD       = 0.40               # Lower = stricter matching (0.0–1.0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tiff", ".webp"}


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def file_hash(path: str) -> str:
    """MD5 hash of file contents — used to detect already-processed photos."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(folder: str) -> list[Path]:
    """Recursively find all image files in a folder."""
    folder = Path(folder)
    return [
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]


def load_db(db_path: str) -> dict:
    """Load the face embeddings database from disk."""
    if os.path.exists(db_path):
        with open(db_path, "rb") as f:
            return pickle.load(f)
    return {"people": {}, "processed": set()}


def save_db(db: dict, db_path: str):
    """Persist the face embeddings database to disk."""
    with open(db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"  Database saved → {db_path}")


# ══════════════════════════════════════════════════════════════════════════
#  ENROLL KNOWN PEOPLE
# ══════════════════════════════════════════════════════════════════════════

def enroll(known_folder: str, db_path: str):
    """
    Build face embeddings for all known people.

    Expected folder structure:
        known_people/
            Alice/
                alice1.jpg
                alice2.jpg
            Bob/
                bob1.jpg
    """
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


# ══════════════════════════════════════════════════════════════════════════
#  MATCH A FACE EMBEDDING AGAINST THE DATABASE
# ══════════════════════════════════════════════════════════════════════════

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))


def match_face(embedding: np.ndarray, people: dict) -> tuple[str, float]:
    """
    Compare an embedding against all known people.
    Returns (best_match_name, distance).  Name is 'Unknown' if no match.
    """
    best_name = "Unknown"
    best_dist = float("inf")

    for name, embeddings in people.items():
        # Average distance across all reference embeddings for this person
        distances = [cosine_distance(embedding, ref) for ref in embeddings]
        avg_dist = float(np.mean(distances))
        if avg_dist < best_dist:
            best_dist = avg_dist
            best_name = name

    if best_dist > THRESHOLD:
        return "Unknown", best_dist
    return best_name, best_dist


# ══════════════════════════════════════════════════════════════════════════
#  SCAN PHOTO LIBRARY
# ══════════════════════════════════════════════════════════════════════════

def scan(photos_folder: str, db_path: str, output_json: str, skip_processed: bool):
    """
    Scan all photos in a folder, detect faces, match against known people,
    and write results to a JSON file.
    """
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
                enforce_detection=False,   # don't crash if no face found
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
    print(f"  Faces found     : {sum(1 for r in results if r['matched_name'] not in ('No face detected','ERROR'))}")
    print(f"  Results saved → {output_json}")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════
#  REPORT: SUMMARY OF JSON RESULTS
# ══════════════════════════════════════════════════════════════════════════

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

    from collections import Counter
    counts = Counter(r["matched_name"] for r in rows if r.get("matched_name"))
    for name, count in counts.most_common():
        print(f"  {name:<25} {count} face(s)")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Local facial recognition photo tagger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    p_enroll = sub.add_parser("enroll", help="Enroll known people from reference photos")
    p_enroll.add_argument("--known",  required=True, help="Folder containing one sub-folder per person")
    p_enroll.add_argument("--db",     default="faces.db", help="Path to face database file (default: faces.db)")

    # scan
    p_scan = sub.add_parser("scan", help="Scan a photo library and tag faces")
    p_scan.add_argument("--photos",          required=True,          help="Folder of photos to scan")
    p_scan.add_argument("--db",              default="faces.db",     help="Path to face database file")
    p_scan.add_argument("--output",          default="results.json",  help="Output JSON file (default: results.json)")
    p_scan.add_argument("--skip-processed",  action="store_true",    help="Skip photos already in the database")

    # report
    p_report = sub.add_parser("report", help="Print a summary of results")
    p_report.add_argument("--output", default="results.json", help="JSON file to summarise")

    args = parser.parse_args()

    configure_gpu()
    print()

    if args.command == "enroll":
        enroll(args.known, args.db)
    elif args.command == "scan":
        scan(args.photos, args.db, args.output, args.skip_processed)
    elif args.command == "report":
        report(args.output)


if __name__ == "__main__":
    main()
