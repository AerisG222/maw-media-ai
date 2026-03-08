#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster.py — Bootstrap the known faces database via unsupervised clustering

Samples a percentage of your photo library, detects all faces, clusters them
by similarity using DBSCAN, and exports representative face crops into the
known_people folder structure ready for enrollment.

After running this command:
  1. Review the output folders in known_people/
  2. Rename each person_NNN folder to the person's real name
  3. Delete any noise or unrecognisable folders
  4. Run:  ./pt.py enroll --known ./known_people --db faces.db

Usage:
    # Sample 10% of photos (default)
    ./pt.py cluster --photos ./my_photos --output ./known_people

    # Sample a specific percentage
    ./pt.py cluster --photos ./my_photos --output ./known_people --sample 20

    # Restrict to a year range (matches your <root>/<year>/... structure)
    ./pt.py cluster --photos ./my_photos --output ./known_people --years 2010 2015

    # Tune clustering sensitivity
    ./pt.py cluster --photos ./my_photos --output ./known_people --eps 0.35 --min-samples 3
"""

import os
import sys
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import (
    DeepFace,
    MODEL_NAME, DETECTOR,
    find_images,
)

# ── Clustering defaults ───────────────────────────────────────────────────
DEFAULT_SAMPLE_PCT  = 10     # % of photos to sample
DEFAULT_EPS         = 0.40   # DBSCAN: max cosine distance to join a cluster
DEFAULT_MIN_SAMPLES = 3      # DBSCAN: min faces to form a cluster
MAX_CROPS_PER_CLUSTER = 10   # max reference crops to export per person


def sample_images(images: list[Path], pct: float, years: tuple | None) -> list[Path]:
    """
    Filter images to a year range if specified, then randomly sample pct%.
    Year is inferred from the grandparent directory name (the <year> level).
    """
    if years:
        year_start, year_end = years
        filtered = []
        for p in images:
            # Walk up path parts to find a 4-digit year component
            for part in p.parts:
                if part.isdigit() and len(part) == 4:
                    y = int(part)
                    if year_start <= y <= year_end:
                        filtered.append(p)
                    break
        images = filtered
        if not images:
            print(f"  ⚠  No images found for years {year_start}–{year_end}.")
            sys.exit(1)

    k = max(1, int(len(images) * pct / 100))
    return random.sample(images, min(k, len(images)))


def extract_embeddings(images: list[Path]) -> tuple[list[np.ndarray], list[dict]]:
    """
    Detect faces in each image and extract ArcFace embeddings.
    Returns parallel lists of embeddings and metadata dicts.
    """
    embeddings = []
    metadata   = []

    for img_path in tqdm(images, desc="  Extracting faces"):
        try:
            faces = DeepFace.represent(
                img_path=str(img_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False,
            )
        except Exception as e:
            continue

        for i, face in enumerate(faces):
            emb    = np.array(face["embedding"])
            region = face.get("facial_area", {})
            embeddings.append(emb)
            metadata.append({
                "img_path":  img_path,
                "face_index": i,
                "region":    region,
            })

    return embeddings, metadata


def cluster_embeddings(
    embeddings: list[np.ndarray],
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """
    Cluster face embeddings using DBSCAN with cosine distance.
    Returns an array of cluster labels (-1 = noise/unmatched).
    """
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import normalize
    except ImportError:
        print("ERROR: scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    if len(embeddings) < 2:
        print("  ⚠  Not enough faces detected to cluster.")
        sys.exit(1)

    # Normalise embeddings to unit length so euclidean distance ≈ cosine distance
    normed = normalize(np.array(embeddings))
    # eps in cosine space; sklearn DBSCAN uses euclidean on normalised vectors
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    return db.fit_predict(normed)


def crop_face(img_path: Path, region: dict, padding: float = 0.20) -> np.ndarray | None:
    """
    Crop a face from an image with optional padding around the detected region.
    Returns the cropped face as a numpy array, or None on failure.
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        h_img, w_img = img.shape[:2]
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        if w <= 0 or h <= 0:
            return None

        # Add padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Resize to a consistent reference size
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        return crop

    except Exception:
        return None


def export_clusters(
    labels: np.ndarray,
    metadata: list[dict],
    output_folder: str,
    max_crops: int,
):
    """
    Export face crops grouped by cluster label into the output folder.
    Noise faces (label -1) go into an 'unmatched' sub-folder.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    n_clusters    = sum(1 for l in unique_labels if l >= 0)
    n_noise       = sum(1 for l in labels if l == -1)

    print(f"\n  Clusters found : {n_clusters}")
    print(f"  Unmatched faces: {n_noise}")
    print(f"\n  Exporting face crops...\n")

    # Pad cluster index for consistent folder naming
    width = len(str(n_clusters))

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]

        if label == -1:
            folder_name = "unmatched"
        else:
            folder_name = f"person_{str(label + 1).zfill(width)}"

        cluster_dir = output_folder / folder_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Pick a diverse spread of crops up to max_crops
        step    = max(1, len(indices) // max_crops)
        to_save = indices[::step][:max_crops]

        saved = 0
        for idx in to_save:
            meta  = metadata[idx]
            crop  = crop_face(meta["img_path"], meta["region"])
            if crop is None:
                continue

            out_path = cluster_dir / f"{folder_name}_{saved + 1:03d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        if label >= 0:
            print(f"  {folder_name}  →  {len(indices)} face(s) detected, {saved} crop(s) saved")

    print(f"\n  Crops saved → {output_folder.resolve()}")


def cluster(
    photos_folder: str,
    output_folder: str,
    sample_pct: float,
    years: tuple | None,
    eps: float,
    min_samples: int,
):
    """Full clustering pipeline: sample → detect → cluster → export."""

    print(f"\n{'─'*50}")
    print(f"  Clustering faces in: {photos_folder}")
    print(f"  Sample rate : {sample_pct}%")
    if years:
        print(f"  Year range  : {years[0]}–{years[1]}")
    print(f"  DBSCAN eps  : {eps}  |  min samples: {min_samples}")
    print(f"  Output      : {output_folder}")
    print(f"{'─'*50}\n")

    # ── Sample images ─────────────────────────────────────────────────────
    all_images = find_images(photos_folder)
    if not all_images:
        print(f"ERROR: No images found in: {photos_folder}")
        sys.exit(1)

    sampled = sample_images(all_images, sample_pct, years)
    print(f"  Total images  : {len(all_images)}")
    print(f"  Sampled       : {len(sampled)} ({sample_pct}%)\n")

    # ── Extract embeddings ────────────────────────────────────────────────
    embeddings, metadata = extract_embeddings(sampled)

    if not embeddings:
        print("ERROR: No faces detected in sampled images.")
        print("  Try increasing --sample or check your photos folder.")
        sys.exit(1)

    print(f"\n  Faces detected: {len(embeddings)}\n")

    # ── Cluster ───────────────────────────────────────────────────────────
    print("  Clustering embeddings...")
    labels = cluster_embeddings(embeddings, eps, min_samples)

    # ── Export ────────────────────────────────────────────────────────────
    export_clusters(labels, metadata, output_folder, MAX_CROPS_PER_CLUSTER)

    # ── Next steps ────────────────────────────────────────────────────────
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\n{'─'*50}")
    print(f"  ✓ Done! {n_clusters} potential people found.")
    print(f"{'─'*50}")
    print(f"\n  Next steps:")
    print(f"  1. Review folders in: {output_folder}")
    print(f"  2. Rename each person_NNN folder to the person's real name")
    print(f"  3. Delete noise or unrecognisable folders")
    print(f"  4. Run enrollment:")
    print(f"       ./pt.py enroll --known {output_folder} --db faces.db\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap the known faces database via unsupervised clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--photos",      required=True,
                        help="Folder of photos to sample from")
    parser.add_argument("--output",      required=True,
                        help="Output folder for clustered face crops (e.g. ./known_people)")
    parser.add_argument("--sample",      type=float, default=DEFAULT_SAMPLE_PCT,
                        help=f"Percentage of photos to sample (default: {DEFAULT_SAMPLE_PCT})")
    parser.add_argument("--years",       type=int, nargs=2, metavar=("START", "END"),
                        help="Restrict to a year range e.g. --years 2010 2015")
    parser.add_argument("--eps",         type=float, default=DEFAULT_EPS,
                        help=f"DBSCAN: max distance to join a cluster (default: {DEFAULT_EPS}, lower = stricter)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN: min faces to form a cluster (default: {DEFAULT_MIN_SAMPLES})")

    args = parser.parse_args()
    cluster(
        photos_folder=args.photos,
        output_folder=args.output,
        sample_pct=args.sample,
        years=tuple(args.years) if args.years else None,
        eps=args.eps,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
