#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster.py — Bootstrap the known faces database via unsupervised clustering

Samples a percentage of your photo library, detects all faces, clusters them
by similarity using DBSCAN, and exports representative face crops into the
known_people folder structure ready for enrollment.

On subsequent runs, existing folders in --output are loaded as anchors.
New clusters are matched against these anchors first — if close enough they
are merged into the existing folder rather than creating a new person_NNN
folder. Repeated runs progressively build up the database without losing
prior work.

After running this command:
  1. Review the output folders in known_people/
  2. Rename each person_NNN folder to the person's real name
  3. Delete any noise or unrecognisable folders
  4. Run:  ./pt.py enroll --known ./known_people --db faces.db

Usage:
    # First run — sample 10% of photos
    ./pt.py cluster --photos ./my_photos --output ./known_people

    # Second run — builds on previous output automatically
    ./pt.py cluster --photos ./my_photos --output ./known_people --sample 20

    # Restrict to a year range (matches your <root>/<year>/... structure)
    ./pt.py cluster --photos ./my_photos --output ./known_people --years 2010 2015

    # Tune clustering sensitivity
    ./pt.py cluster --photos ./my_photos --output ./known_people --eps 0.35 --min-samples 3
"""

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
    find_images, cosine_distance,
)

# ── Clustering defaults ───────────────────────────────────────────────────
DEFAULT_SAMPLE_PCT    = 10    # % of photos to sample
DEFAULT_EPS           = 0.40  # DBSCAN: max cosine distance to join a cluster
DEFAULT_MIN_SAMPLES   = 3     # DBSCAN: min faces to form a cluster
DEFAULT_ANCHOR_THRESH = 0.40  # max distance to merge a new cluster into an anchor
MAX_CROPS_PER_CLUSTER = 10    # max reference crops to export per person


# ── Anchor loading ────────────────────────────────────────────────────────

def load_anchors_from_output(output_folder: Path) -> dict[str, list[np.ndarray]]:
    """
    Load embeddings from face crops already present in the output folder.
    Each sub-folder name becomes an anchor label.
    Returns {folder_name: [embeddings]}.
    """
    anchors = {}
    if not output_folder.exists():
        return anchors

    person_dirs = [d for d in output_folder.iterdir() if d.is_dir() and d.name != "unmatched"]
    if not person_dirs:
        return anchors

    print(f"  Loading anchors from existing output folder...")
    for person_dir in sorted(person_dirs):
        images = find_images(str(person_dir), require_4k=False)
        if not images:
            continue

        embeddings = []
        for img_path in images:
            try:
                result = DeepFace.represent(
                    img_path=str(img_path),
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False,
                )
                for face in result:
                    emb = np.array(face["embedding"])
                    if emb is not None:
                        embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            anchors[person_dir.name] = embeddings

    print(f"  Found {len(anchors)} existing anchor(s): {', '.join(sorted(anchors.keys()))}\n")
    return anchors


def anchor_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """Compute the mean (centroid) embedding for a set of face embeddings."""
    stacked = np.array(embeddings)
    mean    = stacked.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-10)


def match_cluster_to_anchor(
    cluster_embeddings: list[np.ndarray],
    anchors: dict[str, list[np.ndarray]],
    threshold: float,
) -> str | None:
    """
    Compare the centroid of a new cluster against all anchor centroids.
    Returns the name of the closest anchor if within threshold, else None.
    """
    if not anchors:
        return None

    cluster_centroid = anchor_centroid(cluster_embeddings)
    best_name = None
    best_dist = float("inf")

    for name, embeddings in anchors.items():
        centroid = anchor_centroid(embeddings)
        dist     = cosine_distance(cluster_centroid, centroid)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name if best_dist <= threshold else None


# ── Face extraction ───────────────────────────────────────────────────────

def sample_images(images: list[Path], pct: float, years: tuple | None) -> list[Path]:
    """
    Filter images to a year range if specified, then randomly sample pct%.
    Year is inferred from path components (the <year> level).
    """
    if years:
        year_start, year_end = years
        filtered = []
        for p in images:
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
        except Exception:
            continue

        for i, face in enumerate(faces):
            emb    = np.array(face["embedding"])
            region = face.get("facial_area", {})
            embeddings.append(emb)
            metadata.append({
                "img_path":   img_path,
                "face_index": i,
                "region":     region,
            })

    return embeddings, metadata


# ── Clustering ────────────────────────────────────────────────────────────

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

    normed = normalize(np.array(embeddings))
    db     = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    return db.fit_predict(normed)


# ── Face cropping & export ────────────────────────────────────────────────

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

        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        return crop

    except Exception:
        return None


def next_crop_index(folder: Path, folder_name: str) -> int:
    """
    Return the next available crop index for a folder, accounting for
    any crops already saved there from a previous run.
    """
    existing = list(folder.glob(f"{folder_name}_*.jpg"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        stem = f.stem  # e.g. "person_001_003"
        try:
            indices.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


def is_duplicate(
    candidate: np.ndarray,
    existing: list[np.ndarray],
    threshold: float = 0.15,
) -> bool:
    """
    Return True if the candidate embedding is too similar to any existing one.
    A low threshold (default 0.15) catches near-identical frames while still
    allowing genuinely different photos of the same person to be saved.
    """
    for emb in existing:
        if cosine_distance(candidate, emb) < threshold:
            return True
    return False


def export_clusters(
    labels: np.ndarray,
    metadata: list[dict],
    embeddings: list[np.ndarray],
    output_folder: Path,
    anchors: dict[str, list[np.ndarray]],
    anchor_threshold: float,
    max_crops: int,
):
    """
    Export face crops grouped by cluster label into the output folder.

    For each cluster:
      - If it matches an existing anchor, crops are added to that folder
      - Otherwise a new person_NNN folder is created
    Near-duplicate faces (too similar to one already saved) are skipped.
    Noise faces (label -1) go into the 'unmatched' sub-folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    new_clusters  = [l for l in unique_labels if l >= 0]
    n_noise       = int(np.sum(labels == -1))

    print(f"\n  New clusters found : {len(new_clusters)}")
    print(f"  Unmatched faces    : {n_noise}")
    print(f"\n  Resolving clusters against anchors...\n")

    # Figure out next available person index for genuinely new clusters
    existing_person_dirs = [
        d for d in output_folder.iterdir()
        if d.is_dir() and d.name.startswith("person_")
    ]
    next_person_idx = len(existing_person_dirs) + 1

    merged_count   = 0
    new_count      = 0

    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]

        if label == -1:
            folder_name = "unmatched"
        else:
            cluster_embs   = [embeddings[i] for i in indices]
            matched_anchor = match_cluster_to_anchor(cluster_embs, anchors, anchor_threshold)

            if matched_anchor:
                folder_name = matched_anchor
                merged_count += 1
                print(f"  Cluster {label+1:>3}  →  merged into '{folder_name}' ({len(indices)} face(s))")
            else:
                width       = max(3, len(str(next_person_idx + len(new_clusters))))
                folder_name = f"person_{str(next_person_idx).zfill(width)}"
                next_person_idx += 1
                new_count += 1
                print(f"  Cluster {label+1:>3}  →  new folder '{folder_name}' ({len(indices)} face(s))")

        cluster_dir = output_folder / folder_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Seed the seen-embeddings list from the anchor so we also deduplicate
        # against faces saved in previous runs, not just the current batch
        seen_embeddings: list[np.ndarray] = list(anchors.get(folder_name, []))

        # Pick a diverse spread of crops up to max_crops
        step    = max(1, len(indices) // max_crops)
        to_save = indices[::step][:max_crops]

        start_idx = next_crop_index(cluster_dir, folder_name)
        saved     = 0
        skipped   = 0

        for idx in to_save:
            emb  = embeddings[idx]
            meta = metadata[idx]

            if is_duplicate(emb, seen_embeddings):
                skipped += 1
                continue

            crop = crop_face(meta["img_path"], meta["region"])
            if crop is None:
                continue

            out_path = cluster_dir / f"{folder_name}_{start_idx + saved:03d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            seen_embeddings.append(emb)
            saved += 1

        if skipped:
            print(f"             {skipped} near-duplicate(s) skipped")

    print(f"\n  Merged into existing : {merged_count}")
    print(f"  New folders created  : {new_count}")
    print(f"  Crops saved        → {output_folder.resolve()}")


# ── Main pipeline ─────────────────────────────────────────────────────────

def cluster(
    photos_folder: str,
    output_folder: str,
    sample_pct: float,
    years: tuple | None,
    eps: float,
    min_samples: int,
    anchor_threshold: float,
):
    """Full clustering pipeline: load anchors → sample → detect → cluster → export."""

    output_path = Path(output_folder)

    print(f"\n{'─'*50}")
    print(f"  Clustering faces in : {photos_folder}")
    print(f"  Sample rate         : {sample_pct}%")
    if years:
        print(f"  Year range          : {years[0]}–{years[1]}")
    print(f"  DBSCAN eps          : {eps}  |  min samples: {min_samples}")
    print(f"  Anchor threshold    : {anchor_threshold}")
    print(f"  Output              : {output_folder}")
    print(f"{'─'*50}\n")

    # ── Load anchors from any existing output folders ─────────────────────
    anchors = load_anchors_from_output(output_path)

    if anchors:
        print(f"  Total anchors loaded: {len(anchors)} person(s)\n")
    else:
        print(f"  No existing anchors found — all clusters will be new.\n")

    # ── Sample images ─────────────────────────────────────────────────────
    all_images = find_images(photos_folder)
    if not all_images:
        print(f"ERROR: No images found in: {photos_folder}")
        sys.exit(1)

    sampled = sample_images(all_images, sample_pct, years)
    print(f"  Total images : {len(all_images)}")
    print(f"  Sampled      : {len(sampled)} ({sample_pct}%)\n")

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
    export_clusters(
        labels=labels,
        metadata=metadata,
        embeddings=embeddings,
        output_folder=output_path,
        anchors=anchors,
        anchor_threshold=anchor_threshold,
        max_crops=MAX_CROPS_PER_CLUSTER,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\n{'─'*50}")
    print(f"  ✓ Done! {n_clusters} cluster(s) processed.")
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
    parser.add_argument("--photos",           required=True,
                        help="Folder of photos to sample from")
    parser.add_argument("--output",           required=True,
                        help="Output folder for clustered face crops (e.g. ./known_people)")
    parser.add_argument("--sample",           type=float, default=DEFAULT_SAMPLE_PCT,
                        help=f"Percentage of photos to sample (default: {DEFAULT_SAMPLE_PCT})")
    parser.add_argument("--years",            type=int, nargs=2, metavar=("START", "END"),
                        help="Restrict to a year range e.g. --years 2010 2015")
    parser.add_argument("--eps",              type=float, default=DEFAULT_EPS,
                        help=f"DBSCAN: max distance to join a cluster (default: {DEFAULT_EPS}, lower = stricter)")
    parser.add_argument("--min-samples",      type=int, default=DEFAULT_MIN_SAMPLES,
                        help=f"DBSCAN: min faces to form a cluster (default: {DEFAULT_MIN_SAMPLES})")
    parser.add_argument("--anchor-threshold", type=float, default=DEFAULT_ANCHOR_THRESH,
                        help=f"Max distance to merge a new cluster into an existing anchor (default: {DEFAULT_ANCHOR_THRESH})")

    args = parser.parse_args()
    cluster(
        photos_folder=args.photos,
        output_folder=args.output,
        sample_pct=args.sample,
        years=tuple(args.years) if args.years else None,
        eps=args.eps,
        min_samples=args.min_samples,
        anchor_threshold=args.anchor_threshold,
    )


if __name__ == "__main__":
    main()
