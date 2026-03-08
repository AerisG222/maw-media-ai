#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common.py — Shared utilities, constants, and GPU configuration
"""

import os
import sys
import hashlib
import pickle
from pathlib import Path

import numpy as np

# ── Suppress TensorFlow noise before any imports ──────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── DeepFace import with clear error message ──────────────────────────────
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


# ── Constants ─────────────────────────────────────────────────────────────
MODEL_NAME       = "ArcFace"     # Best accuracy for personal collections
DETECTOR         = "retinaface"  # Best detector; fall back to "opencv" if slow
DISTANCE_METRIC  = "cosine"
THRESHOLD        = 0.40          # Lower = stricter matching (0.0–1.0)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tiff", ".webp", ".avif"}

# YOLO models (downloaded automatically on first use)
YOLO_OBJECT_MODEL = "yolo11n.pt"       # COCO — 80 common object classes
YOLO_SCENE_MODEL  = "yolo11n-cls.pt"   # ImageNet — 1000 scene/category classes
YOLO_CONFIDENCE   = 0.30               # Minimum confidence to include a detection


# ── YOLO import ───────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Gracefully degrade — face scanning still works without YOLO


# ── GPU configuration ─────────────────────────────────────────────────────

def configure_gpu():
    """
    Detect and configure GPU acceleration.
    Enables memory growth to prevent TensorFlow grabbing all VRAM at once.
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


# ── File utilities ────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """MD5 hash of file contents — used to detect already-processed photos."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_images(
    folder: str,
    extensions: set[str] | None = None,
    require_4k: bool = True,
) -> list[Path]:
    """
    Recursively find all image files in a folder.

    Args:
        folder:     Root folder to search.
        extensions: Set of lowercase extensions to include e.g. {'.jpg', '.png'}.
                    Defaults to IMAGE_EXTENSIONS if not provided.
        require_4k: When True (default), only returns images whose immediate
                    parent directory is named '4k', matching the library structure:
                        <root>/<year>/<category>/4k/<images>
                    Set to False for flat folders like known_people/ or cluster output.

    Returns:
        Sorted list of matching Path objects.
    """
    exts   = extensions if extensions is not None else IMAGE_EXTENSIONS
    folder = Path(folder)
    return sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in exts
        and p.is_file()
        and (not require_4k or p.parent.name.lower() == "4k")
    ])


# Keep list_images as a backwards-compatible alias
def list_images(folder: str) -> list[Path]:
    return find_images(folder)


# ── Database utilities ────────────────────────────────────────────────────

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


# ── Face matching ─────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(1 - np.dot(a, b))


def match_face(embedding: np.ndarray, people: dict) -> tuple[str, float]:
    """
    Compare an embedding against all known people.
    Returns (best_match_name, distance). Name is 'Unknown' if no match.
    """
    best_name = "Unknown"
    best_dist = float("inf")

    for name, embeddings in people.items():
        distances = [cosine_distance(embedding, ref) for ref in embeddings]
        avg_dist = float(np.mean(distances))
        if avg_dist < best_dist:
            best_dist = avg_dist
            best_name = name

    if best_dist > THRESHOLD:
        return "Unknown", best_dist
    return best_name, best_dist
