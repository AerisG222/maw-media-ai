#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenes.py — Classify scenes in photos using YOLO11

Classifies the overall scene or dominant subject of an image using a YOLO11
model pretrained on ImageNet (1000 categories: beach, kitchen, forest, dog, etc.)
Models are downloaded automatically on first use.

Usage (via pt.py):
    ./pt.py scan --photos ./my_photos --scan-types scenes --output results.json
"""

import sys
from pathlib import Path
from datetime import datetime

from common import YOLO, YOLO_SCENE_MODEL, YOLO_CONFIDENCE


# Number of top scene classifications to return per image
TOP_N = 5


def load_scene_model():
    """Load the YOLO scene classification model, with a clear error if ultralytics is missing."""
    if YOLO is None:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)
    return YOLO(YOLO_SCENE_MODEL)


def classify_scene(img_path: Path, model) -> dict:
    """
    Run scene classification on a single image.
    Returns a dict with the top scene labels and their confidence scores.
    """
    try:
        results = model.predict(
            source=str(img_path),
            verbose=False,
        )

        scenes = []
        for result in results:
            # result.probs contains classification probabilities
            probs  = result.probs
            names  = result.names

            # Get top N predictions sorted by confidence
            top_indices = probs.top5  # indices of top 5 classes
            for idx in top_indices:
                confidence = round(float(probs.data[idx]) * 100, 1)
                if confidence >= (YOLO_CONFIDENCE * 100):
                    scenes.append({
                        "label":      names[int(idx)],
                        "confidence": confidence,
                    })

        # Sort by confidence descending
        scenes = sorted(scenes, key=lambda x: x["confidence"], reverse=True)[:TOP_N]

        return {
            "file_path":     str(img_path),
            "file_name":     img_path.name,
            "scan_type":     "scenes",
            "scenes":        scenes,
            "top_scene":     scenes[0]["label"] if scenes else None,
            "scanned_at":    datetime.now().isoformat(timespec="seconds"),
            "notes":         "",
        }

    except Exception as e:
        return {
            "file_path":   str(img_path),
            "file_name":   img_path.name,
            "scan_type":   "scenes",
            "scenes":      [],
            "top_scene":   None,
            "scanned_at":  datetime.now().isoformat(timespec="seconds"),
            "notes":       f"ERROR: {e}",
        }
