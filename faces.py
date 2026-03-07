#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
faces.py — Detect and match faces in photos

Detects faces using DeepFace and matches them against enrolled people
in the face database. Must run enroll first to populate the database.

Usage (via pt.py):
    ./pt.py scan --photos ./my_photos --scan-types faces --output results.json
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np

from common import (
    DeepFace,
    MODEL_NAME, DETECTOR,
    load_db, match_face,
)


def load_face_model(db_path: str) -> dict:
    """
    Load the face database. Exits with a clear error if no people are enrolled.
    Returns the database dict for use in detect_faces().
    """
    db = load_db(db_path)
    if not db["people"]:
        print("ERROR: No people enrolled yet. Run the 'enroll' command first.")
        print("       Or use --scan-types objects scenes to skip face scanning.")
        sys.exit(1)
    return db


def detect_faces(img_path: Path, db: dict) -> list[dict]:
    """
    Detect and match all faces in a single image.
    Returns a list of result dicts — one per detected face, or one error/no-face entry.
    """
    try:
        faces = DeepFace.represent(
            img_path=str(img_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False,
        )
    except Exception as e:
        return [{
            "file_path":    str(img_path),
            "file_name":    img_path.name,
            "scan_type":    "faces",
            "face_index":   0,
            "matched_name": "ERROR",
            "confidence":   None,
            "distance":     None,
            "face_region":  {},
            "scanned_at":   datetime.now().isoformat(timespec="seconds"),
            "notes":        str(e),
        }]

    if not faces:
        return [{
            "file_path":    str(img_path),
            "file_name":    img_path.name,
            "scan_type":    "faces",
            "face_index":   0,
            "matched_name": "No face detected",
            "confidence":   None,
            "distance":     None,
            "face_region":  {},
            "scanned_at":   datetime.now().isoformat(timespec="seconds"),
            "notes":        "",
        }]

    results = []
    for i, face_data in enumerate(faces):
        embedding  = np.array(face_data["embedding"])
        name, dist = match_face(embedding, db["people"])
        confidence = round((1 - dist) * 100, 1)
        region     = face_data.get("facial_area", {})

        results.append({
            "file_path":    str(img_path),
            "file_name":    img_path.name,
            "scan_type":    "faces",
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

    return results
