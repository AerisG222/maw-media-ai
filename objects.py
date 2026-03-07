#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
objects.py — Detect objects in photos using YOLO11

Detects and labels objects with bounding boxes using a YOLO11 model
pretrained on the COCO dataset (80 classes: people, cars, dogs, chairs, etc.)
Models are downloaded automatically on first use.

Usage (via pt.py):
    ./pt.py scan --photos ./my_photos --scan-types objects --output results.json
"""

import sys
from pathlib import Path
from datetime import datetime

from common import YOLO, YOLO_OBJECT_MODEL, YOLO_CONFIDENCE


def load_object_model():
    """Load the YOLO object detection model, with a clear error if ultralytics is missing."""
    if YOLO is None:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)
    return YOLO(YOLO_OBJECT_MODEL)


def detect_objects(img_path: Path, model) -> dict:
    """
    Run object detection on a single image.
    Returns a dict with detected objects and their bounding boxes.
    """
    try:
        results = model.predict(
            source=str(img_path),
            conf=YOLO_CONFIDENCE,
            verbose=False,
        )

        objects = []
        for result in results:
            names = result.names  # class index → label mapping
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                label      = names[cls_id]
                confidence = round(float(box.conf[0]) * 100, 1)
                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]

                objects.append({
                    "label":      label,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "w":  x2 - x1,
                        "h":  y2 - y1,
                    },
                })

        return {
            "file_path":   str(img_path),
            "file_name":   img_path.name,
            "scan_type":   "objects",
            "objects":     objects,
            "object_count": len(objects),
            "labels":      sorted(set(o["label"] for o in objects)),
            "scanned_at":  datetime.now().isoformat(timespec="seconds"),
            "notes":       "",
        }

    except Exception as e:
        return {
            "file_path":    str(img_path),
            "file_name":    img_path.name,
            "scan_type":    "objects",
            "objects":      [],
            "object_count": 0,
            "labels":       [],
            "scanned_at":   datetime.now().isoformat(timespec="seconds"),
            "notes":        f"ERROR: {e}",
        }
