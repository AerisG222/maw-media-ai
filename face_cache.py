"""Shared helpers for the on-disk face-crop cache.

Both the scanner (scan-faces.py) and the Streamlit explorer (view-clusters-app.py)
locate pre-extracted face crops through this module so the path scheme stays in
sync.  Crops mirror the source photo's directory tree under ``image-cache/faces``
so that no single directory ends up holding tens of thousands of files, e.g.

    /photos/2020/full/img.jpg  +  face <uuid>
      ->  /data/face-cache/faces/photos/2020/full/img.jpg/<uuid>.jpg
"""

import os
from pathlib import Path

# Root of the image cache (thumbnails live at the top level; crops under faces/).
# Override with FACE_CACHE_DIR; defaults to /data/face-cache which has room to grow.
CACHE_ROOT = Path(os.getenv("FACE_CACHE_DIR", "/data/face-cache"))
FACE_CROP_DIR = CACHE_ROOT / "faces"


def face_crop_dir(source_path: str) -> Path:
    """Return the directory holding all crops for one source photo.

    Faces from a photo are grouped in a directory named after the source file
    (extension included, so ``a.jpg`` and ``a.png`` never collide).
    """
    src = Path(source_path)
    # Drop the filesystem anchor ("/" or a drive) so the path nests under the cache.
    parts = src.parts[1:] if src.anchor else src.parts
    return FACE_CROP_DIR.joinpath(*parts)


def face_crop_path(source_path: str, face_id: str) -> Path:
    """Return the on-disk path for a single face crop under its photo's dir."""
    return face_crop_dir(source_path) / f"{face_id}.jpg"
