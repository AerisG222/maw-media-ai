#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list.py — List all image files that would be scanned

Prints a full list of images that would be processed by the scan command,
without actually scanning them. Use this to verify the correct files are
included before running a full scan.

Usage:
    ./pt.py list --photos ./my_photos
    ./pt.py list --photos ./my_photos --ext jpg png
    ./pt.py list --photos ./my_photos --output file-list.txt
"""

import os
import argparse
from pathlib import Path
from collections import Counter

from common import list_images, IMAGE_EXTENSIONS


def list_files(photos_folder: str, extensions: list[str] | None, output_file: str | None):
    """List all image files that would be scanned."""

    # Override extensions if user specified specific ones
    if extensions:
        exts = {f".{e.lower().lstrip('.')}" for e in extensions}
        invalid = exts - IMAGE_EXTENSIONS
        if invalid:
            print(f"  ⚠  Unsupported extensions ignored: {', '.join(sorted(invalid))}")
            exts -= invalid
    else:
        exts = IMAGE_EXTENSIONS

    folder = Path(photos_folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {photos_folder}")
        return

    # Gather matching images
    images = sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in exts and p.is_file()
    ])

    # ── Summary ───────────────────────────────────────────────────────────
    ext_counts = Counter(p.suffix.lower() for p in images)

    print(f"\n{'─'*50}")
    print(f"  Folder  : {folder.resolve()}")
    print(f"  Filter  : {', '.join(sorted(exts))}")
    print(f"  Total   : {len(images)} image(s) found")
    if ext_counts:
        breakdown = "  |  ".join(f"{ext}: {count}" for ext, count in sorted(ext_counts.items()))
        print(f"  Types   : {breakdown}")
    print(f"{'─'*50}\n")

    # ── File listing ──────────────────────────────────────────────────────
    lines = []
    current_dir = None

    for img_path in images:
        parent = str(img_path.parent)

        # Print directory header when we enter a new sub-folder
        if parent != current_dir:
            current_dir = parent
            rel_dir = img_path.parent.relative_to(folder)
            header = f"  📁 {rel_dir}/" if str(rel_dir) != "." else f"  📁 {folder.name}/"
            lines.append("")
            lines.append(header)

        lines.append(f"     {img_path.name}")

    for line in lines:
        print(line)

    print()

    # ── Write to file if requested ────────────────────────────────────────
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Photo Tagger — File List\n")
            f.write(f"# Folder : {folder.resolve()}\n")
            f.write(f"# Filter : {', '.join(sorted(exts))}\n")
            f.write(f"# Total  : {len(images)} image(s)\n")
            f.write("#\n")
            for img_path in images:
                f.write(f"{img_path}\n")
        print(f"  File list saved → {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="List all image files that would be scanned",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--photos",  required=True,
                        help="Folder of photos to list (searched recursively)")
    parser.add_argument("--ext",     nargs="+", default=None,
                        help=f"Limit to specific extensions e.g. --ext jpg png heic "
                             f"(default: {' '.join(sorted(IMAGE_EXTENSIONS))})")
    parser.add_argument("--output",  default=None,
                        help="Optionally save the file list to a text file")

    args = parser.parse_args()
    list_files(args.photos, args.ext, args.output)


if __name__ == "__main__":
    main()
