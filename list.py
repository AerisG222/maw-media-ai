#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list.py — List all image files that would be scanned

Prints the exact set of files that scan.py would process,
without actually scanning them. Use this to verify the correct
files are included before running a full scan.

Usage:
    ./pt.py list --photos ./my_photos
    ./pt.py list --photos ./my_photos --ext jpg png heic
    ./pt.py list --photos ./my_photos --output file-list.txt
"""

import argparse
from pathlib import Path
from collections import Counter

from common import find_images, IMAGE_EXTENSIONS


def list_files(photos_folder: str, extensions: list[str] | None, output_file: str | None):
    """List all image files that would be processed by the scan command."""

    # Resolve extension filter — must match exactly what scan.py would use
    if extensions:
        exts = {f".{e.lower().lstrip('.')}" for e in extensions}
        invalid = exts - IMAGE_EXTENSIONS
        if invalid:
            print(f"  ⚠  Unsupported extensions ignored: {', '.join(sorted(invalid))}")
            exts -= invalid
    else:
        exts = None  # None tells find_images to use IMAGE_EXTENSIONS (same default as scan)

    folder = Path(photos_folder)
    if not folder.exists():
        print(f"ERROR: Folder not found: {photos_folder}")
        return

    # Use the exact same function as scan.py — guaranteed to match
    images = find_images(photos_folder, extensions=exts)

    effective_exts = exts if exts is not None else IMAGE_EXTENSIONS
    ext_counts = Counter(p.suffix.lower() for p in images)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Folder  : {folder.resolve()}")
    print(f"  Filter  : {', '.join(sorted(effective_exts))}")
    print(f"  Total   : {len(images)} image(s) found")
    if ext_counts:
        breakdown = "  |  ".join(f"{ext}: {count}" for ext, count in sorted(ext_counts.items()))
        print(f"  Types   : {breakdown}")
    print(f"{'─'*50}\n")

    # ── File listing grouped by sub-folder ────────────────────────────────
    lines = []
    current_dir = None

    for img_path in images:
        parent = str(img_path.parent)
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

    # ── Optionally save to file ───────────────────────────────────────────
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Photo Tagger — File List\n")
            f.write(f"# Folder : {folder.resolve()}\n")
            f.write(f"# Filter : {', '.join(sorted(effective_exts))}\n")
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
                        help="Limit to specific extensions e.g. --ext jpg png heic "
                             "(default: all supported types)")
    parser.add_argument("--output",  default=None,
                        help="Optionally save the file list to a text file")

    args = parser.parse_args()
    list_files(args.photos, args.ext, args.output)


if __name__ == "__main__":
    main()
