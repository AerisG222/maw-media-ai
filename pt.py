#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pt.py — Photo Tagger entry point

Dispatches to the appropriate command script based on the command entered.

Setup:
    ./prep.sh                       # create conda environment and install dependencies
    conda activate photo-tagger     # enter the environment

Usage:
    ./pt.py gpu

    ./pt.py list   --photos ./my_photos
    ./pt.py list   --photos ./my_photos --ext jpg png heic
    ./pt.py list   --photos ./my_photos --output file-list.txt

    ./pt.py cluster --photos ./my_photos --output ./known_people
    ./pt.py cluster --photos ./my_photos --output ./known_people --sample 20
    ./pt.py cluster --photos ./my_photos --output ./known_people --years 2010 2015
    ./pt.py cluster --photos ./my_photos --output ./known_people --eps 0.35 --min-samples 3

    ./pt.py enroll --known ./known_people --db faces.db

    ./pt.py scan --photos ./my_photos --db faces.db --output results.json
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types objects
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types scenes
    ./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces objects scenes

    ./pt.py report --output results.json
    ./pt.py report --output results.json --type faces
    ./pt.py report --output results.json --type objects
    ./pt.py report --output results.json --type scenes
"""

import sys
import argparse


COMMANDS = {
    "gpu":     "Check GPU availability and configuration",
    "list":    "List all image files that would be scanned",
    "cluster": "Bootstrap known faces database via unsupervised clustering",
    "enroll":  "Enroll known people from reference photos",
    "scan":    "Scan a photo library for faces, objects, and/or scenes",
    "report":  "Print a summary of results",
}


def main():
    parser = argparse.ArgumentParser(
        description="Photo Tagger — local facial recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Command to run: " + ", ".join(COMMANDS.keys()),
    )
    # Pass remaining args through to the sub-script
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Rebuild sys.argv for the sub-script so its own argparse works correctly
    sys.argv = [args.command + ".py"] + args.args

    if args.command == "gpu":
        from gpu import main as run
    elif args.command == "list":
        from list import main as run
    elif args.command == "cluster":
        from cluster import main as run
    elif args.command == "enroll":
        from enroll import main as run
    elif args.command == "scan":
        from scan import main as run
    elif args.command == "report":
        from report import main as run

    run()


if __name__ == "__main__":
    main()
