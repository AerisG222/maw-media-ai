#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pt.py — Photo Tagger entry point

Dispatches to the appropriate command script based on the command entered.

Setup:
    ./prep.sh                       # create conda environment and install dependencies
    conda activate photo-tagger     # enter the environment

Usage:
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

from common import configure_gpu


COMMANDS = {
    "enroll": "Enroll known people from reference photos",
    "scan":   "Scan a photo library and tag faces",
    "report": "Print a summary of results",
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

    configure_gpu()
    print()

    # Rebuild sys.argv for the sub-script so its own argparse works correctly
    sys.argv = [args.command + ".py"] + args.args

    if args.command == "enroll":
        from enroll import main as run
    elif args.command == "scan":
        from scan import main as run
    elif args.command == "report":
        from report import main as run

    run()


if __name__ == "__main__":
    main()
