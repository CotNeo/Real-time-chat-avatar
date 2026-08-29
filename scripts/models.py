#!/usr/bin/env python3
"""
Model management (Section 23) — explicit, one-time, documented downloads.

Models are never auto-downloaded the first time the app happens to need them
at runtime; you run this script once, it tells you exactly what it's fetching
and from where, and the app then loads only from a local path.

Usage:
    python scripts/models.py install face-detection
    python scripts/models.py list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Every installable model is declared here with its real source and license —
# Section 23: identify source/license before downloading, don't guess.
CATALOG = {
    "face-detection": {
        "description": "SCRFD face detector (via the InsightFace 'buffalo_l' "
        "pack — only the detection model is loaded into memory; the pack is "
        "downloaded as one zip regardless, ~326 MB).",
        "source": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "license": "Non-commercial research purposes only "
        "(InsightFace project license — see "
        "https://github.com/deepinsight/insightface#license).",
        "approx_disk_mb": 326,
        "installer": "_install_insightface_buffalo_l",
    },
}


def _install_insightface_buffalo_l() -> None:
    # Deferred import: don't require insightface to be installed just to run
    # `models.py list`.
    from insightface.app import FaceAnalysis

    target_root = MODELS_DIR / "face"
    target_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading into {target_root} (this is the InsightFace project's "
          f"own GitHub Releases URL, not a third-party mirror)...")
    # Instantiating with allowed_modules=['detection'] loads only the SCRFD
    # detector into memory afterwards; the full pack is still downloaded once
    # since it ships as a single zip (Milestone 4's identity/recognition step
    # reuses this same downloaded pack — no second download needed).
    FaceAnalysis(name="buffalo_l", root=str(target_root), allowed_modules=["detection"])
    print("Done.")


def cmd_list() -> None:
    for key, spec in CATALOG.items():
        installed = (MODELS_DIR / "face" / "models" / "buffalo_l").exists() if key == "face-detection" else False
        status = "installed" if installed else "not installed"
        print(f"{key} [{status}]")
        print(f"  {spec['description']}")
        print(f"  source:  {spec['source']}")
        print(f"  license: {spec['license']}")
        print(f"  size:    ~{spec['approx_disk_mb']} MB")
        print()


def cmd_install(name: str, assume_yes: bool) -> int:
    spec = CATALOG.get(name)
    if spec is None:
        print(f"Unknown model '{name}'. Run `python scripts/models.py list` to see options.")
        return 1
    print(f"Installing '{name}':")
    print(f"  source:  {spec['source']}")
    print(f"  license: {spec['license']}")
    print(f"  size:    ~{spec['approx_disk_mb']} MB")
    if not assume_yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return 1
    globals()[spec["installer"]]()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    install_parser = sub.add_parser("install")
    install_parser.add_argument("name")
    install_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip the confirmation prompt."
    )
    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
        return 0
    if args.command == "install":
        return cmd_install(args.name, assume_yes=args.yes)
    return 1


if __name__ == "__main__":
    sys.exit(main())
