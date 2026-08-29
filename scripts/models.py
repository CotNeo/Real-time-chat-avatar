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
    "face-swap": {
        "description": "inswapper_128 (fp16) — Mode A real-time face swap "
        "(Milestone 5). Inputs/outputs are 128x128; source identity is the "
        "512-d ArcFace embedding this project's own Milestone 4 identity "
        "pipeline already produces.",
        "source": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128_fp16.onnx "
        "(community mirror — see LICENSE NOTE below)",
        "license": "UNCLEAR — LICENSE NOTE: InsightFace's own team has "
        "discontinued official maintenance/distribution of this model and "
        "now directs users to their commercial product instead; there is no "
        "clean official license file the way buffalo_l has one. This "
        "download is from Gourieff/ReActor on Hugging Face, a long-standing "
        "(2+ years), widely-used community mirror (the same one the "
        "ReActor/roop/FaceFusion ecosystem builds on), not an "
        "authoritative source. Approved for use in this project only for "
        "strictly local, single-user, consensual avatar experimentation on "
        "the operator's own likeness (Section 21) — never redistributed. "
        "Re-evaluate if this project's scope ever changes.",
        "approx_disk_mb": 265,
        "sha256": "32031dbe50398c1beffa9daadaec8dd7ae9529d8314a0307b45a4987497f8494",
        "installer": "_install_inswapper",
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


def _install_inswapper() -> None:
    import hashlib
    import urllib.request

    target_dir = MODELS_DIR / "face" / "models" / "inswapper"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "inswapper_128_fp16.onnx"
    url = CATALOG["face-swap"]["source"].split(" ")[0]
    print(f"Downloading {url} -> {target_path} ...")
    urllib.request.urlretrieve(url, target_path)

    expected_sha256 = CATALOG["face-swap"]["sha256"]
    actual_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        target_path.unlink()
        raise RuntimeError(
            f"Checksum mismatch! Expected {expected_sha256}, got {actual_sha256}. "
            "Deleted the downloaded file — do not trust it."
        )
    print(f"Checksum verified: {actual_sha256}")
    print("Done.")


_INSTALLED_CHECKS = {
    "face-detection": lambda: (MODELS_DIR / "face" / "models" / "buffalo_l").exists(),
    "face-swap": lambda: (
        MODELS_DIR / "face" / "models" / "inswapper" / "inswapper_128_fp16.onnx"
    ).exists(),
}


def cmd_list() -> None:
    for key, spec in CATALOG.items():
        installed = _INSTALLED_CHECKS.get(key, lambda: False)()
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
