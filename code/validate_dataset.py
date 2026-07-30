#!/usr/bin/env python3
"""Validate the public release contract without changing any files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from mask_dataset import validate_mask


CORE_FILES = ("thermal.tiff", "thermal.png", "video.png", "overlay.png", "mask_80c.png")
EXPECTED_ALIGNED_COUNTS = {
    "0001": 439,
    "0002": 451,
    "0003": 460,
    "0004": 517,
}
EXPECTED_RAW_TIFF_COUNTS = {
    "0001": 482,
    "0002": 468,
    "0003": 651,
    "0004": 535,
}
MANIFEST_PATH_FIELDS = {
    "thermal_tiff_path": "thermal.tiff",
    "thermal_png_path": "thermal.png",
    "video_png_path": "video.png",
    "overlay_png_path": "overlay.png",
    "mask_80c_path": "mask_80c.png",
}


def validate_aligned_sample(aligned_dir: Path) -> None:
    missing = [name for name in CORE_FILES if not (aligned_dir / name).is_file()]
    if missing:
        raise ValueError(f"{aligned_dir}: missing {missing}")
    thermal = np.asarray(tifffile.imread(aligned_dir / "thermal.tiff"))
    if thermal.shape != (512, 640) or thermal.dtype != np.uint16:
        raise ValueError(f"{aligned_dir}: TIFF shape/dtype {thermal.shape}/{thermal.dtype}")
    for name in ("thermal.png", "video.png", "overlay.png", "mask_80c.png"):
        with Image.open(aligned_dir / name) as image:
            if image.size != (640, 512):
                raise ValueError(f"{aligned_dir / name}: size {image.size}")
    validate_mask(aligned_dir / "thermal.tiff", aligned_dir / "mask_80c.png")


def validate_dataset(dataset_dir: Path) -> tuple[int, int]:
    manifest_path = dataset_dir / "processed/manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    aligned_root = dataset_dir / "processed/aligned"
    aligned_dirs = sorted(path for path in aligned_root.iterdir() if path.is_dir())
    if len(rows) != len(aligned_dirs):
        raise ValueError(
            f"{dataset_dir}: manifest {len(rows)} != aligned samples {len(aligned_dirs)}"
        )
    for row in rows:
        aligned_id = row["aligned_id"]
        expected_source = f"raw/thermal/{row['tiff_name']}"
        if row["thermal_source_path"] != expected_source:
            raise ValueError(f"{dataset_dir}: unexpected thermal source path for {aligned_id}")
        if not (dataset_dir / expected_source).is_file():
            raise ValueError(f"{dataset_dir}: missing source TIFF for {aligned_id}")
        for field, filename in MANIFEST_PATH_FIELDS.items():
            expected = f"processed/aligned/{aligned_id}/{filename}"
            if row[field] != expected:
                raise ValueError(
                    f"{dataset_dir}: unexpected {field} for {aligned_id}: {row[field]}"
                )
    for aligned_dir in aligned_dirs:
        validate_aligned_sample(aligned_dir)
    video = cv2.VideoCapture(str(dataset_dir / "raw/video.mp4"))
    if not video.isOpened():
        raise ValueError(f"cannot open {dataset_dir / 'raw/video.mp4'}")
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    if abs(fps - 30.0) > 1e-6 or (width, height) != (3840, 2160):
        raise ValueError(f"{dataset_dir}: unexpected video {fps} fps {width}x{height}")
    return len(rows), len(aligned_dirs) * len(CORE_FILES)


def verify_checksums(root: Path) -> int:
    checksum_path = root / "checksums.sha256"
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum = hashlib.sha256()
        with (root / relative).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                checksum.update(chunk)
        actual = checksum.hexdigest()
        if actual != digest:
            raise ValueError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()
    missing = [
        dataset_id
        for dataset_id in EXPECTED_ALIGNED_COUNTS
        if not (args.root / dataset_id).is_dir()
    ]
    if missing:
        raise ValueError(f"missing datasets: {missing}")
    datasets = [args.root / dataset_id for dataset_id in EXPECTED_ALIGNED_COUNTS]
    total_aligned = total_files = 0
    for dataset in datasets:
        aligned_count, files = validate_dataset(dataset)
        if aligned_count != EXPECTED_ALIGNED_COUNTS[dataset.name]:
            raise ValueError(
                f"{dataset}: expected {EXPECTED_ALIGNED_COUNTS[dataset.name]} aligned samples, "
                f"found {aligned_count}"
            )
        raw_tiff_count = sum(
            path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            for path in (dataset / "raw/thermal").iterdir()
        )
        if raw_tiff_count != EXPECTED_RAW_TIFF_COUNTS[dataset.name]:
            raise ValueError(
                f"{dataset}: expected {EXPECTED_RAW_TIFF_COUNTS[dataset.name]} raw TIFFs, "
                f"found {raw_tiff_count}"
            )
        total_aligned += aligned_count
        total_files += files
        print(f"{dataset.name}: {aligned_count} aligned samples, {files} required files")
    if not args.skip_checksums:
        print(f"checksums: {verify_checksums(args.root)} files")
    print(f"TOTAL: {total_aligned} aligned samples, {total_files} required files")
    if total_aligned != 1867 or total_files != 9335:
        raise ValueError("unexpected total aligned/file count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
