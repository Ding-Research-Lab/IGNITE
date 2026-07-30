#!/usr/bin/env python3
"""Generate and validate indexed 80 °C masks beside each aligned TIFF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


THRESHOLD_DN = 8829
MASK_NAME = "mask_80c.png"
PALETTE = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254


def generate_mask(thermal_path: Path, mask_path: Path) -> tuple[int, int]:
    dn = np.asarray(tifffile.imread(thermal_path))
    if dn.ndim > 2:
        dn = dn[..., 0]
    expected = (dn >= THRESHOLD_DN).astype(np.uint8)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(expected, mode="P")
    image.putpalette(PALETTE)
    image.save(mask_path, format="PNG", optimize=False)
    return int(expected.sum()), int(expected.size)


def validate_mask(thermal_path: Path, mask_path: Path) -> tuple[int, int]:
    dn = np.asarray(tifffile.imread(thermal_path))
    mask = np.asarray(Image.open(mask_path))
    expected = (dn >= THRESHOLD_DN).astype(np.uint8)
    if mask.shape != expected.shape or not np.array_equal(mask, expected):
        raise ValueError(f"mask does not match {thermal_path}")
    if set(np.unique(mask).tolist()) - {0, 1}:
        raise ValueError(f"mask contains values other than 0/1: {mask_path}")
    return int(mask.sum()), int(mask.size)


def update_manifest(dataset_dir: Path) -> None:
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        return
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest_prefix = "processed/aligned" if dataset_dir.name == "processed" else "aligned"
    for row in rows:
        aligned_dir = dataset_dir / "aligned" / row["aligned_id"]
        positive, total = validate_mask(
            aligned_dir / "thermal.tiff",
            aligned_dir / MASK_NAME,
        )
        row["mask_80c_path"] = f"{manifest_prefix}/{row['aligned_id']}/{MASK_NAME}"
        row["mask_positive_pixels"] = str(positive)
        row["mask_positive_fraction"] = f"{positive / total:.12g}"
    fields = list(rows[0]) if rows else []
    for field in ("mask_80c_path", "mask_positive_pixels", "mask_positive_fraction"):
        if field not in fields:
            fields.append(field)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(dataset_dir: Path) -> None:
    aligned_samples = sorted(
        path for path in (dataset_dir / "aligned").iterdir() if path.is_dir()
    )
    for aligned_dir in aligned_samples:
        mask_path = aligned_dir / MASK_NAME
        if not mask_path.is_file():
            generate_mask(aligned_dir / "thermal.tiff", mask_path)
        validate_mask(aligned_dir / "thermal.tiff", mask_path)
    update_manifest(dataset_dir)
    print(f"{dataset_dir}: {len(aligned_samples)} aligned masks validated")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    process_dataset(args.dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
