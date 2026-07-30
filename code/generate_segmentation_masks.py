#!/usr/bin/env python3
"""Generate indexed 80 C segmentation masks from radiometric FLIR TIFFs."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

EXPECTED_SHAPE = (512, 640)
EXPECTED_DTYPE = np.dtype(np.uint16)
EXPECTED_MAKE = "FLIR"
EXPECTED_MODEL = "Vue Pro R 640 13mm"
EXPECTED_IS_NORMALIZED = 1.0
TLINEAR_GAIN = 0.04
CELSIUS_OFFSET = 273.15
THRESHOLD_CELSIUS = 80.0
MASK_FILENAME = "mask_80c.png"
MASK_PALETTE = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254


class MaskGenerationError(RuntimeError):
    """Raised when inputs or existing masks do not meet the dataset contract."""


@dataclass(frozen=True)
class GenerationSummary:
    tiff_count: int
    created_count: int
    preserved_count: int
    positive_pixel_count: int
    total_pixel_count: int
    all_zero_mask_count: int

    @property
    def positive_percent(self) -> float:
        if self.total_pixel_count == 0:
            return 0.0
        return self.positive_pixel_count / self.total_pixel_count * 100.0


def minimum_dn_for_temperature(
    threshold_celsius: float = THRESHOLD_CELSIUS,
    *,
    gain: float = TLINEAR_GAIN,
    celsius_offset: float = CELSIUS_OFFSET,
) -> int:
    """Return the smallest integer DN whose calibrated temperature meets the threshold."""

    if not math.isfinite(threshold_celsius):
        raise ValueError("temperature threshold must be finite")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("TLinear gain must be finite and positive")
    return math.ceil((threshold_celsius + celsius_offset) / gain)


THRESHOLD_DN = minimum_dn_for_temperature()


def parse_xmp_number(xmp: bytes | str, field_name: str) -> float:
    text = xmp.decode("utf-8", errors="replace") if isinstance(xmp, bytes) else xmp
    pattern = rf"<(?:[A-Za-z_][\w.-]*:)?{re.escape(field_name)}>\s*([^<]+?)\s*</"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"missing XMP field {field_name}")
    try:
        return float(match.group(1))
    except ValueError as exc:
        raise ValueError(
            f"invalid XMP field {field_name}={match.group(1)!r}"
        ) from exc


def tag_value(page: tifffile.TiffPage, tag_name: str) -> object | None:
    tag = page.tags.get(tag_name)
    return None if tag is None else tag.value


def validate_thermal_page(page: tifffile.TiffPage) -> list[str]:
    issues: list[str] = []
    expected_tags = {
        "ImageWidth": EXPECTED_SHAPE[1],
        "ImageLength": EXPECTED_SHAPE[0],
        "BitsPerSample": 16,
        "SamplesPerPixel": 1,
        "Make": EXPECTED_MAKE,
        "Model": EXPECTED_MODEL,
    }
    if tuple(page.shape) != EXPECTED_SHAPE:
        issues.append(f"shape={tuple(page.shape)!r}, expected={EXPECTED_SHAPE!r}")
    if np.dtype(page.dtype) != EXPECTED_DTYPE:
        issues.append(f"dtype={page.dtype}, expected={EXPECTED_DTYPE}")
    for name, expected in expected_tags.items():
        actual = tag_value(page, name)
        if actual != expected:
            issues.append(f"{name}={actual!r}, expected={expected!r}")

    xmp_tag = page.tags.get("XMP")
    if xmp_tag is None:
        issues.append("missing XMP metadata")
        return issues
    for field_name, expected in (
        ("TlinearGain", TLINEAR_GAIN),
        ("IsNormalized", EXPECTED_IS_NORMALIZED),
    ):
        try:
            actual = parse_xmp_number(xmp_tag.value, field_name)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            issues.append(f"{field_name}={actual!r}, expected={expected!r}")
    return issues


def mask_from_dn(image: np.ndarray, threshold_dn: int = THRESHOLD_DN) -> np.ndarray:
    """Convert a uint16 radiometric image into literal uint8 class indices 0 and 1."""

    if image.dtype != EXPECTED_DTYPE:
        raise ValueError(f"expected {EXPECTED_DTYPE} pixels, got {image.dtype}")
    return (image >= threshold_dn).astype(np.uint8)


def read_validated_thermal(path: Path) -> np.ndarray:
    """Read one TIFF after validating its radiometric metadata and pixel layout."""

    try:
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != 1:
                raise MaskGenerationError(
                    f"{path}: expected one TIFF page, found {len(tif.pages)}"
                )
            page = tif.pages[0]
            issues = validate_thermal_page(page)
            if issues:
                raise MaskGenerationError(f"{path}: {'; '.join(issues)}")
            image = page.asarray()
    except MaskGenerationError:
        raise
    except Exception as exc:
        raise MaskGenerationError(
            f"{path}: could not read TIFF: {type(exc).__name__}: {exc}"
        ) from exc

    if image.shape != EXPECTED_SHAPE:
        raise MaskGenerationError(
            f"{path}: expected shape {EXPECTED_SHAPE}, got {image.shape}"
        )
    if image.dtype != EXPECTED_DTYPE:
        raise MaskGenerationError(
            f"{path}: expected dtype {EXPECTED_DTYPE}, got {image.dtype}"
        )
    return image


def validate_existing_mask(mask_path: Path, expected: np.ndarray) -> None:
    """Reject an existing mask unless its encoding and class indices are exactly correct."""

    try:
        with Image.open(mask_path) as image:
            image.load()
            mode = image.mode
            palette = image.getpalette()
            actual = np.asarray(image)
    except Exception as exc:
        raise MaskGenerationError(
            f"{mask_path}: could not read existing mask: {type(exc).__name__}: {exc}"
        ) from exc

    problems: list[str] = []
    if mode != "P":
        problems.append(f"mode={mode!r}, expected indexed mode 'P'")
    if palette is None or palette[:6] != MASK_PALETTE[:6]:
        problems.append("palette entries 0/1 are not black/white")
    if actual.shape != expected.shape:
        problems.append(f"shape={actual.shape}, expected={expected.shape}")
    elif not np.array_equal(actual, expected):
        mismatch_count = int(np.count_nonzero(actual != expected))
        problems.append(f"{mismatch_count} pixel values differ from the TIFF threshold")

    if problems:
        raise MaskGenerationError(
            f"{mask_path}: existing mask conflicts with expected output: "
            + "; ".join(problems)
        )


def write_indexed_mask(mask_path: Path, mask: np.ndarray) -> None:
    """Atomically write literal class indices with a black/white display palette."""

    if mask.dtype != np.uint8:
        raise ValueError(f"expected uint8 mask, got {mask.dtype}")
    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(f"mask contains non-binary values: {unique_values.tolist()}")

    image = Image.fromarray(mask)
    image.putpalette(MASK_PALETTE)
    temporary_path = mask_path.with_name(f".{mask_path.stem}.tmp.png")
    try:
        image.save(temporary_path, format="PNG", optimize=True)
        os.replace(temporary_path, mask_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def discover_thermal_tiffs(dataset_dir: Path) -> list[Path]:
    aligned_root = dataset_dir / "aligned"
    if not aligned_root.is_dir():
        raise MaskGenerationError(f"aligned directory does not exist: {aligned_root}")
    paths = sorted(path for path in aligned_root.rglob("thermal.tiff") if path.is_file())
    if not paths:
        raise MaskGenerationError(f"no thermal.tiff files found under {aligned_root}")
    return paths


def generate_masks(dataset_dir: Path) -> GenerationSummary:
    """Preflight all inputs, preserve valid masks, and generate only missing masks."""

    thermal_paths = discover_thermal_tiffs(dataset_dir)
    missing: list[Path] = []
    preserved_count = 0
    positive_pixel_count = 0
    total_pixel_count = 0
    all_zero_mask_count = 0

    # Complete preflight before writing anything so conflicts cannot leave a partial run.
    for thermal_path in thermal_paths:
        thermal = read_validated_thermal(thermal_path)
        expected = mask_from_dn(thermal)
        positives = int(np.count_nonzero(expected))
        positive_pixel_count += positives
        total_pixel_count += int(expected.size)
        all_zero_mask_count += int(positives == 0)

        mask_path = thermal_path.with_name(MASK_FILENAME)
        if mask_path.exists():
            validate_existing_mask(mask_path, expected)
            preserved_count += 1
        else:
            missing.append(thermal_path)

    for thermal_path in missing:
        expected = mask_from_dn(read_validated_thermal(thermal_path))
        write_indexed_mask(thermal_path.with_name(MASK_FILENAME), expected)

    # Re-open every output so successful completion guarantees on-disk correctness.
    for thermal_path in thermal_paths:
        expected = mask_from_dn(read_validated_thermal(thermal_path))
        validate_existing_mask(thermal_path.with_name(MASK_FILENAME), expected)

    return GenerationSummary(
        tiff_count=len(thermal_paths),
        created_count=len(missing),
        preserved_count=preserved_count,
        positive_pixel_count=positive_pixel_count,
        total_pixel_count=total_pixel_count,
        all_zero_mask_count=all_zero_mask_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate indexed 0/1 masks from radiometric thermal.tiff files using "
            "the fixed 80 C threshold."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="dataset root containing aligned/*/thermal.tiff",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = generate_masks(args.dataset_dir)
    except (MaskGenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"threshold: {THRESHOLD_CELSIUS:.1f} C (DN >= {THRESHOLD_DN})")
    print(f"TIFF files: {summary.tiff_count}")
    print(f"masks created: {summary.created_count}")
    print(f"valid masks preserved: {summary.preserved_count}")
    print(f"positive pixels: {summary.positive_pixel_count}")
    print(f"positive fraction: {summary.positive_percent:.6f}%")
    print(f"all-zero masks: {summary.all_zero_mask_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
