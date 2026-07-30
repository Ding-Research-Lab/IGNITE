#!/usr/bin/env python3
"""Stream FLIR TIFF pixels into exact DN and Celsius distributions."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


DN_LEVELS = 1 << 16
EXPECTED_SHAPE = (512, 640)
EXPECTED_DTYPE = np.dtype(np.uint16)
EXPECTED_MAKE = "FLIR"
EXPECTED_MODEL = "Vue Pro R 640 13mm"
TLINEAR_GAIN = 0.04
CELSIUS_OFFSET = 273.15
EXPECTED_IS_NORMALIZED = 1
SATURATION_DN = 11_568
PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)


@dataclass
class DistributionAccumulator:
    """Exact uint16 frequency accumulator for one TIFF collection."""

    file_count: int = 0
    counts: np.ndarray = field(
        default_factory=lambda: np.zeros(DN_LEVELS, dtype=np.uint64)
    )

    def add(self, image: np.ndarray) -> None:
        if image.dtype != EXPECTED_DTYPE:
            raise ValueError(f"expected uint16 pixels, got {image.dtype}")
        frequency = np.bincount(image.reshape(-1), minlength=DN_LEVELS)
        self.counts += frequency.astype(np.uint64, copy=False)
        self.file_count += 1

    @property
    def pixel_count(self) -> int:
        return int(self.counts.sum(dtype=np.uint64))


def discover_tiffs(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def sequence_name(path: Path, input_dir: Path) -> str:
    return path.parent.relative_to(input_dir).as_posix()


def parse_xmp_number(xmp: bytes | str, field_name: str) -> float:
    if isinstance(xmp, bytes):
        text = xmp.decode("utf-8", errors="replace")
    else:
        text = xmp
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


def validate_page(page: tifffile.TiffPage) -> list[str]:
    issues: list[str] = []
    if tuple(page.shape) != EXPECTED_SHAPE:
        issues.append(f"shape={tuple(page.shape)!r}, expected={EXPECTED_SHAPE!r}")
    if np.dtype(page.dtype) != EXPECTED_DTYPE:
        issues.append(f"dtype={page.dtype}, expected={EXPECTED_DTYPE}")

    expected_tags = {
        "ImageWidth": EXPECTED_SHAPE[1],
        "ImageLength": EXPECTED_SHAPE[0],
        "BitsPerSample": 16,
        "SamplesPerPixel": 1,
        "Make": EXPECTED_MAKE,
        "Model": EXPECTED_MODEL,
    }
    for name, expected in expected_tags.items():
        actual = tag_value(page, name)
        if actual != expected:
            issues.append(f"{name}={actual!r}, expected={expected!r}")

    xmp_tag = page.tags.get("XMP")
    if xmp_tag is None:
        issues.append("missing XMP metadata")
        return issues

    try:
        gain = parse_xmp_number(xmp_tag.value, "TlinearGain")
        if not math.isclose(gain, TLINEAR_GAIN, rel_tol=0.0, abs_tol=1e-12):
            issues.append(f"TlinearGain={gain!r}, expected={TLINEAR_GAIN!r}")
    except ValueError as exc:
        issues.append(str(exc))

    try:
        normalized = parse_xmp_number(xmp_tag.value, "IsNormalized")
        if not math.isclose(
            normalized, EXPECTED_IS_NORMALIZED, rel_tol=0.0, abs_tol=1e-12
        ):
            issues.append(
                f"IsNormalized={normalized!r}, expected={EXPECTED_IS_NORMALIZED!r}"
            )
    except ValueError as exc:
        issues.append(str(exc))
    return issues


def weighted_percentile(counts: np.ndarray, percentile: float) -> float:
    """Return a NumPy-compatible linear percentile from integer frequencies."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    total = int(counts.sum(dtype=np.uint64))
    if total == 0:
        raise ValueError("cannot calculate a percentile for an empty distribution")

    rank = (percentile / 100.0) * (total - 1)
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)
    cumulative = np.cumsum(counts, dtype=np.uint64)
    lower_value = int(np.searchsorted(cumulative, lower_rank + 1, side="left"))
    upper_value = int(np.searchsorted(cumulative, upper_rank + 1, side="left"))
    fraction = rank - lower_rank
    return lower_value + fraction * (upper_value - lower_value)


def percentile_field(percentile: float) -> str:
    return f"p{percentile:g}".replace(".", "_")


def distribution_stats(
    counts: np.ndarray,
    *,
    scope: str,
    sequence: str,
    file_count: int,
) -> dict[str, int | float | str]:
    total = int(counts.sum(dtype=np.uint64))
    if total == 0:
        raise ValueError(f"empty distribution for {scope} {sequence!r}")

    observed = np.flatnonzero(counts)
    values = np.arange(counts.size, dtype=np.float64)
    weights = counts.astype(np.float64, copy=False)
    mean_dn = float(np.dot(values, weights) / total)
    mean_square_dn = float(np.dot(values * values, weights) / total)
    std_dn = math.sqrt(max(0.0, mean_square_dn - mean_dn * mean_dn))
    min_dn = int(observed[0])
    max_dn = int(observed[-1])
    saturation_count = int(counts[SATURATION_DN])

    row: dict[str, int | float | str] = {
        "scope": scope,
        "sequence": sequence,
        "file_count": file_count,
        "pixel_count": total,
        "min_dn": min_dn,
        "max_dn": max_dn,
        "mean_dn": mean_dn,
        "std_dn_population": std_dn,
    }
    percentile_values: dict[float, float] = {}
    for percentile in PERCENTILES:
        value = weighted_percentile(counts, percentile)
        percentile_values[percentile] = value
        row[f"{percentile_field(percentile)}_dn"] = value

    row.update(
        {
            "min_celsius": min_dn * TLINEAR_GAIN - CELSIUS_OFFSET,
            "max_celsius": max_dn * TLINEAR_GAIN - CELSIUS_OFFSET,
            "mean_celsius": mean_dn * TLINEAR_GAIN - CELSIUS_OFFSET,
            "std_celsius_population": std_dn * TLINEAR_GAIN,
        }
    )
    for percentile, value in percentile_values.items():
        row[f"{percentile_field(percentile)}_celsius"] = (
            value * TLINEAR_GAIN - CELSIUS_OFFSET
        )

    row.update(
        {
            "tlinear_gain": TLINEAR_GAIN,
            "saturation_dn": SATURATION_DN,
            "saturation_celsius": SATURATION_DN * TLINEAR_GAIN - CELSIUS_OFFSET,
            "saturation_count": saturation_count,
            "saturation_percent": saturation_count / total * 100.0,
        }
    )
    return row


def bin_frequency(
    counts: np.ndarray, bins: int, min_dn: int, max_dn: int
) -> tuple[np.ndarray, np.ndarray]:
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    if max_dn < min_dn:
        raise ValueError(f"invalid DN range [{min_dn}, {max_dn}]")
    edges = np.linspace(float(min_dn), float(max_dn + 1), bins + 1)
    values = np.arange(counts.size)
    binned, _ = np.histogram(values, bins=edges, weights=counts)
    return edges, binned.astype(np.uint64, copy=False)


def read_distributions(
    paths: Sequence[Path], input_dir: Path, progress_every: int = 100
) -> tuple[DistributionAccumulator, dict[str, DistributionAccumulator]]:
    overall = DistributionAccumulator()
    by_sequence: dict[str, DistributionAccumulator] = {}
    errors: list[str] = []

    for index, path in enumerate(paths, start=1):
        try:
            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) != 1:
                    errors.append(f"{path}: page_count={len(tif.pages)}, expected=1")
                    continue
                page = tif.pages[0]
                issues = validate_page(page)
                if issues:
                    errors.append(f"{path}: {'; '.join(issues)}")
                    continue
                image = page.asarray()
        except Exception as exc:  # keep scanning so the error report is complete
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        group = sequence_name(path, input_dir)
        accumulator = by_sequence.setdefault(group, DistributionAccumulator())
        overall.add(image)
        accumulator.add(image)
        if progress_every > 0 and (index % progress_every == 0 or index == len(paths)):
            print(f"Processed {index:,}/{len(paths):,} TIFF files", flush=True)

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"TIFF validation failed for {len(errors)} file(s):\n{detail}")
    return overall, dict(sorted(by_sequence.items()))


def csv_value(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def write_stats_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def frequency_rows(
    distributions: Iterable[tuple[str, DistributionAccumulator]],
) -> Iterable[dict[str, int | float | str]]:
    for group, accumulator in distributions:
        total = accumulator.pixel_count
        for dn in np.flatnonzero(accumulator.counts):
            count = int(accumulator.counts[dn])
            yield {
                "group": group,
                "dn": int(dn),
                "celsius": int(dn) * TLINEAR_GAIN - CELSIUS_OFFSET,
                "pixel_count": count,
                "pixel_percent": count / total * 100.0,
            }


def write_frequency_csv(
    path: Path,
    overall: DistributionAccumulator,
    by_sequence: dict[str, DistributionAccumulator],
) -> None:
    fields = ["group", "dn", "celsius", "pixel_count", "pixel_percent"]
    distributions = [("overall", overall), *by_sequence.items()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in frequency_rows(distributions):
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def annotate_reference_lines(
    axis: plt.Axes,
    stats: dict[str, object],
    *,
    scale: float,
    offset: float,
    include_saturation: bool,
) -> None:
    markers = [
        (float(stats["p1_dn"]) * scale + offset, "P1", "--"),
        (float(stats["p50_dn"]) * scale + offset, "P50", "-"),
        (float(stats["p99_dn"]) * scale + offset, "P99", "--"),
    ]
    if include_saturation:
        markers.append((SATURATION_DN * scale + offset, "DN 11568", ":"))
    for position, label, line_style in markers:
        axis.axvline(
            position,
            color="#4b5563",
            linestyle=line_style,
            linewidth=1.0,
            alpha=0.85,
            label=label,
        )


def plot_overall_histogram(
    path: Path,
    overall: DistributionAccumulator,
    stats: dict[str, object],
    bins: int,
) -> None:
    min_dn = int(stats["min_dn"])
    max_dn = int(stats["max_dn"])
    edges, binned_counts = bin_frequency(overall.counts, bins, min_dn, max_dn)
    if int(binned_counts.sum(dtype=np.uint64)) != overall.pixel_count:
        raise RuntimeError("overall histogram bins do not sum to the pixel count")
    percentages = binned_counts.astype(np.float64) / overall.pixel_count * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    panels = [
        (
            axes[0],
            edges,
            1.0,
            0.0,
            "Raw radiometric values",
            "Digital number (DN)",
            "#2563a6",
        ),
        (
            axes[1],
            edges * TLINEAR_GAIN - CELSIUS_OFFSET,
            TLINEAR_GAIN,
            -CELSIUS_OFFSET,
            "Converted temperature",
            "Temperature (°C)",
            "#d97706",
        ),
    ]
    for axis, panel_edges, scale, offset, title, xlabel, color in panels:
        axis.bar(
            panel_edges[:-1],
            percentages,
            width=np.diff(panel_edges),
            align="edge",
            color=color,
            alpha=0.82,
            linewidth=0,
        )
        annotate_reference_lines(
            axis,
            stats,
            scale=scale,
            offset=offset,
            include_saturation=True,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=4, fontsize=8, loc="upper center")

    axes[0].set_ylabel("Pixels per bin (%)")
    fig.suptitle(
        f"All TIFF pixels (n={overall.pixel_count:,}; {bins} equal-width bins)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_histograms(
    path: Path,
    by_sequence: dict[str, DistributionAccumulator],
    min_dn: int,
    max_dn: int,
    bins: int,
) -> None:
    rows = math.ceil(len(by_sequence) / 2)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(14, 3.2 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    max_percent = 0.0
    plotted: list[tuple[plt.Axes, np.ndarray, np.ndarray, str, int]] = []
    for axis, (group, accumulator) in zip(axes.flat, by_sequence.items(), strict=False):
        edges, binned_counts = bin_frequency(accumulator.counts, bins, min_dn, max_dn)
        if int(binned_counts.sum(dtype=np.uint64)) != accumulator.pixel_count:
            raise RuntimeError(f"histogram bins do not sum to the pixel count for {group}")
        percentages = binned_counts.astype(np.float64) / accumulator.pixel_count * 100.0
        max_percent = max(max_percent, float(percentages.max()))
        plotted.append(
            (
                axis,
                edges * TLINEAR_GAIN - CELSIUS_OFFSET,
                percentages,
                group,
                accumulator.pixel_count,
            )
        )

    for axis, edges_celsius, percentages, group, pixel_count in plotted:
        axis.bar(
            edges_celsius[:-1],
            percentages,
            width=np.diff(edges_celsius),
            align="edge",
            color="#2563a6",
            alpha=0.82,
            linewidth=0,
        )
        axis.set_ylim(0, max_percent * 1.08)
        axis.set_title(f"{group}  (n={pixel_count:,})", fontsize=10)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)

    for axis in axes.flat[len(plotted) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Temperature (°C)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pixels per bin (%)")

    fig.suptitle(
        f"Celsius distributions by TIFF sequence ({bins} shared bins)", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate FLIR TIFFs and calculate exact DN/Celsius distributions."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if args.bins <= 0:
        print(f"--bins must be positive, got {args.bins}", file=sys.stderr)
        return 2

    paths = discover_tiffs(input_dir)
    if not paths:
        print(f"No TIFF files found under {input_dir}", file=sys.stderr)
        return 2
    print(f"Discovered {len(paths):,} TIFF files in {input_dir}", flush=True)

    try:
        overall, by_sequence = read_distributions(paths, input_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    overall_stats = distribution_stats(
        overall.counts,
        scope="overall",
        sequence="all",
        file_count=overall.file_count,
    )
    sequence_stats = [
        distribution_stats(
            accumulator.counts,
            scope="sequence",
            sequence=group,
            file_count=accumulator.file_count,
        )
        for group, accumulator in by_sequence.items()
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_stats_csv(output_dir / "overall_stats.csv", [overall_stats])
    write_stats_csv(output_dir / "sequence_stats.csv", sequence_stats)
    write_frequency_csv(output_dir / "dn_frequency.csv", overall, by_sequence)
    plot_overall_histogram(
        output_dir / "overall_dn_celsius_histogram.png",
        overall,
        overall_stats,
        args.bins,
    )
    plot_sequence_histograms(
        output_dir / "sequence_celsius_histograms.png",
        by_sequence,
        int(overall_stats["min_dn"]),
        int(overall_stats["max_dn"]),
        args.bins,
    )

    print(f"Sequences: {len(by_sequence):,}")
    print(f"Files: {overall.file_count:,}")
    print(f"Pixels: {overall.pixel_count:,}")
    print(
        "DN range: "
        f"{int(overall_stats['min_dn']):,} to {int(overall_stats['max_dn']):,}"
    )
    print(
        "Celsius range: "
        f"{float(overall_stats['min_celsius']):.2f} °C to "
        f"{float(overall_stats['max_celsius']):.2f} °C"
    )
    print(
        f"DN {SATURATION_DN:,}: {int(overall_stats['saturation_count']):,} pixels "
        f"({float(overall_stats['saturation_percent']):.6f}%)"
    )
    print(f"Outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
