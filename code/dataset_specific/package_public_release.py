#!/usr/bin/env python3
"""Build the public, path-portable release tree from the working outputs.

This script deliberately copies data into a new tree.  It never mutates the
working ``raw_data``, ``output`` or ``fire_recoding`` directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tifffile
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
FIRE_RECORDING_ROOT = PROJECT_ROOT.parent / "fire_recoding"
DATA_ROOT = REPO_ROOT / "data"

EXPECTED_VIEWER_COUNTS = {
    "0001": 439,
    "0002": 451,
    "0003": 460,
    "0004": 504,
}


DATASETS: list[dict[str, Any]] = [
    {
        "public_id": "0001",
        "source_sequence_id": "20260226_022619",
        "strategy": "dual_anchor_window_90",
        "tiff_dir": PROJECT_ROOT / "raw_data/20260226_022619",
        "video": PROJECT_ROOT / "raw_data/Runcam6_0000.MP4",
        "processed": PROJECT_ROOT / "output/window5_dual_anchor_dataset",
        "review": PROJECT_ROOT / "output/window5_dual_anchor_dataset",
        "manual_circles": PROJECT_ROOT / "output/window5_dual_anchor_review_manual/pad_circles_manual.json",
        "video_seeds": None,
        "tiff_start": "20260226_022627",
        "tiff_end": "20260226_023345",
        "video_start": "1:18",
        "video_end": "8:37",
        "anchors": [
            {"name": "primary", "requested": "1:23", "selected": "1:23.233"},
            {"name": "secondary", "requested": "8:14", "selected": "8:13.233"},
        ],
    },
    {
        "public_id": "0002",
        "source_sequence_id": "20260226_030445",
        "strategy": "dual_anchor_window_90",
        "tiff_dir": FIRE_RECORDING_ROOT / "20260413/20260226_030445",
        "video": FIRE_RECORDING_ROOT / "20260413/20260226_030445.MP4",
        "processed": PROJECT_ROOT / "output/20260226_030445_dual_anchor_rank1_dataset",
        "review": PROJECT_ROOT / "output/20260226_030445_dual_anchor_review",
        "manual_circles": PROJECT_ROOT / "output/20260226_030445_dual_anchor_review/pad_circles_manual.json",
        "video_seeds": PROJECT_ROOT / "output/20260226_030445_dual_anchor_review/video_pad_seeds_manual.json",
        "tiff_start": "20260226_030455",
        "tiff_end": "20260226_031225",
        "video_start": "4:47",
        "video_end": "12:17",
        "anchors": [
            {"name": "primary", "requested": "4:52", "selected": "4:52.500"},
            {"name": "secondary", "requested": "12:00", "selected": "12:00.567"},
        ],
    },
    {
        "public_id": "0003",
        "source_sequence_id": "20260226_033912",
        "strategy": "anchor_window_90",
        "tiff_dir": FIRE_RECORDING_ROOT / "20260413/20260226_033912",
        "video": FIRE_RECORDING_ROOT / "20260413/20260226_033912.MP4",
        "processed": PROJECT_ROOT / "output/20260226_033912_single_anchor_rank1_dataset",
        "review": PROJECT_ROOT / "output/20260226_033912_single_anchor_review",
        "manual_circles": PROJECT_ROOT / "output/20260226_033912_single_anchor_review/pad_circles_manual.json",
        "video_seeds": PROJECT_ROOT / "output/20260226_033912_single_anchor_review/video_pad_seeds_manual.json",
        "tiff_start": "20260226_033924",
        "tiff_end": "20260226_034703",
        "video_start": "0:24",
        "video_end": "8:03.5",
        "anchors": [{"name": "primary", "requested": "0:29", "selected": "0:29.533"}],
    },
    {
        "public_id": "0004",
        "source_sequence_id": "20260227_023453",
        "strategy": "anchor_window_90",
        "tiff_dir": FIRE_RECORDING_ROOT / "20260414/20260227_023453",
        "video": FIRE_RECORDING_ROOT / "20260414/20260227_023453.MP4",
        "processed": PROJECT_ROOT / "output/20260227_023453_single_anchor_rank1_dataset",
        "review": PROJECT_ROOT / "output/20260227_023453_single_anchor_review",
        "manual_circles": PROJECT_ROOT / "output/20260227_023453_single_anchor_review/pad_circles_manual.json",
        "video_seeds": None,
        "tiff_start": "20260227_023453",
        "tiff_end": "20260227_024316",
        "video_start": "1:08",
        "video_end": "9:31",
        "anchors": [{"name": "primary", "requested": "9:28", "selected": "9:27.833"}],
    },
]


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def clean_string(value: str, *, key: str = "") -> str:
    """Replace machine-specific paths while retaining useful metadata."""

    value = re.sub(
        r"(?:/Users/[^,\s\"'}]+/|raw_data/[^,\s\"'}]+/)(\d{8}_\d{6}\.tiff)",
        r"raw/thermal/\1",
        value,
    )
    value = re.sub(
        r"(?:/Users/[^,\s\"'}]+|raw_data/[^,\s\"'}]+)\.(?:MP4|mp4|MOV|mov)",
        "raw/video.mp4",
        value,
    )
    value = re.sub(
        r"output/[^,\s\"'}]+/transform\.json",
        "metadata/transform.json",
        value,
    )
    lower = value.lower()
    if lower.endswith((".mp4", ".mov")):
        return "raw/video.mp4"
    if lower.endswith((".tif", ".tiff")):
        return f"raw/thermal/{Path(value).name}"
    if "pad_circles_manual" in lower:
        return "metadata/calibration/pad_circles_manual.json"
    if "video_pad_seeds" in lower:
        return "metadata/calibration/video_pad_seeds_manual.json"
    if "transform.json" in lower and ("output" in lower or "review" in lower):
        return "metadata/transform.json"
    if key == "tiff_dir":
        return "raw/thermal"
    if key == "video":
        return "raw/video.mp4"
    if key == "out":
        return "processed"
    if value.startswith("/Users/") or "/output/" in value or value.startswith("output/"):
        return f"metadata/calibration/{Path(value).name}"
    if value.startswith("raw_data/"):
        return "raw/thermal" if value.lower().endswith((".tif", ".tiff")) else "raw/video.mp4"
    return value


def sanitize_json(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            ("max_aligned" if name == "max_pairs" else name): sanitize_json(item, key=name)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json(item, key=key) for item in value]
    if isinstance(value, str):
        return clean_string(value, key=key)
    return value


def sanitize_calibration_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        destination.write_text(json.dumps(sanitize_json(data), indent=2) + "\n", encoding="utf-8")
        return
    if source.suffix == ".jsonl":
        rows = [sanitize_json(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line]
        write_jsonl(destination, rows)
        return
    if source.suffix == ".csv":
        rows = load_rows(source)
        cleaned = [{key: clean_string(value, key=key) for key, value in row.items()} for row in rows]
        write_csv(destination, cleaned)
        return
    copy_file(source, destination)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def filter_rows_for_spec(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the published TIFF boundary to a working or release manifest."""

    selected = [
        row
        for row in rows
        if spec["tiff_start"] <= Path(str(row["tiff_name"])).stem <= spec["tiff_end"]
    ]
    if not selected:
        raise ValueError(f"{spec['public_id']}: no manifest rows inside release boundary")
    observed_boundary = (
        Path(str(selected[0]["tiff_name"])).stem,
        Path(str(selected[-1]["tiff_name"])).stem,
    )
    expected_boundary = (spec["tiff_start"], spec["tiff_end"])
    if observed_boundary != expected_boundary:
        raise ValueError(
            f"{spec['public_id']}: manifest boundary {observed_boundary} "
            f"!= {expected_boundary}"
        )
    expected_count = EXPECTED_VIEWER_COUNTS[spec["public_id"]]
    if len(selected) != expected_count:
        raise ValueError(
            f"{spec['public_id']}: filtered manifest has {len(selected)} rows; "
            f"expected {expected_count}"
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row | {field: "" for field in fields if field not in row} for row in rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_video_time_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"invalid video time: {value}")


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if abs(fps - 30.0) > 1e-6 or (width, height) != (3840, 2160):
        raise ValueError(f"unexpected video properties: {path}: {fps} fps {width}x{height}")
    return {
        "path": "raw/video.mp4",
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps,
    }


def update_metadata_config(spec: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """Refresh release-boundary and canonical-video facts in packaged provenance."""

    dataset_id = spec["public_id"]
    root = DATA_ROOT / dataset_id
    config_path = root / "metadata/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["args"].update(
        {
            "tiff_start": spec["tiff_start"],
            "tiff_end": spec["tiff_end"],
            "video_start": spec["video_start"],
            "video_end": spec["video_end"],
        }
    )
    config["video"] = probe_video(root / "raw/video.mp4")
    raw_tiff_count = sum(
        path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        for path in (root / "raw/thermal").iterdir()
    )
    config["tiff_count_total"] = raw_tiff_count
    config["tiff_count_effective"] = len(rows)
    first, last = rows[0], rows[-1]
    config["tiff_start_selected"] = {
        "path": f"raw/thermal/{first['tiff_name']}",
        "timestamp": first["tiff_time_iso"],
    }
    config["tiff_end_selected"] = {
        "path": f"raw/thermal/{last['tiff_name']}",
        "timestamp": last["tiff_time_iso"],
    }
    model_boundary = {
        "tiff_start": first["tiff_time_iso"],
        "tiff_end": last["tiff_time_iso"],
        "video_start_s": parse_video_time_seconds(spec["video_start"]),
        "video_end_s": parse_video_time_seconds(spec["video_end"]),
    }
    for model_name in ("initial_time_model", "final_time_model"):
        if model_name in config:
            config[model_name].update(model_boundary)
    for model in config.get("final_time_models", {}).values():
        model.update(model_boundary)
    config["manifest_count"] = len(rows)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return raw_tiff_count


def write_release_config(
    spec: dict[str, Any], *, raw_tiff_count: int, aligned_count: int
) -> dict[str, Any]:
    dataset_id = spec["public_id"]
    config = {
        "dataset_id": dataset_id,
        "source_sequence_id": spec["source_sequence_id"],
        "strategy": spec["strategy"],
        "paths": {
            "raw_video": f"data/{dataset_id}/raw/video.mp4",
            "raw_tiff_dir": f"data/{dataset_id}/raw/thermal",
            "processed_dir": f"data/{dataset_id}/processed",
            "transform": f"data/{dataset_id}/metadata/transform.json",
            "pad_circles": f"data/{dataset_id}/metadata/calibration/pad_circles_manual.json",
        },
        "time_mapping": {
            "tiff_start": spec["tiff_start"],
            "tiff_end": spec["tiff_end"],
            "video_start": spec["video_start"],
            "video_end": spec["video_end"],
            "anchors": spec["anchors"],
        },
        "mask": {"threshold_celsius": 80.0, "threshold_dn": 8829},
        "counts": {"raw_tiff": raw_tiff_count, "aligned": aligned_count},
    }
    if spec["video_seeds"]:
        config["paths"]["video_pad_seeds"] = (
            f"data/{dataset_id}/metadata/calibration/video_pad_seeds_manual.json"
        )
    (REPO_ROOT / "configs" / f"{dataset_id}.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return config


def catalog_summary(
    spec: dict[str, Any], *, raw_tiff_count: int, aligned_count: int
) -> dict[str, Any]:
    return {
        "dataset_id": spec["public_id"],
        "source_sequence_id": spec["source_sequence_id"],
        "strategy": spec["strategy"],
        "raw_tiff_count": raw_tiff_count,
        "aligned_count": aligned_count,
        "tiff_start": spec["tiff_start"],
        "tiff_end": spec["tiff_end"],
        "anchors": " / ".join(anchor["selected"] for anchor in spec["anchors"]),
    }


def build_viewer_row(dataset_id: str, row: dict[str, Any]) -> dict[str, Any]:
    """Convert a release manifest row to one ImageFolder metadata row."""

    def repository_path(field: str) -> str:
        relative = Path(row[field])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{dataset_id}/{row['aligned_id']}: unsafe {field}: {relative}")
        return (Path(dataset_id) / relative).as_posix()

    viewer_row = {
        "sequence_id": dataset_id,
        "aligned_id": row["aligned_id"],
        "thermal_file_name": repository_path("thermal_png_path"),
        "thermal_tiff_file_name": repository_path("thermal_tiff_path"),
        "rgb_file_name": repository_path("video_png_path"),
        "overlay_file_name": repository_path("overlay_png_path"),
        "mask_80c_file_name": repository_path("mask_80c_path"),
        "tiff_time_iso": row["tiff_time_iso"],
        "video_time": row["video_time"],
        "video_time_s": float(row["video_time_s"]),
        "video_frame_index": int(row["video_frame_index"]),
        "visual_score": float(row["visual_score"]),
        "mask_positive_pixels": int(row["mask_positive_pixels"]),
        "mask_positive_fraction": float(row["mask_positive_fraction"]),
        "thermal_tiff_path": repository_path("thermal_tiff_path"),
        "thermal_source_path": repository_path("thermal_source_path"),
    }
    for field in (
        "thermal_file_name",
        "thermal_tiff_file_name",
        "rgb_file_name",
        "overlay_file_name",
        "mask_80c_file_name",
        "thermal_tiff_path",
        "thermal_source_path",
    ):
        if not (DATA_ROOT / viewer_row[field]).is_file():
            raise FileNotFoundError(DATA_ROOT / viewer_row[field])
    return viewer_row


def write_viewer_metadata(
    all_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Write the root ImageFolder metadata that defines one aligned pair per row."""

    if all_rows is None:
        all_rows = {
            dataset_id: load_rows(DATA_ROOT / dataset_id / "processed/manifest.csv")
            for dataset_id in EXPECTED_VIEWER_COUNTS
        }
    actual_counts = {dataset_id: len(rows) for dataset_id, rows in all_rows.items()}
    if actual_counts != EXPECTED_VIEWER_COUNTS:
        raise ValueError(
            f"unexpected viewer sequence counts: {actual_counts}; "
            f"expected {EXPECTED_VIEWER_COUNTS}"
        )
    viewer_rows = [
        build_viewer_row(dataset_id, row)
        for dataset_id in EXPECTED_VIEWER_COUNTS
        for row in all_rows[dataset_id]
    ]
    identities = {(row["sequence_id"], row["aligned_id"]) for row in viewer_rows}
    if len(identities) != len(viewer_rows):
        raise ValueError("viewer metadata contains duplicate sequence/aligned IDs")
    write_jsonl(DATA_ROOT / "metadata.jsonl", viewer_rows)
    return viewer_rows


def mask_from_tiff(source: Path, destination: Path) -> tuple[int, int]:
    dn = np.asarray(tifffile.imread(source))
    if dn.ndim > 2:
        dn = dn[..., 0]
    mask = (dn >= 8829).astype(np.uint8)
    image = Image.fromarray(mask, mode="P")
    palette = [0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254
    image.putpalette(palette)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False)
    return int(mask.sum()), int(mask.size)


def render_index(dataset_id: str, out_dir: Path, rows: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in rows:
        aligned_id = row["aligned_id"]
        base = f"aligned/{aligned_id}"
        cards.append(
            "<article class='aligned'>"
            f"<h2>{aligned_id}</h2>"
            f"<p>TIFF {row['tiff_name']} · video {row['video_time']} · frame {row['video_frame_index']}</p>"
            f"<a href='{base}/thermal.tiff'>TIFF</a>"
            f"<div class='grid'><img loading='lazy' src='{base}/thermal.png' alt='thermal'>"
            f"<img loading='lazy' src='{base}/video.png' alt='video'>"
            f"<img loading='lazy' src='{base}/overlay.png' alt='overlay'>"
            f"<img loading='lazy' src='{base}/mask_80c.png' alt='80 C mask'></div>"
            "</article>"
        )
    html = """<!doctype html>
<meta charset="utf-8"><title>DATASET_ID aligned samples</title>
<style>body{font-family:system-ui;margin:2rem;background:#f5f5f5}.aligned{background:white;padding:1rem;margin:1rem 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}.grid img{width:100%;height:auto;background:#111}a{margin-right:1rem}</style>
<h1>DATASET_ID</h1><p>Each aligned sample: radiometric TIFF, thermal rendering, aligned RGB, overlay and 80 °C mask.</p>
BODY
""".replace("DATASET_ID", dataset_id).replace("BODY", "\n".join(cards))
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def percentile_from_counts(counts: np.ndarray, percentile: float) -> float:
    total = int(counts.sum())
    if total <= 0:
        return float("nan")
    rank = (percentile / 100.0) * (total - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    cumulative = np.cumsum(counts, dtype=np.uint64)
    lo = int(np.searchsorted(cumulative, lower + 1))
    hi = int(np.searchsorted(cumulative, upper + 1))
    return float(lo + (hi - lo) * (rank - lower))


def summarize_dataset(dataset_id: str, dataset_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.zeros(1 << 16, dtype=np.uint64)
    mask_positive = 0
    pixel_count = 0
    values_sum = 0.0
    values_sq_sum = 0.0
    for row in rows:
        aligned_dir = dataset_dir / "processed" / "aligned" / row["aligned_id"]
        dn = np.asarray(tifffile.imread(aligned_dir / "thermal.tiff"), dtype=np.uint16)
        counts += np.bincount(dn.reshape(-1), minlength=1 << 16).astype(np.uint64)
        mask = np.asarray(Image.open(aligned_dir / "mask_80c.png"))
        mask_positive += int(np.count_nonzero(mask))
        pixel_count += int(dn.size)
        values_sum += float(dn.sum(dtype=np.float64))
        values_sq_sum += float(np.square(dn.astype(np.float64)).sum(dtype=np.float64))
    mean_dn = values_sum / pixel_count
    variance_dn = max(0.0, values_sq_sum / pixel_count - mean_dn * mean_dn)
    dn_stats = {
        "dataset_id": dataset_id,
        "tiff_count": len(rows),
        "pixel_count": pixel_count,
        "min_celsius": float(np.nonzero(counts)[0][0] * 0.04 - 273.15),
        "p01_celsius": percentile_from_counts(counts, 1.0) * 0.04 - 273.15,
        "median_celsius": percentile_from_counts(counts, 50.0) * 0.04 - 273.15,
        "p99_celsius": percentile_from_counts(counts, 99.0) * 0.04 - 273.15,
        "max_celsius": float(np.nonzero(counts)[0][-1] * 0.04 - 273.15),
        "mean_celsius": mean_dn * 0.04 - 273.15,
        "std_celsius": math.sqrt(variance_dn) * 0.04,
        "mask_positive_pixels": mask_positive,
        "mask_positive_fraction": mask_positive / pixel_count if pixel_count else 0.0,
    }
    stats_dir = dataset_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    write_csv(stats_dir / "temperature_mask_stats.csv", [dn_stats])
    return dn_stats | {"counts": counts}


def write_statistics(summaries: list[dict[str, Any]]) -> None:
    stats_dir = DATA_ROOT / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    public = [{key: value for key, value in item.items() if key != "counts"} for item in summaries]
    write_csv(stats_dir / "dataset_temperature_mask_stats.csv", public)
    counts = np.sum([item["counts"] for item in summaries], axis=0, dtype=np.uint64)
    nonzero = np.nonzero(counts)[0]
    overall = {
        "dataset_id": "all",
        "tiff_count": sum(int(item["tiff_count"]) for item in summaries),
        "pixel_count": sum(int(item["pixel_count"]) for item in summaries),
        "min_celsius": float(nonzero[0] * 0.04 - 273.15),
        "p01_celsius": percentile_from_counts(counts, 1.0) * 0.04 - 273.15,
        "median_celsius": percentile_from_counts(counts, 50.0) * 0.04 - 273.15,
        "p99_celsius": percentile_from_counts(counts, 99.0) * 0.04 - 273.15,
        "max_celsius": float(nonzero[-1] * 0.04 - 273.15),
        "mask_positive_pixels": sum(int(item["mask_positive_pixels"]) for item in summaries),
    }
    overall["mask_positive_fraction"] = overall["mask_positive_pixels"] / overall["pixel_count"]
    write_csv(stats_dir / "overall_temperature_mask_stats.csv", [overall])
    celsius = np.arange(1 << 16, dtype=np.float64) * 0.04 - 273.15
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, item in zip(axes.flat, summaries, strict=True):
        axis.plot(celsius, item["counts"], linewidth=0.7)
        axis.set_title(item["dataset_id"])
        axis.set_xlabel("Temperature (°C)")
        axis.set_ylabel("Pixels")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(stats_dir / "temperature_histograms.png", dpi=160)
    plt.close(fig)


def build_dataset(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_id = spec["public_id"]
    source_sequence_id = spec["source_sequence_id"]
    root = DATA_ROOT / dataset_id
    raw_thermal = root / "raw/thermal"
    processed_aligned = root / "processed/aligned"
    metadata_dir = root / "metadata"
    calibration_dir = metadata_dir / "calibration"
    if root.exists():
        # Preserve an explicitly replaced canonical video across release rebuilds.
        for generated in (
            raw_thermal,
            root / "processed",
            metadata_dir,
            root / "statistics",
        ):
            if generated.exists():
                shutil.rmtree(generated)
    raw_thermal.mkdir(parents=True)
    processed_aligned.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)

    for source in sorted(spec["tiff_dir"].glob("*.tiff")):
        copy_file(source, raw_thermal / source.name)
    canonical_video = root / "raw/video.mp4"
    if not canonical_video.is_file():
        copy_file(spec["video"], canonical_video)

    rows = filter_rows_for_spec(
        spec, load_rows(spec["processed"] / "manifest.csv")
    )
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        # Historical working outputs use pairs/pair_id; the public release uses
        # aligned/aligned_id. This is the only legacy schema boundary.
        aligned_id = row.get("aligned_id") or row["pair_id"]
        source_aligned = spec["processed"] / "pairs" / aligned_id
        target_aligned = processed_aligned / aligned_id
        for filename in ("thermal.tiff", "thermal.png", "video.png", "overlay.png"):
            copy_file(source_aligned / filename, target_aligned / filename)
        positive, total = mask_from_tiff(
            target_aligned / "thermal.tiff",
            target_aligned / "mask_80c.png",
        )
        normalized = {
            "aligned_id": aligned_id,
            **{key: value for key, value in row.items() if key not in {"pair_id", "aligned_id"}},
        }
        normalized.update(
            {
                "thermal_source_path": f"raw/thermal/{row['tiff_name']}",
                "thermal_tiff_path": f"processed/aligned/{aligned_id}/thermal.tiff",
                "thermal_png_path": f"processed/aligned/{aligned_id}/thermal.png",
                "video_png_path": f"processed/aligned/{aligned_id}/video.png",
                "overlay_png_path": f"processed/aligned/{aligned_id}/overlay.png",
                "mask_80c_path": f"processed/aligned/{aligned_id}/mask_80c.png",
                "mask_positive_pixels": positive,
                "mask_positive_fraction": positive / total,
            }
        )
        normalized_rows.append(normalized)
    write_csv(root / "processed/manifest.csv", normalized_rows)
    write_jsonl(root / "processed/manifest.jsonl", normalized_rows)
    render_index(dataset_id, root / "processed", normalized_rows)

    source_config = json.loads((spec["processed"] / "config.json").read_text(encoding="utf-8"))
    source_transform = json.loads((spec["processed"] / "transform.json").read_text(encoding="utf-8"))
    (metadata_dir / "config.json").write_text(json.dumps(sanitize_json(source_config), indent=2) + "\n", encoding="utf-8")
    (metadata_dir / "transform.json").write_text(json.dumps(sanitize_json(source_transform), indent=2) + "\n", encoding="utf-8")

    calibration_files = [
        "anchor_window_candidates.csv",
        "anchor_window_candidates.jsonl",
        "second_anchor_window_candidates.csv",
        "second_anchor_window_candidates.jsonl",
        "pad_circles_manual.json",
        "video_pad_seeds_manual.json",
    ]
    for filename in calibration_files:
        candidates = [spec["review"] / filename]
        if filename == "pad_circles_manual.json" and spec["manual_circles"]:
            candidates.insert(0, spec["manual_circles"])
        if filename == "video_pad_seeds_manual.json" and spec["video_seeds"]:
            candidates.insert(0, spec["video_seeds"])
        if source_sequence_id == "20260226_022619" and filename.startswith(("anchor_window", "second_anchor_window")):
            candidates.insert(0, spec["processed"] / filename)
        source = next((candidate for candidate in candidates if candidate and candidate.is_file()), None)
        if source:
            sanitize_calibration_file(source, calibration_dir / filename)

    raw_tiff_count = update_metadata_config(spec, normalized_rows)
    write_release_config(
        spec, raw_tiff_count=raw_tiff_count, aligned_count=len(normalized_rows)
    )
    summary = catalog_summary(
        spec, raw_tiff_count=raw_tiff_count, aligned_count=len(normalized_rows)
    )
    return summary, normalized_rows


def write_checksums() -> None:
    checksum_path = DATA_ROOT / "checksums.sha256"
    files = sorted(
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_file()
        and path != checksum_path
        and ".cache" not in path.relative_to(DATA_ROOT).parts
        and path.name != ".DS_Store"
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            handle.write(f"{digest.hexdigest()}  {path.relative_to(DATA_ROOT).as_posix()}\n")


def update_checksum_entries(relative_paths: tuple[str, ...]) -> None:
    """Refresh selected derived-file checksums without rehashing the full payload."""

    checksum_path = DATA_ROOT / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        entries[relative] = digest
    for relative in relative_paths:
        path = DATA_ROOT / relative
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        entries[relative] = digest.hexdigest()
    checksum_path.write_text(
        "".join(f"{entries[relative]}  {relative}\n" for relative in sorted(entries)),
        encoding="utf-8",
    )


def refresh_existing_dataset(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply current release boundaries to an existing packaged dataset."""

    dataset_id = spec["public_id"]
    root = DATA_ROOT / dataset_id
    processed = root / "processed"
    aligned_root = processed / "aligned"
    rows = filter_rows_for_spec(spec, load_rows(processed / "manifest.csv"))
    json_rows = filter_rows_for_spec(
        spec, load_jsonl(processed / "manifest.jsonl")
    )
    json_by_id = {str(row["aligned_id"]): row for row in json_rows}
    expected_ids = {str(row["aligned_id"]) for row in rows}
    if set(json_by_id) != expected_ids:
        raise ValueError(f"{dataset_id}: CSV/JSONL manifest identities differ")

    actual_dirs = {path.name: path for path in aligned_root.iterdir() if path.is_dir()}
    missing = sorted(expected_ids - set(actual_dirs))
    if missing:
        raise ValueError(f"{dataset_id}: missing aligned directories: {missing}")
    for stale_id in sorted(set(actual_dirs) - expected_ids):
        shutil.rmtree(actual_dirs[stale_id])

    write_csv(processed / "manifest.csv", rows)
    write_jsonl(
        processed / "manifest.jsonl",
        [json_by_id[str(row["aligned_id"])] for row in rows],
    )
    render_index(dataset_id, processed, rows)
    raw_tiff_count = update_metadata_config(spec, rows)
    write_release_config(
        spec, raw_tiff_count=raw_tiff_count, aligned_count=len(rows)
    )
    return (
        catalog_summary(
            spec, raw_tiff_count=raw_tiff_count, aligned_count=len(rows)
        ),
        rows,
    )


def refresh_existing_release() -> None:
    """Refresh derived release files without recopying the binary payload."""

    summaries: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    computed: list[dict[str, Any]] = []
    for spec in DATASETS:
        summary, rows = refresh_existing_dataset(spec)
        summaries.append(summary)
        all_rows[spec["public_id"]] = rows
        computed.append(
            summarize_dataset(spec["public_id"], DATA_ROOT / spec["public_id"], rows)
        )
        print(f"{spec['public_id']}: {len(rows)} aligned samples refreshed", flush=True)
    write_statistics(computed)
    write_csv(
        DATA_ROOT / "catalog.csv",
        [
            summary | {key: value for key, value in stats.items() if key != "counts"}
            for summary, stats in zip(summaries, computed, strict=True)
        ],
    )
    viewer_rows = write_viewer_metadata(all_rows)
    write_checksums()
    print(
        f"Existing release refreshed: {len(viewer_rows)} viewer rows and checksums written."
    )


def build_release() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in DATASETS:
        summary, rows = build_dataset(spec)
        summaries.append(summary)
        all_rows[spec["public_id"]] = rows
        print(f"{spec['public_id']}: {len(rows)} aligned samples, masks generated", flush=True)
    computed = [
        summarize_dataset(
            spec["public_id"],
            DATA_ROOT / spec["public_id"],
            all_rows[spec["public_id"]],
        )
        for spec in DATASETS
    ]
    write_statistics(computed)
    catalog_rows = []
    for summary, stats in zip(summaries, computed, strict=True):
        catalog_rows.append(summary | {key: value for key, value in stats.items() if key != "counts"})
    write_csv(DATA_ROOT / "catalog.csv", catalog_rows)
    viewer_rows = write_viewer_metadata(all_rows)
    write_checksums()
    print(
        f"Release data, masks, statistics, {len(viewer_rows)} viewer rows and checksums written."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--viewer-metadata-only",
        action="store_true",
        help="Regenerate data/metadata.jsonl and its README/checksum entries only.",
    )
    mode.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Apply release boundaries and regenerate metadata/checksums in the existing data tree.",
    )
    args = parser.parse_args()
    if args.viewer_metadata_only:
        viewer_rows = write_viewer_metadata()
        update_checksum_entries(("README.md", "metadata.jsonl"))
        print(f"Viewer metadata: {len(viewer_rows)} aligned samples")
        return
    if args.refresh_existing:
        refresh_existing_release()
        return
    build_release()


if __name__ == "__main__":
    main()
