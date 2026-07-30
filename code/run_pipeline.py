#!/usr/bin/env python3
"""Config-driven public wrapper around the RGB–thermal alignment pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mask_dataset import process_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_colocated_pairs import main as build_main  # noqa: E402


def parse_time(value: str) -> str:
    return value


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_path"] = path.resolve()
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def common_args(config: dict, out: Path) -> list[str]:
    mapping = config["time_mapping"]
    args = [
        "--tiff-dir", str(resolve_path(config["paths"]["raw_tiff_dir"])),
        "--video", str(resolve_path(config["paths"]["raw_video"])),
        "--tiff-start", mapping["tiff_start"],
        "--tiff-end", mapping["tiff_end"],
        "--video-start", mapping["video_start"],
        "--video-end", mapping["video_end"],
        "--anchor-video", mapping["anchors"][0]["requested"],
        "--out", str(out),
    ]
    if len(mapping["anchors"]) > 1:
        args += ["--second-anchor-video", mapping["anchors"][1]["requested"]]
    return args


def run_stage(config: dict, stage: str) -> None:
    dataset_id = config["dataset_id"]
    paths = config["paths"]
    if stage == "masks":
        process_dataset(resolve_path(paths["processed_dir"]))
        return

    if stage == "calibrate":
        out = REPO_ROOT / "work" / dataset_id / "review"
        args = common_args(config, out) + ["--review-only", "--no-spatial-refine"]
    elif stage == "match":
        out = REPO_ROOT / "work" / dataset_id / "review"
        args = common_args(config, out) + ["--review-only", "--no-spatial-refine"]
        args += ["--pad-circles-in", str(resolve_path(paths["pad_circles"]))]
        if paths.get("video_pad_seeds"):
            args += ["--video-pad-seeds-in", str(resolve_path(paths["video_pad_seeds"]))]
    elif stage == "export":
        out = REPO_ROOT / "reproduced" / dataset_id
        args = common_args(config, out) + [
            "--transform-in", str(resolve_path(paths["transform"])),
            "--dataset-count", "1",
        ]
    else:
        raise ValueError(f"unknown stage: {stage}")
    print("uv pipeline args:", " ".join(args))
    build_main(args)
    if stage == "export":
        process_dataset(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibrate", "match", "export", "masks"), required=True)
    args = parser.parse_args()
    run_stage(load_config(args.config), args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
