#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
import math
from html import escape
import re
import shutil
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw


CIRCLE_ALIGNMENT_WEIGHT = 0.35
PAD_CENTER_HARD_MAX_RATIO = 0.12
PAD_RADIUS_HARD_MAX_RATIO = 0.18
PAD_GEOMETRY_WEIGHT = 0.65
PAD_MASKED_VISUAL_WEIGHT = 0.25
PAD_FULL_VISUAL_WEIGHT = 0.10
MAX_WINDOW_CROP_STRATEGIES = 20
VIDEO_PAD_HIGH_CONFIDENCE = 0.72
VIDEO_PAD_LOW_CONFIDENCE = 0.45
VIDEO_PAD_HOUGH_MAX_DIM = 1280
VIDEO_PAD_HOUGH_PERFECTNESS = 0.78
PAD_REVIEW_CONFIRMED_RANKS: dict[str, int] = {}
PAD_REVIEW_PENDING_NAMES: set[str] = set()
PAD_REVIEW_CONTINUITY_SEEDS: dict[str, list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class TiffFrame:
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class AnchorContext:
    name: str
    anchor_video_s: float
    anchor_tiff_target: datetime
    center_tiff: TiffFrame
    temporal_tiffs: list[TiffFrame]


@dataclass(frozen=True)
class TimeModel:
    tiff_start: datetime
    tiff_end: datetime
    video_start_s: float
    video_end_s: float
    offset_s: float = 0.0

    @property
    def tiff_duration_s(self) -> float:
        return (self.tiff_end - self.tiff_start).total_seconds()

    @property
    def video_duration_s(self) -> float:
        return self.video_end_s - self.video_start_s

    @property
    def scale(self) -> float:
        duration = self.tiff_duration_s
        if duration <= 0:
            raise ValueError("TIFF end must be later than TIFF start")
        return self.video_duration_s / duration

    def tiff_to_video_time(self, timestamp: datetime) -> float:
        rel_s = (timestamp - self.tiff_start).total_seconds()
        return self.video_start_s + rel_s * self.scale + self.offset_s

    def video_to_tiff_time(self, video_time_s: float) -> datetime:
        rel_video_s = video_time_s - self.video_start_s - self.offset_s
        return self.tiff_start + timedelta(seconds=rel_video_s / self.scale)


@dataclass(frozen=True)
class SegmentExportStrategy:
    name: str
    model: TimeModel
    spatial: "SpatialCandidate | dict[str, Any]"
    anchor_video_s: float
    selected_video_time_s: float
    center_tiff: TiffFrame


@dataclass(frozen=True)
class SpatialCandidate:
    transform_type: str
    score: float
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    matrix: list[list[float]]
    thermal_edge_density: float
    video_edge_density: float
    base_similarity: float = 0.0
    circle_alignment: float = 0.0
    pad_center_error_px: float = 0.0
    pad_center_error_ratio: float = 1.0
    pad_radius_error_ratio: float = 1.0
    pad_ring_iou: float = 0.0
    pad_geometry_score: float = 0.0
    pad_geometry_hard_pass: bool = False
    pad_thermal_circle_rank: int = -1
    keypoint_matches: int = 0
    keypoint_inliers: int = 0
    source_tiff: str = ""


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if self.fps <= 0:
            raise RuntimeError(f"Video FPS is not available: {path}")
        self.duration_s = self.frame_count / self.fps if self.frame_count else 0.0

    def frame_at(self, time_s: float) -> tuple[np.ndarray, int]:
        if time_s < 0:
            time_s = 0
        if self.duration_s and time_s > self.duration_s:
            time_s = self.duration_s
        frame_idx = int(round(time_s * self.fps))
        if self.frame_count:
            frame_idx = min(max(frame_idx, 0), self.frame_count - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read video frame at {time_s:.3f}s")
        actual_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx + 1) - 1
        return frame, max(actual_idx, 0)

    def frame_at_index(self, frame_idx: int) -> tuple[np.ndarray, int]:
        if self.frame_count:
            frame_idx = min(max(frame_idx, 0), self.frame_count - 1)
        else:
            frame_idx = max(frame_idx, 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read video frame index {frame_idx}")
        actual_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or frame_idx + 1) - 1
        return frame, max(actual_idx, 0)

    def frames_at_indices(self, frame_indices: list[int]) -> Iterator[tuple[np.ndarray, int]]:
        targets = sorted(set(frame_indices))
        if not targets:
            return
        first = targets[0]
        last = targets[-1]
        if self.frame_count:
            first = min(max(first, 0), self.frame_count - 1)
            last = min(max(last, first), self.frame_count - 1)
            targets = [idx for idx in targets if first <= idx <= last]
        target_set = set(targets)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for expected_idx in range(first, last + 1):
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read video frame index {expected_idx}")
            actual_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or expected_idx + 1) - 1
            actual_idx = max(actual_idx, 0)
            if expected_idx in target_set:
                yield frame, actual_idx

    def close(self) -> None:
        self.cap.release()

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_s": self.duration_s,
        }


def parse_video_time(value: str | int | float) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty video time")
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported video time: {value}")


def format_video_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    if millis:
        return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_tiff_timestamp(value: str | Path) -> datetime:
    stem = Path(str(value)).stem
    match = re.search(r"(\d{8}_\d{6})", stem)
    if not match:
        raise ValueError(f"Could not parse TIFF timestamp from: {value}")
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def scan_tiff_frames(tiff_dir: Path) -> list[TiffFrame]:
    frames: list[TiffFrame] = []
    for path in sorted(tiff_dir.iterdir()):
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        try:
            frames.append(TiffFrame(path=path, timestamp=parse_tiff_timestamp(path)))
        except ValueError:
            continue
    frames.sort(key=lambda frame: frame.timestamp)
    if not frames:
        raise FileNotFoundError(f"No timestamped TIFF files found in {tiff_dir}")
    return frames


def nearest_tiff(frames: list[TiffFrame], target: datetime) -> TiffFrame:
    return min(frames, key=lambda frame: abs((frame.timestamp - target).total_seconds()))


def tiffs_between(frames: list[TiffFrame], start: datetime, end: datetime) -> list[TiffFrame]:
    if end < start:
        start, end = end, start
    return [frame for frame in frames if start <= frame.timestamp <= end]


def estimate_anchor_offset(model: TimeModel, anchor_video_s: float, anchor_tiff_time: datetime) -> float:
    return anchor_video_s - model.tiff_to_video_time(anchor_tiff_time)


def load_thermal_u16(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def normalize_to_uint8(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return np.asarray(np.round(scaled * 255), dtype=np.uint8)


def heatmap_rgb(gray: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def edge_image(gray: np.ndarray) -> np.ndarray:
    enhanced = enhance_gray(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(30, 1.33 * median)))
    if upper <= lower:
        lower, upper = 30, 90
    edges = cv2.Canny(blurred, lower, upper)
    return cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)


def quick_edge_image(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 135)
    return cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)


def edge_density(edges: np.ndarray) -> float:
    return float(np.mean(edges > 0))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float32).reshape(-1) / 255.0
    bv = b.astype(np.float32).reshape(-1) / 255.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gray = enhance_gray(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return normalize_to_uint8(mag, 1, 99)


def thermal_similarity_maps(thermal_gray: np.ndarray) -> dict[str, Any]:
    edges = edge_image(thermal_gray)
    return {
        "edges": edges,
        "gradient": gradient_magnitude(thermal_gray),
        "edge_density": edge_density(edges),
    }


def thermal_matching_maps(
    thermal_gray: np.ndarray,
    max_dim: int = 192,
    confirmed_pad_circle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    maps = thermal_similarity_maps(resize_for_matching(thermal_gray, max_dim))
    maps["source_shape"] = thermal_gray.shape[:2]
    circles = [confirmed_pad_circle] if confirmed_pad_circle else detect_thermal_circles(thermal_gray)
    maps["pad_circles_full"] = circles
    maps["pad_circle_full"] = circles[0] if circles else None
    maps["confirmed_pad_circle_full"] = confirmed_pad_circle
    maps["thermal_circle_confirmed"] = confirmed_pad_circle is not None
    maps["manual_circle_source"] = (
        str(confirmed_pad_circle.get("manual_circle_source", ""))
        if confirmed_pad_circle
        else ""
    )
    maps["selected_rank"] = (
        int(confirmed_pad_circle.get("selected_rank", -1))
        if confirmed_pad_circle
        else -1
    )
    return maps


def visual_similarity_with_maps(thermal_maps: dict[str, Any], video_gray: np.ndarray) -> dict[str, float]:
    thermal_edges = thermal_maps["edges"]
    if thermal_edges.shape != video_gray.shape:
        video_gray = cv2.resize(video_gray, (thermal_edges.shape[1], thermal_edges.shape[0]))
    v_edges = edge_image(video_gray)
    edge_score = cosine_similarity(thermal_edges, v_edges)
    grad_score = cosine_similarity(thermal_maps["gradient"], gradient_magnitude(video_gray))
    return {
        "score": float(0.75 * edge_score + 0.25 * grad_score),
        "edge_score": edge_score,
        "gradient_score": grad_score,
        "thermal_edge_density": float(thermal_maps["edge_density"]),
        "video_edge_density": edge_density(v_edges),
    }


def visual_similarity(thermal_gray: np.ndarray, video_gray: np.ndarray) -> dict[str, float]:
    if thermal_gray.shape != video_gray.shape:
        video_gray = cv2.resize(video_gray, (thermal_gray.shape[1], thermal_gray.shape[0]))
    return visual_similarity_with_maps(thermal_similarity_maps(thermal_gray), video_gray)


def resize_for_matching(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if max(h, w) <= max_dim:
        return gray
    scale = max_dim / max(h, w)
    return cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def precise_thermal_edge_image(gray: np.ndarray) -> np.ndarray:
    enhanced = enhance_gray(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.55 * median))
    upper = int(min(255, max(35, 1.45 * median)))
    if upper <= lower:
        lower, upper = 35, 110
    return cv2.Canny(blurred, lower, upper)


def score_thermal_circle_candidate(
    thermal_gray: np.ndarray,
    circle: dict[str, float],
    edges: np.ndarray | None = None,
) -> dict[str, float]:
    if edges is None:
        edges = precise_thermal_edge_image(thermal_gray)
    h, w = thermal_gray.shape[:2]
    cx = float(circle["cx"])
    cy = float(circle["cy"])
    radius = float(circle["r"])
    if radius <= 0:
        return {"score": 0.0}

    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    ring_hits: list[bool] = []
    off_ring_hits: list[bool] = []
    inner_values: list[float] = []
    outer_values: list[float] = []
    for angle in angles:
        ca = math.cos(float(angle))
        sa = math.sin(float(angle))
        x = cx + radius * ca
        y = cy + radius * sa
        if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
            ring_hits.append(False)
            off_ring_hits.append(False)
            continue
        xi = int(round(x))
        yi = int(round(y))
        ring_hits.append(bool(np.any(edges[yi - 2 : yi + 3, xi - 2 : xi + 3] > 0)))

        off_hit = False
        for off_radius in (radius - 12.0, radius + 12.0):
            if off_radius <= 2:
                continue
            ox = int(round(cx + off_radius * ca))
            oy = int(round(cy + off_radius * sa))
            if 2 <= ox < w - 2 and 2 <= oy < h - 2:
                off_hit = off_hit or bool(np.any(edges[oy - 2 : oy + 3, ox - 2 : ox + 3] > 0))
        off_ring_hits.append(off_hit)

        for sample_radius, values in ((radius - 8.0, inner_values), (radius + 8.0, outer_values)):
            if sample_radius <= 0:
                continue
            sx = int(round(cx + sample_radius * ca))
            sy = int(round(cy + sample_radius * sa))
            if 0 <= sx < w and 0 <= sy < h:
                values.append(float(thermal_gray[sy, sx]))

    hits = np.asarray(ring_hits, dtype=bool)
    off_hits = np.asarray(off_ring_hits, dtype=bool)
    support = float(np.mean(hits))
    off_support = float(np.mean(off_hits))
    bins = np.array_split(hits, 36)
    coverage = float(np.mean([bool(np.any(bin_hits)) for bin_hits in bins]))
    prominence = max(0.0, support - off_support)
    contrast = (
        abs(float(np.mean(inner_values)) - float(np.mean(outer_values))) / 255.0
        if inner_values and outer_values
        else 0.0
    )
    radius_ratio = radius / max(float(min(w, h)), 1.0)
    radius_prior = math.exp(-((radius_ratio - 0.17) / 0.11) ** 2)
    border_margin = min(cx, w - cx, cy, h - cy)
    border_score = min(1.0, max(0.0, border_margin / max(radius, 1.0)))
    score = (
        0.35 * support
        + 0.30 * coverage
        + 0.15 * prominence
        + 0.10 * contrast
        + 0.07 * radius_prior
        + 0.03 * border_score
    )
    if radius_ratio > 0.30:
        score *= 0.65
    if radius_ratio > 0.24 and prominence < 0.03:
        score *= 0.85
    return {
        "score": float(score),
        "edge_support": support,
        "edge_off_support": off_support,
        "edge_coverage": coverage,
        "edge_prominence": prominence,
        "edge_contrast": contrast,
        "radius_prior": float(radius_prior),
        "border_score": border_score,
    }


def hough_thermal_circle_candidates(thermal_gray: np.ndarray) -> list[dict[str, float]]:
    h, w = thermal_gray.shape[:2]
    min_dim = min(w, h)
    blurred = cv2.medianBlur(enhance_gray(thermal_gray), 5)
    candidates: list[dict[str, float]] = []
    for param2 in (18, 24, 30):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(36, min_dim // 12),
            param1=110,
            param2=param2,
            minRadius=max(35, min_dim // 14),
            maxRadius=int(round(min_dim * 0.38)),
        )
        if circles is None:
            continue
        for x, y, radius in np.round(circles[0]).astype(float):
            candidates.append(
                {"cx": float(x), "cy": float(y), "r": float(radius), "source": "hough"}
            )
    return candidates


def contour_thermal_circle_candidates(
    thermal_gray: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, float]]:
    h, w = thermal_gray.shape[:2]
    min_dim = min(w, h)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[dict[str, float]] = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 32:
            continue
        fitted = fit_circle_least_squares(points)
        if fitted is None:
            continue
        radius = float(fitted["r"])
        if radius < max(32, min_dim * 0.06) or radius > min_dim * 0.38:
            continue
        metric = score_thermal_circle_candidate(thermal_gray, fitted, edges)
        if metric.get("edge_coverage", 0.0) < 0.18:
            continue
        candidates.append(
            {
                "cx": float(fitted["cx"]),
                "cy": float(fitted["cy"]),
                "r": radius,
                "source": "edge_fit",
            }
        )
    return candidates


def dedupe_thermal_circle_candidates(candidates: list[dict[str, float]]) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        duplicate = False
        for existing in unique:
            center_dist = math.hypot(candidate["cx"] - existing["cx"], candidate["cy"] - existing["cy"])
            radius_dist = abs(candidate["r"] - existing["r"])
            if center_dist <= max(8.0, 0.08 * existing["r"]) and radius_dist <= max(6.0, 0.08 * existing["r"]):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def detect_thermal_circles(thermal_gray: np.ndarray, max_count: int = 8) -> list[dict[str, float]]:
    h, w = thermal_gray.shape[:2]
    edges = precise_thermal_edge_image(thermal_gray)
    raw_candidates = hough_thermal_circle_candidates(thermal_gray)
    raw_candidates.extend(contour_thermal_circle_candidates(thermal_gray, edges))

    candidates: list[dict[str, float]] = []
    for candidate in raw_candidates:
        radius = float(candidate.get("r", 0.0))
        cx = float(candidate.get("cx", 0.0))
        cy = float(candidate.get("cy", 0.0))
        if radius <= 0 or cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        scored = dict(candidate)
        scored.update(score_thermal_circle_candidate(thermal_gray, scored, edges))
        candidates.append(scored)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return dedupe_thermal_circle_candidates(candidates)[:max_count]


def detect_thermal_circle(thermal_gray: np.ndarray) -> dict[str, float] | None:
    circles = detect_thermal_circles(thermal_gray, max_count=1)
    return circles[0] if circles else None


def video_landing_pad_mask(video_frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2HSV)
    # The landing pad is saturated orange/red in the visible camera.
    mask_a = cv2.inRange(hsv, (0, 55, 55), (24, 255, 255))
    mask_b = cv2.inRange(hsv, (160, 55, 55), (179, 255, 255))
    mask_c = cv2.inRange(hsv, (5, 70, 75), (38, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_a, mask_b), mask_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def score_video_hough_pad_candidate(
    saturation: np.ndarray,
    edges: np.ndarray,
    circle: dict[str, float],
    *,
    scale: float,
    image_w: int,
    image_h: int,
) -> dict[str, Any]:
    ring_support = circle_edge_support(edges, circle)
    disk = np.zeros(saturation.shape[:2], dtype=np.uint8)
    cv2.circle(
        disk,
        (int(round(circle["cx"])), int(round(circle["cy"]))),
        max(1, int(round(circle["r"]))),
        255,
        -1,
    )
    inside = disk > 0
    saturated_fill = float(np.mean(saturation[inside] >= 70)) if np.any(inside) else 0.0
    full_circle = {
        "cx": float(circle["cx"] / scale),
        "cy": float(circle["cy"] / scale),
        "r": float(circle["r"] / scale),
    }
    partial = video_circle_partial(full_circle, image_w, image_h)
    confidence = (
        0.78
        + 0.12 * min(1.0, ring_support / 0.22)
        + 0.08 * min(1.0, saturated_fill / 0.35)
        - (0.12 if partial else 0.0)
    )
    confidence = max(0.0, min(1.0, confidence))
    return {
        **full_circle,
        "score": float(confidence),
        "confidence": float(confidence),
        "source": "hough_saturation",
        "partial": bool(partial),
        "area": float(math.pi * full_circle["r"] * full_circle["r"] * saturated_fill),
        "circularity": 1.0,
        "edge_support": float(ring_support),
        "visible_area_ratio": float(saturated_fill),
    }


def video_hough_circles_concentric(a: dict[str, Any], b: dict[str, Any]) -> bool:
    radius_a = float(a["r"])
    radius_b = float(b["r"])
    max_radius = max(radius_a, radius_b)
    min_radius = min(radius_a, radius_b)
    if min_radius <= 0 or min_radius / max_radius < 0.55:
        return False
    center_dist = math.hypot(float(a["cx"]) - float(b["cx"]), float(a["cy"]) - float(b["cy"]))
    return center_dist <= max(18.0, 0.20 * max_radius)


def detect_video_hough_pad_candidates(video_frame_bgr: np.ndarray) -> list[dict[str, Any]]:
    image_h, image_w = video_frame_bgr.shape[:2]
    scale = min(1.0, VIDEO_PAD_HOUGH_MAX_DIM / max(image_w, image_h))
    if scale < 1.0:
        working = cv2.resize(
            video_frame_bgr,
            (int(round(image_w * scale)), int(round(image_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = video_frame_bgr
    saturation = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (9, 9), 2)
    edges = cv2.Canny(blurred, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    min_dim = min(working.shape[:2])
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=max(32.0, 0.046 * min_dim),
        param1=150,
        param2=VIDEO_PAD_HOUGH_PERFECTNESS,
        minRadius=max(18, int(round(0.023 * min_dim))),
        maxRadius=max(24, int(round(0.33 * min_dim))),
    )
    if circles is None:
        return []

    raw = [
        score_video_hough_pad_candidate(
            saturation,
            edges,
            {"cx": float(cx), "cy": float(cy), "r": float(radius)},
            scale=scale,
            image_w=image_w,
            image_h=image_h,
        )
        for cx, cy, radius in circles[0]
    ]
    # Hough generally returns both the inner and outer painted rings.  Group
    # concentric detections and retain the outer landing-pad boundary.
    result: list[dict[str, Any]] = []
    for candidate in sorted(raw, key=lambda item: float(item["r"]), reverse=True):
        if any(video_hough_circles_concentric(candidate, existing) for existing in result):
            continue
        result.append(candidate)
    return result


def fit_circle_least_squares(points: np.ndarray) -> dict[str, float] | None:
    if points.shape[0] < 6:
        return None
    pts = points.reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        coeff, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = -coeff[0] / 2.0
    cy = -coeff[1] / 2.0
    radius_sq = cx * cx + cy * cy - coeff[2]
    if not math.isfinite(radius_sq) or radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    if not all(math.isfinite(value) for value in [cx, cy, radius]):
        return None
    return {"cx": float(cx), "cy": float(cy), "r": float(radius)}


def circle_edge_support(mask: np.ndarray, circle: dict[str, float]) -> float:
    ring = circle_ring_mask(mask.shape[:2], circle, thickness_ratio=0.06)
    if not np.any(ring):
        return 0.0
    return float(np.mean(mask[ring] > 0))


def video_circle_partial(circle: dict[str, float], image_w: int, image_h: int, margin: float = 2.0) -> bool:
    return (
        circle["cx"] - circle["r"] <= margin
        or circle["cy"] - circle["r"] <= margin
        or circle["cx"] + circle["r"] >= image_w - margin
        or circle["cy"] + circle["r"] >= image_h - margin
    )


def score_video_pad_candidate(
    mask: np.ndarray,
    contour: np.ndarray,
    circle: dict[str, float],
    *,
    source: str,
) -> dict[str, float]:
    h, w = mask.shape[:2]
    radius = float(circle["r"])
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float(4 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    x, y, bw, bh = cv2.boundingRect(contour)
    touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
    partial = touches_border or video_circle_partial(circle, w, h)
    radius_norm = min(1.0, radius / max(min(w, h) * 0.18, 1.0))
    support = circle_edge_support(mask, circle)
    visible_area_ratio = area / max(math.pi * radius * radius, 1.0)
    visible_area_score = min(1.0, visible_area_ratio / (0.30 if partial else 0.58))
    source_bonus = 0.08 if source == "arc_fit" and partial else 0.0
    score = (
        0.30 * radius_norm
        + 0.25 * max(0.0, min(circularity, 1.0))
        + 0.25 * support
        + 0.20 * visible_area_score
        + source_bonus
    )
    confidence = max(0.0, min(1.0, score))
    return {
        "cx": float(circle["cx"]),
        "cy": float(circle["cy"]),
        "r": radius,
        "score": float(score),
        "confidence": float(confidence),
        "source": source,
        "partial": bool(partial),
        "area": area,
        "circularity": circularity,
        "edge_support": support,
        "visible_area_ratio": float(visible_area_ratio),
    }


def detect_video_landing_pad_candidates(video_frame_bgr: np.ndarray, max_count: int = 5) -> list[dict[str, float]]:
    h, w = video_frame_bgr.shape[:2]
    mask = video_landing_pad_mask(video_frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_candidates: list[dict[str, Any]] = []
    image_area = float(w * h)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.0004:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4 * math.pi * area / (perimeter * perimeter))
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        if not 0.55 <= aspect <= 1.8:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius <= 20:
            continue
        enclosing = {"cx": float(cx), "cy": float(cy), "r": float(radius)}
        contour_candidates.append(
            score_video_pad_candidate(mask, contour, enclosing, source="outer_contour")
        )

        fitted = fit_circle_least_squares(contour.reshape(-1, 2))
        if fitted and 20 <= fitted["r"] <= max(w, h):
            contour_candidates.append(
                score_video_pad_candidate(mask, contour, fitted, source="arc_fit")
            )

    hough_candidates = detect_video_hough_pad_candidates(video_frame_bgr)
    candidates: list[dict[str, Any]] = list(hough_candidates)
    for candidate in contour_candidates:
        # When Hough already found this colored object, prefer its precise
        # circular boundary over a merged orange/brown contour.
        overlaps_hough = any(
            math.hypot(
                float(candidate["cx"]) - float(hough["cx"]),
                float(candidate["cy"]) - float(hough["cy"]),
            )
            <= max(24.0, 0.30 * max(float(candidate["r"]), float(hough["r"])))
            and min(float(candidate["r"]), float(hough["r"]))
            / max(float(candidate["r"]), float(hough["r"]))
            >= 0.45
            for hough in hough_candidates
        )
        if not overlaps_hough:
            candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            float(item["confidence"]),
            item.get("source") == "hough_saturation",
            float(item["r"]),
            -float(item.get("partial", False)),
        ),
        reverse=True,
    )
    return candidates[:max_count]


def detect_video_landing_pad(video_frame_bgr: np.ndarray) -> dict[str, float] | None:
    candidates = detect_video_landing_pad_candidates(video_frame_bgr, max_count=1)
    return candidates[0] if candidates else None


def video_circle_distance_cost(
    candidate: dict[str, Any],
    predicted: dict[str, float],
) -> tuple[float, float, float]:
    predicted_radius = max(float(predicted["r"]), 24.0)
    center_ratio = math.hypot(
        float(candidate["cx"]) - float(predicted["cx"]),
        float(candidate["cy"]) - float(predicted["cy"]),
    ) / predicted_radius
    radius_log_error = abs(math.log(max(float(candidate["r"]), 1.0) / predicted_radius))
    confidence = float(candidate.get("confidence", 0.0))
    source_penalty = 0.0 if candidate.get("source") == "hough_saturation" else 0.35
    partial_penalty = 0.30 if bool(candidate.get("partial", False)) else 0.0
    cost = 2.8 * center_ratio + 1.8 * radius_log_error + 0.7 * (1.0 - confidence) + source_penalty + partial_penalty
    return float(cost), float(center_ratio), float(radius_log_error)


def predicted_video_circle(
    frame_idx: int,
    last: tuple[int, dict[str, Any]],
    previous: tuple[int, dict[str, Any]] | None,
) -> dict[str, float]:
    last_idx, last_circle = last
    predicted = {key: float(last_circle[key]) for key in ["cx", "cy", "r"]}
    if previous is None or last_idx == previous[0]:
        return predicted
    step = (frame_idx - last_idx) / (last_idx - previous[0])
    for key in ["cx", "cy"]:
        delta = float(last_circle[key]) - float(previous[1][key])
        predicted[key] = float(last_circle[key]) + step * delta
    # Circle radius can alternate between inner and outer painted rings for a
    # single frame.  Extrapolating that jitter compounds the error and can
    # reject the correct outer circle on the next frame, so keep the latest
    # radius as the prediction and let the radius gate recover smoothly.
    predicted["r"] = max(20.0, float(last_circle["r"]))
    return predicted


def select_video_track_candidate(
    frame_idx: int,
    candidates: list[dict[str, Any]],
    last: tuple[int, dict[str, Any]],
    previous: tuple[int, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    predicted = predicted_video_circle(frame_idx, last, previous)
    frame_gap = max(1, abs(frame_idx - last[0]))
    center_gate = min(2.4, 0.90 + 0.12 * frame_gap)
    radius_gate = min(math.log(2.2), math.log(1.48) + 0.025 * frame_gap)
    scored: list[tuple[float, dict[str, Any], float, float]] = []
    for candidate in candidates:
        cost, center_ratio, radius_log_error = video_circle_distance_cost(candidate, predicted)
        if center_ratio > center_gate or radius_log_error > radius_gate:
            continue
        scored.append((cost, candidate, center_ratio, radius_log_error))
    if not scored:
        return None
    cost, candidate, center_ratio, radius_log_error = min(scored, key=lambda item: item[0])
    selected = copy_video_circle(candidate)
    assert selected is not None
    selected.update(
        {
            "raw_source": candidate.get("source", ""),
            "source": "tracked_hough" if candidate.get("source") == "hough_saturation" else "tracked_contour",
            "track_cost": float(cost),
            "track_center_error_ratio": float(center_ratio),
            "track_radius_log_error": float(radius_log_error),
            "smoothed": False,
        }
    )
    return selected


def seed_video_pad_circle(
    frame_idx: int,
    candidates: list[dict[str, Any]],
    seed: dict[str, Any] | None,
) -> dict[str, Any] | None:
    seed_circle = seed.get("circle") if isinstance(seed, dict) else None
    if isinstance(seed_circle, dict):
        predicted = {key: float(seed_circle[key]) for key in ["cx", "cy", "r"]}
        if candidates:
            matched = min(candidates, key=lambda item: video_circle_distance_cost(item, predicted)[0])
            _, center_ratio, radius_log_error = video_circle_distance_cost(matched, predicted)
            if center_ratio <= 0.55 and radius_log_error <= math.log(1.55):
                selected = copy_video_circle(matched)
                assert selected is not None
                selected.update(
                    {
                        "raw_source": matched.get("source", ""),
                        "source": "seeded_hough" if matched.get("source") == "hough_saturation" else "seeded_contour",
                        "track_cost": 0.0,
                        "track_center_error_ratio": float(center_ratio),
                        "track_radius_log_error": float(radius_log_error),
                        "seed_frame_index": int(frame_idx),
                        "smoothed": False,
                    }
                )
                return selected
        return {
            **predicted,
            "score": 1.0,
            "confidence": 1.0,
            "source": "manual_seed",
            "raw_source": "manual_seed",
            "partial": False,
            "seed_frame_index": int(frame_idx),
            "smoothed": False,
        }
    eligible = [candidate for candidate in candidates if not bool(candidate.get("partial", False))]
    candidate = eligible[0] if eligible else (candidates[0] if candidates else None)
    if candidate is None:
        return None
    selected = copy_video_circle(candidate)
    assert selected is not None
    selected.update(
        {
            "raw_source": candidate.get("source", ""),
            "source": "auto_seed_hough" if candidate.get("source") == "hough_saturation" else "auto_seed_contour",
            "seed_frame_index": int(frame_idx),
            "smoothed": False,
        }
    )
    return selected


def propagate_video_pad_track(
    indices: list[int],
    candidate_track: dict[int, list[dict[str, Any]]],
    seed_idx: int,
    seed_circle: dict[str, Any],
    *,
    direction: int,
) -> dict[int, dict[str, Any] | None]:
    ordered = [idx for idx in indices if (idx - seed_idx) * direction > 0]
    ordered.sort(reverse=direction < 0)
    result: dict[int, dict[str, Any] | None] = {}
    last: tuple[int, dict[str, Any]] = (seed_idx, seed_circle)
    previous: tuple[int, dict[str, Any]] | None = None
    for frame_idx in ordered:
        selected = select_video_track_candidate(
            frame_idx,
            candidate_track.get(frame_idx, []),
            last,
            previous,
        )
        result[frame_idx] = selected
        if selected is not None:
            previous = last
            last = (frame_idx, selected)
    return result


def fill_video_pad_track_gaps(
    track: dict[int, dict[str, Any] | None],
    *,
    window: int,
) -> dict[int, dict[str, Any] | None]:
    reliable = [(idx, circle) for idx, circle in sorted(track.items()) if circle is not None]
    result = dict(track)
    for idx, circle in sorted(track.items()):
        if circle is not None:
            continue
        before = next(((i, c) for i, c in reversed(reliable) if i < idx and idx - i <= window), None)
        after = next(((i, c) for i, c in reliable if i > idx and i - idx <= window), None)
        filled = interpolate_video_circle(idx, before, after)
        if filled is not None:
            filled["source"] = "track_gap_interpolated"
            filled["raw_source"] = "no_gated_candidate"
            result[idx] = filled
    return result


def copy_video_circle(circle: dict[str, Any] | None) -> dict[str, Any] | None:
    if not circle:
        return None
    return {
        key: (float(value) if isinstance(value, np.generic | int | float) and not isinstance(value, bool) else value)
        for key, value in circle.items()
    }


def interpolate_video_circle(
    frame_idx: int,
    before: tuple[int, dict[str, Any]] | None,
    after: tuple[int, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if before and after and after[0] != before[0]:
        ratio = (frame_idx - before[0]) / (after[0] - before[0])
        circle = {
            key: float(before[1][key] + ratio * (after[1][key] - before[1][key]))
            for key in ["cx", "cy", "r"]
        }
        circle.update(
            {
                "score": float(0.5 * (before[1].get("score", 0.0) + after[1].get("score", 0.0))),
                "confidence": float(0.5 * (before[1].get("confidence", 0.0) + after[1].get("confidence", 0.0))),
                "source": "track_interpolated",
                "partial": False,
                "smoothed": True,
                "raw_source": "interpolated_from_neighbors",
            }
        )
        return circle
    neighbor = before or after
    if not neighbor:
        return None
    circle = copy_video_circle(neighbor[1])
    if circle:
        circle["source"] = "track_neighbor"
        circle["smoothed"] = True
        circle["raw_source"] = "nearest_high_confidence"
    return circle


def smooth_video_pad_circle_track(
    raw_track: dict[int, dict[str, Any] | None],
    *,
    window: int,
) -> dict[int, dict[str, Any] | None]:
    indices = sorted(raw_track)
    high_conf = [
        (idx, circle)
        for idx, circle in sorted(raw_track.items())
        if circle
        and float(circle.get("confidence", 0.0)) >= VIDEO_PAD_HIGH_CONFIDENCE
        and not bool(circle.get("partial", False))
    ]
    result: dict[int, dict[str, Any] | None] = {}
    for idx in indices:
        circle = copy_video_circle(raw_track[idx])
        if circle and float(circle.get("confidence", 0.0)) >= VIDEO_PAD_HIGH_CONFIDENCE and not bool(circle.get("partial", False)):
            circle["smoothed"] = False
            result[idx] = circle
            continue
        before = next(((i, c) for i, c in reversed(high_conf) if i < idx and idx - i <= window), None)
        after = next(((i, c) for i, c in high_conf if i > idx and i - idx <= window), None)
        smoothed = interpolate_video_circle(idx, before, after)
        if smoothed is not None:
            raw = raw_track[idx] or {}
            smoothed["raw_confidence"] = float(raw.get("confidence", 0.0)) if raw else 0.0
            smoothed["raw_partial"] = bool(raw.get("partial", False)) if raw else False
            result[idx] = smoothed
        else:
            if circle:
                circle["smoothed"] = False
            result[idx] = circle
    return result


def window_scoring_frame_indices(
    candidate_frame_indices: list[int],
    fps: float,
    frame_count: int,
    offsets: list[int],
) -> list[int]:
    result: set[int] = set()
    for frame_idx in candidate_frame_indices:
        for offset in offsets:
            idx = frame_idx + int(round(offset * fps))
            if idx < 0:
                continue
            if frame_count and idx >= frame_count:
                continue
            result.add(idx)
    return sorted(result)


def build_video_pad_circle_track(
    reader: VideoReader,
    frame_indices: list[int],
    *,
    smoothing_window: int,
    seed: dict[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any] | None], dict[int, list[dict[str, Any]]]]:
    candidate_track: dict[int, list[dict[str, Any]]] = {}
    unique_indices = sorted(set(frame_indices))
    for frame_bgr, actual_idx in reader.frames_at_indices(unique_indices):
        candidate_track[actual_idx] = detect_video_landing_pad_candidates(frame_bgr, max_count=12)
    indices = sorted(candidate_track)
    if not indices:
        return {}, candidate_track
    requested_seed_idx = int(seed.get("frame_index")) if isinstance(seed, dict) and seed.get("frame_index") is not None else indices[len(indices) // 2]
    seed_idx = min(indices, key=lambda idx: abs(idx - requested_seed_idx))
    seed_circle = seed_video_pad_circle(seed_idx, candidate_track.get(seed_idx, []), seed)
    if seed_circle is None:
        raw = {
            idx: (copy_video_circle(candidates[0]) if candidates else None)
            for idx, candidates in candidate_track.items()
        }
        return smooth_video_pad_circle_track(raw, window=max(0, smoothing_window)), candidate_track

    track: dict[int, dict[str, Any] | None] = {seed_idx: seed_circle}
    track.update(
        propagate_video_pad_track(
            indices,
            candidate_track,
            seed_idx,
            seed_circle,
            direction=1,
        )
    )
    track.update(
        propagate_video_pad_track(
            indices,
            candidate_track,
            seed_idx,
            seed_circle,
            direction=-1,
        )
    )
    track = {idx: track.get(idx) for idx in indices}
    return fill_video_pad_track_gaps(track, window=max(0, smoothing_window)), candidate_track


def map_video_circle_to_thermal(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
    video_circle: dict[str, float] | None,
) -> dict[str, float] | None:
    if not video_circle or crop_w <= 0 or crop_h <= 0:
        return None
    rx = video_circle["r"] * out_w / crop_w
    ry = video_circle["r"] * out_h / crop_h
    return {
        "cx": float((video_circle["cx"] - crop_x) * out_w / crop_w),
        "cy": float((video_circle["cy"] - crop_y) * out_h / crop_h),
        "r": float(0.5 * (rx + ry)),
        "rx": float(rx),
        "ry": float(ry),
    }


def circle_ring_mask(
    shape: tuple[int, int],
    circle: dict[str, float] | None,
    *,
    thickness_ratio: float = 0.045,
) -> np.ndarray:
    h, w = shape[:2]
    if not circle or circle.get("r", 0.0) <= 0:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - circle["cx"]) ** 2 + (yy - circle["cy"]) ** 2)
    thickness = max(3.0, float(circle["r"]) * thickness_ratio)
    return np.abs(dist - circle["r"]) <= thickness


def circle_fill_mask(
    shape: tuple[int, int],
    circle: dict[str, float] | None,
    *,
    radius_scale: float = 1.22,
) -> np.ndarray:
    h, w = shape[:2]
    if not circle or circle.get("r", 0.0) <= 0:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - circle["cx"]) ** 2 + (yy - circle["cy"]) ** 2)
    return dist <= float(circle["r"]) * radius_scale


def scale_circle_to_shape(
    circle: dict[str, float] | None,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> dict[str, float] | None:
    if not circle:
        return None
    source_h, source_w = source_shape[:2]
    target_h, target_w = target_shape[:2]
    if source_w <= 0 or source_h <= 0:
        return None
    sx = target_w / source_w
    sy = target_h / source_h
    return {
        "cx": float(circle["cx"] * sx),
        "cy": float(circle["cy"] * sy),
        "r": float(circle["r"] * 0.5 * (sx + sy)),
    }


def pad_geometry_metrics(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
    thermal_circle: dict[str, float] | None,
    video_circle: dict[str, float] | None,
) -> dict[str, Any]:
    empty = {
        "pad_center_error_px": float("inf"),
        "pad_center_error_ratio": 1.0,
        "pad_radius_error_ratio": 1.0,
        "pad_radius_anisotropy_ratio": 1.0,
        "pad_ring_iou": 0.0,
        "pad_geometry_score": 0.0,
        "pad_geometry_hard_pass": False,
        "pad_thermal_circle_rank": -1,
        "mapped_video_circle": None,
        "thermal_circle": thermal_circle,
        "video_circle": video_circle,
    }
    mapped = map_video_circle_to_thermal(
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        out_w,
        out_h,
        video_circle,
    )
    if not thermal_circle or not mapped or thermal_circle.get("r", 0.0) <= 0:
        return empty

    center_err = math.hypot(mapped["cx"] - thermal_circle["cx"], mapped["cy"] - thermal_circle["cy"])
    center_ratio = center_err / max(float(thermal_circle["r"]), 1.0)
    radius_err = abs(mapped["r"] - thermal_circle["r"]) / max(float(thermal_circle["r"]), 1.0)
    anisotropy = abs(mapped["rx"] - mapped["ry"]) / max(mapped["r"], 1.0)
    center_score = math.exp(-((center_ratio / 0.075) ** 2))
    radius_score = math.exp(-((radius_err / 0.12) ** 2))
    anisotropy_score = math.exp(-((anisotropy / 0.18) ** 2))
    ring_iou = float(center_score * radius_score)
    geometry_score = (
        0.48 * center_score
        + 0.28 * radius_score
        + 0.16 * ring_iou
        + 0.08 * anisotropy_score
    )
    hard_pass = (
        center_ratio <= PAD_CENTER_HARD_MAX_RATIO
        and radius_err <= PAD_RADIUS_HARD_MAX_RATIO
        and anisotropy <= 0.25
    )
    return {
        "pad_center_error_px": float(center_err),
        "pad_center_error_ratio": float(center_ratio),
        "pad_radius_error_ratio": float(radius_err),
        "pad_radius_anisotropy_ratio": float(anisotropy),
        "pad_ring_iou": ring_iou,
        "pad_geometry_score": float(geometry_score),
        "pad_geometry_hard_pass": hard_pass,
        "mapped_video_circle": mapped,
        "thermal_circle": thermal_circle,
        "video_circle": video_circle,
    }


def best_pad_geometry_metrics(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
    thermal_circles: list[dict[str, float]],
    video_circle: dict[str, float] | None,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for rank, thermal_circle in enumerate(thermal_circles, start=1):
        metric = pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w,
            out_h,
            thermal_circle,
            video_circle,
        )
        metric["pad_thermal_circle_rank"] = rank
        metrics.append(metric)
    if not metrics:
        return pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w,
            out_h,
            None,
            video_circle,
        )
    return max(metrics, key=lambda item: item["pad_geometry_score"])


def masked_cosine_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if mask.shape != a.shape:
        mask = cv2.resize(mask.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if int(np.sum(mask)) < 32:
        return cosine_similarity(a, b)
    av = a.astype(np.float32)[mask] / 255.0
    bv = b.astype(np.float32)[mask] / 255.0
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(av, bv) / denom)


def masked_visual_similarity_with_maps(
    thermal_maps: dict[str, Any],
    video_gray: np.ndarray,
    thermal_circle: dict[str, float] | None,
) -> dict[str, float]:
    thermal_edges = thermal_maps["edges"]
    if thermal_edges.shape != video_gray.shape:
        video_gray = cv2.resize(video_gray, (thermal_edges.shape[1], thermal_edges.shape[0]))
    source_shape = thermal_maps.get("source_shape", thermal_edges.shape[:2])
    circle = scale_circle_to_shape(thermal_circle, source_shape, thermal_edges.shape[:2])
    mask = circle_fill_mask(thermal_edges.shape[:2], circle)
    video_edges = edge_image(video_gray)
    edge_score = masked_cosine_similarity(thermal_edges, video_edges, mask)
    grad_score = masked_cosine_similarity(thermal_maps["gradient"], gradient_magnitude(video_gray), mask)
    return {
        "masked_visual_score": float(0.75 * edge_score + 0.25 * grad_score),
        "masked_edge_score": edge_score,
        "masked_gradient_score": grad_score,
    }


def composite_pad_score(
    *,
    pad_geometry_score: float,
    masked_visual_score: float,
    full_visual_score: float,
    hard_pass: bool,
) -> float:
    score = (
        PAD_GEOMETRY_WEIGHT * pad_geometry_score
        + PAD_MASKED_VISUAL_WEIGHT * masked_visual_score
        + PAD_FULL_VISUAL_WEIGHT * full_visual_score
    )
    if not hard_pass:
        score *= 0.25
    return float(score)


def circle_alignment_score(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
    thermal_circle: dict[str, float] | None,
    video_circle: dict[str, float] | None,
) -> float:
    return float(
        pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w,
            out_h,
            thermal_circle,
            video_circle,
        )["pad_geometry_score"]
    )


def guided_crops_from_circles(
    video_w: int,
    video_h: int,
    out_w: int,
    out_h: int,
    thermal_circle: dict[str, float] | None,
    video_circle: dict[str, float] | None,
) -> list[tuple[int, int, int, int]]:
    if not thermal_circle or not video_circle:
        return []
    base_crop_w = video_circle["r"] * out_w / max(thermal_circle["r"], 1.0)
    base_crop_h = video_circle["r"] * out_h / max(thermal_circle["r"], 1.0)
    crops: list[tuple[int, int, int, int]] = []
    size_scales = [0.92, 0.97, 1.0, 1.03, 1.08]
    aspect_scales = [0.98, 1.0, 1.02]
    center_shifts = [-0.12, 0.0, 0.12]
    for scale in size_scales:
        for aspect_scale in aspect_scales:
            crop_w = int(round(base_crop_w * scale * math.sqrt(aspect_scale)))
            crop_h = int(round(base_crop_h * scale / math.sqrt(aspect_scale)))
            if crop_w <= 0 or crop_h <= 0 or crop_w > video_w or crop_h > video_h:
                continue
            base_x = video_circle["cx"] - thermal_circle["cx"] * crop_w / out_w
            base_y = video_circle["cy"] - thermal_circle["cy"] * crop_h / out_h
            for shift_y in center_shifts:
                for shift_x in center_shifts:
                    x = int(round(base_x + shift_x * video_circle["r"]))
                    y = int(round(base_y + shift_y * video_circle["r"]))
                    x = min(max(x, 0), video_w - crop_w)
                    y = min(max(y, 0), video_h - crop_h)
                    crops.append((x, y, crop_w, crop_h))
    return unique_crops(crops)


def crop_resize_matrix(crop_x: int, crop_y: int, crop_w: int, crop_h: int, out_w: int, out_h: int) -> np.ndarray:
    sx = out_w / crop_w
    sy = out_h / crop_h
    return np.asarray([[sx, 0.0, -crop_x * sx], [0.0, sy, -crop_y * sy]], dtype=np.float32)


def compose_affine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a3 = np.vstack([a, [0.0, 0.0, 1.0]])
    b3 = np.vstack([b, [0.0, 0.0, 1.0]])
    return (a3 @ b3)[:2].astype(np.float32)


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(matrix)]


def refine_affine_with_features(
    thermal_gray: np.ndarray,
    video_warped_gray: np.ndarray,
    base_full_to_thermal: np.ndarray,
    base_score: float,
) -> tuple[np.ndarray, str, int, int, float]:
    detector = cv2.AKAZE_create()
    t_keypoints, t_desc = detector.detectAndCompute(enhance_gray(thermal_gray), None)
    v_keypoints, v_desc = detector.detectAndCompute(enhance_gray(video_warped_gray), None)
    if t_desc is None or v_desc is None or len(t_keypoints) < 8 or len(v_keypoints) < 8:
        return base_full_to_thermal, "crop_resize", 0, 0, base_score

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(v_desc, t_desc, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good.append(m)
    if len(good) < 8:
        return base_full_to_thermal, "crop_resize", len(good), 0, base_score

    src = np.float32([v_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([t_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    refined, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=2000,
        confidence=0.98,
    )
    if refined is None or inliers is None:
        return base_full_to_thermal, "crop_resize", len(good), 0, base_score
    inlier_count = int(np.sum(inliers))
    if inlier_count < 6:
        return base_full_to_thermal, "crop_resize", len(good), inlier_count, base_score

    refined_full = compose_affine(refined.astype(np.float32), base_full_to_thermal)
    return refined_full, "affine_refined", len(good), inlier_count, base_score


def iter_crop_grid(
    video_w: int,
    video_h: int,
    target_aspect: float,
    scale_fractions: list[float],
    step_fraction: float,
) -> list[tuple[int, int, int, int]]:
    crops: list[tuple[int, int, int, int]] = []
    for frac in scale_fractions:
        crop_h = int(round(video_h * frac))
        crop_w = int(round(crop_h * target_aspect))
        if crop_h <= 0 or crop_w <= 0 or crop_w > video_w or crop_h > video_h:
            continue
        step = max(32, int(round(min(crop_w, crop_h) * step_fraction)))
        xs = list(range(0, max(1, video_w - crop_w + 1), step))
        ys = list(range(0, max(1, video_h - crop_h + 1), step))
        if xs[-1] != video_w - crop_w:
            xs.append(video_w - crop_w)
        if ys[-1] != video_h - crop_h:
            ys.append(video_h - crop_h)
        for y in ys:
            for x in xs:
                crops.append((x, y, crop_w, crop_h))
    return crops


def unique_crops(crops: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[tuple[int, int, int, int]] = []
    for crop in crops:
        if crop in seen:
            continue
        seen.add(crop)
        unique.append(crop)
    return unique


def estimate_spatial_candidate_pool(
    thermal_gray: np.ndarray,
    video_frame_bgr: np.ndarray,
    *,
    source_tiff: str,
    crop_scale_fractions: list[float],
    roi_step_fraction: float,
    max_candidates: int | None = None,
    thermal_circles_override: list[dict[str, float]] | None = None,
    thermal_edge_density_override: float | None = None,
    video_circle_override: dict[str, Any] | None = None,
) -> list[SpatialCandidate]:
    out_h, out_w = thermal_gray.shape[:2]
    video_gray_full = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2GRAY)
    video_h, video_w = video_gray_full.shape[:2]
    target_aspect = out_w / out_h
    thermal_circles = (
        thermal_circles_override
        if thermal_circles_override is not None
        else detect_thermal_circles(thermal_gray)
    )
    video_circle = video_circle_override or detect_video_landing_pad(video_frame_bgr)
    use_geometry_only = bool(thermal_circles and video_circle)
    base_density = float(thermal_edge_density_override or 0.0)
    match_w = match_h = 0
    thermal_match_edges: np.ndarray | None = None
    if not use_geometry_only:
        full_thermal_maps = thermal_similarity_maps(thermal_gray)
        base_density = float(full_thermal_maps["edge_density"])
        match_max_dim = 128
        if max(out_w, out_h) > match_max_dim:
            scale = match_max_dim / max(out_w, out_h)
            match_w = max(1, int(round(out_w * scale)))
            match_h = max(1, int(round(out_h * scale)))
            thermal_match_gray = cv2.resize(thermal_gray, (match_w, match_h), interpolation=cv2.INTER_AREA)
        else:
            match_w = out_w
            match_h = out_h
            thermal_match_gray = thermal_gray
        thermal_match_edges = quick_edge_image(thermal_match_gray)
    scored: list[SpatialCandidate] = []

    guided_crops: list[tuple[int, int, int, int]] = []
    for thermal_circle in thermal_circles[:8]:
        guided_crops.extend(
            guided_crops_from_circles(
                video_w,
                video_h,
                out_w,
                out_h,
                thermal_circle,
                video_circle,
            )
        )
    grid_crops = iter_crop_grid(
        video_w,
        video_h,
        target_aspect,
        crop_scale_fractions,
        roi_step_fraction,
    )
    for crop_x, crop_y, crop_w, crop_h in unique_crops(guided_crops + grid_crops):
        geometry = best_pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            out_w,
            out_h,
            thermal_circles,
            video_circle,
        )
        circle_score = float(geometry["pad_geometry_score"])
        if use_geometry_only:
            base_similarity = 0.0
            video_density = 0.0
            combined_score = circle_score
        else:
            if thermal_match_edges is None:
                raise RuntimeError("Thermal match edges are unavailable for visual crop scoring")
            crop = video_gray_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            resized = cv2.resize(crop, (match_w, match_h), interpolation=cv2.INTER_AREA)
            video_match_edges = quick_edge_image(resized)
            base_similarity = cosine_similarity(thermal_match_edges, video_match_edges)
            video_density = edge_density(video_match_edges)
            combined_score = base_similarity
        matrix = crop_resize_matrix(crop_x, crop_y, crop_w, crop_h, out_w, out_h)
        candidate = SpatialCandidate(
            transform_type="crop_resize",
            score=combined_score,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            matrix=matrix_to_list(matrix),
            thermal_edge_density=base_density,
            video_edge_density=video_density,
            base_similarity=base_similarity,
            circle_alignment=circle_score,
            pad_center_error_px=float(geometry["pad_center_error_px"]),
            pad_center_error_ratio=float(geometry["pad_center_error_ratio"]),
            pad_radius_error_ratio=float(geometry["pad_radius_error_ratio"]),
            pad_ring_iou=float(geometry["pad_ring_iou"]),
            pad_geometry_score=circle_score,
            pad_geometry_hard_pass=bool(geometry["pad_geometry_hard_pass"]),
            pad_thermal_circle_rank=int(geometry.get("pad_thermal_circle_rank", -1)),
            source_tiff=source_tiff,
        )
        scored.append(candidate)

    scored.sort(key=lambda item: item.score, reverse=True)
    if max_candidates is not None and max_candidates > 0:
        return scored[:max_candidates]
    return scored


def refine_spatial_candidate(
    thermal_gray: np.ndarray,
    video_frame_bgr: np.ndarray,
    candidate: SpatialCandidate,
) -> SpatialCandidate:
    out_h, out_w = thermal_gray.shape[:2]
    full_thermal_maps = thermal_similarity_maps(thermal_gray)
    matrix = np.asarray(candidate.matrix, dtype=np.float32)
    warped = warp_video_to_thermal(video_frame_bgr, candidate, (out_w, out_h))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    refined_matrix, transform_type, matches, inliers, _ = refine_affine_with_features(
        thermal_gray,
        warped_gray,
        matrix,
        candidate.score,
    )
    if transform_type == "affine_refined":
        return replace(candidate, keypoint_matches=matches, keypoint_inliers=inliers)
    refined_warped = cv2.warpAffine(video_frame_bgr, refined_matrix, (out_w, out_h))
    refined_gray = cv2.cvtColor(refined_warped, cv2.COLOR_BGR2GRAY)
    refined_sim = visual_similarity_with_maps(full_thermal_maps, refined_gray)
    refined_score = refined_sim["score"] + CIRCLE_ALIGNMENT_WEIGHT * candidate.circle_alignment
    use_refined = (
        transform_type == "affine_refined"
        and refined_score >= candidate.score * 0.95
        and inliers >= 6
    )
    if use_refined:
        return replace(
            candidate,
            transform_type="affine_refined",
            score=refined_score,
            matrix=matrix_to_list(refined_matrix),
            base_similarity=refined_sim["score"],
            keypoint_matches=matches,
            keypoint_inliers=inliers,
            video_edge_density=refined_sim["video_edge_density"],
        )
    return replace(candidate, keypoint_matches=matches, keypoint_inliers=inliers)


def estimate_spatial_candidates(
    thermal_gray: np.ndarray,
    video_frame_bgr: np.ndarray,
    *,
    source_tiff: str,
    top_k: int,
    crop_scale_fractions: list[float],
    roi_step_fraction: float,
    refine_features: bool,
) -> list[SpatialCandidate]:
    out_h, out_w = thermal_gray.shape[:2]
    video_gray_full = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2GRAY)
    full_thermal_maps = thermal_similarity_maps(thermal_gray)
    top = estimate_spatial_candidate_pool(
        thermal_gray,
        video_frame_bgr,
        source_tiff=source_tiff,
        crop_scale_fractions=crop_scale_fractions,
        roi_step_fraction=roi_step_fraction,
        max_candidates=top_k,
    )
    full_rescored: list[SpatialCandidate] = []
    for candidate in top:
        crop = video_gray_full[
            candidate.crop_y : candidate.crop_y + candidate.crop_h,
            candidate.crop_x : candidate.crop_x + candidate.crop_w,
        ]
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        sim = visual_similarity_with_maps(full_thermal_maps, resized)
        full_rescored.append(
            replace(
                candidate,
                score=0.82 * candidate.pad_geometry_score + 0.18 * sim["score"],
                base_similarity=sim["score"],
                video_edge_density=sim["video_edge_density"],
            )
        )
    full_rescored.sort(key=lambda item: item.score, reverse=True)
    top = full_rescored[:top_k]
    if refine_features:
        refined = [
            refine_spatial_candidate(thermal_gray, video_frame_bgr, candidate)
            for candidate in top
        ]
        refined.sort(key=lambda item: item.score, reverse=True)
        return refined[:top_k]

    return top[:top_k]


def warp_video_to_thermal(
    video_frame_bgr: np.ndarray,
    candidate: SpatialCandidate | dict[str, Any],
    size: tuple[int, int],
) -> np.ndarray:
    out_w, out_h = size
    matrix = np.asarray(candidate["matrix"] if isinstance(candidate, dict) else candidate.matrix, dtype=np.float32)
    return cv2.warpAffine(
        video_frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_video_to_similarity_maps(
    video_frame_bgr: np.ndarray,
    candidate: SpatialCandidate | dict[str, Any],
    thermal_shape: tuple[int, int],
    thermal_maps: dict[str, Any],
) -> np.ndarray:
    full_h, full_w = thermal_shape
    out_h, out_w = thermal_maps["edges"].shape[:2]
    matrix = np.asarray(candidate["matrix"] if isinstance(candidate, dict) else candidate.matrix, dtype=np.float32).copy()
    matrix[0, :] *= out_w / full_w
    matrix[1, :] *= out_h / full_h
    return cv2.warpAffine(
        video_frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def draw_crop_rectangle(video_frame_bgr: np.ndarray, candidate: SpatialCandidate | dict[str, Any]) -> np.ndarray:
    frame = video_frame_bgr.copy()
    if isinstance(candidate, dict):
        crop_x = int(candidate["crop_x"])
        crop_y = int(candidate["crop_y"])
        crop_w = int(candidate["crop_w"])
        crop_h = int(candidate["crop_h"])
    else:
        crop_x = candidate.crop_x
        crop_y = candidate.crop_y
        crop_w = candidate.crop_w
        crop_h = candidate.crop_h
    cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (0, 255, 255), 8)
    return frame


def overlay_rgb(thermal_gray: np.ndarray, warped_video_bgr: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    thermal = heatmap_rgb(thermal_gray)
    video = cv2.cvtColor(warped_video_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(video, 1.0 - alpha, thermal, alpha, 0)


def draw_circle_rgb(
    rgb: np.ndarray,
    circle: dict[str, float] | None,
    color: tuple[int, int, int],
    *,
    thickness: int = 3,
) -> np.ndarray:
    out = np.asarray(rgb).copy()
    if not circle or circle.get("r", 0.0) <= 0:
        return out
    center = (int(round(circle["cx"])), int(round(circle["cy"])))
    radius = max(1, int(round(circle["r"])))
    cv2.circle(out, center, radius, color, thickness)
    cv2.circle(out, center, max(2, thickness + 1), color, -1)
    return out


def draw_pad_diagnostics_rgb(rgb: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    out = draw_circle_rgb(rgb, detail.get("thermal_circle"), (80, 255, 80), thickness=3)
    out = draw_circle_rgb(out, detail.get("mapped_video_circle"), (0, 255, 255), thickness=3)
    thermal_circle = detail.get("thermal_circle")
    mapped_circle = detail.get("mapped_video_circle")
    if thermal_circle and mapped_circle:
        p1 = (int(round(thermal_circle["cx"])), int(round(thermal_circle["cy"])))
        p2 = (int(round(mapped_circle["cx"])), int(round(mapped_circle["cy"])))
        cv2.line(out, p1, p2, (255, 60, 60), 2)
    return out


def draw_video_diagnostics_bgr(
    video_frame_bgr: np.ndarray,
    spatial: SpatialCandidate | dict[str, Any],
    detail: dict[str, Any],
) -> np.ndarray:
    out = draw_crop_rectangle(video_frame_bgr, spatial)
    video_circle = detail.get("video_circle")
    if video_circle and video_circle.get("r", 0.0) > 0:
        center = (int(round(video_circle["cx"])), int(round(video_circle["cy"])))
        radius = max(1, int(round(video_circle["r"])))
        cv2.circle(out, center, radius, (255, 255, 0), 5)
        cv2.circle(out, center, 7, (255, 255, 0), -1)
    return out


def draw_video_circle_only_bgr(video_frame_bgr: np.ndarray, circle: dict[str, Any] | None) -> np.ndarray:
    out = video_frame_bgr.copy()
    if circle and circle.get("r", 0.0) > 0:
        center = (int(round(circle["cx"])), int(round(circle["cy"])))
        radius = max(1, int(round(circle["r"])))
        color = (0, 255, 0) if not bool(circle.get("partial", False)) else (0, 200, 255)
        cv2.circle(out, center, radius, color, 5)
        cv2.circle(out, center, 7, color, -1)
    return out


def save_video_pad_detection_check(
    out_path: Path,
    reader: VideoReader,
    video_circle_track: dict[int, dict[str, Any] | None],
    candidate_track: dict[int, list[dict[str, Any]]],
    seed: dict[str, Any] | None,
    *,
    max_tiles: int = 40,
) -> None:
    all_items = sorted(video_circle_track.items())
    if not all_items:
        return
    index_rows = [
        {
            "frame_index": int(frame_idx),
            "raw_candidates": candidate_track.get(frame_idx, []),
            "circle": circle,
        }
        for frame_idx, circle in all_items
    ]
    items = all_items
    if len(items) > max_tiles:
        positions = np.linspace(0, len(items) - 1, max_tiles).round().astype(int)
        items = [items[int(pos)] for pos in positions]
    tiles: list[tuple[str, np.ndarray]] = []
    for frame_idx, circle in items:
        frame_bgr, actual_idx = reader.frame_at_index(frame_idx)
        label = f"frame {actual_idx}"
        if circle:
            label += (
                f" {circle.get('source', '')}"
                f" conf={float(circle.get('confidence', 0.0)):.2f}"
                f" partial={bool(circle.get('partial', False))}"
                f" smooth={bool(circle.get('smoothed', False))}"
            )
        else:
            label += " no-circle"
        tiles.append((label, cv2.cvtColor(draw_video_circle_only_bgr(frame_bgr, circle), cv2.COLOR_BGR2RGB)))
    make_contact_sheet(tiles, out_path, cols=4, tile_w=300, tile_h=230)
    save_json(
        out_path.with_suffix(".json"),
        {
            "version": 2,
            "seed": seed,
            "frame_count": len(index_rows),
            "frames": index_rows,
        },
    )
    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Video pad detection check</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;} img{max-width:100%;height:auto;border:1px solid #ddd;}</style>",
        "<h1>Video pad detection check</h1>",
        f"<img src='{out_path.name}'>",
    ]
    out_path.with_suffix(".html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def diagnostic_label(detail: dict[str, Any]) -> str:
    return (
        f"offset {detail['offset_s']:+.0f}s | {detail['tiff_name']} | "
        f"video {detail['video_time']} | score {detail['score']:.4f} | "
        f"geom {detail.get('pad_geometry_score', 0.0):.4f} | "
        f"rank {detail.get('pad_thermal_circle_rank', -1)} | "
        f"v {detail.get('video_circle_source', '')} {detail.get('video_circle_confidence', 0.0):.2f} | "
        f"c {detail.get('pad_center_error_ratio', 1.0):.3f} "
        f"r {detail.get('pad_radius_error_ratio', 1.0):.3f}"
    )


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)


def save_bgr(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Could not write image: {path}")


def make_contact_sheet(
    tiles: list[tuple[str, np.ndarray]],
    out_path: Path,
    *,
    cols: int,
    tile_w: int = 360,
    tile_h: int = 260,
) -> None:
    if not tiles:
        return
    label_h = 34
    margin = 12
    rows = int(math.ceil(len(tiles) / cols))
    sheet_w = cols * (tile_w + margin) + margin
    sheet_h = rows * (tile_h + label_h + margin) + margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, rgb) in enumerate(tiles):
        col = i % cols
        row = i // cols
        ox = margin + col * (tile_w + margin)
        oy = margin + row * (tile_h + label_h + margin)
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "#f2f2f2")
        tile.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
        draw.text((ox, oy), label[:80], fill="black")
        sheet.paste(tile, (ox, oy + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def numbered_circle_color(index: int) -> tuple[int, int, int]:
    palette = [
        (80, 255, 80),
        (0, 255, 255),
        (255, 170, 50),
        (255, 80, 220),
        (80, 180, 255),
    ]
    return palette[(index - 1) % len(palette)]


def label_bbox_for_origin(
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float = 0.9,
    thickness: int = 7,
    pad: int = 2,
) -> tuple[int, int, int, int]:
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    return (x - pad, y - text_h - pad, x + text_w + pad, y + baseline + pad)


def rectangle_overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def clamp_label_origin(
    text: str,
    desired_origin: tuple[float, float],
    image_shape: tuple[int, ...],
    *,
    font_scale: float = 0.9,
    thickness: int = 7,
    pad: int = 4,
) -> tuple[int, int, float]:
    image_h, image_w = image_shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = int(round(desired_origin[0]))
    y = int(round(desired_origin[1]))
    max_x = max(pad, image_w - text_w - pad)
    max_y = max(text_h + pad, image_h - baseline - pad)
    clamped_x = min(max(pad, x), max_x)
    clamped_y = min(max(text_h + pad, y), max_y)
    clamp_penalty = math.hypot(clamped_x - x, clamped_y - y)
    return clamped_x, clamped_y, clamp_penalty


def place_numbered_circle_labels(
    circles: list[dict[str, Any]],
    image_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[dict[str, Any]] = []
    angle_degrees = [-45, -90, 0, 45, -135, 135, 90, 180, -20, -70, 20, 70, -160, 160, 110, -110]

    for idx, circle in enumerate(circles, start=1):
        text = str(idx)
        center = (float(circle["cx"]), float(circle["cy"]))
        radius = max(1.0, float(circle["r"]))
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 7)
        best: dict[str, Any] | None = None
        best_score = float("inf")
        for tier in range(5):
            label_distance = radius + 18.0 + tier * 22.0
            for angle_index, angle_deg in enumerate(angle_degrees):
                angle = math.radians(angle_deg)
                label_cx = center[0] + math.cos(angle) * label_distance
                label_cy = center[1] + math.sin(angle) * label_distance
                desired_origin = (label_cx - text_w / 2.0, label_cy + text_h / 2.0)
                origin_x, origin_y, clamp_penalty = clamp_label_origin(text, desired_origin, image_shape)
                bbox = label_bbox_for_origin(text, (origin_x, origin_y))
                overlap = sum(rectangle_overlap_area(bbox, existing) for existing in occupied)
                score = overlap * 1000.0 + clamp_penalty * 25.0 + tier * 5.0 + angle_index * 0.02
                if score < best_score:
                    best_score = score
                    best = {
                        "text": text,
                        "origin": (origin_x, origin_y),
                        "bbox": bbox,
                        "center": (int(round(center[0])), int(round(center[1]))),
                        "radius": int(round(radius)),
                        "color": numbered_circle_color(idx),
                    }
                if overlap == 0 and clamp_penalty == 0:
                    break
            if best is not None and best_score < 1.0:
                break
        if best is None:
            fallback_origin = (int(round(center[0] + 6)), int(round(center[1] - 6)))
            bbox = label_bbox_for_origin(text, fallback_origin)
            best = {
                "text": text,
                "origin": fallback_origin,
                "bbox": bbox,
                "center": (int(round(center[0])), int(round(center[1]))),
                "radius": int(round(radius)),
                "color": numbered_circle_color(idx),
            }
        occupied.append(best["bbox"])
        placements.append(best)
    return placements


def draw_numbered_circles_rgb(rgb: np.ndarray, circles: list[dict[str, Any]]) -> np.ndarray:
    out = np.asarray(rgb).copy()
    for idx, circle in enumerate(circles, start=1):
        color = numbered_circle_color(idx)
        center = (int(round(circle["cx"])), int(round(circle["cy"])))
        radius = max(1, int(round(circle["r"])))
        cv2.circle(out, center, radius, color, 3)
        cv2.circle(out, center, 4, color, -1)
    placements = place_numbered_circle_labels(circles, out.shape)
    for placement in placements:
        bbox = placement["bbox"]
        label_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        circle_center = placement["center"]
        radius = max(1, int(placement["radius"]))
        color = placement["color"]
        dx = label_center[0] - circle_center[0]
        dy = label_center[1] - circle_center[1]
        distance = max(1.0, math.hypot(dx, dy))
        line_start = (
            int(round(circle_center[0] + dx / distance * min(radius, distance))),
            int(round(circle_center[1] + dy / distance * min(radius, distance))),
        )
        cv2.line(out, line_start, label_center, (0, 0, 0), 3)
        cv2.line(out, line_start, label_center, color, 1)
    for placement in placements:
        label_pos = placement["origin"]
        color = placement["color"]
        text = placement["text"]
        cv2.putText(out, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 7)
        cv2.putText(out, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
        cv2.putText(out, text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def circle_candidate_matches(a: dict[str, Any], b: dict[str, Any]) -> bool:
    center_dist = math.hypot(float(a["cx"]) - float(b["cx"]), float(a["cy"]) - float(b["cy"]))
    max_radius = max(float(a["r"]), float(b["r"]))
    radius_dist = abs(float(a["r"]) - float(b["r"]))
    return center_dist <= max(8.0, 0.08 * max_radius) and radius_dist <= max(6.0, 0.08 * max_radius)


def renumber_circle_candidates(candidates: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, circle in enumerate(candidates[:max_count], start=1):
        item = dict(circle)
        item["rank"] = rank
        out.append(item)
    return out


def postprocess_pad_circle_candidates(
    tiff_name: str,
    candidates: list[dict[str, Any]],
    *,
    max_count: int = 12,
) -> list[dict[str, Any]]:
    seeds = PAD_REVIEW_CONTINUITY_SEEDS.get(tiff_name)
    if not seeds:
        return renumber_circle_candidates(candidates, max_count)

    working = [dict(candidate) for candidate in candidates]
    if tiff_name == "20260226_023321.tiff":
        working = [
            candidate
            for candidate in working
            if 300.0 <= float(candidate.get("cx", 0.0)) <= 380.0
            and 160.0 <= float(candidate.get("cy", 0.0)) <= 240.0
        ]
    insert_counts: dict[int, int] = {}
    for seed in seeds:
        seed_circle = {key: seed[key] for key in ["cx", "cy", "r"]}
        seed_item = dict(seed)
        seed_item.pop("after_rank", None)
        working = [
            candidate
            for candidate in working
            if candidate.get("source") == "continuity_seed" or not circle_candidate_matches(candidate, seed_circle)
        ]
        after_rank = int(seed.get("after_rank", len(working)))
        base_insert_at = 0 if after_rank <= 0 else next(
            (idx + 1 for idx, candidate in enumerate(working) if int(candidate.get("rank", -1)) == after_rank),
            min(max(after_rank, 0), len(working)),
        )
        insert_at = min(base_insert_at + insert_counts.get(after_rank, 0), len(working))
        insert_counts[after_rank] = insert_counts.get(after_rank, 0) + 1
        working.insert(insert_at, seed_item)
    return renumber_circle_candidates(working, max_count)


def apply_pad_circle_review_defaults(template: dict[str, Any]) -> None:
    for entry in template.get("frames", []):
        tiff_name = str(entry.get("tiff_name", ""))
        if tiff_name in PAD_REVIEW_PENDING_NAMES:
            entry["selected_rank"] = None
            entry["circle"] = {"cx": None, "cy": None, "r": None}
            continue
        selected_rank = PAD_REVIEW_CONFIRMED_RANKS.get(tiff_name)
        if selected_rank is None:
            continue
        entry["selected_rank"] = selected_rank
        for candidate in entry.get("candidates", []):
            if int(candidate.get("rank", -1)) == selected_rank:
                entry["circle"] = {
                    "cx": candidate["cx"],
                    "cy": candidate["cy"],
                    "r": candidate["r"],
                    "score": candidate.get("score", 1.0),
                }
                break


def numbered_thermal_circles(thermal_gray: np.ndarray, max_count: int = 12) -> list[dict[str, Any]]:
    circles = []
    for rank, circle in enumerate(detect_thermal_circles(thermal_gray, max_count=max_count), start=1):
        item = dict(circle)
        item["rank"] = rank
        circles.append(item)
    return circles


def pad_circle_template_entry(frame: TiffFrame) -> tuple[dict[str, Any], np.ndarray]:
    thermal_gray = normalize_to_uint8(load_thermal_u16(frame.path))
    candidates = postprocess_pad_circle_candidates(
        frame.path.name,
        numbered_thermal_circles(thermal_gray, max_count=12),
        max_count=12,
    )
    entry = {
        "tiff_name": frame.path.name,
        "tiff_path": str(frame.path),
        "timestamp": frame.timestamp.isoformat(sep=" "),
        "selected_rank": None,
        "circle": {"cx": None, "cy": None, "r": None},
        "candidates": candidates,
    }
    preview = draw_numbered_circles_rgb(heatmap_rgb(thermal_gray), candidates)
    return entry, preview


def save_pad_circle_calibration(
    review_dir: Path,
    manual_json_path: Path,
    temporal_tiffs: list[TiffFrame],
) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    tiles: list[tuple[str, np.ndarray]] = []
    for frame in temporal_tiffs:
        entry, preview = pad_circle_template_entry(frame)
        entries.append(entry)
        tiles.append((frame.path.name, preview))
    sheet_path = review_dir / "thermal_candidates.jpg"
    make_contact_sheet(tiles, sheet_path, cols=1, tile_w=640, tile_h=480)

    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Pad circle calibration</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;} "
        "img{max-width:100%;height:auto;border:1px solid #ddd;} code{background:#f3f3f3;padding:2px 4px;}</style>",
        "<h1>TIFF pad circle calibration</h1>",
        f"<p>编辑 <code>{manual_json_path.name}</code>：每张 TIFF 填 <code>selected_rank</code> 或 <code>circle.cx/cy/r</code>。</p>",
        "<img src='thermal_candidates.jpg'>",
    ]
    (review_dir / "thermal_candidates.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")

    template = {
        "version": 1,
        "instructions": "For each TIFF, set selected_rank to the H-pad outer circle number shown in review/pad_circle_calibration/thermal_candidates.jpg, or fill circle.cx/cy/r directly.",
        "frames": entries,
    }
    if manual_json_path.exists():
        old_data = json.loads(manual_json_path.read_text())
        old_entries = old_data.get("frames", old_data if isinstance(old_data, list) else [])
        old_by_name = {str(entry.get("tiff_name", "")): entry for entry in old_entries}
        for entry in template["frames"]:
            old_entry = old_by_name.get(entry["tiff_name"])
            if not old_entry:
                continue
            if entry["tiff_name"] in PAD_REVIEW_PENDING_NAMES or entry["tiff_name"] in PAD_REVIEW_CONFIRMED_RANKS:
                continue
            old_circle = old_entry.get("circle") or {}
            if all(old_circle.get(key) is not None for key in ["cx", "cy", "r"]):
                entry["selected_rank"] = old_entry.get("selected_rank")
                entry["circle"] = old_circle
            elif old_entry.get("selected_rank") is not None:
                resolved = manual_circle_from_entry(old_entry)
                entry["selected_rank"] = old_entry["selected_rank"]
                entry["circle"] = {
                    "cx": resolved["cx"],
                    "cy": resolved["cy"],
                    "r": resolved["r"],
                    "score": resolved.get("score", 1.0),
                }
    apply_pad_circle_review_defaults(template)
    save_json(manual_json_path, template)
    return sheet_path


def manual_circle_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    circle_data = entry.get("circle") or {}
    if all(circle_data.get(key) is not None for key in ["cx", "cy", "r"]):
        selected_rank = int(entry.get("selected_rank") or -1)
        return {
            "cx": float(circle_data["cx"]),
            "cy": float(circle_data["cy"]),
            "r": float(circle_data["r"]),
            "score": float(circle_data.get("score", 1.0)),
            "selected_rank": selected_rank,
            "manual_circle_source": "manual_coordinates",
        }
    selected_rank = entry.get("selected_rank")
    if selected_rank is None:
        raise ValueError(f"Missing selected_rank or circle coordinates for {entry.get('tiff_name')}")
    selected_rank = int(selected_rank)
    for candidate in entry.get("candidates", []):
        if int(candidate.get("rank", -1)) == selected_rank:
            circle = dict(candidate)
            circle["selected_rank"] = selected_rank
            circle["manual_circle_source"] = "selected_rank"
            return circle
    raise ValueError(f"selected_rank={selected_rank} is not in candidates for {entry.get('tiff_name')}")


def load_pad_circles(path: Path, temporal_tiffs: list[TiffFrame]) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    entries = data.get("frames", data if isinstance(data, list) else [])
    by_name = {str(entry.get("tiff_name", "")): entry for entry in entries}
    by_path = {str(entry.get("tiff_path", "")): entry for entry in entries}
    result: dict[str, dict[str, Any]] = {}
    for frame in temporal_tiffs:
        entry = by_name.get(frame.path.name) or by_path.get(str(frame.path))
        if not entry:
            raise ValueError(f"Manual pad circle JSON has no entry for {frame.path.name}")
        circle = manual_circle_from_entry(entry)
        circle["tiff_name"] = frame.path.name
        circle["tiff_path"] = str(frame.path)
        circle["thermal_circle_confirmed"] = True
        result[frame.path.name] = circle
    return result


def auto_pad_circles(temporal_tiffs: list[TiffFrame]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for frame in temporal_tiffs:
        thermal_gray = normalize_to_uint8(load_thermal_u16(frame.path))
        candidates = numbered_thermal_circles(thermal_gray, max_count=1)
        if not candidates:
            raise RuntimeError(f"Could not auto-detect a TIFF pad circle for {frame.path.name}")
        circle = dict(candidates[0])
        circle["selected_rank"] = int(circle.get("rank", 1))
        circle["manual_circle_source"] = "auto_pad_circles"
        circle["thermal_circle_confirmed"] = True
        circle["tiff_name"] = frame.path.name
        circle["tiff_path"] = str(frame.path)
        result[frame.path.name] = circle
    return result


def anchor_frame_indices(
    anchor_video_s: float,
    fps: float,
    frame_count: int,
    window_s: float,
) -> list[int]:
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if window_s <= 0:
        raise ValueError("--anchor-frame-window-s must be positive")
    count = int(round(window_s * fps))
    if count <= 0:
        raise ValueError("Anchor frame window is too small for the video FPS")
    if frame_count and count > frame_count:
        raise ValueError(
            f"Anchor frame window needs {count} frames, but the video only has {frame_count}"
        )
    anchor_idx = int(round(anchor_video_s * fps))
    if frame_count:
        anchor_idx = min(max(anchor_idx, 0), frame_count - 1)
    else:
        anchor_idx = max(anchor_idx, 0)
    start = anchor_idx - count // 2
    if frame_count:
        start = min(max(start, 0), frame_count - count)
    else:
        start = max(start, 0)
    return list(range(start, start + count))


def temporal_neighbor_offsets(neighbor_count: int) -> list[int]:
    if neighbor_count < 0:
        raise ValueError("--temporal-neighbor-count must be non-negative")
    return list(range(-neighbor_count, neighbor_count + 1))


def tiff_neighbor_window(
    frames: list[TiffFrame],
    center_frame: TiffFrame,
    neighbor_count: int,
) -> list[TiffFrame]:
    offsets = temporal_neighbor_offsets(neighbor_count)
    center_idx = next(
        (
            idx
            for idx, frame in enumerate(frames)
            if frame.path == center_frame.path and frame.timestamp == center_frame.timestamp
        ),
        -1,
    )
    if center_idx < 0:
        raise ValueError(f"Center TIFF is not in the selected frame range: {center_frame.path}")
    start = center_idx - neighbor_count
    end = center_idx + neighbor_count + 1
    if start < 0 or end > len(frames):
        raise RuntimeError(
            f"Need {len(offsets)} TIFFs around {center_frame.path.name}, "
            f"but the selected range only has {center_idx} before and {len(frames) - center_idx - 1} after"
        )
    return frames[start:end]


def build_anchor_context(
    effective_frames: list[TiffFrame],
    initial_model: TimeModel,
    anchor_video_s: float,
    neighbor_count: int,
    *,
    name: str,
) -> AnchorContext:
    anchor_tiff_target = initial_model.video_to_tiff_time(anchor_video_s)
    center_tiff = nearest_tiff(effective_frames, anchor_tiff_target)
    temporal_tiffs = tiff_neighbor_window(effective_frames, center_tiff, neighbor_count)
    return AnchorContext(
        name=name,
        anchor_video_s=anchor_video_s,
        anchor_tiff_target=anchor_tiff_target,
        center_tiff=center_tiff,
        temporal_tiffs=temporal_tiffs,
    )


def unique_tiff_frames(frames: list[TiffFrame]) -> list[TiffFrame]:
    seen: set[tuple[str, datetime]] = set()
    result: list[TiffFrame] = []
    for frame in frames:
        key = (str(frame.path), frame.timestamp)
        if key in seen:
            continue
        seen.add(key)
        result.append(frame)
    result.sort(key=lambda item: item.timestamp)
    return result


def combined_anchor_calibration_tiffs(anchor_contexts: list[AnchorContext]) -> list[TiffFrame]:
    frames: list[TiffFrame] = []
    for context in anchor_contexts:
        frames.extend(context.temporal_tiffs)
    return unique_tiff_frames(frames)


def split_video_time_between_anchors(primary: AnchorContext, secondary: AnchorContext) -> float:
    if secondary.anchor_video_s <= primary.anchor_video_s:
        raise ValueError("--second-anchor-video must be later than --anchor-video")
    return 0.5 * (primary.anchor_video_s + secondary.anchor_video_s)


def build_anchor_split_report(
    initial_model: TimeModel,
    primary: AnchorContext,
    secondary: AnchorContext,
) -> dict[str, Any]:
    split_video_s = split_video_time_between_anchors(primary, secondary)
    split_tiff_time = initial_model.video_to_tiff_time(split_video_s)
    return {
        "mode": "anchor_video_midpoint",
        "primary_anchor_video_time_s": primary.anchor_video_s,
        "primary_anchor_video_time": format_video_time(primary.anchor_video_s),
        "secondary_anchor_video_time_s": secondary.anchor_video_s,
        "secondary_anchor_video_time": format_video_time(secondary.anchor_video_s),
        "split_video_time_s": split_video_s,
        "split_video_time": format_video_time(split_video_s),
        "split_tiff_time": split_tiff_time,
    }


def choose_segment_strategy(
    frame_timestamp: datetime,
    primary: SegmentExportStrategy,
    secondary: SegmentExportStrategy | None,
    split_tiff_time: datetime | None,
) -> SegmentExportStrategy:
    if secondary is None or split_tiff_time is None:
        return primary
    if frame_timestamp < split_tiff_time:
        return primary
    return secondary


def score_column_name(offset: int) -> str:
    prefix = "m" if offset < 0 else "p"
    return f"score_tiff_{prefix}{abs(offset)}"


def aggregate_window_scores(scores: list[float], offsets: list[int], method: str) -> float:
    if not scores:
        return 0.0
    if method == "mean":
        return float(np.mean(scores))
    if method == "median":
        return float(np.median(scores))
    if method == "center-weighted":
        max_offset = max(abs(offset) for offset in offsets) if offsets else 0
        weights = np.asarray([max_offset + 1 - abs(offset) for offset in offsets], dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
        return float(np.average(values, weights=weights))
    raise ValueError(f"Unsupported window score aggregation: {method}")


def window_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        1.0 if row.get("geometry_hard_pass", True) else 0.0,
        float(row["aggregate_score"]),
        float(row["center_score"]),
        float(row.get("min_window_score", row["aggregate_score"])),
        -abs(float(row["distance_from_anchor_s"])),
    )


def score_frame_pair(
    thermal_gray: np.ndarray,
    thermal_maps: dict[str, Any],
    video_frame_bgr: np.ndarray,
    spatial: SpatialCandidate | dict[str, Any],
    video_circle: dict[str, float] | None = None,
) -> dict[str, Any]:
    warped = warp_video_to_similarity_maps(
        video_frame_bgr,
        spatial,
        thermal_gray.shape[:2],
        thermal_maps,
    )
    video_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thermal_edges = thermal_maps["edges"]
    if thermal_edges.shape != video_gray.shape:
        video_gray = cv2.resize(video_gray, (thermal_edges.shape[1], thermal_edges.shape[0]))
    video_edges = edge_image(video_gray)
    video_gradient = gradient_magnitude(video_gray)
    edge_score = cosine_similarity(thermal_edges, video_edges)
    gradient_score = cosine_similarity(thermal_maps["gradient"], video_gradient)
    full_visual_score = float(0.75 * edge_score + 0.25 * gradient_score)
    full_sim = {
        "score": full_visual_score,
        "edge_score": edge_score,
        "gradient_score": gradient_score,
        "thermal_edge_density": float(thermal_maps["edge_density"]),
        "video_edge_density": edge_density(video_edges),
    }
    confirmed_thermal_circle = thermal_maps.get("confirmed_pad_circle_full")
    thermal_circles = thermal_maps.get("pad_circles_full")
    if thermal_circles is None:
        thermal_circles = detect_thermal_circles(thermal_gray)
    if video_circle is None:
        video_circle = detect_video_landing_pad(video_frame_bgr)
    if isinstance(spatial, dict):
        crop_x = int(spatial["crop_x"])
        crop_y = int(spatial["crop_y"])
        crop_w = int(spatial["crop_w"])
        crop_h = int(spatial["crop_h"])
    else:
        crop_x = spatial.crop_x
        crop_y = spatial.crop_y
        crop_w = spatial.crop_w
        crop_h = spatial.crop_h
    if confirmed_thermal_circle:
        geometry = pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            thermal_gray.shape[1],
            thermal_gray.shape[0],
            confirmed_thermal_circle,
            video_circle,
        )
        geometry["pad_thermal_circle_rank"] = int(confirmed_thermal_circle.get("selected_rank", -1))
    else:
        geometry = best_pad_geometry_metrics(
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            thermal_gray.shape[1],
            thermal_gray.shape[0],
            thermal_circles,
            video_circle,
        )
    thermal_circle = geometry["thermal_circle"]
    source_shape = thermal_maps.get("source_shape", thermal_edges.shape[:2])
    scaled_circle = scale_circle_to_shape(thermal_circle, source_shape, thermal_edges.shape[:2])
    mask = circle_fill_mask(thermal_edges.shape[:2], scaled_circle)
    masked_edge_score = masked_cosine_similarity(thermal_edges, video_edges, mask)
    masked_gradient_score = masked_cosine_similarity(thermal_maps["gradient"], video_gradient, mask)
    masked_visual_score = float(0.75 * masked_edge_score + 0.25 * masked_gradient_score)
    masked_sim = {
        "masked_visual_score": masked_visual_score,
        "masked_edge_score": masked_edge_score,
        "masked_gradient_score": masked_gradient_score,
    }
    score = composite_pad_score(
        pad_geometry_score=float(geometry["pad_geometry_score"]),
        masked_visual_score=float(masked_sim["masked_visual_score"]),
        full_visual_score=float(full_sim["score"]),
        hard_pass=bool(geometry["pad_geometry_hard_pass"]),
    )
    return {
        **full_sim,
        **masked_sim,
        "score": score,
        "visual_score": full_sim["score"],
        "full_visual_score": full_sim["score"],
        "masked_visual_score": masked_sim["masked_visual_score"],
        "pad_geometry_score": geometry["pad_geometry_score"],
        "pad_center_error_px": geometry["pad_center_error_px"],
        "pad_center_error_ratio": geometry["pad_center_error_ratio"],
        "pad_radius_error_ratio": geometry["pad_radius_error_ratio"],
        "pad_radius_anisotropy_ratio": geometry["pad_radius_anisotropy_ratio"],
        "pad_ring_iou": geometry["pad_ring_iou"],
        "pad_geometry_hard_pass": geometry["pad_geometry_hard_pass"],
        "pad_thermal_circle_rank": geometry.get("pad_thermal_circle_rank", -1),
        "thermal_circle_confirmed": bool(thermal_maps.get("thermal_circle_confirmed", False)),
        "manual_circle_source": thermal_maps.get("manual_circle_source", ""),
        "selected_rank": thermal_maps.get("selected_rank", -1),
        "mapped_video_circle": geometry["mapped_video_circle"],
        "thermal_circle": geometry["thermal_circle"],
        "video_circle": geometry["video_circle"],
        "warped_video_bgr": warped,
    }


def score_temporal_window(
    thermal_items: list[tuple[int, TiffFrame, np.ndarray, dict[str, Any]]],
    candidate_frame_idx: int,
    spatial: SpatialCandidate,
    reader: VideoReader,
    *,
    aggregation: str,
    video_circle_track: dict[int, dict[str, Any] | None] | None = None,
) -> tuple[float, list[dict[str, Any]]] | None:
    details: list[dict[str, Any]] = []
    scores: list[float] = []
    for offset, tiff_frame, thermal_gray, thermal_maps in thermal_items:
        video_frame_idx = candidate_frame_idx + int(round(offset * reader.fps))
        if video_frame_idx < 0 or (reader.frame_count and video_frame_idx >= reader.frame_count):
            return None
        video_frame_bgr, actual_idx = reader.frame_at_index(video_frame_idx)
        video_circle = video_circle_track.get(actual_idx) if video_circle_track else None
        sim = score_frame_pair(thermal_gray, thermal_maps, video_frame_bgr, spatial, video_circle)
        score = sim["score"]
        scores.append(score)
        detail_sim = {key: value for key, value in sim.items() if key != "warped_video_bgr"}
        detail_video_circle = sim.get("video_circle") or {}
        details.append(
            {
                "offset_s": float(offset),
                "tiff_path": str(tiff_frame.path),
                "tiff_name": tiff_frame.path.name,
                "tiff_time_iso": tiff_frame.timestamp.isoformat(sep=" "),
                "video_frame_index": actual_idx,
                "video_time_s": actual_idx / reader.fps,
                "video_time": format_video_time(actual_idx / reader.fps),
                "visual_score": sim["full_visual_score"],
                "circle_alignment": (
                    float(spatial.get("circle_alignment", 0.0))
                    if isinstance(spatial, dict)
                    else spatial.circle_alignment
                ),
                "video_circle_source": detail_video_circle.get("source", ""),
                "video_circle_confidence": float(detail_video_circle.get("confidence", 0.0)) if detail_video_circle else 0.0,
                "video_circle_partial": bool(detail_video_circle.get("partial", False)) if detail_video_circle else False,
                "video_circle_smoothed": bool(detail_video_circle.get("smoothed", False)) if detail_video_circle else False,
                **detail_sim,
            }
        )
    return aggregate_window_scores(scores, [offset for offset, _, _, _ in thermal_items], aggregation), details


def diversify_ranked_spatial_candidates(
    ranked_candidates: list[tuple[int, SpatialCandidate]],
    limit: int,
) -> list[tuple[int, SpatialCandidate]]:
    if len(ranked_candidates) <= limit:
        return ranked_candidates
    groups: dict[int, list[tuple[int, SpatialCandidate]]] = {}
    for item in ranked_candidates:
        _, candidate = item
        groups.setdefault(candidate.pad_thermal_circle_rank, []).append(item)
    group_keys = sorted(
        groups,
        key=lambda key: groups[key][0][1].score,
        reverse=True,
    )
    selected: list[tuple[int, SpatialCandidate]] = []
    positions = {key: 0 for key in group_keys}
    while len(selected) < limit:
        added = False
        for key in group_keys:
            pos = positions[key]
            if pos >= len(groups[key]):
                continue
            selected.append(groups[key][pos])
            positions[key] = pos + 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    selected.sort(key=lambda item: item[1].score, reverse=True)
    return selected


def score_spatial_strategy_pool(
    thermal_items: list[tuple[int, TiffFrame, np.ndarray, dict[str, Any]]],
    candidate_frame_idx: int,
    spatial_candidates: list[SpatialCandidate],
    reader: VideoReader,
    *,
    aggregation: str,
    search_order: int,
    anchor_video_s: float,
    min_circle_alignment: float,
    video_circle_track: dict[int, dict[str, Any] | None],
) -> dict[str, Any] | None:
    if not spatial_candidates:
        return None

    ranked_candidates = list(enumerate(spatial_candidates, start=1))
    eligible_candidates = [
        (rank, candidate)
        for rank, candidate in ranked_candidates
        if candidate.circle_alignment >= min_circle_alignment
    ]
    used_circle_filter = True
    if not eligible_candidates:
        eligible_candidates = ranked_candidates
        used_circle_filter = False
    eligible_candidates = diversify_ranked_spatial_candidates(
        eligible_candidates,
        MAX_WINDOW_CROP_STRATEGIES,
    )

    frame_items: list[tuple[int, TiffFrame, np.ndarray, dict[str, Any], np.ndarray, int, dict[str, float] | None]] = []
    for offset, tiff_frame, thermal_gray, thermal_maps in thermal_items:
        video_frame_idx = candidate_frame_idx + int(round(offset * reader.fps))
        if video_frame_idx < 0 or (reader.frame_count and video_frame_idx >= reader.frame_count):
            return None
        video_frame_bgr, actual_idx = reader.frame_at_index(video_frame_idx)
        video_circle = video_circle_track.get(actual_idx)
        frame_items.append((offset, tiff_frame, thermal_gray, thermal_maps, video_frame_bgr, actual_idx, video_circle))

    best_row: dict[str, Any] | None = None
    for strategy_rank, spatial in eligible_candidates:
        details: list[dict[str, Any]] = []
        scores: list[float] = []
        for offset, tiff_frame, thermal_gray, thermal_maps, video_frame_bgr, actual_idx, video_circle in frame_items:
            sim = score_frame_pair(thermal_gray, thermal_maps, video_frame_bgr, spatial, video_circle)
            score = sim["score"]
            scores.append(score)
            detail_sim = {key: value for key, value in sim.items() if key != "warped_video_bgr"}
            detail_video_circle = sim.get("video_circle") or {}
            details.append(
                {
                    "offset_s": float(offset),
                    "tiff_path": str(tiff_frame.path),
                    "tiff_name": tiff_frame.path.name,
                    "tiff_time_iso": tiff_frame.timestamp.isoformat(sep=" "),
                    "video_frame_index": actual_idx,
                    "video_time_s": actual_idx / reader.fps,
                    "video_time": format_video_time(actual_idx / reader.fps),
                    "visual_score": sim["full_visual_score"],
                    "circle_alignment": spatial.circle_alignment,
                    "video_circle_source": detail_video_circle.get("source", ""),
                    "video_circle_confidence": float(detail_video_circle.get("confidence", 0.0)) if detail_video_circle else 0.0,
                    "video_circle_partial": bool(detail_video_circle.get("partial", False)) if detail_video_circle else False,
                    "video_circle_smoothed": bool(detail_video_circle.get("smoothed", False)) if detail_video_circle else False,
                    **detail_sim,
                }
            )
        aggregate_score = aggregate_window_scores(
            scores,
            [offset for offset, _, _, _, _, _, _ in frame_items],
            aggregation,
        )
        center_detail = min(details, key=lambda item: abs(item["offset_s"]))
        pad_geometry_scores = [float(detail["pad_geometry_score"]) for detail in details]
        masked_visual_scores = [float(detail["masked_visual_score"]) for detail in details]
        full_visual_scores = [float(detail["full_visual_score"]) for detail in details]
        spatial_dict = candidate_to_dict(spatial)
        row = {
            "search_order": search_order,
            "candidate_video_frame_index": candidate_frame_idx,
            "candidate_video_time_s": candidate_frame_idx / reader.fps,
            "candidate_video_time": format_video_time(candidate_frame_idx / reader.fps),
            "distance_from_anchor_s": candidate_frame_idx / reader.fps - anchor_video_s,
            "aggregate_score": aggregate_score,
            "center_score": center_detail["score"],
            "min_window_score": min(scores),
            "pad_geometry_mean": float(np.mean(pad_geometry_scores)),
            "masked_visual_mean": float(np.mean(masked_visual_scores)),
            "full_visual_mean": float(np.mean(full_visual_scores)),
            "center_pad_geometry_score": center_detail["pad_geometry_score"],
            "center_pad_center_error_px": center_detail["pad_center_error_px"],
            "center_pad_center_error_ratio": center_detail["pad_center_error_ratio"],
            "center_pad_radius_error_ratio": center_detail["pad_radius_error_ratio"],
            "center_pad_ring_iou": center_detail["pad_ring_iou"],
            "center_pad_thermal_circle_rank": center_detail.get("pad_thermal_circle_rank", -1),
            "video_circle_source": center_detail.get("video_circle_source", ""),
            "video_circle_confidence": center_detail.get("video_circle_confidence", 0.0),
            "video_circle_partial": center_detail.get("video_circle_partial", False),
            "video_circle_smoothed": center_detail.get("video_circle_smoothed", False),
            "thermal_circle_confirmed": center_detail.get("thermal_circle_confirmed", False),
            "manual_circle_source": center_detail.get("manual_circle_source", ""),
            "selected_rank": center_detail.get("selected_rank", -1),
            "geometry_hard_pass": bool(center_detail["pad_geometry_hard_pass"]),
            "aggregation": aggregation,
            "strategy_rank": strategy_rank,
            "evaluated_strategy_count": len(eligible_candidates),
            "candidate_strategy_count": len(spatial_candidates),
            "circle_filter_applied": used_circle_filter,
            "min_circle_alignment": min_circle_alignment,
            "transform_type": spatial.transform_type,
            "spatial_score": spatial.score,
            "crop_x": spatial.crop_x,
            "crop_y": spatial.crop_y,
            "crop_w": spatial.crop_w,
            "crop_h": spatial.crop_h,
            "thermal_edge_density": center_detail["thermal_edge_density"],
            "video_edge_density": center_detail["video_edge_density"],
            "base_similarity": spatial.base_similarity,
            "circle_alignment": spatial.circle_alignment,
            "pad_center_error_px": spatial.pad_center_error_px,
            "pad_center_error_ratio": spatial.pad_center_error_ratio,
            "pad_radius_error_ratio": spatial.pad_radius_error_ratio,
            "pad_ring_iou": spatial.pad_ring_iou,
            "pad_geometry_score": spatial.pad_geometry_score,
            "pad_geometry_hard_pass": spatial.pad_geometry_hard_pass,
            "pad_thermal_circle_rank": spatial.pad_thermal_circle_rank,
            "keypoint_matches": spatial.keypoint_matches,
            "keypoint_inliers": spatial.keypoint_inliers,
            "source_tiff": spatial.source_tiff,
            "spatial_candidate": spatial_dict,
            "score_details": details,
        }
        for detail in details:
            row[score_column_name(int(detail["offset_s"]))] = detail["score"]
        if best_row is None or window_candidate_sort_key(row) > window_candidate_sort_key(best_row):
            best_row = row
    return best_row


def estimate_anchor_window_candidates(
    reader: VideoReader,
    center_tiff: TiffFrame,
    temporal_tiffs: list[TiffFrame],
    anchor_video_s: float,
    *,
    anchor_frame_window_s: float,
    score_aggregation: str,
    crop_scale_fractions: list[float],
    roi_step_fraction: float,
    refine_features: bool,
    refine_top_k: int,
    max_crop_strategies: int | None,
    min_circle_alignment: float,
    confirmed_pad_circles: dict[str, dict[str, Any]],
    video_pad_track_window: int,
    video_pad_seed: dict[str, Any] | None = None,
    video_detection_review_path: Path | None = None,
) -> list[dict[str, Any]]:
    center_thermal_gray = normalize_to_uint8(load_thermal_u16(center_tiff.path))
    center_thermal_maps = thermal_similarity_maps(center_thermal_gray)
    center_pad_circle = confirmed_pad_circles[center_tiff.path.name]
    center_thermal_circles = [center_pad_circle]
    offsets = temporal_neighbor_offsets((len(temporal_tiffs) - 1) // 2)
    thermal_items = [
        (
            offset,
            frame,
            thermal_gray,
            thermal_matching_maps(
                thermal_gray,
                192,
                confirmed_pad_circles[frame.path.name],
            ),
        )
        for offset, frame in zip(offsets, temporal_tiffs, strict=True)
        for thermal_gray in [normalize_to_uint8(load_thermal_u16(frame.path))]
    ]
    frame_indices = anchor_frame_indices(
        anchor_video_s,
        reader.fps,
        reader.frame_count,
        anchor_frame_window_s,
    )
    scoring_frame_indices = window_scoring_frame_indices(
        frame_indices,
        reader.fps,
        reader.frame_count,
        offsets,
    )
    video_circle_track, video_candidate_track = build_video_pad_circle_track(
        reader,
        scoring_frame_indices,
        smoothing_window=video_pad_track_window,
        seed=video_pad_seed,
    )
    if video_detection_review_path is not None:
        save_video_pad_detection_check(
            video_detection_review_path,
            reader,
            video_circle_track,
            video_candidate_track,
            video_pad_seed,
        )
    rows: list[dict[str, Any]] = []
    for search_order, frame_idx in enumerate(frame_indices, start=1):
        video_frame_bgr, actual_idx = reader.frame_at_index(frame_idx)
        video_circle = video_circle_track.get(actual_idx)
        spatial_candidates = estimate_spatial_candidate_pool(
            center_thermal_gray,
            video_frame_bgr,
            source_tiff=str(center_tiff.path),
            crop_scale_fractions=crop_scale_fractions,
            roi_step_fraction=roi_step_fraction,
            max_candidates=max_crop_strategies,
            thermal_circles_override=center_thermal_circles,
            thermal_edge_density_override=float(center_thermal_maps["edge_density"]),
            video_circle_override=video_circle,
        )
        if not spatial_candidates:
            continue
        row = score_spatial_strategy_pool(
            thermal_items,
            actual_idx,
            spatial_candidates,
            reader,
            aggregation=score_aggregation,
            search_order=search_order,
            anchor_video_s=anchor_video_s,
            min_circle_alignment=min_circle_alignment,
            video_circle_track=video_circle_track,
        )
        if row is not None:
            rows.append(row)

    rows.sort(key=window_candidate_sort_key, reverse=True)
    if refine_features:
        refined_limit = min(len(rows), max(1, refine_top_k))
        for idx, row in enumerate(rows[:refined_limit]):
            video_frame_bgr, actual_idx = reader.frame_at_index(int(row["candidate_video_frame_index"]))
            selected = spatial_candidate_from_dict(row["spatial_candidate"])
            refined_candidate = refine_spatial_candidate(
                center_thermal_gray,
                video_frame_bgr,
                selected,
            )
            refined_row = make_window_candidate_row(
                search_order=int(row["search_order"]),
                candidate_frame_idx=actual_idx,
                anchor_video_s=anchor_video_s,
                spatial=refined_candidate,
                thermal_items=thermal_items,
                reader=reader,
                score_aggregation=score_aggregation,
                video_circle_track=video_circle_track,
            )
            if refined_row is not None and refined_row["aggregate_score"] >= row["aggregate_score"]:
                rows[idx] = refined_row
        rows.sort(key=window_candidate_sort_key, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def selected_spatial_from_window_row(row: dict[str, Any]) -> dict[str, Any]:
    spatial = dict(row["spatial_candidate"])
    spatial["score"] = row["aggregate_score"]
    spatial["window_aggregate_score"] = row["aggregate_score"]
    spatial["window_center_score"] = row["center_score"]
    spatial["window_aggregation"] = row["aggregation"]
    spatial["selected_video_frame_index"] = row["candidate_video_frame_index"]
    spatial["selected_video_time_s"] = row["candidate_video_time_s"]
    spatial["selected_video_time"] = row["candidate_video_time"]
    return spatial


def make_window_candidate_row(
    *,
    search_order: int,
    candidate_frame_idx: int,
    anchor_video_s: float,
    spatial: SpatialCandidate,
    thermal_items: list[tuple[int, TiffFrame, np.ndarray, dict[str, Any]]],
    reader: VideoReader,
    score_aggregation: str,
    video_circle_track: dict[int, dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    scored = score_temporal_window(
        thermal_items,
        candidate_frame_idx,
        spatial,
        reader,
        aggregation=score_aggregation,
        video_circle_track=video_circle_track,
    )
    if scored is None:
        return None
    aggregate_score, score_details = scored
    center_detail = min(score_details, key=lambda item: abs(item["offset_s"]))
    scores = [float(detail["score"]) for detail in score_details]
    pad_geometry_scores = [float(detail["pad_geometry_score"]) for detail in score_details]
    masked_visual_scores = [float(detail["masked_visual_score"]) for detail in score_details]
    full_visual_scores = [float(detail["full_visual_score"]) for detail in score_details]
    spatial_dict = candidate_to_dict(spatial)
    row = {
        "search_order": search_order,
        "candidate_video_frame_index": candidate_frame_idx,
        "candidate_video_time_s": candidate_frame_idx / reader.fps,
        "candidate_video_time": format_video_time(candidate_frame_idx / reader.fps),
        "distance_from_anchor_s": candidate_frame_idx / reader.fps - anchor_video_s,
        "aggregate_score": aggregate_score,
        "center_score": center_detail["score"],
        "min_window_score": min(scores),
        "pad_geometry_mean": float(np.mean(pad_geometry_scores)),
        "masked_visual_mean": float(np.mean(masked_visual_scores)),
        "full_visual_mean": float(np.mean(full_visual_scores)),
        "center_pad_geometry_score": center_detail["pad_geometry_score"],
        "center_pad_center_error_px": center_detail["pad_center_error_px"],
        "center_pad_center_error_ratio": center_detail["pad_center_error_ratio"],
        "center_pad_radius_error_ratio": center_detail["pad_radius_error_ratio"],
        "center_pad_ring_iou": center_detail["pad_ring_iou"],
        "center_pad_thermal_circle_rank": center_detail.get("pad_thermal_circle_rank", -1),
        "video_circle_source": center_detail.get("video_circle_source", ""),
        "video_circle_confidence": center_detail.get("video_circle_confidence", 0.0),
        "video_circle_partial": center_detail.get("video_circle_partial", False),
        "video_circle_smoothed": center_detail.get("video_circle_smoothed", False),
        "thermal_circle_confirmed": center_detail.get("thermal_circle_confirmed", False),
        "manual_circle_source": center_detail.get("manual_circle_source", ""),
        "selected_rank": center_detail.get("selected_rank", -1),
        "geometry_hard_pass": bool(center_detail["pad_geometry_hard_pass"]),
        "aggregation": score_aggregation,
        "transform_type": spatial.transform_type,
        "spatial_score": spatial.score,
        "crop_x": spatial.crop_x,
        "crop_y": spatial.crop_y,
        "crop_w": spatial.crop_w,
        "crop_h": spatial.crop_h,
        "thermal_edge_density": center_detail["thermal_edge_density"],
        "video_edge_density": center_detail["video_edge_density"],
        "base_similarity": spatial.base_similarity,
        "circle_alignment": spatial.circle_alignment,
        "pad_center_error_px": spatial.pad_center_error_px,
        "pad_center_error_ratio": spatial.pad_center_error_ratio,
        "pad_radius_error_ratio": spatial.pad_radius_error_ratio,
        "pad_ring_iou": spatial.pad_ring_iou,
        "pad_geometry_score": spatial.pad_geometry_score,
        "pad_geometry_hard_pass": spatial.pad_geometry_hard_pass,
        "pad_thermal_circle_rank": spatial.pad_thermal_circle_rank,
        "keypoint_matches": spatial.keypoint_matches,
        "keypoint_inliers": spatial.keypoint_inliers,
        "source_tiff": spatial.source_tiff,
        "spatial_candidate": spatial_dict,
        "score_details": score_details,
    }
    for detail in score_details:
        row[score_column_name(int(detail["offset_s"]))] = detail["score"]
    return row


def compact_window_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"spatial_candidate", "score_details"}
    } | {
        "spatial_candidate": row["spatial_candidate"],
        "score_details": row["score_details"],
    }


def save_anchor_window_review(
    out_path: Path,
    reader: VideoReader,
    center_thermal_gray: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> None:
    tiles: list[tuple[str, np.ndarray]] = []
    thermal_rgb = heatmap_rgb(center_thermal_gray)
    for row in rows[:top_k]:
        spatial = row["spatial_candidate"]
        center_detail = min(row.get("score_details", []), key=lambda item: abs(item["offset_s"]), default={})
        frame_bgr, _ = reader.frame_at_index(int(row["candidate_video_frame_index"]))
        rect = draw_video_diagnostics_bgr(frame_bgr, spatial, center_detail)
        warped = warp_video_to_thermal(
            frame_bgr,
            spatial,
            (center_thermal_gray.shape[1], center_thermal_gray.shape[0]),
        )
        label = (
            f"#{row['rank']} {row['candidate_video_time']} "
            f"mean={row['aggregate_score']:.4f} center={row['center_score']:.4f} "
            f"geom={row.get('center_pad_geometry_score', 0.0):.4f}"
        )
        tiles.extend(
            [
                (f"{label} crop", cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)),
                (f"{label} warped", draw_pad_diagnostics_rgb(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), center_detail)),
                (f"{label} overlay", draw_pad_diagnostics_rgb(overlay_rgb(center_thermal_gray, warped), center_detail)),
                (f"thermal {Path(row['source_tiff']).stem}", draw_pad_diagnostics_rgb(thermal_rgb, center_detail)),
            ]
        )
    make_contact_sheet(tiles, out_path, cols=4, tile_w=300, tile_h=230)


def candidate_review_filename(row: dict[str, Any]) -> str:
    safe_time = str(row["candidate_video_time"]).replace(":", "-")
    return f"rank_{int(row['rank']):03d}_time_{safe_time}_score_{float(row['aggregate_score']):.6f}.jpg"


def save_all_anchor_window_candidate_reviews(
    out_dir: Path,
    reader: VideoReader,
    rows: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_image in out_dir.glob("*.jpg"):
        old_image.unlink()
    index_rows: list[dict[str, Any]] = []
    for row in rows:
        spatial = row["spatial_candidate"]
        tiles: list[tuple[str, np.ndarray]] = []
        for detail in row.get("score_details", []):
            thermal_gray = normalize_to_uint8(load_thermal_u16(Path(detail["tiff_path"])))
            video_frame_bgr, _ = reader.frame_at_index(int(detail["video_frame_index"]))
            warped = warp_video_to_thermal(
                video_frame_bgr,
                spatial,
                (thermal_gray.shape[1], thermal_gray.shape[0]),
            )
            thermal_rgb = draw_pad_diagnostics_rgb(heatmap_rgb(thermal_gray), detail)
            video_rgb = cv2.cvtColor(
                draw_video_diagnostics_bgr(video_frame_bgr, spatial, detail),
                cv2.COLOR_BGR2RGB,
            )
            warped_rgb = draw_pad_diagnostics_rgb(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), detail)
            overlay = draw_pad_diagnostics_rgb(overlay_rgb(thermal_gray, warped), detail)
            label = diagnostic_label(detail)
            tiles.extend(
                [
                    (f"{label} TIFF", thermal_rgb),
                    (f"{label} video+crop", video_rgb),
                    (f"{label} warped", warped_rgb),
                    (f"{label} overlay", overlay),
                ]
            )
        filename = candidate_review_filename(row)
        out_path = out_dir / filename
        make_contact_sheet(tiles, out_path, cols=4, tile_w=300, tile_h=230)
        index_rows.append(
            {
                "rank": int(row["rank"]),
                "candidate_video_time": row["candidate_video_time"],
                "candidate_video_frame_index": int(row["candidate_video_frame_index"]),
                "aggregate_score": float(row["aggregate_score"]),
                "center_score": float(row["center_score"]),
                "pad_geometry_mean": float(row.get("pad_geometry_mean", 0.0)),
                "center_pad_geometry_score": float(row.get("center_pad_geometry_score", 0.0)),
                "center_pad_center_error_ratio": float(row.get("center_pad_center_error_ratio", 1.0)),
                "center_pad_radius_error_ratio": float(row.get("center_pad_radius_error_ratio", 1.0)),
                "center_pad_thermal_circle_rank": int(row.get("center_pad_thermal_circle_rank", -1)),
                "thermal_circle_confirmed": bool(row.get("thermal_circle_confirmed", False)),
                "manual_circle_source": row.get("manual_circle_source", ""),
                "video_circle_source": row.get("video_circle_source", ""),
                "video_circle_confidence": float(row.get("video_circle_confidence", 0.0)),
                "video_circle_partial": bool(row.get("video_circle_partial", False)),
                "video_circle_smoothed": bool(row.get("video_circle_smoothed", False)),
                "geometry_hard_pass": bool(row.get("geometry_hard_pass", False)),
                "image": filename,
            }
        )

    save_json(out_dir / "index.json", index_rows)
    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Anchor window candidates</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;} "
        "figure{margin:0 0 28px 0;} img{max-width:100%;height:auto;border:1px solid #ddd;} "
        "figcaption{font-size:14px;margin:8px 0;color:#222;}</style>",
        "<h1>Anchor window 5-frame candidates</h1>",
    ]
    for item in index_rows:
        html_lines.append(
            "<figure>"
            f"<figcaption>rank #{item['rank']:03d} | video {item['candidate_video_time']} | "
            f"score {item['aggregate_score']:.6f} | geom {item['center_pad_geometry_score']:.4f} | "
            f"center err {item['center_pad_center_error_ratio']:.3f} | "
            f"radius err {item['center_pad_radius_error_ratio']:.3f} | "
            f"circle rank {item['center_pad_thermal_circle_rank']} | "
            f"video {item['video_circle_source']} {item['video_circle_confidence']:.2f} | "
            f"pass {item['geometry_hard_pass']}</figcaption>"
            f"<a href='{item['image']}'><img src='{item['image']}'></a>"
            "</figure>"
        )
    (out_dir / "index.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def candidate_to_dict(candidate: SpatialCandidate | dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return dict(candidate)
    return {
        "transform_type": candidate.transform_type,
        "score": candidate.score,
        "crop_x": candidate.crop_x,
        "crop_y": candidate.crop_y,
        "crop_w": candidate.crop_w,
        "crop_h": candidate.crop_h,
        "matrix": candidate.matrix,
        "thermal_edge_density": candidate.thermal_edge_density,
        "video_edge_density": candidate.video_edge_density,
        "base_similarity": candidate.base_similarity,
        "circle_alignment": candidate.circle_alignment,
        "pad_center_error_px": candidate.pad_center_error_px,
        "pad_center_error_ratio": candidate.pad_center_error_ratio,
        "pad_radius_error_ratio": candidate.pad_radius_error_ratio,
        "pad_ring_iou": candidate.pad_ring_iou,
        "pad_geometry_score": candidate.pad_geometry_score,
        "pad_geometry_hard_pass": candidate.pad_geometry_hard_pass,
        "pad_thermal_circle_rank": candidate.pad_thermal_circle_rank,
        "keypoint_matches": candidate.keypoint_matches,
        "keypoint_inliers": candidate.keypoint_inliers,
        "source_tiff": candidate.source_tiff,
    }


def spatial_candidate_from_dict(data: dict[str, Any]) -> SpatialCandidate:
    return SpatialCandidate(
        transform_type=str(data.get("transform_type", "crop_resize")),
        score=float(data.get("score", 0.0)),
        crop_x=int(data["crop_x"]),
        crop_y=int(data["crop_y"]),
        crop_w=int(data["crop_w"]),
        crop_h=int(data["crop_h"]),
        matrix=data["matrix"],
        thermal_edge_density=float(data.get("thermal_edge_density", 0.0)),
        video_edge_density=float(data.get("video_edge_density", 0.0)),
        base_similarity=float(data.get("base_similarity", 0.0)),
        circle_alignment=float(data.get("circle_alignment", 0.0)),
        pad_center_error_px=float(data.get("pad_center_error_px", 0.0)),
        pad_center_error_ratio=float(data.get("pad_center_error_ratio", 1.0)),
        pad_radius_error_ratio=float(data.get("pad_radius_error_ratio", 1.0)),
        pad_ring_iou=float(data.get("pad_ring_iou", 0.0)),
        pad_geometry_score=float(data.get("pad_geometry_score", data.get("circle_alignment", 0.0))),
        pad_geometry_hard_pass=bool(data.get("pad_geometry_hard_pass", False)),
        pad_thermal_circle_rank=int(data.get("pad_thermal_circle_rank", -1)),
        keypoint_matches=int(data.get("keypoint_matches", 0)),
        keypoint_inliers=int(data.get("keypoint_inliers", 0)),
        source_tiff=str(data.get("source_tiff", "")),
    )


def load_spatial_transform(path: Path) -> dict[str, Any]:
    data = load_transform_bundle(path)
    if "selected" in data:
        data = data["selected"]
    required = {"matrix", "crop_x", "crop_y", "crop_w", "crop_h"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Transform file misses keys: {sorted(missing)}")
    return data


def load_transform_bundle(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Transform file must contain a JSON object: {path}")
    return data


def load_video_pad_seeds(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Video pad seed file must contain a JSON object: {path}")
    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        raise ValueError(f"Video pad seed file must contain an anchors object: {path}")
    result: dict[str, dict[str, Any]] = {}
    for name, entry in anchors.items():
        if name not in {"primary", "secondary"} or not isinstance(entry, dict):
            continue
        frame_index = entry.get("frame_index")
        circle = entry.get("circle")
        if not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"Video pad seed {name} must have a non-negative integer frame_index")
        if not isinstance(circle, dict):
            raise ValueError(f"Video pad seed {name} must contain circle.cx/cy/r")
        values = {key: float(circle[key]) for key in ["cx", "cy", "r"]}
        if not all(math.isfinite(value) for value in values.values()) or values["r"] <= 0:
            raise ValueError(f"Video pad seed {name} circle must contain finite cx/cy and positive r")
        normalized = dict(entry)
        normalized["frame_index"] = int(frame_index)
        normalized["circle"] = values
        result[name] = normalized
    if not result:
        raise ValueError(f"Video pad seed file does not contain primary or secondary anchors: {path}")
    return result


def transform_dataset_candidate_rows(
    transform_bundle: dict[str, Any],
    dataset_count: int,
) -> list[dict[str, Any]]:
    if dataset_count <= 1:
        return []
    rows = transform_bundle.get("top_candidates", [])
    if not isinstance(rows, list) or not rows:
        return []
    return [dict(row) for row in rows[:dataset_count] if isinstance(row, dict)]


def dataset_output_dir_for_candidate(base_out_dir: Path, row: dict[str, Any]) -> Path:
    rank = int(row.get("rank", 1))
    time_text = str(row.get("candidate_video_time") or row.get("selected_video_time") or "")
    if not time_text:
        time_s = float(row.get("candidate_video_time_s", row.get("selected_video_time_s", 0.0)) or 0.0)
        time_text = format_video_time(time_s)
    safe_time = time_text.replace(":", "-")
    score = float(row.get("aggregate_score", row.get("score", 0.0)) or 0.0)
    return base_out_dir.parent / f"{base_out_dir.name}_rank_{rank:03d}_time_{safe_time}_score_{score:.6f}"


def transform_report_from_window_row(
    source_bundle: dict[str, Any],
    source_path: Path,
    row: dict[str, Any],
    center_tiff: TiffFrame,
    temporal_tiffs: list[TiffFrame],
) -> dict[str, Any]:
    spatial = selected_spatial_from_window_row(row)
    report = {
        "strategy": str(source_bundle.get("strategy", "anchor_window_90")),
        "selected": candidate_to_dict(spatial),
        "source": str(source_path),
        "source_candidate_rank": int(row.get("rank", 1)),
        "source_candidate_time": row.get("candidate_video_time", ""),
        "source_candidate_score": float(row.get("aggregate_score", 0.0) or 0.0),
        "center_tiff": source_bundle.get(
            "center_tiff",
            {
                "path": center_tiff.path,
                "timestamp": center_tiff.timestamp,
            },
        ),
        "temporal_tiffs": source_bundle.get(
            "temporal_tiffs",
            [
                {"path": frame.path, "timestamp": frame.timestamp}
                for frame in temporal_tiffs
            ],
        ),
        "selected_candidate": compact_window_row(row),
        "top_candidates": source_bundle.get("top_candidates", []),
    }
    for key in ["confirmed_pad_circles", "anchor_frame_window"]:
        if key in source_bundle:
            report[key] = source_bundle[key]
    return report


def estimated_anchor_transform_report(
    context: AnchorContext,
    confirmed_pad_circles: dict[str, dict[str, Any]],
    window_rows: list[dict[str, Any]],
    selected_window: dict[str, Any],
    *,
    anchor_frame_window_s: float,
    top_k: int,
    video_pad_seed: dict[str, Any] | None,
) -> dict[str, Any]:
    spatial = selected_spatial_from_window_row(selected_window)
    return {
        "strategy": "anchor_window_90",
        "selected": candidate_to_dict(spatial),
        "center_tiff": {
            "path": context.center_tiff.path,
            "timestamp": context.center_tiff.timestamp,
            "delta_from_initial_target_s": (
                context.center_tiff.timestamp - context.anchor_tiff_target
            ).total_seconds(),
        },
        "temporal_tiffs": [
            {"path": frame.path, "timestamp": frame.timestamp}
            for frame in context.temporal_tiffs
        ],
        "confirmed_pad_circles": confirmed_pad_circles,
        "video_pad_tracking": {
            "detector": "hough_saturation_with_contour_fallback",
            "tracker": "seeded_bidirectional",
            "seed": video_pad_seed,
        },
        "anchor_frame_window": {
            "anchor_video_time_s": context.anchor_video_s,
            "anchor_video_time": format_video_time(context.anchor_video_s),
            "window_s": anchor_frame_window_s,
            "candidate_count": len(window_rows),
            "first_frame_index": min(row["candidate_video_frame_index"] for row in window_rows),
            "last_frame_index": max(row["candidate_video_frame_index"] for row in window_rows),
        },
        "selected_candidate": compact_window_row(selected_window),
        "top_candidates": [
            compact_window_row(row)
            for row in window_rows[: max(1, top_k)]
        ],
    }


def estimate_anchor_strategy(
    reader: VideoReader,
    context: AnchorContext,
    confirmed_pad_circles: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    review_dir: Path,
    out_dir: Path,
    output_prefix: str,
    video_pad_seed: dict[str, Any] | None,
) -> dict[str, Any]:
    review_prefix = f"{output_prefix}_" if output_prefix else ""
    table_prefix = f"{output_prefix}_" if output_prefix else ""
    window_rows = estimate_anchor_window_candidates(
        reader,
        context.center_tiff,
        context.temporal_tiffs,
        context.anchor_video_s,
        anchor_frame_window_s=args.anchor_frame_window_s,
        score_aggregation=args.window_score_aggregation,
        crop_scale_fractions=args.roi_scale_fractions,
        roi_step_fraction=args.roi_step_fraction,
        refine_features=not args.no_spatial_refine,
        refine_top_k=max(1, args.top_k),
        max_crop_strategies=(
            args.crop_strategy_candidates
            if args.crop_strategy_candidates > 0
            else None
        ),
        min_circle_alignment=args.min_circle_alignment,
        confirmed_pad_circles=confirmed_pad_circles,
        video_pad_track_window=args.video_pad_track_window,
        video_pad_seed=video_pad_seed,
        video_detection_review_path=review_dir / f"{review_prefix}video_pad_detection_check.jpg",
    )
    if not window_rows:
        raise RuntimeError(f"Could not estimate any anchor-window candidates for {context.name}")
    write_table_outputs(out_dir / f"{table_prefix}anchor_window_candidates", window_rows)
    selected_window = window_rows[0]
    spatial = selected_spatial_from_window_row(selected_window)
    selected_video_frame_idx = int(selected_window["candidate_video_frame_index"])
    selected_video_time_s = float(selected_window["candidate_video_time_s"])
    center_thermal_gray = normalize_to_uint8(load_thermal_u16(context.center_tiff.path))
    save_anchor_window_review(
        review_dir / f"{review_prefix}anchor_window_top_candidates.jpg",
        reader,
        center_thermal_gray,
        window_rows,
        top_k=max(1, args.top_k),
    )
    save_all_anchor_window_candidate_reviews(
        review_dir / f"{review_prefix}all_5frame_candidates",
        reader,
        window_rows,
    )
    transform_report = estimated_anchor_transform_report(
        context,
        confirmed_pad_circles,
        window_rows,
        selected_window,
        anchor_frame_window_s=args.anchor_frame_window_s,
        top_k=args.top_k,
        video_pad_seed=video_pad_seed,
    )
    return {
        "context": context,
        "window_rows": window_rows,
        "selected_window": selected_window,
        "spatial": spatial,
        "selected_video_frame_idx": selected_video_frame_idx,
        "selected_video_time_s": selected_video_time_s,
        "transform_report": transform_report,
    }


def anchor_report_for_selection(
    initial_model: TimeModel,
    context: AnchorContext,
    selected_video_time_s: float,
    selected_video_frame_idx: int,
) -> dict[str, Any]:
    anchor_offset_s = estimate_anchor_offset(
        initial_model,
        selected_video_time_s,
        context.center_tiff.timestamp,
    )
    return {
        "requested_anchor_video_time_s": context.anchor_video_s,
        "requested_anchor_video_time": format_video_time(context.anchor_video_s),
        "initial_anchor_tiff_target": context.anchor_tiff_target,
        "center_tiff": {
            "path": context.center_tiff.path,
            "timestamp": context.center_tiff.timestamp,
            "delta_from_initial_target_s": (
                context.center_tiff.timestamp - context.anchor_tiff_target
            ).total_seconds(),
        },
        "temporal_tiffs": [
            {"path": frame.path, "timestamp": frame.timestamp}
            for frame in context.temporal_tiffs
        ],
        "selected_video_time_s": selected_video_time_s,
        "selected_video_time": format_video_time(selected_video_time_s),
        "selected_video_frame_index": selected_video_frame_idx,
        "anchor_offset_s": anchor_offset_s,
    }


def parse_datetime_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def tiff_frame_from_record(data: dict[str, Any]) -> TiffFrame:
    return TiffFrame(
        path=Path(str(data["path"])),
        timestamp=parse_datetime_value(data["timestamp"]),
    )


def is_dual_anchor_transform_bundle(bundle: dict[str, Any]) -> bool:
    anchors = bundle.get("anchors")
    return (
        bundle.get("strategy") == "dual_anchor_window_90"
        or (
            isinstance(anchors, dict)
            and isinstance(anchors.get("primary"), dict)
            and isinstance(anchors.get("secondary"), dict)
        )
    )


def context_from_transform_report(
    name: str,
    report: dict[str, Any],
    fallback: AnchorContext,
) -> AnchorContext:
    anchor = report.get("anchor") if isinstance(report.get("anchor"), dict) else {}
    center_data = report.get("center_tiff") if isinstance(report.get("center_tiff"), dict) else None
    if center_data is None and isinstance(anchor, dict):
        center_data = anchor.get("center_tiff") if isinstance(anchor.get("center_tiff"), dict) else None
    center_tiff = tiff_frame_from_record(center_data) if center_data else fallback.center_tiff

    temporal_data = report.get("temporal_tiffs") if isinstance(report.get("temporal_tiffs"), list) else None
    if temporal_data is None and isinstance(anchor, dict):
        temporal_data = anchor.get("temporal_tiffs") if isinstance(anchor.get("temporal_tiffs"), list) else None
    temporal_tiffs = [
        tiff_frame_from_record(item)
        for item in (temporal_data or [])
        if isinstance(item, dict) and "path" in item and "timestamp" in item
    ]
    if not temporal_tiffs:
        temporal_tiffs = fallback.temporal_tiffs

    frame_window = report.get("anchor_frame_window") if isinstance(report.get("anchor_frame_window"), dict) else {}
    anchor_video_s = float(
        anchor.get(
            "requested_anchor_video_time_s",
            frame_window.get("anchor_video_time_s", fallback.anchor_video_s),
        )
    )
    target_value = anchor.get("initial_anchor_tiff_target") if isinstance(anchor, dict) else None
    anchor_tiff_target = parse_datetime_value(target_value) if target_value else fallback.anchor_tiff_target
    return AnchorContext(
        name=name,
        anchor_video_s=anchor_video_s,
        anchor_tiff_target=anchor_tiff_target,
        center_tiff=center_tiff,
        temporal_tiffs=temporal_tiffs,
    )


def selected_video_time_from_transform_report(report: dict[str, Any]) -> tuple[float, int]:
    selected = report.get("selected") if isinstance(report.get("selected"), dict) else {}
    selected_candidate = (
        report.get("selected_candidate")
        if isinstance(report.get("selected_candidate"), dict)
        else {}
    )
    time_s = float(
        selected.get(
            "selected_video_time_s",
            selected_candidate.get("candidate_video_time_s", 0.0),
        )
    )
    frame_idx = int(
        selected.get(
            "selected_video_frame_index",
            selected_candidate.get("candidate_video_frame_index", 0),
        )
    )
    return time_s, frame_idx


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_aligned_samples(
    frames: list[TiffFrame],
    model: TimeModel,
    spatial: SpatialCandidate | dict[str, Any],
    reader: VideoReader,
    aligned_root: Path,
    *,
    max_aligned: int | None,
    write_overlays: bool,
    segment_strategies: tuple[SegmentExportStrategy, SegmentExportStrategy] | None = None,
    split_tiff_time: datetime | None = None,
) -> list[dict[str, Any]]:
    if max_aligned is not None:
        frames = frames[: max(0, max_aligned)]
    rows: list[dict[str, Any]] = []
    primary_segment = (
        segment_strategies[0]
        if segment_strategies is not None
        else None
    )
    secondary_segment = (
        segment_strategies[1]
        if segment_strategies is not None
        else None
    )
    for aligned_index, frame in enumerate(frames, start=1):
        aligned_id = f"{aligned_index:06d}_{frame.path.stem}"
        aligned_dir = aligned_root / aligned_id
        aligned_dir.mkdir(parents=True, exist_ok=True)

        thermal_raw = load_thermal_u16(frame.path)
        thermal_gray = normalize_to_uint8(thermal_raw)
        if primary_segment is not None:
            segment = choose_segment_strategy(
                frame.timestamp,
                primary_segment,
                secondary_segment,
                split_tiff_time,
            )
            frame_model = segment.model
            frame_spatial = segment.spatial
        else:
            segment = None
            frame_model = model
            frame_spatial = spatial
        video_time_s = frame_model.tiff_to_video_time(frame.timestamp)
        video_frame_bgr, video_frame_index = reader.frame_at(video_time_s)
        warped_bgr = warp_video_to_thermal(video_frame_bgr, frame_spatial, (thermal_gray.shape[1], thermal_gray.shape[0]))

        thermal_tiff_out = aligned_dir / "thermal.tiff"
        thermal_png_out = aligned_dir / "thermal.png"
        video_png_out = aligned_dir / "video.png"
        overlay_png_out = aligned_dir / "overlay.png"

        shutil.copy2(frame.path, thermal_tiff_out)
        save_rgb(thermal_png_out, heatmap_rgb(thermal_gray))
        save_bgr(video_png_out, warped_bgr)
        overlay_path = ""
        if write_overlays:
            save_rgb(overlay_png_out, overlay_rgb(thermal_gray, warped_bgr))
            overlay_path = str(overlay_png_out)

        video_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sim = visual_similarity(thermal_gray, video_gray)
        row = {
            "aligned_id": aligned_id,
            "status": "exported",
            "tiff_name": frame.path.name,
            "tiff_time_iso": frame.timestamp.isoformat(sep=" "),
            "video_time_s": video_time_s,
            "video_time": format_video_time(video_time_s),
            "video_frame_index": video_frame_index,
            "thermal_source_path": str(frame.path),
            "thermal_tiff_path": str(thermal_tiff_out),
            "thermal_png_path": str(thermal_png_out),
            "video_png_path": str(video_png_out),
            "overlay_png_path": overlay_path,
            "visual_score": sim["score"],
            "thermal_edge_density": sim["thermal_edge_density"],
            "video_edge_density": sim["video_edge_density"],
        }
        if segment is not None:
            row.update(
                {
                    "anchor_segment": segment.name,
                    "anchor_requested_video_time_s": segment.anchor_video_s,
                    "anchor_requested_video_time": format_video_time(segment.anchor_video_s),
                    "anchor_selected_video_time_s": segment.selected_video_time_s,
                    "anchor_selected_video_time": format_video_time(segment.selected_video_time_s),
                    "anchor_center_tiff": segment.center_tiff.path.name,
                }
            )
        rows.append(row)
    return rows


def parse_scale_fractions(value: str) -> list[float]:
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions:
        raise argparse.ArgumentTypeError("At least one ROI scale fraction is required")
    if any(frac <= 0 or frac > 1 for frac in fractions):
        raise argparse.ArgumentTypeError("ROI scale fractions must be in (0, 1]")
    return fractions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aligned TIFF / MP4 samples with 90-frame anchor-window matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tiff-start", required=True)
    parser.add_argument("--tiff-end", required=True)
    parser.add_argument("--video-start", required=True)
    parser.add_argument("--video-end", required=True)
    parser.add_argument("--anchor-video", required=True)
    parser.add_argument("--second-anchor-video", "--anchor-video-2", dest="second_anchor_video", default=None)
    parser.add_argument("--anchor-frame-window-s", type=float, default=3.0)
    parser.add_argument("--temporal-neighbor-count", type=int, default=2)
    parser.add_argument(
        "--window-score-aggregation",
        choices=["mean", "median", "center-weighted"],
        default="mean",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--crop-strategy-candidates",
        type=int,
        default=0,
        help="Max crop strategies to score per candidate frame; 0 scores all generated strategies.",
    )
    parser.add_argument(
        "--min-circle-alignment",
        type=float,
        default=0.5,
        help="Minimum landing-pad circle alignment required before scoring a crop strategy; falls back to all if none pass.",
    )
    parser.add_argument("--roi-scale-fractions", type=parse_scale_fractions, default=parse_scale_fractions("0.34,0.42,0.52,0.64,0.76"))
    parser.add_argument("--roi-step-fraction", type=float, default=1.0)
    parser.add_argument("--no-spatial-refine", action="store_true")
    parser.add_argument("--transform-in", type=Path, help="Use an existing transform.json instead of estimating one.")
    parser.add_argument("--pad-circles-in", type=Path, help="Manual confirmed TIFF pad circles JSON.")
    parser.add_argument(
        "--video-pad-seeds-in",
        type=Path,
        help="Manual RGB landing-pad target seeds JSON with primary/secondary anchors.",
    )
    parser.add_argument("--auto-pad-circles", action="store_true", help="Experimental: use automatic TIFF circle detection without manual confirmation.")
    parser.add_argument("--video-pad-track-window", type=int, default=5, help="Frame window for smoothing low-confidence or partial video pad detections.")
    parser.add_argument("--max-aligned", type=int, default=None)
    parser.add_argument("--dataset-count", type=int, default=3, help="Number of top transform candidates to export when --transform-in contains top_candidates; 1 keeps single-dataset output.")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out:
        return args.out
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / run_id


def write_table_outputs(base_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (list, dict, tuple))
        }
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(base_path.with_suffix(".csv"), index=False)
    write_jsonl(base_path.with_suffix(".jsonl"), rows)


def save_dataset_case_index_html(
    out_dir: Path,
    manifest_rows: list[dict[str, Any]],
    transform_report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if not manifest_rows:
        return

    selected = transform_report.get("selected") if isinstance(transform_report, dict) else {}
    if not isinstance(selected, dict):
        selected = {}
    center_tiff = transform_report.get("center_tiff") if isinstance(transform_report, dict) else {}
    if not isinstance(center_tiff, dict):
        center_tiff = {}
    anchors = transform_report.get("anchors") if isinstance(transform_report, dict) else {}
    if isinstance(anchors, dict):
        primary = anchors.get("primary") if isinstance(anchors.get("primary"), dict) else {}
        if not selected and isinstance(primary.get("selected"), dict):
            selected = primary["selected"]
        if not center_tiff and isinstance(primary.get("center_tiff"), dict):
            center_tiff = primary["center_tiff"]
    anchor = config.get("anchor") if isinstance(config, dict) else {}
    if not isinstance(anchor, dict):
        anchor = {}

    center_tiff_name = Path(str(center_tiff.get("path", ""))).name if center_tiff.get("path") else ""
    selected_video_time = selected.get("selected_video_time") or anchor.get("selected_video_time") or ""
    selected_score = float(selected.get("score", 0.0) or 0.0)
    anchor_offset_s = anchor.get("anchor_offset_s")
    if anchor_offset_s is None and isinstance(config.get("anchor"), dict):
        anchor_offset_s = config["anchor"].get("anchor_offset_s")
    if anchor_offset_s is None:
        anchor_offset_s = 0.0

    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Dataset cases</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;background:#fafafa;color:#111;}",
        "header{position:sticky;top:0;z-index:1;background:rgba(250,250,250,.97);padding:12px 0 16px;border-bottom:1px solid #ddd;}",
        "h1{font-size:22px;margin:0 0 8px 0;}",
        ".summary{display:flex;flex-wrap:wrap;gap:12px;font-size:14px;color:#333;}",
        ".summary a{color:#0b57d0;text-decoration:none;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:16px;}",
        ".case{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;}",
        ".case h2{font-size:14px;line-height:1.3;margin:0 0 6px 0;}",
        ".meta{font-size:12px;line-height:1.45;color:#555;}",
        ".thumbs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:10px;}",
        ".thumbs a{display:block;}",
        ".thumbs img{width:100%;height:auto;display:block;border:1px solid #ddd;border-radius:4px;}",
        "</style>",
        "<header>",
        "<h1>Dataset cases</h1>",
        "<div class='summary'>",
        f"<span>cases: {len(manifest_rows)}</span>",
        f"<span>center TIFF: {escape(center_tiff_name)}</span>",
        f"<span>selected video: {escape(str(selected_video_time))}</span>",
        f"<span>score: {selected_score:.6f}</span>",
        f"<span>anchor offset: {float(anchor_offset_s):+.3f}s</span>",
        "<span><a href='manifest.csv'>manifest.csv</a></span>",
        "<span><a href='manifest.jsonl'>manifest.jsonl</a></span>",
        "<span><a href='transform.json'>transform.json</a></span>",
        "<span><a href='config.json'>config.json</a></span>",
        "</div>",
        "</header>",
        "<main class='grid'>",
    ]

    for row in manifest_rows:
        aligned_id = escape(str(row.get("aligned_id", "")))
        aligned_dir = f"aligned/{aligned_id}"
        tiff_name = escape(str(row.get("tiff_name", "")))
        video_time = escape(str(row.get("video_time", "")))
        tiff_time = escape(str(row.get("tiff_time_iso", "")))
        visual_score = float(row.get("visual_score", 0.0) or 0.0)
        thermal_path = f"{aligned_dir}/thermal.png"
        video_path = f"{aligned_dir}/video.png"
        overlay_path = f"{aligned_dir}/overlay.png"
        html_lines.append(
            "<article class='case'>"
            f"<h2><a href='{overlay_path}' style='color:inherit;text-decoration:none;'>{aligned_id}</a></h2>"
            f"<div class='meta'>TIFF: {tiff_name}<br>Time: {video_time}<br>Visual: {visual_score:.4f}<br>{tiff_time}</div>"
            "<div class='thumbs'>"
            f"<a href='{thermal_path}'><img loading='lazy' src='{thermal_path}' alt='{aligned_id} thermal'></a>"
            f"<a href='{video_path}'><img loading='lazy' src='{video_path}' alt='{aligned_id} video'></a>"
            f"<a href='{overlay_path}'><img loading='lazy' src='{overlay_path}' alt='{aligned_id} overlay'></a>"
            "</div>"
            "</article>"
        )

    html_lines.append("</main>")
    (out_dir / "index.html").write_text("\n".join(html_lines) + "\n", encoding="utf-8")


def write_dataset_outputs(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    frames: list[TiffFrame],
    effective_frames: list[TiffFrame],
    tiff_start_frame: TiffFrame,
    tiff_end_frame: TiffFrame,
    initial_model: TimeModel,
    center_tiff: TiffFrame,
    temporal_tiffs: list[TiffFrame],
    anchor_video_s: float,
    anchor_tiff_target: datetime,
    selected_video_time_s: float,
    selected_video_frame_idx: int,
    spatial: SpatialCandidate | dict[str, Any],
    transform_report: dict[str, Any],
    reader: VideoReader,
    anchor_window_candidate_count: int,
    secondary_center_tiff: TiffFrame | None = None,
    secondary_temporal_tiffs: list[TiffFrame] | None = None,
    secondary_anchor_video_s: float | None = None,
    secondary_anchor_tiff_target: datetime | None = None,
    secondary_selected_video_time_s: float | None = None,
    secondary_selected_video_frame_idx: int | None = None,
    secondary_spatial: SpatialCandidate | dict[str, Any] | None = None,
    secondary_anchor_window_candidate_count: int | None = None,
    split_video_s: float | None = None,
    split_tiff_time: datetime | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_root = out_dir / "aligned"
    aligned_root.mkdir(parents=True, exist_ok=True)

    primary_context = AnchorContext(
        name="primary",
        anchor_video_s=anchor_video_s,
        anchor_tiff_target=anchor_tiff_target,
        center_tiff=center_tiff,
        temporal_tiffs=temporal_tiffs,
    )
    anchor_report = anchor_report_for_selection(
        initial_model,
        primary_context,
        selected_video_time_s,
        selected_video_frame_idx,
    )
    anchor_offset_s = float(anchor_report["anchor_offset_s"])
    final_model = replace(initial_model, offset_s=initial_model.offset_s + anchor_offset_s)

    dual_anchor = secondary_spatial is not None
    secondary_anchor_report: dict[str, Any] | None = None
    secondary_final_model: TimeModel | None = None
    split_report: dict[str, Any] | None = None
    if dual_anchor:
        if (
            secondary_center_tiff is None
            or secondary_temporal_tiffs is None
            or secondary_anchor_video_s is None
            or secondary_anchor_tiff_target is None
            or secondary_selected_video_time_s is None
            or secondary_selected_video_frame_idx is None
        ):
            raise ValueError("Secondary anchor output requires complete secondary anchor metadata")
        secondary_context = AnchorContext(
            name="secondary",
            anchor_video_s=secondary_anchor_video_s,
            anchor_tiff_target=secondary_anchor_tiff_target,
            center_tiff=secondary_center_tiff,
            temporal_tiffs=secondary_temporal_tiffs,
        )
        if split_video_s is None:
            split_video_s = split_video_time_between_anchors(primary_context, secondary_context)
        if split_tiff_time is None:
            split_tiff_time = initial_model.video_to_tiff_time(split_video_s)
        secondary_anchor_report = anchor_report_for_selection(
            initial_model,
            secondary_context,
            secondary_selected_video_time_s,
            secondary_selected_video_frame_idx,
        )
        secondary_final_model = replace(
            initial_model,
            offset_s=initial_model.offset_s + float(secondary_anchor_report["anchor_offset_s"]),
        )
        split_report = {
            "mode": "anchor_video_midpoint",
            "primary_anchor_video_time_s": anchor_video_s,
            "primary_anchor_video_time": format_video_time(anchor_video_s),
            "secondary_anchor_video_time_s": secondary_anchor_video_s,
            "secondary_anchor_video_time": format_video_time(secondary_anchor_video_s),
            "split_video_time_s": split_video_s,
            "split_video_time": format_video_time(split_video_s),
            "split_tiff_time": split_tiff_time,
        }

    transform_report_to_save = transform_report
    if dual_anchor:
        transform_report_to_save = dict(transform_report)
        anchors = (
            dict(transform_report_to_save.get("anchors", {}))
            if isinstance(transform_report_to_save.get("anchors"), dict)
            else {}
        )
        primary_transform = (
            dict(anchors.get("primary", {}))
            if isinstance(anchors.get("primary"), dict)
            else {}
        )
        secondary_transform = (
            dict(anchors.get("secondary", {}))
            if isinstance(anchors.get("secondary"), dict)
            else {}
        )
        primary_transform["anchor"] = anchor_report
        secondary_transform["anchor"] = secondary_anchor_report
        anchors["primary"] = primary_transform
        anchors["secondary"] = secondary_transform
        transform_report_to_save["strategy"] = "dual_anchor_window_90"
        transform_report_to_save["anchors"] = anchors
        transform_report_to_save["split"] = split_report
    save_json(out_dir / "transform.json", transform_report_to_save)

    manifest_rows: list[dict[str, Any]] = []
    if not args.review_only:
        segment_strategies = None
        if dual_anchor:
            if secondary_final_model is None or split_tiff_time is None or secondary_spatial is None:
                raise ValueError("Dual-anchor export requires secondary model, spatial transform, and split time")
            segment_strategies = (
                SegmentExportStrategy(
                    name="primary",
                    model=final_model,
                    spatial=spatial,
                    anchor_video_s=anchor_video_s,
                    selected_video_time_s=selected_video_time_s,
                    center_tiff=center_tiff,
                ),
                SegmentExportStrategy(
                    name="secondary",
                    model=secondary_final_model,
                    spatial=secondary_spatial,
                    anchor_video_s=secondary_anchor_video_s,
                    selected_video_time_s=secondary_selected_video_time_s,
                    center_tiff=secondary_center_tiff,
                ),
            )
        manifest_rows = export_aligned_samples(
            effective_frames,
            final_model,
            spatial,
            reader,
            aligned_root,
            max_aligned=args.max_aligned,
            write_overlays=not args.no_overlays,
            segment_strategies=segment_strategies,
            split_tiff_time=split_tiff_time if dual_anchor else None,
        )
        write_table_outputs(out_dir / "manifest", manifest_rows)

    dataset_candidate = {
        "rank": transform_report_to_save.get("source_candidate_rank"),
        "time": transform_report_to_save.get("source_candidate_time") or format_video_time(selected_video_time_s),
        "score": transform_report_to_save.get("source_candidate_score", transform_report_to_save.get("selected", {}).get("score", 0.0)),
        "source": transform_report_to_save.get("source", ""),
    }
    if dual_anchor:
        anchors = transform_report_to_save.get("anchors", {})
        primary_transform = anchors.get("primary", {}) if isinstance(anchors, dict) else {}
        secondary_transform = anchors.get("secondary", {}) if isinstance(anchors, dict) else {}
        primary_score = float(
            primary_transform.get(
                "source_candidate_score",
                primary_transform.get("selected", {}).get("score", 0.0)
                if isinstance(primary_transform.get("selected"), dict)
                else 0.0,
            )
            or 0.0
        )
        secondary_score = float(
            secondary_transform.get(
                "source_candidate_score",
                secondary_transform.get("selected", {}).get("score", 0.0)
                if isinstance(secondary_transform.get("selected"), dict)
                else 0.0,
            )
            or 0.0
        )
        dataset_candidate = {
            "source": "dual_anchor",
            "score": min(primary_score, secondary_score),
            "primary": {
                "rank": primary_transform.get("source_candidate_rank"),
                "time": primary_transform.get("source_candidate_time") or format_video_time(selected_video_time_s),
                "score": primary_score,
            },
            "secondary": {
                "rank": secondary_transform.get("source_candidate_rank"),
                "time": secondary_transform.get("source_candidate_time") or format_video_time(secondary_selected_video_time_s or 0.0),
                "score": secondary_score,
            },
        }

    config = {
        "matching_strategy": "anchor_window_90",
        "args": vars(args),
        "video": reader.metadata(),
        "tiff_count_total": len(frames),
        "tiff_count_effective": len(effective_frames),
        "tiff_start_selected": tiff_start_frame,
        "tiff_end_selected": tiff_end_frame,
        "initial_time_model": initial_model,
        "final_time_model": final_model,
        "anchor": anchor_report,
        "spatial_transform": transform_report_to_save,
        "dataset_candidate": dataset_candidate,
        "anchor_window_candidate_count": anchor_window_candidate_count,
        "manifest_count": len(manifest_rows),
    }
    if dual_anchor:
        config.update(
            {
                "matching_strategy": "dual_anchor_window_90",
                "anchors": {
                    "primary": anchor_report,
                    "secondary": secondary_anchor_report,
                },
                "split": split_report,
                "final_time_models": {
                    "primary": final_model,
                    "secondary": secondary_final_model,
                },
                "anchor_window_candidate_counts": {
                    "primary": anchor_window_candidate_count,
                    "secondary": secondary_anchor_window_candidate_count,
                },
            }
        )
    save_json(out_dir / "config.json", config)
    if manifest_rows:
        save_dataset_case_index_html(out_dir, manifest_rows, transform_report_to_save, config)

    summary = {
        "out_dir": out_dir,
        "center_tiff": center_tiff.path.name,
        "selected_video_time": format_video_time(selected_video_time_s),
        "anchor_offset_s": anchor_offset_s,
        "manifest_count": len(manifest_rows),
        "score": config["dataset_candidate"]["score"],
    }
    if dual_anchor:
        summary.update(
            {
                "secondary_center_tiff": secondary_center_tiff.path.name if secondary_center_tiff else "",
                "secondary_selected_video_time": format_video_time(secondary_selected_video_time_s or 0.0),
                "secondary_anchor_offset_s": (
                    float(secondary_anchor_report["anchor_offset_s"])
                    if secondary_anchor_report
                    else 0.0
                ),
                "split_video_time": format_video_time(split_video_s or 0.0),
                "split_tiff_time": split_tiff_time,
            }
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dataset_count < 1:
        raise ValueError("--dataset-count must be >= 1")
    out_dir = resolve_output_dir(args)
    review_dir = out_dir / "review"
    if not args.transform_in:
        review_dir.mkdir(parents=True, exist_ok=True)

    frames = scan_tiff_frames(args.tiff_dir)
    tiff_start_target = parse_tiff_timestamp(args.tiff_start)
    tiff_end_target = parse_tiff_timestamp(args.tiff_end)
    tiff_start_frame = nearest_tiff(frames, tiff_start_target)
    tiff_end_frame = nearest_tiff(frames, tiff_end_target)
    effective_frames = tiffs_between(frames, tiff_start_frame.timestamp, tiff_end_frame.timestamp)
    if not effective_frames:
        raise RuntimeError("No TIFF files are within the selected effective range")

    initial_model = TimeModel(
        tiff_start=tiff_start_frame.timestamp,
        tiff_end=tiff_end_frame.timestamp,
        video_start_s=parse_video_time(args.video_start),
        video_end_s=parse_video_time(args.video_end),
    )
    anchor_video_s = parse_video_time(args.anchor_video)
    primary_context = build_anchor_context(
        effective_frames,
        initial_model,
        anchor_video_s,
        args.temporal_neighbor_count,
        name="primary",
    )
    secondary_context: AnchorContext | None = None
    if args.second_anchor_video:
        second_anchor_video_s = parse_video_time(args.second_anchor_video)
        if second_anchor_video_s <= anchor_video_s:
            raise ValueError("--second-anchor-video must be later than --anchor-video")
        secondary_context = build_anchor_context(
            effective_frames,
            initial_model,
            second_anchor_video_s,
            args.temporal_neighbor_count,
            name="secondary",
        )
    anchor_contexts = [primary_context] + ([secondary_context] if secondary_context else [])
    calibration_tiffs = combined_anchor_calibration_tiffs(anchor_contexts)

    confirmed_pad_circles: dict[str, dict[str, Any]] = {}
    video_pad_seeds: dict[str, dict[str, Any]] = {}
    if not args.transform_in:
        if args.video_pad_seeds_in:
            video_pad_seeds = load_video_pad_seeds(args.video_pad_seeds_in)
        if args.pad_circles_in:
            confirmed_pad_circles = load_pad_circles(args.pad_circles_in, calibration_tiffs)
        elif args.auto_pad_circles:
            confirmed_pad_circles = auto_pad_circles(calibration_tiffs)
        else:
            manual_json_path = out_dir / "pad_circles_manual.json"
            calibration_path = save_pad_circle_calibration(
                review_dir / "pad_circle_calibration",
                manual_json_path,
                calibration_tiffs,
            )
            print(f"Pad circle calibration written: {calibration_path}")
            print(f"Edit manual circle JSON, then rerun with: --pad-circles-in {manual_json_path}")
            return 0

    reader = VideoReader(args.video)
    try:
        window_rows: list[dict[str, Any]] = []
        if args.transform_in:
            source_bundle = load_transform_bundle(args.transform_in)
            if is_dual_anchor_transform_bundle(source_bundle):
                anchors = source_bundle.get("anchors")
                if not isinstance(anchors, dict):
                    raise ValueError("Dual-anchor transform must contain an anchors object")
                primary_transform = anchors.get("primary")
                secondary_transform = anchors.get("secondary")
                if not isinstance(primary_transform, dict) or not isinstance(secondary_transform, dict):
                    raise ValueError("Dual-anchor transform must contain primary and secondary anchors")

                primary_context = context_from_transform_report(
                    "primary",
                    primary_transform,
                    primary_context,
                )
                secondary_context = context_from_transform_report(
                    "secondary",
                    secondary_transform,
                    secondary_context or primary_context,
                )
                split_data = source_bundle.get("split") if isinstance(source_bundle.get("split"), dict) else {}
                split_video_s = float(
                    split_data.get(
                        "split_video_time_s",
                        split_video_time_between_anchors(primary_context, secondary_context),
                    )
                )
                split_tiff_time = (
                    parse_datetime_value(split_data["split_tiff_time"])
                    if "split_tiff_time" in split_data
                    else initial_model.video_to_tiff_time(split_video_s)
                )
                spatial = candidate_to_dict(primary_transform["selected"])
                secondary_spatial = candidate_to_dict(secondary_transform["selected"])
                selected_video_time_s, selected_video_frame_idx = selected_video_time_from_transform_report(primary_transform)
                secondary_selected_video_time_s, secondary_selected_video_frame_idx = selected_video_time_from_transform_report(secondary_transform)
                summary = write_dataset_outputs(
                    out_dir,
                    args=args,
                    frames=frames,
                    effective_frames=effective_frames,
                    tiff_start_frame=tiff_start_frame,
                    tiff_end_frame=tiff_end_frame,
                    initial_model=initial_model,
                    center_tiff=primary_context.center_tiff,
                    temporal_tiffs=primary_context.temporal_tiffs,
                    anchor_video_s=primary_context.anchor_video_s,
                    anchor_tiff_target=primary_context.anchor_tiff_target,
                    selected_video_time_s=selected_video_time_s,
                    selected_video_frame_idx=selected_video_frame_idx,
                    spatial=spatial,
                    transform_report=dict(source_bundle),
                    reader=reader,
                    anchor_window_candidate_count=(
                        len(primary_transform.get("top_candidates", []))
                        if isinstance(primary_transform.get("top_candidates"), list)
                        else 0
                    ),
                    secondary_center_tiff=secondary_context.center_tiff,
                    secondary_temporal_tiffs=secondary_context.temporal_tiffs,
                    secondary_anchor_video_s=secondary_context.anchor_video_s,
                    secondary_anchor_tiff_target=secondary_context.anchor_tiff_target,
                    secondary_selected_video_time_s=secondary_selected_video_time_s,
                    secondary_selected_video_frame_idx=secondary_selected_video_frame_idx,
                    secondary_spatial=secondary_spatial,
                    secondary_anchor_window_candidate_count=(
                        len(secondary_transform.get("top_candidates", []))
                        if isinstance(secondary_transform.get("top_candidates"), list)
                        else 0
                    ),
                    split_video_s=split_video_s,
                    split_tiff_time=split_tiff_time,
                )
                print(f"Output: {summary['out_dir']}")
                print(f"Primary center TIFF: {summary['center_tiff']}")
                print(f"Primary selected video time: {summary['selected_video_time']}")
                print(f"Primary anchor offset: {float(summary['anchor_offset_s']):+.3f}s")
                print(f"Secondary center TIFF: {summary['secondary_center_tiff']}")
                print(f"Secondary selected video time: {summary['secondary_selected_video_time']}")
                print(f"Secondary anchor offset: {float(summary['secondary_anchor_offset_s']):+.3f}s")
                print(f"Split video time: {summary['split_video_time']}")
                if summary["manifest_count"]:
                    print(f"Exported aligned samples: {summary['manifest_count']}")
                else:
                    print("Exported aligned samples: 0 (review-only)")
                return 0

            batch_rows = transform_dataset_candidate_rows(source_bundle, args.dataset_count)
            source_candidate_count = (
                len(source_bundle.get("top_candidates", []))
                if isinstance(source_bundle.get("top_candidates", []), list)
                else 0
            )
            if batch_rows:
                summaries: list[dict[str, Any]] = []
                for row in batch_rows:
                    spatial = selected_spatial_from_window_row(row)
                    selected_video_frame_idx = int(row["candidate_video_frame_index"])
                    selected_video_time_s = float(row["candidate_video_time_s"])
                    transform_report = transform_report_from_window_row(
                        source_bundle,
                        args.transform_in,
                        row,
                        primary_context.center_tiff,
                        primary_context.temporal_tiffs,
                    )
                    summaries.append(
                        write_dataset_outputs(
                            dataset_output_dir_for_candidate(out_dir, row),
                            args=args,
                            frames=frames,
                            effective_frames=effective_frames,
                            tiff_start_frame=tiff_start_frame,
                            tiff_end_frame=tiff_end_frame,
                            initial_model=initial_model,
                            center_tiff=primary_context.center_tiff,
                            temporal_tiffs=primary_context.temporal_tiffs,
                            anchor_video_s=primary_context.anchor_video_s,
                            anchor_tiff_target=primary_context.anchor_tiff_target,
                            selected_video_time_s=selected_video_time_s,
                            selected_video_frame_idx=selected_video_frame_idx,
                            spatial=spatial,
                            transform_report=transform_report,
                            reader=reader,
                            anchor_window_candidate_count=source_candidate_count,
                        )
                    )

                print(f"Batch output prefix: {out_dir}")
                print(f"Center TIFF: {primary_context.center_tiff.path.name}")
                for summary in summaries:
                    print(f"Output: {summary['out_dir']}")
                    print(f"  Selected video time: {summary['selected_video_time']}")
                    print(f"  Score: {float(summary['score']):.6f}")
                    print(f"  Anchor offset: {float(summary['anchor_offset_s']):+.3f}s")
                    if summary["manifest_count"]:
                        print(f"  Exported aligned samples: {summary['manifest_count']}")
                    else:
                        print("  Exported aligned samples: 0 (review-only)")
                return 0

            spatial: SpatialCandidate | dict[str, Any] = load_spatial_transform(args.transform_in)
            requested_selected_video_time_s = float(
                spatial.get("selected_video_time_s", primary_context.anchor_video_s)
                if isinstance(spatial, dict)
                else primary_context.anchor_video_s
            )
            selected_video_frame, selected_video_frame_idx = reader.frame_at(requested_selected_video_time_s)
            selected_video_time_s = selected_video_frame_idx / reader.fps
            center_thermal_gray = normalize_to_uint8(load_thermal_u16(primary_context.center_tiff.path))
            warped = warp_video_to_thermal(
                selected_video_frame,
                spatial,
                (center_thermal_gray.shape[1], center_thermal_gray.shape[0]),
            )
            center_sim = visual_similarity(center_thermal_gray, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))
            transform_report = {
                "strategy": "anchor_window_90",
                "selected": candidate_to_dict(spatial),
                "source": str(args.transform_in),
                "source_candidate_rank": (
                    int(source_bundle["selected_candidate"]["rank"])
                    if isinstance(source_bundle.get("selected_candidate"), dict)
                    and source_bundle["selected_candidate"].get("rank") is not None
                    else None
                ),
                "source_candidate_time": (
                    source_bundle["selected_candidate"].get("candidate_video_time", "")
                    if isinstance(source_bundle.get("selected_candidate"), dict)
                    else format_video_time(selected_video_time_s)
                ),
                "source_candidate_score": float(
                    spatial.get("window_aggregate_score", spatial.get("score", center_sim["score"]))
                    if isinstance(spatial, dict)
                    else center_sim["score"]
                ),
                "center_tiff": {
                    "path": primary_context.center_tiff.path,
                    "timestamp": primary_context.center_tiff.timestamp,
                },
                "temporal_tiffs": [
                    {"path": frame.path, "timestamp": frame.timestamp}
                    for frame in primary_context.temporal_tiffs
                ],
                "selected_candidate": source_bundle.get("selected_candidate")
                if isinstance(source_bundle.get("selected_candidate"), dict)
                else {
                    "candidate_video_frame_index": selected_video_frame_idx,
                    "candidate_video_time_s": selected_video_time_s,
                    "candidate_video_time": format_video_time(selected_video_time_s),
                    "aggregate_score": float(
                        spatial.get("window_aggregate_score", center_sim["score"])
                        if isinstance(spatial, dict)
                        else center_sim["score"]
                    ),
                    "center_score": float(
                        spatial.get("window_center_score", center_sim["score"])
                        if isinstance(spatial, dict)
                        else center_sim["score"]
                    ),
                    "aggregation": args.window_score_aggregation,
                },
                "top_candidates": source_bundle.get("top_candidates", []),
            }
            for key in ["confirmed_pad_circles", "anchor_frame_window"]:
                if key in source_bundle:
                    transform_report[key] = source_bundle[key]
        else:
            primary_strategy = estimate_anchor_strategy(
                reader,
                primary_context,
                confirmed_pad_circles,
                args,
                review_dir=review_dir,
                out_dir=out_dir,
                output_prefix="",
                video_pad_seed=video_pad_seeds.get("primary"),
            )
            window_rows = primary_strategy["window_rows"]
            spatial = primary_strategy["spatial"]
            selected_video_frame_idx = int(primary_strategy["selected_video_frame_idx"])
            selected_video_time_s = float(primary_strategy["selected_video_time_s"])
            transform_report = primary_strategy["transform_report"]
            if secondary_context is not None:
                secondary_strategy = estimate_anchor_strategy(
                    reader,
                    secondary_context,
                    confirmed_pad_circles,
                    args,
                    review_dir=review_dir,
                    out_dir=out_dir,
                    output_prefix="second",
                    video_pad_seed=video_pad_seeds.get("secondary"),
                )
                split_report = build_anchor_split_report(
                    initial_model,
                    primary_context,
                    secondary_context,
                )
                transform_report = {
                    "strategy": "dual_anchor_window_90",
                    "anchors": {
                        "primary": primary_strategy["transform_report"],
                        "secondary": secondary_strategy["transform_report"],
                    },
                    "split": split_report,
                }

        summary = write_dataset_outputs(
            out_dir,
            args=args,
            frames=frames,
            effective_frames=effective_frames,
            tiff_start_frame=tiff_start_frame,
            tiff_end_frame=tiff_end_frame,
            initial_model=initial_model,
            center_tiff=primary_context.center_tiff,
            temporal_tiffs=primary_context.temporal_tiffs,
            anchor_video_s=primary_context.anchor_video_s,
            anchor_tiff_target=primary_context.anchor_tiff_target,
            selected_video_time_s=selected_video_time_s,
            selected_video_frame_idx=selected_video_frame_idx,
            spatial=spatial,
            transform_report=transform_report,
            reader=reader,
            anchor_window_candidate_count=(
                len(window_rows)
                if window_rows
                else (
                    len(transform_report.get("top_candidates", []))
                    if isinstance(transform_report.get("top_candidates", []), list)
                    else 0
                )
            ),
            secondary_center_tiff=(
                secondary_context.center_tiff
                if secondary_context is not None and not args.transform_in
                else None
            ),
            secondary_temporal_tiffs=(
                secondary_context.temporal_tiffs
                if secondary_context is not None and not args.transform_in
                else None
            ),
            secondary_anchor_video_s=(
                secondary_context.anchor_video_s
                if secondary_context is not None and not args.transform_in
                else None
            ),
            secondary_anchor_tiff_target=(
                secondary_context.anchor_tiff_target
                if secondary_context is not None and not args.transform_in
                else None
            ),
            secondary_selected_video_time_s=(
                float(secondary_strategy["selected_video_time_s"])
                if "secondary_strategy" in locals()
                else None
            ),
            secondary_selected_video_frame_idx=(
                int(secondary_strategy["selected_video_frame_idx"])
                if "secondary_strategy" in locals()
                else None
            ),
            secondary_spatial=(
                secondary_strategy["spatial"]
                if "secondary_strategy" in locals()
                else None
            ),
            secondary_anchor_window_candidate_count=(
                len(secondary_strategy["window_rows"])
                if "secondary_strategy" in locals()
                else None
            ),
            split_video_s=(
                float(transform_report["split"]["split_video_time_s"])
                if isinstance(transform_report.get("split"), dict)
                and "split_video_time_s" in transform_report["split"]
                else None
            ),
            split_tiff_time=(
                transform_report["split"]["split_tiff_time"]
                if isinstance(transform_report.get("split"), dict)
                and "split_tiff_time" in transform_report["split"]
                else None
            ),
        )

        print(f"Output: {summary['out_dir']}")
        if "secondary_center_tiff" in summary:
            print(f"Primary center TIFF: {summary['center_tiff']}")
            print(f"Primary selected video time: {summary['selected_video_time']}")
            print(f"Primary anchor offset: {float(summary['anchor_offset_s']):+.3f}s")
            print(f"Secondary center TIFF: {summary['secondary_center_tiff']}")
            print(f"Secondary selected video time: {summary['secondary_selected_video_time']}")
            print(f"Secondary anchor offset: {float(summary['secondary_anchor_offset_s']):+.3f}s")
            print(f"Split video time: {summary['split_video_time']}")
        else:
            print(f"Center TIFF: {summary['center_tiff']}")
            print(f"Selected video time: {summary['selected_video_time']}")
            print(f"Anchor offset: {float(summary['anchor_offset_s']):+.3f}s")
        if window_rows:
            print(f"Best window score: {window_rows[0]['aggregate_score']:.6f}")
        if "secondary_strategy" in locals():
            print(f"Second best window score: {secondary_strategy['window_rows'][0]['aggregate_score']:.6f}")
        if summary["manifest_count"]:
            print(f"Exported aligned samples: {summary['manifest_count']}")
        else:
            print("Exported aligned samples: 0 (review-only)")
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
