from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any

import cv2

from build_colocated_pairs import (
    draw_circle_rgb,
    heatmap_rgb,
    load_thermal_u16,
    make_contact_sheet,
    manual_circle_from_entry,
    normalize_to_uint8,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and validate the manually selected TIFF landing-pad circles."
    )
    parser.add_argument("--pad-circles", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JPG. Defaults to review/pad_circle_calibration/landing_pad_circle_check.jpg.",
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument(
        "--candidate-frame",
        default=None,
        help="Render the top candidates for one TIFF instead of selected circles.",
    )
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument(
        "--candidate-ranks",
        default=None,
        help="Optional comma-separated candidate ranks, for example 1,2,4 or 10.",
    )
    return parser


def load_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("frames", data if isinstance(data, list) else [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No frame entries found in {path}")
    return [dict(entry) for entry in entries]


def validate_circle(
    entry: dict[str, Any],
    circle: dict[str, Any],
    *,
    image_w: int,
    image_h: int,
) -> None:
    name = str(entry.get("tiff_name", entry.get("tiff_path", "unknown TIFF")))
    cx = float(circle["cx"])
    cy = float(circle["cy"])
    radius = float(circle["r"])
    if radius <= 0:
        raise ValueError(f"{name}: circle radius must be positive")
    if not (0 <= cx < image_w and 0 <= cy < image_h):
        raise ValueError(f"{name}: circle center ({cx}, {cy}) is outside {image_w}x{image_h}")
    if cx - radius < 0 or cy - radius < 0 or cx + radius >= image_w or cy + radius >= image_h:
        raise ValueError(f"{name}: circle extends outside {image_w}x{image_h}")


def resolve_output_path(manual_path: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    return (
        manual_path.parent
        / "review"
        / "pad_circle_calibration"
        / "landing_pad_circle_check.jpg"
    )


def resolve_candidate_output_path(
    manual_path: Path,
    requested: Path | None,
    frame_name: str,
) -> Path:
    if requested is not None:
        return requested
    return (
        manual_path.parent
        / "review"
        / "pad_circle_calibration"
        / f"{Path(frame_name).stem}_refined_candidates.jpg"
    )


def square_crop_bounds(
    circles: list[dict[str, Any]],
    *,
    image_w: int,
    image_h: int,
    margin: int = 16,
) -> tuple[int, int, int, int]:
    left = min(float(circle["cx"]) - float(circle["r"]) for circle in circles) - margin
    right = max(float(circle["cx"]) + float(circle["r"]) for circle in circles) + margin
    top = min(float(circle["cy"]) - float(circle["r"]) for circle in circles) - margin
    bottom = max(float(circle["cy"]) + float(circle["r"]) for circle in circles) + margin
    size = min(float(max(right - left, bottom - top)), float(min(image_w, image_h)))
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    x0 = int(round(center_x - 0.5 * size))
    y0 = int(round(center_y - 0.5 * size))
    x0 = min(max(x0, 0), max(0, image_w - int(round(size))))
    y0 = min(max(y0, 0), max(0, image_h - int(round(size))))
    side = max(1, int(round(size)))
    return x0, y0, min(image_w, x0 + side), min(image_h, y0 + side)


def render_candidate_review(
    manual_path: Path,
    output_path: Path,
    *,
    frame_name: str,
    candidate_count: int,
    candidate_ranks: list[int] | None = None,
) -> list[dict[str, Any]]:
    if candidate_count < 1:
        raise ValueError("--candidate-count must be positive")
    entries = load_entries(manual_path)
    entry = next(
        (
            item
            for item in entries
            if str(item.get("tiff_name", "")) == frame_name
            or Path(str(item.get("tiff_path", ""))).stem == Path(frame_name).stem
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"No TIFF entry matching {frame_name} in {manual_path}")
    all_candidates = [dict(item) for item in entry.get("candidates", [])]
    if candidate_ranks:
        by_rank = {int(item.get("rank", -1)): item for item in all_candidates}
        missing = [rank for rank in candidate_ranks if rank not in by_rank]
        if missing:
            raise ValueError(f"Missing candidate ranks for {frame_name}: {missing}")
        candidates = [by_rank[rank] for rank in candidate_ranks]
    else:
        candidates = all_candidates[:candidate_count]
    if not candidates:
        raise ValueError(f"No circle candidates found for {frame_name}")

    tiff_path = Path(str(entry["tiff_path"]))
    thermal_gray = normalize_to_uint8(load_thermal_u16(tiff_path))
    image_h, image_w = thermal_gray.shape[:2]
    for candidate in candidates:
        validate_circle(entry, candidate, image_w=image_w, image_h=image_h)
    x0, y0, x1, y1 = square_crop_bounds(
        candidates,
        image_w=image_w,
        image_h=image_h,
    )

    base_rgb = heatmap_rgb(thermal_gray)
    tiles: list[tuple[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rank = int(candidate.get("rank", len(rows) + 1))
        rgb = draw_circle_rgb(base_rgb, candidate, (80, 255, 80), thickness=3)
        center = (int(round(candidate["cx"])), int(round(candidate["cy"])))
        cv2.drawMarker(
            rgb,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        crop = cv2.resize(rgb[y0:y1, x0:x1], (640, 640), interpolation=cv2.INTER_CUBIC)
        label = (
            f"Candidate {rank}: cx={float(candidate['cx']):.1f} "
            f"cy={float(candidate['cy']):.1f} r={float(candidate['r']):.1f}"
        )
        tiles.append((label, crop))
        rows.append(
            {
                "rank": rank,
                "cx": float(candidate["cx"]),
                "cy": float(candidate["cy"]),
                "r": float(candidate["r"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_contact_sheet(tiles, output_path, cols=1, tile_w=640, tile_h=640)
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row['rank']}</td><td>{row['cx']:.1f}</td>"
        f"<td>{row['cy']:.1f}</td><td>{row['r']:.1f}</td>"
        "</tr>"
        for row in rows
    )
    output_path.with_suffix(".html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<meta charset='utf-8'>",
                f"<title>{escape(tiff_path.name)} refined candidates</title>",
                "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;}"
                "img{max-width:100%;height:auto;border:1px solid #ddd;}"
                "table{border-collapse:collapse;margin-top:20px;}th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;}</style>",
                f"<h1>{escape(tiff_path.name)} refined candidates</h1>",
                "<p>Green: candidate circle; white cross: circle center.</p>",
                f"<img src='{escape(output_path.name)}' alt='Refined circle candidates'>",
                "<table><thead><tr><th>Candidate</th><th>cx</th><th>cy</th><th>r</th></tr></thead><tbody>",
                table_rows,
                "</tbody></table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def render_review(
    manual_path: Path,
    output_path: Path,
    *,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    entries = load_entries(manual_path)
    if expected_count is not None and len(entries) != expected_count:
        raise ValueError(
            f"Expected {expected_count} circle entries in {manual_path}, found {len(entries)}"
        )

    tiles: list[tuple[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        tiff_path = Path(str(entry["tiff_path"]))
        thermal_gray = normalize_to_uint8(load_thermal_u16(tiff_path))
        image_h, image_w = thermal_gray.shape[:2]
        circle = manual_circle_from_entry(entry)
        validate_circle(entry, circle, image_w=image_w, image_h=image_h)

        rgb = heatmap_rgb(thermal_gray)
        rgb = draw_circle_rgb(rgb, circle, (80, 255, 80), thickness=3)
        center = (int(round(circle["cx"])), int(round(circle["cy"])))
        cv2.drawMarker(
            rgb,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        label = (
            f"{tiff_path.name}  cx={float(circle['cx']):.1f} "
            f"cy={float(circle['cy']):.1f} r={float(circle['r']):.1f}"
        )
        tiles.append((label, rgb))
        rows.append(
            {
                "tiff_name": tiff_path.name,
                "cx": float(circle["cx"]),
                "cy": float(circle["cy"]),
                "r": float(circle["r"]),
                "image_w": image_w,
                "image_h": image_h,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_contact_sheet(tiles, output_path, cols=1, tile_w=640, tile_h=512)

    table_rows = "\n".join(
        "<tr>"
        f"<td>{escape(row['tiff_name'])}</td>"
        f"<td>{row['cx']:.1f}</td><td>{row['cy']:.1f}</td><td>{row['r']:.1f}</td>"
        "</tr>"
        for row in rows
    )
    html_path = output_path.with_suffix(".html")
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<meta charset='utf-8'>",
                "<title>Landing pad circle check</title>",
                "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;}"
                "img{max-width:100%;height:auto;border:1px solid #ddd;}"
                "table{border-collapse:collapse;margin-top:20px;}th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;}"
                "th:first-child,td:first-child{text-align:left;}</style>",
                "<h1>Landing pad circle check</h1>",
                f"<p>{len(rows)} frames. Green: selected outer circle; white cross: circle center.</p>",
                f"<img src='{escape(output_path.name)}' alt='Selected landing-pad circles'>",
                "<table><thead><tr><th>TIFF</th><th>cx</th><th>cy</th><th>r</th></tr></thead><tbody>",
                table_rows,
                "</tbody></table>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.candidate_frame:
        candidate_ranks = None
        if args.candidate_ranks:
            candidate_ranks = [
                int(part.strip())
                for part in str(args.candidate_ranks).split(",")
                if part.strip()
            ]
        output_path = resolve_candidate_output_path(
            args.pad_circles,
            args.out,
            args.candidate_frame,
        )
        rows = render_candidate_review(
            args.pad_circles,
            output_path,
            frame_name=args.candidate_frame,
            candidate_count=args.candidate_count,
            candidate_ranks=candidate_ranks,
        )
        print(f"Rendered candidates: {len(rows)}")
        print(f"Candidate image: {output_path}")
        print(f"Candidate HTML: {output_path.with_suffix('.html')}")
        return 0
    output_path = resolve_output_path(args.pad_circles, args.out)
    rows = render_review(
        args.pad_circles,
        output_path,
        expected_count=args.expected_count,
    )
    print(f"Validated circles: {len(rows)}")
    print(f"Review image: {output_path}")
    print(f"Review HTML: {output_path.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
