---
license: cc-by-4.0
task_categories:
  - image-classification
  - image-to-image
tags:
  - thermal-imaging
  - rgb-video
  - multimodal
  - computer-vision
pretty_name: IGNITE RGB–Thermal Wildfire Dataset
---

# IGNITE: A Multimodal UAV-Collected Dataset for Wildfire Detection

IGNITE contains four time-aligned FLIR radiometric TIFF and RGB video sequences
with 1,867 approved aligned samples. Each sample contains the original
radiometric TIFF, its thermal rendering, an aligned RGB frame, an overlay, and
an indexed 80 °C mask.

![Aligned RGB–thermal sample](assets/sample_plot.png)

## Download and review

```bash
hf download Kyoma001/IGNITE-fire-dataset \
  --repo-type dataset \
  --local-dir data
```

For a quick visual review after downloading, open
`data/<dataset_id>/processed/index.html`, for example
`data/0001/processed/index.html`.

## Dataset structure

```text
<dataset_id>/
├── raw/
│   ├── video.mp4
│   └── thermal/*.tiff
├── processed/
│   ├── aligned/<aligned_id>/
│   │   ├── thermal.tiff
│   │   ├── thermal.png
│   │   ├── video.png
│   │   ├── overlay.png
│   │   └── mask_80c.png
│   ├── manifest.csv
│   ├── manifest.jsonl
│   └── index.html
├── metadata/
└── statistics/
```

Original TIFF filenames and aligned sample IDs retain their acquisition
timestamps.

## Dataset summary

| Dataset ID | Source sequence ID | Alignment strategy | Raw TIFF | Aligned samples | Anchor time(s) |
|---|---|---|---:|---:|---|
| `0001` | `20260226_022619` | dual anchor | 482 | 439 | `01:23.233`, `08:13.233` |
| `0002` | `20260226_030445` | dual anchor | 468 | 451 | `04:52.500`, `12:00.567` |
| `0003` | `20260226_033912` | single anchor | 651 | 460 | `00:29.533` |
| `0004` | `20260227_023453` | single anchor | 535 | 517 | `09:27.833` |
| **Total** |  |  | **2,136** | **1,867** |  |

Due to limitations in the acquisition setup, `0003` does not include RGB
video of the landing, while `0004` does not include thermal imagery of the
takeoff. These sequences therefore use a single-anchor strategy.

## Thermal data and mask

All sequences use a FLIR Vue Pro R 640 13mm radiometric camera (`640×512`,
16-bit, one frame per second) and H.264 RGB video (`3840×2160`, 30 FPS). The
thermal conversion is:

```text
temperature_C = DN * 0.04 - 273.15
```

The indexed `mask_80c.png` contains only values `{0, 1}` and marks pixels at or
above 80 °C, equivalently `DN >= 8829`.

## Processing flow

```text
raw TIFF + MP4
      │
      ├─ timestamp scan and initial time model
      ├─ landing-pad circle detection + manual confirmation
      ├─ seeded RGB circle tracking
      ├─ 90-frame anchor-window crop scoring
      ├─ Rank 1 transform selection (single or dual anchor)
      └─ full-range aligned sample export + 80 °C mask
```

## Known alignment notes

`0003` follows the left RGB landing pad while its confirmed TIFF circle points
to a different physical pad. Its final mapped frame is clamped to the final
MP4 frame. `0004` uses an inferred video start.

## Acknowledgment

The authors thank the prescribed-fire personnel and land-management partners
who made the data collection possible, especially Deborah Landau, Gabriel
Cahalan, and Chase McLean from The Nature Conservancy, and Miles Roy of
American University. This research is based upon work supported in part by the
NSF (#2536664) and NASA ESTO Program (80NSSC25K7777). The views and conclusions
contained herein are those of the authors and should not be interpreted as
necessarily representing the official policies, either expressed or implied,
of the U.S. Government.

## Citation

Citation information will be added to the companion repository. Please retain
the CC BY 4.0 attribution when using or redistributing the dataset.
