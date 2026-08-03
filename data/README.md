---
license: cc-by-4.0
task_categories:
  - image-to-image
  - image-segmentation
tags:
  - thermal-imaging
  - rgb-video
  - multimodal
  - computer-vision
  - wildfire-detection
  - uav
pretty_name: IGNITE RGB–Thermal Wildfire Dataset
---

# IGNITE: A Multimodal UAV-Collected Dataset for Wildfire Detection

IGNITE contains radiometric FLIR TIFF frames aligned with RGB video frames from four UAV-collected prescribed-fire sequences. The release includes 1,854 approved aligned samples. Processing code and release provenance are available in the companion [GitHub repository](https://github.com/Ding-Research-Lab/IGNITE).

<p align="center">
  <img src="https://raw.githubusercontent.com/Ding-Research-Lab/IGNITE/main/notebooks/sample_plot.png" alt="Aligned thermal and RGB examples" width="75%">
</p>

## Dataset Viewer

Each Viewer row is one aligned sample. The five image columns are:

- `thermal`: display rendering of the radiometric TIFF
- `thermal_tiff`: original 16-bit radiometric TIFF
- `rgb`: aligned RGB video frame
- `overlay`: thermal/RGB visual overlay
- `mask_80c`: binary mask for temperatures greater than or equal to 80 °C

`thermal_tiff` is decoded losslessly as a Pillow `I;16` image. Convert it to a NumPy array to access the calibrated `uint16` digital numbers; an ordinary browser preview may not preserve their intensity semantics. The backward-compatible `thermal_tiff_path` column identifies the processed TIFF, while `thermal_source_path` points to the corresponding TIFF in the raw sequence.

The default `train` view contains all 1,854 approved samples. `sequence_id` identifies the source sequence (`0001`–`0004`); these acquisition sequences are not train/test partitions.

```python
import numpy as np
from datasets import load_dataset

dataset = load_dataset(
    "Kyoma001/IGNITE-fire-dataset",
    split="train",
)

sample = dataset[0]
sample["thermal"], sample["thermal_tiff"], sample["rgb"], sample["overlay"], sample["mask_80c"]

thermal_dn = np.asarray(sample["thermal_tiff"], dtype=np.uint16)
temperature_c = thermal_dn * 0.04 - 273.15
```

In addition to the five image columns, each row includes alignment identity and timing, RGB frame index, visual alignment score, fire-mask statistics, and paths to the processed and source radiometric TIFFs.

## Quick start for complete download

```bash
git clone https://github.com/Ding-Research-Lab/IGNITE.git
cd IGNITE

hf download Kyoma001/IGNITE-fire-dataset \
  0001/ 0002/ 0003/ 0004/ metadata.jsonl catalog.csv checksums.sha256 statistics/ \
  --repo-type dataset \
  --local-dir data
```

- The complete `data/` directory is about 26.1 GB.
- For offline visual review, open `data/<sequence_id>/processed/index.html` in a browser.
- `data/<sequence_id>/raw/` contains the source RGB video and TIFF sequence.
- `data/<sequence_id>/processed/` contains aligned samples, manifests and the browser index.
- `data/<sequence_id>/metadata/` contains path-portable configuration and calibration provenance.

## Dataset summary

| Sequence ID | Alignment strategy | Raw TIFF | Aligned samples | Anchor time(s) |
|---|---|---:|---:|---|
| `0001` | dual anchor | 482 | 439 | `01:23.233`, `08:13.233` |
| `0002` | dual anchor | 468 | 451 | `04:52.500`, `12:00.567` |
| `0003` | single anchor\* | 651 | 460 | `00:29.533` (take-off anchor) |
| `0004` | single anchor\* | 535 | 504 | `09:27.833` (landing anchor) |
| **Total** |  | **2,136** | **1,854** |  |

\* Sequence `0003` does not include RGB video of the landing, while `0004` does not include thermal imagery of the takeoff. Both were therefore aligned with a single-anchor strategy.

All sequences use a FLIR Vue Pro R 640 13mm radiometric TIFF (`640×512`, 16-bit, one frame per second) and H.264 RGB video (`3840×2160`, 30 FPS). The thermal calibration stored in the TIFF XMP metadata is:

```text
temperature_C = DN * 0.04 - 273.15
```

The mask threshold is 80 °C, equivalently `DN >= 8829`. Each aligned sample contains:

```text
thermal.tiff   original radiometric uint16 TIFF
thermal.png    display rendering of the TIFF
video.png      RGB frame after the approved crop/resize transform
overlay.png    thermal/RGB visual overlay
mask_80c.png   indexed 0/1 mask for temperature >= 80 °C
```

## Notebook and validation

`notebooks/inspect_aligned_data.ipynb` displays a radiometric TIFF with a Celsius color bar together with its RGB frame, overlay and 80 °C mask. A small sample is bundled in `examples/sample_aligned/` so the notebook can be previewed without downloading the complete dataset.

```bash
uv sync
uv run python code/validate_dataset.py --root data
```

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

## Acknowledgment

The authors thank the prescribed-fire personnel and land-management partners who made the data collection possible, especially Dr. Deborah Landau, Mr. Gabriel Cahalan, and Mr. Chase McLean from The Nature Conservancy, and Mr. Miles Roy of American University. This research is based upon work supported in part by the NSF (\#2536664) and NASA ESTO Program (80NSSC25K7777). The views and conclusions contained herein should not be interpreted as representing the official policies of the U.S. Government.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{wang2026ignite,
  title  = {IGNITE: A Multimodal UAV-Collected Dataset for Wildfire Detection},
  author = {Yiding Wang and Zhaoxi Zhang and Chenzhi Zhao and Zhangyu Guan and Dong L. Wu and Leah Ding},
  year   = {2026}
}
```
