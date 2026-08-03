# IGNITE: A Multimodal UAV-Collected Dataset for Wildfire Detection

This repository contains code and release metadata for aligning FLIR TIFF frames with RGB video frames. The four released sequences contain 1,854 approved aligned samples. The complete binary payload is intended for the companion [Hugging Face Dataset](https://huggingface.co/datasets/Kyoma001/IGNITE-fire-dataset); the local `data/` directory is the upload-ready copy.

<p align="center">
  <img src="https://raw.githubusercontent.com/Ding-Research-Lab/IGNITE/main/notebooks/sample_plot.png" alt="Sample plot" width="75%">
</p>

## Quick start for dataset download

```bash
git clone https://github.com/Ding-Research-Lab/IGNITE.git
cd IGNITE

# Download the complete data payload from Hugging Face into data/.
hf download Kyoma001/IGNITE-fire-dataset \
  0001/ 0002/ 0003/ 0004/ metadata.jsonl catalog.csv checksums.sha256 statistics/ \
  --repo-type dataset \
  --local-dir data
```

- The `data/` directory is about 26.1 GB.
- For a **quick visual review**, open `data/<dataset_id>/processed/index.html` in a web browser, for example `data/0001/processed/index.html`.
- `data/<dataset_id>/raw/` directory contains the complete source video and TIFF sequence.
- `data/<dataset_id>/processed/` contains the aligned samples, manifest and browser index.
- `data/<dataset_id>/metadata/` contains path-portable configuration and calibration provenance.



## Notebook

`notebooks/inspect_aligned_data.ipynb` loads one aligned sample and displays the radiometric TIFF with a Celsius color bar, the aligned RGB frame, the overlay and the 80 °C mask. A small copy is in `examples/sample_aligned/` (`000189_20260226_030803`) so the notebook can also be previewed without downloading the full dataset.


## Dataset summary

| Dataset ID  | Alignment strategy | Raw TIFF | Aligned samples | Anchor time(s) |
|---|---|---:|---:|---|
| `0001` |  dual anchor | 482 | 439 | `01:23.233`, `08:13.233` |
| `0002` |  dual anchor | 468 | 451 | `04:52.500`, `12:00.567` |
| `0003` |  single anchor * | 651 | 460 | `00:29.533 (take-off anchor)` |
| `0004` |  single anchor * | 535 | 504 | `09:27.833 (landing anchor)` |
| **Total** |  | **2,136** | **1,854** |  |

*: Due to limitations in the data acquisition setup, `0003` does not include RGB video of the landing, while `0004` does not include thermal imagery of the takeoff. Therefore, both sequences were aligned using a single-anchor strategy.

All sequences use a FLIR Vue Pro R 640 13mm radiometric TIFF (`640×512`, 16-bit, one frame per second) and H.264 RGB video (`3840×2160`, 30 FPS). The thermal calibration in the TIFF XMP metadata is:

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


## Quick start for Processing flow

```bash
uv sync
uv run python code/validate_dataset.py --root data
```

The exact published aligned samples can be regenerated from their locked Rank 1
transforms:

```bash
uv run python code/dataset_specific/reproduce_all.py --stage export
```

To repeat the review stages for one sequence:

```bash
uv run python code/run_pipeline.py \
  --config configs/0002.json --stage calibrate
uv run python code/run_pipeline.py \
  --config configs/0002.json --stage match
uv run python code/run_pipeline.py \
  --config configs/0002.json --stage export
```

The `masks` stage is idempotent and writes `mask_80c.png` beside each aligned TIFF. `export` also runs it automatically.


## Acknowledgment
The authors thank the prescribed-fire personnel and land-management partners who made the data collection possible, especially Dr. Deborah Landau, Mr. Gabriel Cahalan, and Mr. Chase McLean from The Nature Conservancy, and Mr. Miles Roy of American University. This research is based upon work supported in part by the NSF (\#2536664) and NASA ESTO Program (80NSSC25K7777). The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the U.S. Government.


## Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@article{wang2026ignite,
  title  = {IGNITE: A Multimodal UAV-Collected Dataset for Wildfire Detection},
  author = {Yiding Wang and Zhaoxi Zhang and Chenzhi Zhao and Zhangyu Guan and Dong L. Wu and Leah Ding},
  year   = {2026}
}
```
