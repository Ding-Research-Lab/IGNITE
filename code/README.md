# Reusable code

`run_pipeline.py` is the supported config-driven entry point.  The other
modules provide landing-pad calibration, 90-frame ranking, export, thermal
statistics, mask generation and release validation.

No dataset path, manual candidate rank or local username is embedded in this
directory; dataset-specific values live in `configs/`.
