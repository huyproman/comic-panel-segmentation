# Changelog

All notable changes to the `comics_panel_extraction` package will be documented in this file.

## [0.2.0] - 2026-09-09

### Removed
- **M2 Manuscript-Faithful Research Path**: Pruned `src/comics_panel_extraction/manuscript_faithful/postprocess/` and associated unit tests to prevent maintaining superseded research variants.
- **M1 Retraining Scaffolding**: Pruned from-scratch training loops, data sequences, learning-rate schedules, and rasterizers.
- **Round-06 Retraining Code**: Pruned legacy-first experimental training implementation (`research/legacy_first_model/`).
- **Migration & Archaeology Debris**: Pruned Slices 01–06 root JSON migration plans, superseded configs, and one-off scaffolding tests.
- **Duplicate Wrappers**: Pruned redundant postprocessing legacy duplicates and dead runner stubs.

### Retained
- **M1 Operational Model Inference**: Consolidated M1 architecture (`comics_panel_extraction.model.m1`) and inference pipeline (`comics_panel_extraction.inference`).
- **Canonical Recovered Historical Post-Processing**: Retained bit-exact line-segment postprocessing (`comics_panel_extraction.historical_legacy`).
- **Historical Evaluator**: Retained deterministic evaluation suite reproducing Table 1 & Table 2 metrics (`comics_panel_extraction.evaluation`).

### Changed
- Streamlined CLI subcommands to four core operations: `infer`, `postprocess`, `evaluate`, and `historical-replay`.
- Updated package version to `0.2.0` representing final repository convergence.

---

## [0.1.1] - 2026-09-09
- Verified fixture path portability and release packaging.

## [0.1.0] - 2026-09-09
- Initial standalone historical reconstruction release.
