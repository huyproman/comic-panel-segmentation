# Artifact Management & Registry

This document defines the storage layout, hashing, and distribution policies for all data, model, and benchmark artifacts.

---

## 1. Authoritative Artifact Registry

| Logical Name | Path / Location | SHA-256 / Manifest Hash | Size | Role |
| :--- | :--- | :--- | :--- | :--- |
| **M1 Checkpoint** | `markdowns_dir/.../best_m1_unetpp_val_loss.keras` | `b6e4e63bc98408a482b9b00d11c8a21b42e47f850ceae4a7154c2e0145dba57b` | 109.1 MB | Reconstructed Comic model |
| **Historical Raw Masks** | `comics_analysis/preds/.../pred_mask/` | `cca252f9b38b3258bf64b298b5191e94778d1873f0c44d05f6fc9770da5b62e8` | 74 files | Table 1 Empirical Oracle |
| **Historical Final Masks**| `comics_analysis/preds/.../final_mask/` | `bd1df82ed693fd8167f336ae97c3848b84f66259bc96a1913f0173ba29456244` | 74 files | Table 2 Empirical Oracle |
| **Historical Ground Truth**| `comics_analysis/preds/.../true_mask/` | `824b22c748c90fe3453b3c4f74d0813083db19ee4ad3286f3769c8bfbe0d3bb3` | 74 files | Evaluation Ground Truth |
| **Comic Dataset Archive** | `marvel.v8-augment-x3-again.yolov12.zip` | `1c1a829380bfef975065cb1a979506232c8ac6b714d0ba9b2f4d96764e5f40b9` | 103.7 MB | Training / Evaluation Data |

---

## 2. Path Portability & Resolution Contract

All runtime code in `src/comics_panel_extraction/` resolves paths portably:
1. **Explicit CLI / Function Arguments**: Primary mechanism (`--input-dir`, `--output-dir`, `--checkpoint`).
2. **Environment Overrides**: `COMICS_CHECKPOINT_PATH`, `COMICS_ARTIFACTS_DIR`.
3. **Repository-Relative Fallbacks**: Default configuration templates in `configs/legacy_2024.yaml`.
Zero hardcoded developer absolute paths (`/home/ubuntu/...`) exist in runtime source files.
