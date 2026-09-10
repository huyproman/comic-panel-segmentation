# Reproduction Guide

This guide details the step-by-step procedures to reproduce all benchmarks and tables from the paper.

---

## 1. Historical Table 1 Oracle Evaluation
Evaluates the archived raw prediction masks against archived ground truth to confirm the baseline metric reproduction:

```bash
comics-panel-extraction evaluate \
    --predictions path/to/historical_raw_masks/ \
    --ground-truth path/to/historical_ground_truth/
```
Expected result: **mIoU 90.97%**, **Panel Accuracy 86.25%**, **Page Accuracy 71.62%** (Bit-exact match to published Table 1).

---

## 2. Historical Table 2 Oracle Evaluation
Evaluates the archived final masks against archived ground truth:

```bash
comics-panel-extraction evaluate \
    --predictions path/to/historical_final_masks/ \
    --ground-truth path/to/historical_ground_truth/
```
Expected result: **mIoU 92.34%**, **Panel Accuracy 93.19%**, **Page Accuracy 81.08%** (Bit-exact match to published Table 2).

---

## 3. Canonical Historical Legacy Replay (Mode A)
Executes the source-faithful legacy line segment analysis pipeline on raw prediction masks:

```bash
comics-panel-extraction historical-replay \
    --input-dir path/to/historical_raw_masks/ \
    --output-dir path/to/candidate_final_masks/ \
    --ground-truth path/to/historical_ground_truth/
```
Expected result: **mIoU 92.46%**, **Panel Accuracy 93.42%**, **Page Accuracy 82.43%**, **Precision 91.11%**, **F1 91.95%**. Matches published Table 2 within 0.12 pp.

---

## 4. Reconstructed End-to-End Pipeline (Mode B)
Runs the from-scratch trained M1 Comic model followed by canonical historical legacy post-processing.
Expected result: **mIoU 89.46%**, **Panel Accuracy 89.25%**, **Page Accuracy 74.32%**.
