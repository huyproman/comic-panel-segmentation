# Comprehensive Metrics Matrix

This document provides the authoritative metric benchmark comparisons across all evaluated execution modes on the 74 test comic pages.

---

## 1. Metric Summary Table

| Metric | Historical Raw Oracle (Table 1) | Reconstructed Raw (M1) | Canonical Historical Replay (Mode A) | Reconstructed End-to-End (Mode B) | Manuscript-Faithful (Mode C) | Historical Published Benchmark (Table 2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **mIoU** | 90.9661% | 88.6334% | **92.4611%** | 89.4571% | 87.7013% | **92.3440%** |
| **mDice** | 93.6560% | 91.6440% | **95.3047%** | 92.9296% | 91.3390% | **95.2264%** |
| **Panel Acc** | 86.2515% | 83.9463% | **93.4192%** | 89.2493% | 84.9908% | **93.1940%** |
| **Page Acc** | 71.6216% | 66.2162% | **82.4324%** | 74.3243% | 68.9189% | **81.0811%** |
| **Precision** | 86.9064% | 80.2118% | **91.1105%** | 87.6432% | 89.0246% | **93.6931%** |
| **Recall** | 86.2515% | 83.9463% | **93.4192%** | 89.2493% | 84.9908% | **93.1940%** |
| **F1 Score** | 85.8604% | 80.0999% | **91.9477%** | 87.7985% | 86.5631% | **93.2542%** |

---

## 2. Evaluation Protocol & Semantics

- **Lane A (Historical Evaluator Baseline)**: Computes connected components using 8-connectivity. Ground truth panels and predicted components are matched greedily using Dice coefficient. A match is declared valid if $\text{Dice} \ge 0.90$.
- **Panel Accuracy**: Proportion of ground truth panels successfully matched by predicted components ($\ge 0.90\text{ Dice}$).
- **Page Accuracy**: Proportion of pages in the test set where 100% of ground truth panels are successfully segmented without omission.
