from dataclasses import dataclass, field
from typing import Dict, List, Optional
@dataclass(frozen=True)
class PageEvaluationResult:
    mean_iou: float
    mean_dice: float
    panel_accuracy: float
    precision: float
    recall: float
    f1: float
    page_correct: bool
    gt_panel_count: int
    pred_panel_count: int
    iou_per_gt: List[float] = field(default_factory=list)
    dice_per_gt: List[float] = field(default_factory=list)
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "panel_accuracy": self.panel_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "page_accuracy": 1.0 if self.page_correct else 0.0,
            "gt_panel_count": float(self.gt_panel_count),
            "pred_panel_count": float(self.pred_panel_count),
        }
@dataclass(frozen=True)
class DatasetEvaluationResult:
    corpus_id: str
    page_count: int
    mean_iou: float
    mean_dice: float
    panel_accuracy: float
    precision: float
    recall: float
    f1: float
    page_accuracy: float
    per_page_results: Dict[str, PageEvaluationResult] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "panel_accuracy": self.panel_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "page_accuracy": self.page_accuracy,
            "page_count": float(self.page_count),
        }
