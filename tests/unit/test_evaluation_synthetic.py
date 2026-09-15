import numpy as np
import pytest
from comics_panel_extraction.evaluation import (
    calculate_dice,
    calculate_iou,
    calculate_mean_dice,
    calculate_mean_iou,
    calculate_panel_accuracy,
    calculate_precision_recall_f1,
    evaluate_page,
    extract_components,
)
@pytest.mark.unit
def test_case_01_perfect_match():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:40, 10:40] = 255
    gt[50:80, 50:80] = 255
    pred = gt.copy()
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 1.0
    assert res.mean_dice == 1.0
    assert res.panel_accuracy == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.page_correct is True
    assert res.gt_panel_count == 2
    assert res.pred_panel_count == 2
@pytest.mark.unit
def test_case_02_no_overlap():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[60:80, 60:80] = 255
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.mean_dice == 0.0
    assert res.panel_accuracy == 0.0
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.page_correct is False
@pytest.mark.unit
def test_case_03_partial_overlap():
    b1 = np.zeros((50, 50), dtype=bool)
    b1[10:30, 10:30] = True
    b2 = np.zeros((50, 50), dtype=bool)
    b2[10:30, 20:40] = True
    iou = calculate_iou(b2, b1)
    dice = calculate_dice(b2, b1)
    assert abs(iou - (1.0 / 3.0)) < 1e-6
    assert abs(dice - 0.5) < 1e-6
@pytest.mark.unit
def test_case_04_extra_false_positive():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:40, 10:40] = 255              
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:40, 10:40] = 255                            
    pred[60:90, 60:90] = 255                              
    res = evaluate_page(pred, gt)
    assert res.gt_panel_count == 1
    assert res.pred_panel_count == 2
    assert res.panel_accuracy == 1.0
    assert res.recall == 1.0
    assert res.precision == 0.5
    assert abs(res.f1 - (2.0 * 0.5 * 1.0 / 1.5)) < 1e-6
    assert res.page_correct is True                                                    
@pytest.mark.unit
def test_case_05_missing_panel():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255        
    gt[50:70, 50:70] = 255        
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:30] = 255                                       
    res = evaluate_page(pred, gt)
    assert res.gt_panel_count == 2
    assert res.pred_panel_count == 1
    assert res.panel_accuracy == 0.5
    assert res.recall == 0.5
    assert res.precision == 1.0
    assert abs(res.f1 - (2.0 * 1.0 * 0.5 / 1.5)) < 1e-6
    assert res.page_correct is False
@pytest.mark.unit
def test_case_06_merged_prediction():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255          
    gt[10:30, 50:70] = 255          
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:70] = 255                          
    res = evaluate_page(pred, gt)
    assert res.panel_accuracy == 0.0
    assert res.recall == 0.0
    assert res.precision == 0.0
    assert res.page_correct is False
@pytest.mark.unit
def test_case_07_multiple_predictions_around_one_gt():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:50, 10:50] = 255           
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:50, 10:50] = 255                 
    pred[60:70, 60:70] = 255                      
    mean_iou, _ = calculate_mean_iou(pred, gt)
    mean_dice, _ = calculate_mean_dice(pred, gt)
    assert mean_iou == 1.0
    assert mean_dice == 1.0
@pytest.mark.unit
def test_case_08_both_empty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.mean_dice == 0.0
    assert res.panel_accuracy == 0.0
    assert res.page_correct is False
@pytest.mark.unit
def test_case_09_gt_nonempty_pred_empty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 255
    pred = np.zeros((50, 50), dtype=np.uint8)
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.panel_accuracy == 0.0
    assert res.page_correct is False
@pytest.mark.unit
def test_case_10_gt_empty_pred_nonempty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:20, 10:20] = 255
    res = evaluate_page(pred, gt)
    assert res.panel_accuracy == 0.0
    assert res.precision == 0.0
    assert res.page_correct is False
@pytest.mark.unit
def test_case_11_diagonal_touch_connectivity():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 255                       
    labeled, labels = extract_components(mask)
    assert len(labels) == 1
@pytest.mark.unit
def test_case_12_dtype_representations():
    gt1 = np.zeros((20, 20), dtype=np.uint8)
    gt1[5:15, 5:15] = 1
    gt255 = np.zeros((20, 20), dtype=np.uint8)
    gt255[5:15, 5:15] = 255
    res = evaluate_page(gt1, gt255)
    assert res.mean_iou == 1.0
    assert res.mean_dice == 1.0
@pytest.mark.unit
def test_case_13_dice_threshold_exact_boundary():
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True                             
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:9] = True                             
    pred[10, 0:10] = True                                              
    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.9) < 1e-9
    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 1.0
@pytest.mark.unit
def test_case_14_dice_threshold_just_below():
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True              
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:8] = True              
    pred[0:9, 8] = True                                  
    pred[10, 0:11] = True                                           
    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.89) < 1e-9
    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 0.0
@pytest.mark.unit
def test_case_15_dice_threshold_just_above():
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True              
    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:9] = True              
    pred[0, 9] = True                                    
    pred[10, 0:9] = True                                           
    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.91) < 1e-9
    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 1.0
