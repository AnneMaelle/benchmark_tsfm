"""Unit tests for map_iou, f1_det and their helpers."""

import math

import numpy as np
import pytest

from benchmark_utils.metrics import _ap_from_tp_fp, _iou_1d, f1_det, map_iou


def _evt(start, width, *class_cols):
    """Single-row event array: [start, width, *class_cols]."""
    return np.array([[start, width, *class_cols]], dtype=float)


def _no_evts(n_classes):
    return np.zeros((0, 2 + n_classes))


# ---------------------------------------------------------------------------
# _iou_1d
# ---------------------------------------------------------------------------

def test_iou_identical():
    assert _iou_1d(0.0, 0.5, 0.0, 0.5) == pytest.approx(1.0)

def test_iou_no_overlap():
    assert _iou_1d(0.0, 0.3, 0.4, 0.3) == pytest.approx(0.0)

def test_iou_adjacent():
    # Segments touch at a single point — inter = 0
    assert _iou_1d(0.0, 0.5, 0.5, 0.5) == pytest.approx(0.0)

def test_iou_partial():
    # [0, 1) and [0.5, 1.5) → inter=0.5, union=1.5
    assert _iou_1d(0.0, 1.0, 0.5, 1.0) == pytest.approx(0.5 / 1.5)

def test_iou_contained():
    # [0, 1) contains [0.2, 0.6) → inter=0.4, union=1.0
    assert _iou_1d(0.0, 1.0, 0.2, 0.4) == pytest.approx(0.4 / 1.0)

def test_iou_zero_width_returns_zero():
    assert _iou_1d(0.5, 0.0, 0.5, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _ap_from_tp_fp
# ---------------------------------------------------------------------------

def test_ap_no_gt_returns_nan():
    assert math.isnan(_ap_from_tp_fp(np.array([1.0]), np.array([0.0]), 0))

def test_ap_all_tp():
    ap = _ap_from_tp_fp(np.array([1.0, 1.0]), np.array([0.0, 0.0]), n_gt=2)
    assert ap == pytest.approx(1.0)

def test_ap_all_fp():
    ap = _ap_from_tp_fp(np.array([0.0, 0.0]), np.array([1.0, 1.0]), n_gt=2)
    assert ap == pytest.approx(0.0)

def test_ap_tp_then_fp():
    # Rank 1: TP (precision=1, recall=0.5), Rank 2: FP (recall still 0.5)
    # AP = Δrecall × precision = 0.5 × 1.0 = 0.5
    ap = _ap_from_tp_fp(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n_gt=2)
    assert ap == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# map_iou
# ---------------------------------------------------------------------------

def test_map_empty_y_true_returns_nan():
    assert math.isnan(map_iou([], []))

def test_map_no_predictions_returns_zero():
    assert map_iou([_evt(0.0, 0.5, 1)], [_no_evts(1)]) == pytest.approx(0.0)

def test_map_no_gt_events_returns_nan():
    # Series has no GT events — AP is undefined
    assert math.isnan(map_iou([_no_evts(1)], [_evt(0.0, 0.5, 0.9)]))

def test_map_perfect_match():
    assert map_iou([_evt(0.1, 0.4, 1)], [_evt(0.1, 0.4, 1.0)]) == pytest.approx(1.0)

def test_map_no_overlap():
    assert map_iou([_evt(0.0, 0.3, 1)], [_evt(0.7, 0.3, 1.0)]) == pytest.approx(0.0)

def test_map_overlap_below_threshold_is_miss():
    # IoU ≈ 0.333 < 0.5
    assert map_iou([_evt(0.0, 1.0, 1)], [_evt(0.5, 1.0, 1.0)], iou_threshold=0.5) == pytest.approx(0.0)

def test_map_overlap_above_custom_threshold_is_hit():
    # Same IoU ≈ 0.333 > 0.3
    assert map_iou([_evt(0.0, 1.0, 1)], [_evt(0.5, 1.0, 1.0)], iou_threshold=0.3) == pytest.approx(1.0)

def test_map_duplicate_pred_only_first_matched():
    # Two identical predictions for one GT: first TP, second FP → AP=1.0
    preds = [np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.9]])]
    assert map_iou([_evt(0.0, 0.5, 1)], preds) == pytest.approx(1.0)

def test_map_predictions_ranked_by_score():
    # High-score pred misses (FP at rank 1), low-score pred hits (TP at rank 2)
    # AP = Δrecall × precision at rank 2 = 1.0 × 0.5 = 0.5
    preds = [np.array([[0.8, 0.1, 0.9], [0.0, 0.5, 0.5]])]
    assert map_iou([_evt(0.0, 0.5, 1)], preds) == pytest.approx(0.5)

def test_map_two_classes_both_perfect():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([[0.0, 0.3, 1.0, 0.0], [0.5, 0.3, 0.0, 1.0]])
    assert map_iou([gt], [pred]) == pytest.approx(1.0)

def test_map_two_classes_one_missed():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([
        [0.0, 0.3, 1.0, 0.0],   # class 0: perfect hit
        [0.0, 0.1, 0.0, 1.0],   # class 1: no overlap with GT at [0.5, 0.3]
    ])
    assert map_iou([gt], [pred]) == pytest.approx(0.5)

def test_map_multi_series_both_matched():
    y_true = [_evt(0.0, 0.5, 1), _evt(0.2, 0.4, 1)]
    y_pred = [_evt(0.0, 0.5, 1.0), _evt(0.2, 0.4, 1.0)]
    assert map_iou(y_true, y_pred) == pytest.approx(1.0)

def test_map_prediction_in_wrong_series_is_fp():
    # GT in series 0, pred only in series 1 — should not match
    y_true = [_evt(0.0, 0.5, 1), _no_evts(1)]
    y_pred = [_no_evts(1), _evt(0.0, 0.5, 1.0)]
    assert map_iou(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# f1_det
# ---------------------------------------------------------------------------

def test_f1det_empty_y_true_returns_nan():
    assert math.isnan(f1_det([], []))

def test_f1det_no_gt_events_returns_nan():
    assert math.isnan(f1_det([_no_evts(1)], [_evt(0.0, 0.5, 0.9)]))

def test_f1det_perfect_match_fixed_threshold():
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.9)], score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_no_overlap_fixed_threshold():
    # TP=0, FP=1, FN=1 → F1=0
    assert f1_det([_evt(0.0, 0.3, 1)], [_evt(0.7, 0.3, 0.9)], score_threshold=0.5) == pytest.approx(0.0)

def test_f1det_prediction_below_threshold_becomes_fn():
    # Pred score 0.3 < threshold 0.5 → filtered out → TP=0, FP=0, FN=1 → F1=0
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.3)], score_threshold=0.5) == pytest.approx(0.0)

def test_f1det_oracle_finds_best_threshold():
    # Low-score pred (0.2) would be filtered by fixed threshold=0.5 but oracle should find it
    assert f1_det([_evt(0.0, 0.5, 1)], [_evt(0.0, 0.5, 0.2)], score_threshold=None) == pytest.approx(1.0)

def test_f1det_duplicate_pred_second_is_fp():
    # TP=1, FP=1, FN=0 → F1 = 2/(2+1+0) = 2/3
    preds = [np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.9]])]
    assert f1_det([_evt(0.0, 0.5, 1)], preds, score_threshold=0.5) == pytest.approx(2 / 3)

def test_f1det_two_classes_both_perfect():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([[0.0, 0.3, 1.0, 0.0], [0.5, 0.3, 0.0, 1.0]])
    assert f1_det([gt], [pred], score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_two_classes_one_missed():
    gt = np.array([[0.0, 0.3, 1, 0], [0.5, 0.3, 0, 1]])
    pred = np.array([
        [0.0, 0.3, 1.0, 0.0],  # class 0: perfect hit
        [0.0, 0.1, 0.0, 0.9],  # class 1: no overlap with GT at [0.5, 0.3]
    ])
    # class 0: F1=1.0, class 1: TP=0 FP=1 FN=1 → F1=0 → mean=0.5
    assert f1_det([gt], [pred], score_threshold=0.5) == pytest.approx(0.5)

def test_f1det_multi_series_both_matched():
    y_true = [_evt(0.0, 0.5, 1), _evt(0.2, 0.4, 1)]
    y_pred = [_evt(0.0, 0.5, 0.9), _evt(0.2, 0.4, 0.8)]
    assert f1_det(y_true, y_pred, score_threshold=0.5) == pytest.approx(1.0)

def test_f1det_prediction_in_wrong_series_is_fp():
    y_true = [_evt(0.0, 0.5, 1), _no_evts(1)]
    y_pred = [_no_evts(1), _evt(0.0, 0.5, 0.9)]
    # TP=0, FP=1, FN=1 → F1=0
    assert f1_det(y_true, y_pred, score_threshold=0.5) == pytest.approx(0.0)
