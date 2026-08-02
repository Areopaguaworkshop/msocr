import pytest

from msocr.evaluation.layout_metrics import evaluate_layout


def test_layout_metrics_report_detection_order_rows_and_gaps():
    ground_truth = {
        "lines": [
            {"id": "g1", "baseline": [[100, 10], [50, 10]], "rowId": "row-1"},
            {"id": "g2", "baseline": [[40, 10], [10, 10]], "rowId": "row-1"},
            {"id": "g3", "baseline": [[100, 40], [10, 40]], "rowId": "row-2"},
        ],
        "gaps": [{"polygon": [[42, 0], [48, 0], [48, 20], [42, 20]]}],
    }
    prediction = {
        "lines": [
            {"id": "p1", "baseline": [[100, 11], [50, 11]], "rowId": "pred-row-1"},
            {"id": "p2", "baseline": [[40, 11], [10, 11]], "rowId": "pred-row-1"},
            {"id": "p3", "baseline": [[100, 41], [10, 41]], "rowId": "pred-row-2"},
        ],
        "gaps": [{"polygon": [[42, 0], [48, 0], [48, 20], [42, 20]]}],
    }

    result = evaluate_layout(ground_truth, prediction, distance_tolerance=3)

    assert result["baseline_f1"] == 1.0
    assert result["reading_order_pair_accuracy"] == 1.0
    assert result["row_grouping"]["f1"] == 1.0
    assert result["gap_localization"]["mean_iou"] == 1.0
    assert result["split_count"] == 0
    assert result["merge_count"] == 0


def test_layout_metrics_expose_split_without_counting_full_match():
    ground_truth = {
        "lines": [{"id": "g", "baseline": [[0, 10], [100, 10]]}],
        "gaps": [],
    }
    prediction = {
        "lines": [
            {"id": "left", "baseline": [[0, 10], [40, 10]]},
            {"id": "right", "baseline": [[60, 10], [100, 10]]},
        ],
        "gaps": [],
    }

    result = evaluate_layout(ground_truth, prediction, distance_tolerance=3)

    assert result["matched_lines"] == 0
    assert result["split_count"] == 1
    assert result["baseline_recall"] == 0.0
    assert result["gap_localization"]["f1"] == pytest.approx(1.0)
