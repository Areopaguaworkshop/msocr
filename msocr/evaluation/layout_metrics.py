"""Model-neutral layout metrics for BLLA/Orli segmentation JSON."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _sample_polyline(points: list[list[float]], spacing: float = 5.0) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1:
        return np.asarray(points, dtype=float)
    samples: list[np.ndarray] = []
    for raw_start, raw_end in zip(points, points[1:]):
        start = np.asarray(raw_start, dtype=float)
        end = np.asarray(raw_end, dtype=float)
        steps = max(1, math.ceil(float(np.linalg.norm(end - start)) / spacing))
        samples.extend(
            start + (end - start) * fraction
            for fraction in np.linspace(0, 1, steps, endpoint=False)
        )
    samples.append(np.asarray(points[-1], dtype=float))
    return np.asarray(samples)


def _directed_distance(source: np.ndarray, target: np.ndarray) -> float:
    if not len(source) or not len(target):
        return math.inf
    distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def _coverage(source: np.ndarray, target: np.ndarray, tolerance: float) -> float:
    if not len(source) or not len(target):
        return 0.0
    distances = np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1) <= tolerance))


def _pairwise_order_accuracy(matches: list[tuple[int, int]]) -> float | None:
    if len(matches) < 2:
        return None
    correct = 0
    total = 0
    for index, (gt_a, pred_a) in enumerate(matches):
        for gt_b, pred_b in matches[index + 1 :]:
            correct += (gt_a - gt_b) * (pred_a - pred_b) > 0
            total += 1
    return correct / total if total else None


def _row_grouping_scores(
    gt_lines: list[dict[str, Any]],
    pred_lines: list[dict[str, Any]],
    matches: list[tuple[int, int]],
) -> dict[str, float] | None:
    if not matches or not any(pred_lines[pred].get("rowId") for _, pred in matches):
        return None
    true_positive = false_positive = false_negative = 0
    for index, (gt_a, pred_a) in enumerate(matches):
        for gt_b, pred_b in matches[index + 1 :]:
            gt_same = bool(gt_lines[gt_a].get("rowId")) and (
                gt_lines[gt_a].get("rowId") == gt_lines[gt_b].get("rowId")
            )
            pred_same = bool(pred_lines[pred_a].get("rowId")) and (
                pred_lines[pred_a].get("rowId") == pred_lines[pred_b].get("rowId")
            )
            if gt_same and pred_same:
                true_positive += 1
            elif pred_same:
                false_positive += 1
            elif gt_same:
                false_negative += 1
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


def _polygon_iou(first: list[list[int]], second: list[list[int]]) -> float:
    if len(first) < 3 or len(second) < 3:
        return 0.0
    all_points = np.asarray(first + second, dtype=np.int32)
    left, top = all_points.min(axis=0)
    right, bottom = all_points.max(axis=0)
    shape = (int(bottom - top + 3), int(right - left + 3))
    first_mask = np.zeros(shape, dtype=np.uint8)
    second_mask = np.zeros(shape, dtype=np.uint8)
    offset = np.asarray([left - 1, top - 1], dtype=np.int32)
    cv2.fillPoly(first_mask, [np.asarray(first, dtype=np.int32) - offset], 1)
    cv2.fillPoly(second_mask, [np.asarray(second, dtype=np.int32) - offset], 1)
    union = np.count_nonzero(first_mask | second_mask)
    return np.count_nonzero(first_mask & second_mask) / union if union else 0.0


def _gap_scores(
    gt_gaps: list[dict[str, Any]], pred_gaps: list[dict[str, Any]], threshold: float
) -> dict[str, float | int]:
    candidates = sorted(
        (
            (
                _polygon_iou(gt.get("polygon") or [], pred.get("polygon") or []),
                gt_index,
                pred_index,
            )
            for gt_index, gt in enumerate(gt_gaps)
            for pred_index, pred in enumerate(pred_gaps)
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    ious: list[float] = []
    for iou, gt_index, pred_index in candidates:
        if iou < threshold or gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        ious.append(iou)
    precision = (
        len(ious) / len(pred_gaps) if pred_gaps else (1.0 if not gt_gaps else 0.0)
    )
    recall = len(ious) / len(gt_gaps) if gt_gaps else (1.0 if not pred_gaps else 0.0)
    return {
        "matched": len(ious),
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def evaluate_layout(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    *,
    distance_tolerance: float = 15.0,
    clipping_coverage: float = 0.9,
    gap_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare PAGE-derived GT with common BLLA/Orli segmentation JSON.

    Full-baseline matching is one-to-one and requires symmetric mean distance
    within the tolerance. Split/merge diagnostics use directed proximity so
    partial baselines are still counted as failure associations.
    """
    gt_lines = ground_truth.get("lines") or []
    pred_lines = prediction.get("lines") or []
    gt_samples = [_sample_polyline(line.get("baseline") or []) for line in gt_lines]
    pred_samples = [_sample_polyline(line.get("baseline") or []) for line in pred_lines]

    symmetric: dict[tuple[int, int], float] = {}
    gt_to_pred: dict[tuple[int, int], float] = {}
    pred_to_gt: dict[tuple[int, int], float] = {}
    for gt_index, gt in enumerate(gt_samples):
        for pred_index, pred in enumerate(pred_samples):
            forward = _directed_distance(gt, pred)
            reverse = _directed_distance(pred, gt)
            gt_to_pred[(gt_index, pred_index)] = forward
            pred_to_gt[(gt_index, pred_index)] = reverse
            symmetric[(gt_index, pred_index)] = max(forward, reverse)

    candidates = sorted(
        (distance, gt, pred)
        for (gt, pred), distance in symmetric.items()
        if distance <= distance_tolerance
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    coverages: list[float] = []
    for _, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index))
        coverages.append(
            _coverage(
                gt_samples[gt_index], pred_samples[pred_index], distance_tolerance
            )
        )
    precision = (
        len(matches) / len(pred_lines) if pred_lines else (1.0 if not gt_lines else 0.0)
    )
    recall = (
        len(matches) / len(gt_lines) if gt_lines else (1.0 if not pred_lines else 0.0)
    )
    split_count = sum(
        sum(
            pred_to_gt[(gt, pred)] <= distance_tolerance
            for pred in range(len(pred_lines))
        )
        > 1
        for gt in range(len(gt_lines))
    )
    merge_count = sum(
        sum(gt_to_pred[(gt, pred)] <= distance_tolerance for gt in range(len(gt_lines)))
        > 1
        for pred in range(len(pred_lines))
    )
    return {
        "ground_truth_lines": len(gt_lines),
        "predicted_lines": len(pred_lines),
        "matched_lines": len(matches),
        "baseline_precision": precision,
        "baseline_recall": recall,
        "baseline_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "split_count": split_count,
        "merge_count": merge_count,
        "mean_baseline_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "clipped_match_count": sum(
            coverage < clipping_coverage for coverage in coverages
        ),
        "reading_order_pair_accuracy": _pairwise_order_accuracy(matches),
        "row_grouping": _row_grouping_scores(gt_lines, pred_lines, matches),
        "gap_localization": _gap_scores(
            ground_truth.get("gaps") or [],
            prediction.get("gaps") or [],
            gap_iou_threshold,
        ),
        "matches": [
            {
                "ground_truth_id": gt_lines[gt].get("id"),
                "prediction_id": pred_lines[pred].get("id"),
                "coverage": coverage,
            }
            for (gt, pred), coverage in zip(matches, coverages)
        ],
    }
