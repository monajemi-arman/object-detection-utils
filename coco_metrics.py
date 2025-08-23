import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import tempfile
import csv
import os
from contextlib import redirect_stdout
from io import StringIO
from tabulate import tabulate


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def filter_predictions(pred_anns, gt_img_ids):
    """Filter predictions to only include images present in ground truth."""
    return [ann for ann in pred_anns if int(ann["image_id"]) in gt_img_ids]


def validate_bbox_format(bbox, source="unknown"):
    """Check if bounding box is in [x, y, w, h] format and has valid values."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox format in {source}: {bbox}")
    x, y, w, h = bbox
    if any(v < 0 for v in [x, y, w, h]):
        raise ValueError(f"Negative values in bbox from {source}: {bbox}")
    return True


def compute_per_image_metrics(
    coco_gt, coco_pred, img_id, iou_threshold, compute_agnostic=False
):
    """Compute metrics for a single image, optionally class-agnostic."""
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

    # Initialize metrics
    TP = 0
    FP = 0
    FN = 0
    class_error = 0
    iou_list = []  # <-- track IoUs

    # Sort predictions by confidence (descending)
    pred_anns_sorted = sorted(
        pred_anns, key=lambda x: x.get("score", 1.0), reverse=True
    )

    # Track which GTs have at least one match
    gt_matched = {gt["id"]: False for gt in gt_anns}

    for pred in pred_anns_sorted:
        best_iou = 0.0
        best_gt = None
        for gt in gt_anns:
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        iou_list.append(best_iou)  # log best IoU for this prediction

        if best_iou >= iou_threshold:
            if compute_agnostic or pred["category_id"] == best_gt["category_id"]:
                if not gt_matched[best_gt["id"]]:
                    TP += 1
                    gt_matched[best_gt["id"]] = True  # mark GT as detected
                else:
                    FP += 1  # duplicate detection of same GT
            else:
                # class mismatch → still count GT as detected, but log class error
                if not gt_matched[best_gt["id"]]:
                    gt_matched[best_gt["id"]] = True
                    class_error += 1
                    TP += 1
                else:
                    FP += 1
        else:
            FP += 1

    # Any GT never matched at all → FN
    FN = sum(1 for matched in gt_matched.values() if not matched)

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    cls_accuracy = TP / (TP + class_error) if (TP + class_error) > 0 else 0
    detection_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Compute mAP for this image
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.params.imgIds = [img_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_50 = coco_eval.stats[0] if coco_eval.stats[0] >= 0 else 0.0
    map_75 = coco_eval.stats[1] if coco_eval.stats[1] >= 0 else 0.0

    return {
        "image_id": img_id,
        "gt_count": len(gt_anns),
        "pred_count": len(pred_anns),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "class_error": class_error,
        "precision": precision,
        "recall": recall,
        "cls_accuracy": cls_accuracy,
        "detection_accuracy": detection_accuracy,
        "mAP_50": map_50,
        "mAP_75": map_75,
        "ious": iou_list,
    }


def write_csv_results(results, csv_path):
    """Write per-image results to a CSV file."""
    headers = [
        "image_id",
        "gt_count",
        "pred_count",
        "TP",
        "FP",
        "FN",
        "class_error",
        "precision",
        "recall",
        "cls_accuracy",
        "detection_accuracy",
        "mAP_50",
        "mAP_75",
        "ious",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            row = result.copy()
            # Convert IoUs list into a string like "0.712;0.423;0.000"
            row["ious"] = ";".join([f"{x:.3f}" for x in result.get("ious", [])])
            writer.writerow(row)


def silent_coco_summarize(coco_eval):
    """Run COCOeval.summarize() without printing output."""
    buf = StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    return buf.getvalue()


def main(args):
    # Load ground truth
    coco_gt = COCO(args.labels)
    gt_img_ids = set(coco_gt.getImgIds())
    print(f"Number of ground truth images: {len(gt_img_ids)}")
    print(f"Number of ground truth annotations: {len(coco_gt.getAnnIds())}")

    # Load predictions and filter
    with open(args.predictions, "r") as f:
        pred_data = json.load(f)

    if not isinstance(pred_data, dict) or "annotations" not in pred_data:
        raise ValueError("Predictions must be in COCO format with 'annotations' key")

    # Filter predictions to only include images present in ground truth
    filtered_preds = filter_predictions(pred_data["annotations"], gt_img_ids)
    print(f"Total predictions: {len(pred_data['annotations'])}")
    print(f"Predictions after image ID filtering: {len(filtered_preds)}")

    # Validate bounding box format for ground truth and predictions
    for ann in coco_gt.dataset["annotations"]:
        validate_bbox_format(ann["bbox"], "ground truth")
    for ann in filtered_preds:
        validate_bbox_format(ann["bbox"], "predictions")

    # Filter out low-confidence predictions
    filtered_preds = [
        ann for ann in filtered_preds if ann.get("score", 1.0) >= args.confidence
    ]
    print(
        f"Predictions after confidence filtering (>= {args.confidence}): {len(filtered_preds)}"
    )

    if not filtered_preds:
        print(
            "No valid predictions found that match ground truth images or pass confidence threshold."
        )
        return

    # Load filtered predictions into COCO API
    coco_pred = coco_gt.loadRes(filtered_preds)
    print(f"Number of predicted annotations loaded: {len(coco_pred.getAnnIds())}")

    # Compute per-image metrics
    per_image_results = []
    per_image_agnostic_results = []

    # Initialize aggregate metrics
    TP = 0
    FP = 0
    FN = 0
    class_error = 0
    TP_agnostic = 0
    FP_agnostic = 0
    FN_agnostic = 0

    for img_id in coco_gt.getImgIds():
        # Standard metrics
        result = compute_per_image_metrics(
            coco_gt, coco_pred, img_id, args.iou_threshold
        )
        per_image_results.append(result)
        TP += result["TP"]
        FP += result["FP"]
        FN += result["FN"]
        class_error += result["class_error"]

        # Class-agnostic metrics
        result_agnostic = compute_per_image_metrics(
            coco_gt, coco_pred, img_id, args.iou_threshold, compute_agnostic=True
        )
        per_image_agnostic_results.append(
            {**result_agnostic, "image_id": f"{img_id}_agnostic"}
        )
        TP_agnostic += result_agnostic["TP"]
        FP_agnostic += result_agnostic["FP"]
        FN_agnostic += result_agnostic["FN"]

    # Write per-image results to CSV if requested
    if args.csv:
        write_csv_results(per_image_results + per_image_agnostic_results, args.csv)
        print(f"Per-image results written to {args.csv}")

    # Calculate aggregate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    cls_accuracy = TP / (TP + class_error) if (TP + class_error) > 0 else 0
    detection_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Compute aggregate mAP
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    silent_coco_summarize(coco_eval)
    map_50 = coco_eval.stats[0] if coco_eval.stats[0] >= 0 else 0.0
    map_75 = coco_eval.stats[1] if coco_eval.stats[1] >= 0 else 0.0

    # Prepare aggregate results for table
    table_data = [
        ["mAP@0.5", f"{map_50:.4f}"],
        ["mAP@0.75", f"{map_75:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["Classification Accuracy", f"{cls_accuracy:.4f}"],
        ["Detection Accuracy", f"{detection_accuracy:.4f}"],
        ["True Positives (TP)", TP],
        ["False Positives (FP)", FP],
        ["False Negatives (FN)", FN],
        ["Classification Errors", class_error],
    ]

    # --- Class-agnostic aggregate metrics ---
    precision_agnostic = (
        TP_agnostic / (TP_agnostic + FP_agnostic)
        if (TP_agnostic + FP_agnostic) > 0
        else 0
    )
    recall_agnostic = (
        TP_agnostic / (TP_agnostic + FN_agnostic)
        if (TP_agnostic + FN_agnostic) > 0
        else 0
    )
    detection_accuracy_agnostic = (
        TP_agnostic / (TP_agnostic + FP_agnostic + FN_agnostic)
        if (TP_agnostic + FP_agnostic + FN_agnostic) > 0
        else 0
    )

    # --- Class-agnostic mAP ---
    def map_to_one_category(anns):
        anns_new = []
        for ann in anns:
            ann_new = ann.copy()
            ann_new["category_id"] = 1
            anns_new.append(ann_new)
        return anns_new

    # Prepare new GT and prediction dicts
    gt_dict = coco_gt.dataset.copy()
    gt_dict["annotations"] = map_to_one_category(coco_gt.dataset["annotations"])
    gt_dict["categories"] = [{"id": 1, "name": "object"}]
    print("Ground truth dataset keys:", gt_dict.keys())

    # Build new COCO objects using tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as gt_temp:
        json.dump(gt_dict, gt_temp)
        gt_temp.flush()
        coco_gt_agnostic = COCO(gt_temp.name)

    pred_anns = map_to_one_category(filtered_preds)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as pred_temp:
        json.dump(pred_anns, pred_temp)
        pred_temp.flush()
        coco_pred_agnostic = coco_gt_agnostic.loadRes(pred_temp.name)

    coco_eval_agnostic = COCOeval(coco_gt_agnostic, coco_pred_agnostic, "bbox")
    coco_eval_agnostic.evaluate()
    coco_eval_agnostic.accumulate()
    silent_coco_summarize(coco_eval_agnostic)
    map_50_agnostic = (
        coco_eval_agnostic.stats[0] if coco_eval_agnostic.stats[0] >= 0 else 0.0
    )
    map_75_agnostic = (
        coco_eval_agnostic.stats[1] if coco_eval_agnostic.stats[1] >= 0 else 0.0
    )

    table_data_agnostic = [
        ["Class-agnostic mAP@0.5", f"{map_50_agnostic:.4f}"],
        ["Class-agnostic mAP@0.75", f"{map_75_agnostic:.4f}"],
        ["Class-agnostic Precision", f"{precision_agnostic:.4f}"],
        ["Class-agnostic Recall", f"{recall_agnostic:.4f}"],
        ["Class-agnostic Detection Accuracy", f"{detection_accuracy_agnostic:.4f}"],
    ]

    print("\nAggregate Evaluation Metrics:")
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="github"))
    print("\nAggregate Class-agnostic Detection Metrics (ignore class):")
    print(tabulate(table_data_agnostic, headers=["Metric", "Value"], tablefmt="github"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate object detection metrics from COCO format files."
    )
    parser.add_argument(
        "-l", "--labels", required=True, help="Path to ground truth JSON (COCO format)"
    )
    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        help="Path to predictions JSON (COCO format)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for custom metrics (default: 0.5)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to output CSV file for per-image results (optional)",
    )
    args = parser.parse_args()
    main(args)
