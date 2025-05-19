import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

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
    return intersection_area / union_area if union_area > 0 else 0.0

def filter_predictions(pred_anns, gt_img_ids):
    """Filter predictions to only include images present in ground truth"""
    return [ann for ann in pred_anns if int(ann['image_id']) in gt_img_ids]

def main(args):
    # Load ground truth
    coco_gt = COCO(args.labels)
    gt_img_ids = set(coco_gt.getImgIds())

    # Load predictions and filter
    with open(args.predictions, 'r') as f:
        pred_data = json.load(f)
    
    if not isinstance(pred_data, dict) or 'annotations' not in pred_data:
        raise ValueError("Predictions must be in COCO format with 'annotations' key")
    
    # Filter predictions to only include images present in ground truth
    filtered_preds = filter_predictions(pred_data['annotations'], gt_img_ids)
    
    if not filtered_preds:
        raise ValueError("No valid predictions found that match ground truth images")
    
    # Load filtered predictions into COCO API
    coco_pred = coco_gt.loadRes(filtered_preds)

    # Initialize metrics
    TP = 0
    FP = 0
    FN = 0
    class_error = 0

    # Process each image
    for img_id in coco_gt.getImgIds():
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        # Sort predictions by confidence (descending)
        pred_anns_sorted = sorted(pred_anns, key=lambda x: x['score'], reverse=True)
        matched_gt = set()

        for pred in pred_anns_sorted:
            best_iou = 0.0
            best_gt = None
            for gt in gt_anns:
                if gt['id'] in matched_gt:
                    continue
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_iou >= 0.5:
                if pred['category_id'] == best_gt['category_id']:
                    TP += 1
                    matched_gt.add(best_gt['id'])
                else:
                    class_error += 1
                    FP += 1
            else:
                FP += 1

        # Count unmatched GTs as FN
        FN += len(gt_anns) - len(matched_gt)

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    cls_accuracy = TP / (TP + class_error) if (TP + class_error) > 0 else 0
    detection_accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Compute mAP using COCOeval
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_50 = coco_eval.stats[0]
    map_75 = coco_eval.stats[1]

    # Output results
    print("\nEvaluation Metrics:")
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.75: {map_75:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Classification Accuracy: {cls_accuracy:.4f}")
    print(f"Detection Accuracy: {detection_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate object detection metrics from COCO format files.')
    parser.add_argument('-l', '--labels', required=True, help='Path to ground truth JSON (COCO format)')
    parser.add_argument('-p', '--predictions', required=True, help='Path to predictions JSON (COCO format)')
    args = parser.parse_args()
    main(args)