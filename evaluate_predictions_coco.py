import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

# ** Notice **
# * Annotation and image ids must start from 0 and go up consecutively
# * Annotations must have "area"
# Scripts are given to correct the above if not present in your custom COCO JSON

def load_json(file_path):
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)

def convert_annotations_to_coco_format(coco_json):
    """Convert COCO-style annotations to a complete predictions COCO format."""
    predictions_coco_format = {
        'images': coco_json['images'],
        'annotations': [],
        'categories': coco_json['categories'],
    }

    for annotation in coco_json['annotations']:
        prediction = {
            'image_id': annotation['image_id'],
            'category_id': annotation['category_id'],
            'score': annotation.get('score', 0.0),
            'bbox': annotation['bbox'],
            'id': len(predictions_coco_format['annotations']),
        }
        predictions_coco_format['annotations'].append(prediction)

    return predictions_coco_format


def calculate_map(predictions_file, ground_truth_file):
    """Calculate mAP."""
    coco_gt = COCO(ground_truth_file)
    coco_dt = COCO(predictions_file)

    # Convert predictions to COCO format for evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]


def plot_precision_recall(predictions_file, ground_truth_file):
    """Plot Precision-Recall curve."""
    coco_gt = COCO(ground_truth_file)
    coco_json = load_json(predictions_file)
    predictions = convert_annotations_to_coco_format(coco_json)

    scores = np.array([ann['score'] for ann in predictions['annotations']])
    thresholds = np.linspace(0.0, 1.0, 101)
    precision, recall = [], []

    for t in thresholds:
        tp = sum(score >= t for score in scores)
        fp = len(scores) - tp
        fn = len(coco_gt.anns) - tp
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='o', linestyle='-', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


def main(predictions_file, ground_truth_file):
    mAP = calculate_map(predictions_file, ground_truth_file)
    print(f'mAP: {mAP:.4f}')
    plot_precision_recall(predictions_file, ground_truth_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate COCO-style predictions.')
    parser.add_argument('predictions', type=str, help='Path to predictions JSON file')
    parser.add_argument('ground_truth', type=str, help='Path to ground truth JSON file')
    args = parser.parse_args()

    main(args.predictions, args.ground_truth)