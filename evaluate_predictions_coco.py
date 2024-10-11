import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def load_json(file_path):
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def calculate_map(predictions_file, ground_truth_file):
    """Calculate mAP using COCOeval."""
    coco_gt = COCO(ground_truth_file)

    # Load predictions and convert to COCO format using loadRes
    predictions = load_json(predictions_file)
    coco_dt = coco_gt.loadRes(predictions['annotations'])

    # Run COCO evaluation per class
    for catId in coco_gt.getCatIds():
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.catIds = [catId]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    # Run COCO evaluation overall
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def plot_precision_recall(coco_eval, categories):
    """Plot Precision-Recall as an x-y coordinate chart."""
    precisions = coco_eval.eval['precision']

    # Create the plot for x (recall) and y (precision) coordinates
    plt.figure(figsize=(10, 8))

    for idx, cat in enumerate(categories):
        precision = precisions[:, :, idx, 0, -1]  # Take slice for IoU threshold of 0.5
        precision = precision[precision > -1]  # Filter out invalid precision values
        recall = np.linspace(0, 1, num=precision.size)

        if precision.size > 0:
            # Plot recall (x-axis) vs precision (y-axis) with markers
            plt.plot(recall, precision, marker='o', label=cat['name'], linestyle='-', markersize=5)
        else:
            print(f"Warning: No valid precision-recall data for category {cat['name']}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


def print_dataset_stats(coco):
    """Print statistics about the dataset."""
    print("\nDataset Statistics:")
    print(f"Number of images: {len(coco.imgs)}")
    print(f"Number of categories: {len(coco.cats)}")
    print(f"Number of annotations: {len(coco.anns)}")

    print("\nCategories:")
    for cat in coco.loadCats(coco.getCatIds()):
        print(f"- {cat['name']} (id: {cat['id']}): {len(coco.getAnnIds(catIds=cat['id']))} annotations")


def print_sample_annotations(coco, num_samples=5):
    """Print sample annotations for debugging."""
    print("\nSample Annotations:")
    for i, ann_id in enumerate(list(coco.anns.keys())[:num_samples]):
        ann = coco.anns[ann_id]
        print(f"Annotation {i + 1}:")
        print(f"  Image ID: {ann['image_id']}")
        print(f"  Category ID: {ann['category_id']}")
        print(f"  Bbox: {ann['bbox']}")
        print(f"  Area: {ann['area']}")
        if 'score' in ann:
            print(f"  Score: {ann['score']}")
        print()


def main(predictions_file, ground_truth_file):
    print("Loading ground truth...")
    coco_gt = COCO(ground_truth_file)
    print_dataset_stats(coco_gt)
    print_sample_annotations(coco_gt)

    print("\nLoading predictions...")
    predictions = load_json(predictions_file)
    coco_dt = coco_gt.loadRes(predictions['annotations'])
    print_dataset_stats(coco_dt)
    print_sample_annotations(coco_dt)

    coco_eval = calculate_map(predictions_file, ground_truth_file)
    print(f'\nmAP: {coco_eval.stats[0]:.4f}')

    plot_precision_recall(coco_eval, coco_gt.loadCats(coco_gt.getCatIds()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate COCO-style predictions.')
    parser.add_argument('predictions', type=str, help='Path to predictions JSON file')
    parser.add_argument('ground_truth', type=str, help='Path to ground truth JSON file')
    args = parser.parse_args()

    main(args.predictions, args.ground_truth)
