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
            'score': annotation.get('score', 1.0),  # Default score to 1.0 if not present
            'bbox': annotation['bbox'],
            'area': annotation.get('area', annotation['bbox'][2] * annotation['bbox'][3]),  # Calculate area if not present
            'id': len(predictions_coco_format['annotations']),
        }
        predictions_coco_format['annotations'].append(prediction)

    return predictions_coco_format

def calculate_map(predictions_file, ground_truth_file):
    """Calculate mAP."""
    coco_gt = COCO(ground_truth_file)
    
    # Load predictions and convert to COCO format
    predictions = load_json(predictions_file)
    coco_dt = COCO()
    coco_dt.dataset = convert_annotations_to_coco_format(predictions)
    coco_dt.createIndex()

    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

def plot_precision_recall(coco_eval, categories):
    """Plot Precision-Recall curve."""
    precisions = coco_eval.eval['precision']
    
    # Plotting
    plt.figure(figsize=(10, 8))
    for idx, cat in enumerate(categories):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        recall = np.linspace(0, 1, num=precision.size)
        
        if precision.size > 0:
            plt.plot(recall, precision, label=cat['name'])
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
        print(f"Annotation {i+1}:")
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
    coco_dt = COCO()
    coco_dt.dataset = convert_annotations_to_coco_format(predictions)
    coco_dt.createIndex()
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
