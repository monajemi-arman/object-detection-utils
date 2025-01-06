import json
from collections import defaultdict

def validate_coco_annotations(file_path):
    """
    Validates a COCO-style dataset JSON annotations file.

    Args:
        file_path (str): Path to the COCO annotations file.

    Returns:
        dict: Contains errors and notices categorized by their type.
    """
    REQUIRED_KEYS = {
        "info": ["year", "version", "description", "contributor", "url", "date_created"],
        "licenses": ["id", "name", "url"],
        "images": ["id", "width", "height", "file_name", "license", "date_captured"],
        "annotations": ["id", "image_id", "category_id", "bbox", "area", "iscrowd"],
        "categories": ["id", "name", "supercategory"],
    }

    OPTIONAL_KEYS = {
        "info": [],
        "licenses": [],
        "images": ["flickr_url", "coco_url"],
        "annotations": ["segmentation"],
        "categories": ["keypoints", "skeleton"],
    }

    results = {
        "errors": defaultdict(set),
        "notices": defaultdict(set),
    }

    try:
        with open(file_path, "r") as file:
            coco_data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        results["errors"]["file"].add(f"[ ! ] Error reading file: {e}")
        return results

    for section, keys in REQUIRED_KEYS.items():
        if section not in coco_data:
            results["errors"]["structure"].add(f"[ ! ] Missing required section: {section}")
            continue

        for item in coco_data.get(section, []):
            if not isinstance(item, dict):
                results["errors"]["structure"].add(f"[ ! ] Items in '{section}' must be dictionaries.")
                continue

            for key in keys:
                if key not in item:
                    results["errors"]["fields"].add(f"[ ! ] Missing required key '{key}' in section '{section}'.")

            for opt_key in OPTIONAL_KEYS.get(section, []):
                if opt_key not in item:
                    results["notices"]["fields"].add(f"[ * ] Optional key '{opt_key}' missing in section '{section}'.")

    # Additional checks
    image_ids = {img["id"] for img in coco_data.get("images", []) if "id" in img}
    annotation_image_ids = {ann["image_id"] for ann in coco_data.get("annotations", []) if "image_id" in ann}

    # Check for orphan annotations
    for ann in coco_data.get("annotations", []):
        if ann["image_id"] not in image_ids:
            results["errors"]["annotations"].add(
                f"[ ! ] Annotation {ann.get('id')} references missing image_id {ann.get('image_id')}.")

    # Check for duplicate image IDs
    if len(image_ids) != len(coco_data.get("images", [])):
        results["errors"]["images"].add("[ ! ] Duplicate image IDs detected.")

    # Check for duplicate annotation IDs
    annotation_ids = [ann.get("id") for ann in coco_data.get("annotations", [])]
    if len(annotation_ids) != len(set(annotation_ids)):
        results["errors"]["annotations"].add("[ ! ] Duplicate annotation IDs detected.")

    # Convert sets to lists for consistent output
    results["errors"] = {k: list(v) for k, v in results["errors"].items()}
    results["notices"] = {k: list(v) for k, v in results["notices"].items()}

    return results

def display_results(results):
    """Displays the validation results."""
    print("Validation Results:\n")

    for category, messages in results["errors"].items():
        print(f"Errors in {category}:")
        for message in messages:
            print(f"  {message}")

    for category, messages in results["notices"].items():
        print(f"Notices in {category}:")
        for message in messages:
            print(f"  {message}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a COCO-style dataset JSON annotations file.")
    parser.add_argument("file", help="Path to the COCO annotations file.")
    args = parser.parse_args()

    results = validate_coco_annotations(args.file)
    display_results(results)

