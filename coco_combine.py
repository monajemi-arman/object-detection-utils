import json
import sys

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def combine_coco_files(file_paths, output_file):
    combined = {"images": [], "annotations": [], "categories": []}
    category_name_to_id = {}  # Maps category name to new ID
    category_id_mapping = {}  # Maps old category ID to new ID
    next_category_id = 1
    next_image_id = 1
    next_annotation_id = 1

    for file_path in file_paths:
        data = load_json(file_path)
        image_id_map = {}  # Maps original image IDs to new IDs for the current file

        # Process images to assign new unique IDs
        for image in data["images"]:
            original_id = image["id"]
            new_image = image.copy()
            new_image["id"] = next_image_id
            image_id_map[original_id] = next_image_id
            combined["images"].append(new_image)
            next_image_id += 1

        # Process categories to merge by name and create ID mappings
        for category in data["categories"]:
            name = category["name"]
            old_id = category["id"]
            if name not in category_name_to_id:
                category_name_to_id[name] = next_category_id
                combined["categories"].append({
                    "id": next_category_id,
                    "name": name,
                    "supercategory": category.get("supercategory", "")
                })
                category_id_mapping[old_id] = next_category_id
                next_category_id += 1
            else:
                # If category exists, map the old ID to the existing new ID
                category_id_mapping[old_id] = category_name_to_id[name]

        # Process annotations to update image_id, category_id, and assign new IDs
        for annotation in data["annotations"]:
            new_ann = annotation.copy()
            new_ann["id"] = next_annotation_id
            new_ann["image_id"] = image_id_map[annotation["image_id"]]
            new_ann["category_id"] = category_id_mapping[annotation["category_id"]]
            combined["annotations"].append(new_ann)
            next_annotation_id += 1

    save_json(combined, output_file)
    print(f"Combined COCO JSON saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_file.json> <input1.json> <input2.json> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    combine_coco_files(input_files, output_file)