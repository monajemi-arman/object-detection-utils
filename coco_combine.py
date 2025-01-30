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
    category_name_to_id, category_id_mapping = {}, {}
    next_category_id = 1

    for file_path in file_paths:
        data = load_json(file_path)
        combined["images"].extend(data["images"])

        # Merge categories and update ID mapping
        for category in data["categories"]:
            name = category["name"]
            if name not in category_name_to_id:
                category_name_to_id[name] = next_category_id
                category_id_mapping[category["id"]] = next_category_id
                combined["categories"].append({
                    "id": next_category_id, "name": name,
                    "supercategory": category.get("supercategory", "")
                })
                next_category_id += 1
            else:
                category_id_mapping[category["id"]] = category_name_to_id[name]

        # Update category IDs in annotations
        for annotation in data["annotations"]:
            annotation["category_id"] = category_id_mapping[annotation["category_id"]]
            combined["annotations"].append(annotation)

    save_json(combined, output_file)
    print(f"Combined COCO JSON saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_file.json> <input1.json> <input2.json> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    file_paths = sys.argv[2:]
    combine_coco_files(file_paths, output_file)
