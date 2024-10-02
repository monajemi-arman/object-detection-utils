import json
import argparse


def update_coco_ids(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create mappings for old to new IDs
    image_id_map = {}
    annotation_id_map = {}

    # Update image IDs
    for new_id, image in enumerate(data['images']):
        old_id = image['id']
        image_id_map[old_id] = new_id
        image['id'] = new_id

    # Update annotation IDs and image_id references
    for new_id, annotation in enumerate(data['annotations']):
        old_id = annotation['id']
        annotation_id_map[old_id] = new_id
        annotation['id'] = new_id

        # Update image_id reference
        old_image_id = annotation['image_id']
        annotation['image_id'] = image_id_map[old_image_id]

    # Write the updated data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated COCO JSON file has been saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Update COCO JSON file IDs")
    parser.add_argument('input', help="Path to the input COCO JSON file")
    parser.add_argument('output', help="Path to save the output COCO JSON file")

    args = parser.parse_args()

    update_coco_ids(args.input, args.output)


if __name__ == "__main__":
    main()