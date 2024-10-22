import json
import argparse
import os
import shutil


def update_coco_ids_and_rename_images(input_file, output_file, image_dir):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create mappings for old to new IDs
    image_id_map = {}
    annotation_id_map = {}

    # Update image IDs and filenames
    for new_id, image in enumerate(data['images']):
        old_id = image['id']
        image_id_map[old_id] = new_id
        image['id'] = new_id

        # Keep the original filename and extension
        original_file_name = image['file_name']
        file_extension = os.path.splitext(original_file_name)[1]  # Get file extension (e.g., .jpg, .png)

        # New file name based on image ID
        new_file_name = f"{new_id}{file_extension}"

        # Rename the image file on disk
        old_image_path = os.path.join(image_dir, original_file_name)
        new_image_path = os.path.join(image_dir, new_file_name)
        shutil.move(old_image_path, new_image_path)

        # Update the file_name in the JSON
        image['file_name'] = new_file_name

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
    print("Images have been renamed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Update COCO JSON file IDs, filenames, and rename image files")
    parser.add_argument('input', help="Path to the input COCO JSON file")
    parser.add_argument('output', help="Path to save the output COCO JSON file")
    parser.add_argument('image_dir', help="Directory where the images are stored")

    args = parser.parse_args()

    update_coco_ids_and_rename_images(args.input, args.output, args.image_dir)


if __name__ == "__main__":
    main()

