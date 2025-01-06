import json
import argparse
import os
import shutil
import sys

def update_coco_ids_and_rename_images(input_file, output_file, image_dir):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create mappings for old to new IDs
    image_id_map = {}
    annotation_id_map = {}

    # Verify that image_dir exists
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        sys.exit(1)

    # Verify that all image files exist
    missing_files = []
    for image in data['images']:
        original_file_name = image['file_name']
        old_image_path = os.path.join(image_dir, original_file_name)
        if not os.path.isfile(old_image_path):
            missing_files.append(original_file_name)

    if missing_files:
        print("Error: The following image files referenced in the JSON do not exist in the image directory:")
        for filename in missing_files:
            print(f"  - {filename}")
        sys.exit(1)

    # First Pass: Rename all images to temporary filenames to avoid conflicts
    temp_suffix = "__temp__"
    temp_names = {}  # Mapping from new_id to temporary filename

    for new_id, image in enumerate(data['images']):
        old_id = image['id']
        image_id_map[old_id] = new_id
        image['id'] = new_id

        # Keep the original filename and extension
        original_file_name = image['file_name']
        file_extension = os.path.splitext(original_file_name)[1]  # e.g., .jpg, .png

        # Temporary file name
        temp_file_name = f"{temp_suffix}{new_id}{file_extension}"
        temp_image_path = os.path.join(image_dir, temp_file_name)
        old_image_path = os.path.join(image_dir, original_file_name)

        try:
            os.rename(old_image_path, temp_image_path)
            temp_names[new_id] = temp_file_name
        except Exception as e:
            print(f"Error renaming '{original_file_name}' to temporary name '{temp_file_name}': {e}")
            sys.exit(1)

    # Second Pass: Rename temporary filenames to final new filenames
    for new_id, image in enumerate(data['images']):
        temp_file_name = temp_names[new_id]
        temp_image_path = os.path.join(image_dir, temp_file_name)

        # New file name based on image ID
        new_file_name = f"{new_id}{os.path.splitext(temp_file_name)[1]}"
        new_image_path = os.path.join(image_dir, new_file_name)

        try:
            os.rename(temp_image_path, new_image_path)
            image['file_name'] = new_file_name
        except Exception as e:
            print(f"Error renaming temporary file '{temp_file_name}' to final name '{new_file_name}': {e}")
            sys.exit(1)

    # Update annotation IDs and image_id references
    for new_id, annotation in enumerate(data['annotations']):
        old_id = annotation['id']
        annotation_id_map[old_id] = new_id
        annotation['id'] = new_id

        # Update image_id reference
        old_image_id = annotation['image_id']
        if old_image_id not in image_id_map:
            print(f"Error: Annotation {new_id} references unknown image_id {old_image_id}")
            sys.exit(1)
        annotation['image_id'] = image_id_map[old_image_id]

    # Optionally, reindex category IDs if needed
    # This example assumes category IDs are already consecutive and start from 0
    # If not, you can add similar mapping logic as above

    # Write the updated data to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error writing to output JSON file '{output_file}': {e}")
        sys.exit(1)

    print(f"✅ Updated COCO JSON file has been saved to '{output_file}'")
    print("✅ Images have been renamed successfully and start from 0 without gaps.")

def main():
    parser = argparse.ArgumentParser(description="Update COCO JSON file IDs, filenames, and rename image files.")
    parser.add_argument('input', help="Path to the input COCO JSON file.")
    parser.add_argument('output', help="Path to save the output COCO JSON file.")
    parser.add_argument('image_dir', help="Directory where the images are stored.")

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    update_coco_ids_and_rename_images(args.input, args.output, args.image_dir)

if __name__ == "__main__":
    main()

