import json
import os
import cv2
import argparse

# Between YOLO and COCO
class_id_difference = 0

def coco_to_yolo(coco_annotation_file, output_dir, image_dir):
    # Load COCO annotations
    with open(coco_annotation_file) as f:
        coco_data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over images
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        image_path = os.path.join(image_dir, file_name)

        # Load image
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        valid_bboxes = []

        # Check annotations for the current image
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                x, y, w, h = annotation['bbox']
                class_id = annotation['category_id'] - class_id_difference

                # Convert COCO bounding box format to YOLO format
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                relative_width = w / img_width
                relative_height = h / img_height

                # Ignore boxes if center coords are > 1 or if they exceed image dimensions
                if center_x > 1 or center_y > 1:
                    continue
                if (x + w) > img_width or (y + h) > img_height:
                    continue

                # Add valid bbox to the list
                valid_bboxes.append(f"{class_id} {center_x} {center_y} {relative_width} {relative_height}\n")

        # Write YOLO annotation file only if there are valid bboxes
        if valid_bboxes:
            yolo_file = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
            with open(yolo_file, 'w') as f:
                f.writelines(valid_bboxes)

def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("-c", "--coco_annotation_file", required=True, help="Path to the COCO annotation file")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for the converted YOLO files")
    parser.add_argument("-i", "--image_dir", required=True, help="Directory containing the COCO images")
    args = parser.parse_args()

    coco_to_yolo(args.coco_annotation_file, args.output_dir, args.image_dir)

if __name__ == '__main__':
    main()
