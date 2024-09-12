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

        # Create YOLO annotation file
        yolo_file = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
        with open(yolo_file, 'w') as f:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    x, y, w, h = annotation['bbox']
                    class_id = annotation['category_id'] - class_id_difference

                    # Convert COCO bounding box format to YOLO format
                    center_x = (x + w / 2) / img.shape[1]
                    center_y = (y + h / 2) / img.shape[0]
                    relative_width = w / img.shape[1]
                    relative_height = h / img.shape[0]

                    # Write YOLO annotation to file
                    f.write(f"{class_id} {center_x} {center_y} {relative_width} {relative_height}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("-c", "--coco_annotation_file", required=True, help="Path to the COCO annotation file")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for the converted YOLO files")
    parser.add_argument("-i", "--image_dir", required=True, help="Directory containing the COCO images")
    args = parser.parse_args()

    coco_to_yolo(args.coco_annotation_file, args.output_dir, args.image_dir)

if __name__ == '__main__':
    main()
